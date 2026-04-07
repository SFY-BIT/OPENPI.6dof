import dataclasses
import functools
import logging
import os
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


def _tree_all_finite(pytree) -> jax.Array:
    leaves = jax.tree.leaves(pytree)
    if not leaves:
        return jnp.asarray(True)
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in leaves]))


def _summarize_array(name: str, value: Any) -> str:
    array = np.asarray(jax.device_get(value))
    finite_mask = np.isfinite(array) if np.issubdtype(array.dtype, np.number) else None
    parts = [f"{name}: shape={array.shape}, dtype={array.dtype}"]
    if finite_mask is not None:
        parts.append(f" finite={bool(finite_mask.all())}")
        parts.append(f" nan={int(np.isnan(array).sum())}")
        parts.append(f" inf={int(np.isinf(array).sum())}")
        finite_values = array[finite_mask]
        if finite_values.size:
            parts.append(
                " min={:.6g} max={:.6g} mean={:.6g}".format(
                    float(finite_values.min()),
                    float(finite_values.max()),
                    float(finite_values.mean()),
                )
            )
    return "".join(parts)


def _summarize_batch(batch: tuple[_model.Observation, _model.Actions]) -> str:
    observation, actions = batch
    summaries = []
    summaries.append(_summarize_array("observation.state", observation.state))
    summaries.append(_summarize_array("actions", actions))
    for name, image in observation.images.items():
        summaries.append(_summarize_array(f"observation.images[{name}]", image))
    for name, image_mask in observation.image_masks.items():
        summaries.append(_summarize_array(f"observation.image_masks[{name}]", image_mask))
    return "\n".join(summaries)


def _describe_debug_batch(raw_dataset, batch_size: int, batch_idx: int) -> str:
    sample_start = batch_idx * batch_size
    sample_end = sample_start + batch_size
    episode_ids = [int(raw_dataset[i]["episode_index"]) for i in range(sample_start, sample_end)]
    frame_ids = [int(raw_dataset[i]["frame_index"]) for i in range(sample_start, sample_end)]
    return (
        f"samples=[{sample_start},{sample_end}) "
        f"episodes=[{min(episode_ids)},{max(episode_ids)}] "
        f"frames=[{min(frame_ids)},{max(frame_ids)}]"
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)
    grad_norm = optax.global_norm(grads)
    finite_update = jnp.isfinite(loss) & jnp.isfinite(grad_norm) & _tree_all_finite(grads)

    current_kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(
        state,
        step=state.step + 1,
        params=jax.tree.map(lambda new, old: jnp.where(finite_update, new, old), new_params, state.params),
        opt_state=jax.tree.map(lambda new, old: jnp.where(finite_update, new, old), new_opt_state, state.opt_state),
    )
    if state.ema_decay is not None:
        updated_ema_params = jax.tree.map(
            lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
        )
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda new, old: jnp.where(finite_update, new, old), updated_ema_params, state.ema_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    param_norm = jnp.where(finite_update, optax.global_norm(kernel_params), optax.global_norm(current_kernel_params))
    info = {
        "loss": loss,
        "grad_norm": grad_norm,
        "param_norm": param_norm,
        "skipped_nonfinite": 1.0 - finite_update.astype(jnp.float32),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_config = config.data.create(config.assets_dirs, config.model)

    debug_disable_shuffle = os.environ.get("OPENPI_DEBUG_DISABLE_SHUFFLE") == "1"
    if debug_disable_shuffle:
        logging.warning("OPENPI_DEBUG_DISABLE_SHUFFLE=1, disabling data shuffling for reproducible debugging")

    debug_batch_meta_steps = int(os.environ.get("OPENPI_DEBUG_BATCH_META_STEPS", "0"))
    debug_raw_dataset = None
    debug_batches_per_epoch = None
    if debug_batch_meta_steps > 0 and debug_disable_shuffle and data_config.rlds_data_dir is None:
        debug_raw_dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
        effective_len = len(debug_raw_dataset) - (len(debug_raw_dataset) % config.batch_size)
        debug_batches_per_epoch = effective_len // config.batch_size if effective_len > 0 else None
        logging.warning(
            "OPENPI_DEBUG_BATCH_META_STEPS=%s, logging deterministic batch metadata for the first steps",
            debug_batch_meta_steps,
        )

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=not debug_disable_shuffle,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    reported_nonfinite = False
    for step in pbar:
        if (
            debug_raw_dataset is not None
            and debug_batches_per_epoch is not None
            and step < debug_batch_meta_steps
        ):
            batch_in_epoch = step % debug_batches_per_epoch
            logging.warning(
                "Debug batch meta at loop_step=%s: %s",
                step,
                _describe_debug_batch(debug_raw_dataset, config.batch_size, batch_in_epoch),
            )
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        skipped_nonfinite = float(jax.device_get(info["skipped_nonfinite"]))
        if skipped_nonfinite > 0.0 and not reported_nonfinite:
            batch_meta = ""
            if debug_raw_dataset is not None and debug_batches_per_epoch is not None:
                batch_in_epoch = step % debug_batches_per_epoch
                batch_meta = _describe_debug_batch(debug_raw_dataset, config.batch_size, batch_in_epoch)
            logging.warning(
                "Detected non-finite train step at loop_step=%s train_state_step=%s loss=%s grad_norm=%s%s\n%s",
                step,
                int(jax.device_get(train_state.step)),
                float(jax.device_get(info["loss"])),
                float(jax.device_get(info["grad_norm"])),
                f" batch_meta=({batch_meta})" if batch_meta else "",
                _summarize_batch(batch),
            )
            reported_nonfinite = True
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
