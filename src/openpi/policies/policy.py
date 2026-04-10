from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
            self._has_flow_breakdown = False
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)
            self._has_flow_breakdown = all(
                hasattr(model, name)
                for name in (
                    "sample_actions_embed_prefix",
                    "sample_actions_prefix_prefill",
                    "sample_actions_flow_loop",
                )
            )
            if self._has_flow_breakdown:
                self._sample_actions_embed_prefix = nnx_utils.module_jit(model.sample_actions_embed_prefix)
                self._sample_actions_prefix_prefill = nnx_utils.module_jit(model.sample_actions_prefix_prefill)
                self._sample_actions_flow_loop = nnx_utils.module_jit(model.sample_actions_flow_loop)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        total_start = time.monotonic()
        inputs = jax.tree.map(lambda x: x, obs)

        inputs = self._input_transform(inputs)

        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)

        flow_preprocess_ms = None
        flow_noise_ms = None
        flow_prefix_embed_ms = None
        flow_prefix_prefill_ms = None
        flow_loop_ms = None

        if not self._is_pytorch_model and self._has_flow_breakdown:
            flow_preprocess_start = time.monotonic()
            flow_observation = _model.preprocess_observation(None, observation, train=False)
            flow_preprocess_ms = (time.monotonic() - flow_preprocess_start) * 1000

            flow_noise_start = time.monotonic()
            flow_num_steps = sample_kwargs.get("num_steps", 10)
            flow_noise = sample_kwargs.get("noise")
            if flow_noise is None:
                flow_noise = jax.random.normal(
                    sample_rng_or_pytorch_device,
                    (flow_observation.state.shape[0], self._model.action_horizon, self._model.action_dim),
                )
            flow_noise = jax.block_until_ready(flow_noise)
            flow_noise_ms = (time.monotonic() - flow_noise_start) * 1000

            flow_prefix_embed_start = time.monotonic()
            prefix_tokens, prefix_mask, prefix_ar_mask = self._sample_actions_embed_prefix(flow_observation)
            prefix_tokens, prefix_mask, prefix_ar_mask = jax.block_until_ready((prefix_tokens, prefix_mask, prefix_ar_mask))
            flow_prefix_embed_ms = (time.monotonic() - flow_prefix_embed_start) * 1000

            flow_prefix_prefill_start = time.monotonic()
            prefix_mask, kv_cache = self._sample_actions_prefix_prefill(prefix_tokens, prefix_mask, prefix_ar_mask)
            prefix_mask, kv_cache = jax.block_until_ready((prefix_mask, kv_cache))
            flow_prefix_prefill_ms = (time.monotonic() - flow_prefix_prefill_start) * 1000

            flow_loop_start = time.monotonic()
            sampled_actions = self._sample_actions_flow_loop(
                flow_observation,
                prefix_mask,
                kv_cache,
                flow_noise,
                num_steps=flow_num_steps,
            )
            sampled_actions = jax.block_until_ready(sampled_actions)
            flow_loop_ms = (time.monotonic() - flow_loop_start) * 1000
            sample_total_ms = flow_preprocess_ms + flow_noise_ms + flow_prefix_embed_ms + flow_prefix_prefill_ms + flow_loop_ms
        else:
            sample_start = time.monotonic()
            sampled_actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
            if self._is_pytorch_model:
                if hasattr(sampled_actions, "is_cuda") and sampled_actions.is_cuda:
                    torch.cuda.synchronize(device=sampled_actions.device)
            else:
                sampled_actions = jax.block_until_ready(sampled_actions)
            sample_total_ms = (time.monotonic() - sample_start) * 1000

        outputs = {
            "state": inputs["state"],
            "actions": sampled_actions,
        }

        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)

        total_ms = (time.monotonic() - total_start) * 1000
        outputs["policy_timing"] = {
            "infer_ms": sample_total_ms,
            "flow_preprocess_ms": flow_preprocess_ms,
            "flow_noise_ms": flow_noise_ms,
            "flow_prefix_embed_ms": flow_prefix_embed_ms,
            "flow_prefix_prefill_ms": flow_prefix_prefill_ms,
            "flow_loop_ms": flow_loop_ms,
            "flow_total_ms": sample_total_ms if flow_preprocess_ms is not None else None,
            "total_ms": total_ms,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
