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
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        total_start = time.monotonic()

        # Make a copy since transformations may modify the inputs in place.
        input_copy_start = time.monotonic()
        inputs = jax.tree.map(lambda x: x, obs)
        input_copy_ms = (time.monotonic() - input_copy_start) * 1000

        input_transform_start = time.monotonic()
        inputs = self._input_transform(inputs)
        input_transform_ms = (time.monotonic() - input_transform_start) * 1000

        tensor_convert_start = time.monotonic()
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device
        tensor_convert_ms = (time.monotonic() - tensor_convert_start) * 1000

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        noise_prepare_start = time.monotonic()
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise
        noise_prepare_ms = (time.monotonic() - noise_prepare_start) * 1000

        observation_build_start = time.monotonic()
        observation = _model.Observation.from_dict(inputs)
        observation_build_ms = (time.monotonic() - observation_build_start) * 1000

        sample_start = time.monotonic()
        sampled_actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        sample_dispatch_ms = (time.monotonic() - sample_start) * 1000

        sample_sync_start = time.monotonic()
        if self._is_pytorch_model:
            if hasattr(sampled_actions, "is_cuda") and sampled_actions.is_cuda:
                torch.cuda.synchronize(device=sampled_actions.device)
        else:
            sampled_actions = jax.block_until_ready(sampled_actions)
        sample_sync_ms = (time.monotonic() - sample_sync_start) * 1000

        outputs = {
            "state": inputs["state"],
            "actions": sampled_actions,
        }

        to_numpy_start = time.monotonic()
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        to_numpy_ms = (time.monotonic() - to_numpy_start) * 1000

        output_transform_start = time.monotonic()
        outputs = self._output_transform(outputs)
        output_transform_ms = (time.monotonic() - output_transform_start) * 1000

        total_ms = (time.monotonic() - total_start) * 1000
        outputs["policy_timing"] = {
            "infer_ms": sample_dispatch_ms + sample_sync_ms,
            "input_copy_ms": input_copy_ms,
            "input_transform_ms": input_transform_ms,
            "tensor_convert_ms": tensor_convert_ms,
            "noise_prepare_ms": noise_prepare_ms,
            "observation_build_ms": observation_build_ms,
            "sample_dispatch_ms": sample_dispatch_ms,
            "sample_sync_ms": sample_sync_ms,
            "to_numpy_ms": to_numpy_ms,
            "output_transform_ms": output_transform_ms,
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
