import argparse
import logging
import pathlib
import sys
import time
from typing import Any

import numpy as np
import torch

OPENPI_ROOT = pathlib.Path(__file__).resolve().parents[1]
OPENPI_CLIENT_SRC = OPENPI_ROOT / "packages" / "openpi-client" / "src"
if str(OPENPI_CLIENT_SRC) not in sys.path:
    sys.path.insert(0, str(OPENPI_CLIENT_SRC))

from openpi_client import image_tools
from openpi_client import websocket_client_policy


DEFAULT_LEROBOT_ROOT = pathlib.Path("/mnt/hdd/sfy/lerobot.act")
DEFAULT_BASE_IMAGE_KEY = "observation.images.one"
DEFAULT_WRIST_IMAGE_KEY = "observation.images.two"
DEFAULT_STATE_KEY = "observation.state"
DEFAULT_MASKED_JOINT_INDEX = 3


def _load_make_robot(lerobot_root: pathlib.Path):
    if not lerobot_root.exists():
        raise FileNotFoundError(f"LeRobot root does not exist: {lerobot_root}")

    sys.path.insert(0, str(lerobot_root))

    try:
        from lerobot.common.robot_devices.robots.utils import make_robot
    except ImportError as exc:
        raise ImportError(
            "Failed to import Piper runtime from lerobot.act. "
            f"Make sure {lerobot_root} is the repo root and its dependencies are installed."
        ) from exc

    return make_robot


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _prepare_image(value: Any, image_size: int) -> np.ndarray:
    image = _to_numpy(value)
    image = image_tools.resize_with_pad(image, image_size, image_size)
    return image_tools.convert_to_uint8(image)


def _extract_first_action(action_chunk: Any) -> np.ndarray:
    action_chunk = _to_numpy(action_chunk)
    if action_chunk.ndim == 1:
        return action_chunk.astype(np.float32)
    if action_chunk.ndim != 2:
        raise ValueError(f"Expected action chunk rank 1 or 2, got shape {action_chunk.shape}")
    return action_chunk[0].astype(np.float32)


def _extract_action_chunk(action_chunk: Any) -> np.ndarray:
    action_chunk = _to_numpy(action_chunk).astype(np.float32)
    if action_chunk.ndim == 1:
        return action_chunk[None, :]
    if action_chunk.ndim != 2:
        raise ValueError(f"Expected action chunk rank 1 or 2, got shape {action_chunk.shape}")
    return action_chunk


def _read_current_state(robot: Any, expected_state_dim: int) -> np.ndarray:
    if not hasattr(robot, "arm") or not hasattr(robot.arm, "read"):
        raise AttributeError("Piper remote client expects the robot runtime to expose robot.arm.read().")

    state = np.asarray(list(robot.arm.read().values()), dtype=np.float32)
    if expected_state_dim > 0 and state.shape[0] != expected_state_dim:
        raise ValueError(f"Expected {expected_state_dim}-D robot state, got {state.shape[0]}")
    return state


def _clamp_action_delta(
    target_action: np.ndarray,
    current_state: np.ndarray,
    *,
    max_abs_joint_delta: float | None,
    max_abs_gripper_delta: float | None,
) -> tuple[np.ndarray, bool]:
    if target_action.shape != current_state.shape:
        raise ValueError(
            f"Target action shape {target_action.shape} does not match current state shape {current_state.shape}"
        )

    clamped = np.array(target_action, copy=True)
    was_clamped = False
    gripper_index = clamped.shape[0] - 1

    for idx, (target_value, current_value) in enumerate(zip(target_action, current_state, strict=True)):
        max_delta = max_abs_gripper_delta if idx == gripper_index else max_abs_joint_delta
        if max_delta is None or max_delta <= 0:
            continue

        bounded = np.clip(float(target_value), float(current_value - max_delta), float(current_value + max_delta))
        if bounded != float(target_value):
            was_clamped = True
        clamped[idx] = bounded

    return clamped, was_clamped


def _build_policy_observation(
    observation: dict[str, Any],
    task: str,
    image_size: int,
    expected_state_dim: int,
) -> dict[str, Any]:
    missing_keys = [
        key
        for key in (DEFAULT_BASE_IMAGE_KEY, DEFAULT_WRIST_IMAGE_KEY, DEFAULT_STATE_KEY)
        if key not in observation
    ]
    if missing_keys:
        raise KeyError(f"Observation is missing required keys: {missing_keys}")

    state = _to_numpy(observation[DEFAULT_STATE_KEY]).astype(np.float32)
    if state.ndim != 1:
        raise ValueError(f"Expected 1-D state, got shape {state.shape}")
    if expected_state_dim > 0 and state.shape[0] != expected_state_dim:
        raise ValueError(f"Expected {expected_state_dim}-D state for openpi Piper policy, got {state.shape[0]}")

    return {
        "observation/image": _prepare_image(observation[DEFAULT_BASE_IMAGE_KEY], image_size),
        "observation/wrist_image": _prepare_image(observation[DEFAULT_WRIST_IMAGE_KEY], image_size),
        "observation/state": state,
        "prompt": task,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Piper robot locally and query an openpi websocket policy server remotely."
    )
    parser.add_argument(
        "--lerobot-root",
        type=pathlib.Path,
        default=DEFAULT_LEROBOT_ROOT,
        help="Path to the local lerobot.act repository that contains the Piper robot runtime.",
    )
    parser.add_argument("--server-host", default="127.0.0.1", help="Hostname or IP of the openpi policy server.")
    parser.add_argument("--server-port", type=int, default=8000, help="Port of the openpi policy server.")
    parser.add_argument("--task", required=True, help="Task prompt to send to the policy.")
    parser.add_argument("--fps", type=float, default=5.0, help="Requested local control loop frequency.")
    parser.add_argument("--duration-s", type=float, default=60.0, help="How long to run the control loop.")
    parser.add_argument("--robot-type", default="piper", help="Robot type exposed by lerobot.act.")
    parser.add_argument(
        "--chunk-execute-steps",
        type=int,
        default=5,
        help="Maximum number of actions to execute from each returned action chunk before refreshing from the server.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Resize both camera images to this square size before sending them to the server.",
    )
    parser.add_argument(
        "--expected-state-dim",
        type=int,
        default=7,
        help="Expected robot state dimension for the served openpi policy. Piper configs in this repo use 7.",
    )
    parser.add_argument(
        "--hold-joint-index",
        type=int,
        default=DEFAULT_MASKED_JOINT_INDEX,
        help=(
            "Action index to overwrite with the current joint value before sending to the robot. "
            "Defaults to 3 for joint3mask checkpoints."
        ),
    )
    parser.add_argument(
        "--max-abs-joint-delta",
        type=float,
        default=0.2,
        help="Maximum per-step absolute change for each arm joint target relative to the current state.",
    )
    parser.add_argument(
        "--max-abs-gripper-delta",
        type=float,
        default=0.02,
        help="Maximum per-step absolute change for the gripper target relative to the current state.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    make_robot = _load_make_robot(args.lerobot_root.resolve())
    robot = make_robot(args.robot_type, inference_time=True)
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.server_host, port=args.server_port)

    logging.info("Connected to policy server: %s", policy.get_server_metadata())
    logging.info(
        "Applying joint3mask execution rule: action[%s] will be replaced with the current joint value.",
        args.hold_joint_index,
    )
    logging.info(
        "Action chunks will be executed sequentially before the next server inference request (max %s steps).",
        args.chunk_execute_steps,
    )

    robot.connect()
    try:
        start_t = time.perf_counter()
        step_idx = 0
        pending_actions: np.ndarray | None = None
        pending_action_index = 0
        latest_policy_timing: dict[str, Any] = {}
        latest_server_timing: dict[str, Any] = {}
        latest_client_timing: dict[str, float] = {}
        raw_chunk_len = 0
        executed_chunk_len = 0

        while time.perf_counter() - start_t < args.duration_s:
            loop_start_t = time.perf_counter()
            infer_requested = False
            capture_ms = 0.0
            prepare_ms = 0.0
            read_state_ms = 0.0
            policy_roundtrip_ms = 0.0
            chunk_extract_ms = 0.0
            action_send_ms = 0.0
            clamp_ms = 0.0

            if pending_actions is None or pending_action_index >= len(pending_actions):
                capture_start_t = time.perf_counter()
                observation = robot.capture_observation()
                capture_ms = (time.perf_counter() - capture_start_t) * 1000

                prepare_start_t = time.perf_counter()
                policy_observation = _build_policy_observation(
                    observation,
                    args.task,
                    args.image_size,
                    args.expected_state_dim,
                )
                prepare_ms = (time.perf_counter() - prepare_start_t) * 1000
                current_state = np.asarray(policy_observation["observation/state"], dtype=np.float32)

                policy_start_t = time.perf_counter()
                policy_result = policy.infer(policy_observation)
                policy_roundtrip_ms = (time.perf_counter() - policy_start_t) * 1000

                extract_start_t = time.perf_counter()
                pending_actions = _extract_action_chunk(policy_result["actions"])
                raw_chunk_len = len(pending_actions)
                if args.chunk_execute_steps > 0:
                    pending_actions = pending_actions[: args.chunk_execute_steps]
                executed_chunk_len = len(pending_actions)
                chunk_extract_ms = (time.perf_counter() - extract_start_t) * 1000
                pending_action_index = 0
                latest_policy_timing = policy_result.get("policy_timing", {})
                latest_server_timing = policy_result.get("server_timing", {})
                latest_client_timing = {
                    "capture_ms": capture_ms,
                    "prepare_ms": prepare_ms,
                    "policy_roundtrip_ms": policy_roundtrip_ms,
                    "chunk_extract_ms": chunk_extract_ms,
                }
                infer_requested = True
                logging.info(
                    "Fetched action chunk from policy server raw_chunk_len=%s executed_chunk_len=%s "
                    "capture_ms=%.2f prepare_ms=%.2f policy_roundtrip_ms=%.2f chunk_extract_ms=%.2f "
                    "server_infer_ms=%s policy_total_ms=%s "
                    "flow_prefix_image_embed_ms=%s flow_prompt_embed_ms=%s flow_prefix_concat_ms=%s "
                    "flow_prefix_embed_ms=%s "
                    "flow_prefix_prefill_ms=%s flow_loop_ms=%s flow_total_ms=%s",
                    raw_chunk_len,
                    executed_chunk_len,
                    capture_ms,
                    prepare_ms,
                    policy_roundtrip_ms,
                    chunk_extract_ms,
                    latest_server_timing.get("infer_ms"),
                    latest_policy_timing.get("total_ms"),
                    latest_policy_timing.get("flow_prefix_image_embed_ms"),
                    latest_policy_timing.get("flow_prompt_embed_ms"),
                    latest_policy_timing.get("flow_prefix_concat_ms"),
                    latest_policy_timing.get("flow_prefix_embed_ms"),
                    latest_policy_timing.get("flow_prefix_prefill_ms"),
                    latest_policy_timing.get("flow_loop_ms"),
                    latest_policy_timing.get("flow_total_ms"),
                )
            else:
                read_state_start_t = time.perf_counter()
                current_state = _read_current_state(robot, args.expected_state_dim)
                read_state_ms = (time.perf_counter() - read_state_start_t) * 1000

            action = np.array(pending_actions[pending_action_index], copy=True)
            chunk_step = pending_action_index + 1
            chunk_size = len(pending_actions)
            pending_action_index += 1

            if action.shape != current_state.shape:
                raise ValueError(
                    f"Policy returned {action.shape[0]} action dims, but Piper runtime is using {current_state.shape[0]}"
                )

            if not 0 <= args.hold_joint_index < action.shape[0]:
                raise ValueError(
                    f"hold_joint_index={args.hold_joint_index} is out of range for action dimension {action.shape[0]}"
                )
            action[args.hold_joint_index] = current_state[args.hold_joint_index]

            clamp_start_t = time.perf_counter()
            safe_action, was_clamped = _clamp_action_delta(
                action,
                current_state,
                max_abs_joint_delta=args.max_abs_joint_delta,
                max_abs_gripper_delta=args.max_abs_gripper_delta,
            )
            clamp_ms = (time.perf_counter() - clamp_start_t) * 1000
            if was_clamped:
                logging.warning(
                    "step=%s action clamp triggered | current_state=%s raw_action=%s safe_action=%s",
                    step_idx,
                    current_state.tolist(),
                    action.tolist(),
                    safe_action.tolist(),
                )

            action_send_start_t = time.perf_counter()
            executed_action = robot.send_action(torch.tensor(safe_action, dtype=torch.float32))
            action_send_ms = (time.perf_counter() - action_send_start_t) * 1000
            executed_action = _to_numpy(executed_action).astype(np.float32)

            if executed_action.shape == safe_action.shape and not np.allclose(executed_action, safe_action):
                logging.warning(
                    "step=%s robot-level action clip triggered | safe_action=%s executed_action=%s",
                    step_idx,
                    safe_action.tolist(),
                    executed_action.tolist(),
                )

            elapsed_s = time.perf_counter() - loop_start_t
            sleep_s = 0.0
            if args.fps > 0:
                sleep_s = max(0.0, (1.0 / args.fps) - elapsed_s)
                if sleep_s > 0:
                    time.sleep(sleep_s)

            loop_compute_ms = elapsed_s * 1000
            cycle_ms = (time.perf_counter() - loop_start_t) * 1000
            effective_hz = 1000.0 / cycle_ms if cycle_ms > 0 else float("inf")
            network_overhead_ms = None
            if infer_requested and latest_server_timing.get("total_ms") is not None:
                network_overhead_ms = policy_roundtrip_ms - float(latest_server_timing["total_ms"])
            elif infer_requested and latest_server_timing.get("infer_ms") is not None:
                network_overhead_ms = policy_roundtrip_ms - float(latest_server_timing["infer_ms"])

            logging.info(
                "step=%s chunk_step=%s/%s infer_requested=%s raw_chunk_len=%s executed_chunk_len=%s "
                "cycle_ms=%.2f effective_hz=%.2f loop_compute_ms=%.2f sleep_ms=%.2f "
                "capture_ms=%.2f prepare_ms=%.2f read_state_ms=%.2f policy_roundtrip_ms=%.2f "
                "server_infer_ms=%s policy_ms=%s policy_total_ms=%s "
                "flow_prefix_image_embed_ms=%s flow_prompt_embed_ms=%s flow_prefix_concat_ms=%s "
                "flow_prefix_embed_ms=%s "
                "flow_prefix_prefill_ms=%s flow_loop_ms=%s flow_total_ms=%s "
                "network_overhead_ms=%s chunk_extract_ms=%.2f clamp_ms=%.2f action_send_ms=%.2f",
                step_idx,
                chunk_step,
                chunk_size,
                infer_requested,
                raw_chunk_len,
                executed_chunk_len,
                cycle_ms,
                effective_hz,
                loop_compute_ms,
                sleep_s * 1000,
                latest_client_timing.get("capture_ms", capture_ms),
                latest_client_timing.get("prepare_ms", prepare_ms),
                read_state_ms,
                latest_client_timing.get("policy_roundtrip_ms", policy_roundtrip_ms),
                latest_server_timing.get("infer_ms"),
                latest_policy_timing.get("infer_ms"),
                latest_policy_timing.get("total_ms"),
                latest_policy_timing.get("flow_prefix_image_embed_ms"),
                latest_policy_timing.get("flow_prompt_embed_ms"),
                latest_policy_timing.get("flow_prefix_concat_ms"),
                latest_policy_timing.get("flow_prefix_embed_ms"),
                latest_policy_timing.get("flow_prefix_prefill_ms"),
                latest_policy_timing.get("flow_loop_ms"),
                latest_policy_timing.get("flow_total_ms"),
                network_overhead_ms,
                latest_client_timing.get("chunk_extract_ms", chunk_extract_ms),
                clamp_ms,
                action_send_ms,
            )

            step_idx += 1
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
