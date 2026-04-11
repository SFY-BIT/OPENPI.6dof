# Piper Remote Inference

This repo can already act as the GPU-side `pi0.5` inference server for Piper.

What was missing is the robot-side runtime: camera capture, joint readback, and Piper SDK control over CAN. For that, use [`scripts/piper_remote_client.py`](../../scripts/piper_remote_client.py), which reuses the Piper runtime from your local `lerobot.act` checkout and talks to the openpi policy server over websocket.

## 1. Start the policy server on the GPU machine

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_piper_low_mem_finetune_warmup1k \
  --policy.dir=/path/to/your/pi05_piper_checkpoint
```

The current client is adapted for `joint3mask` checkpoints by default: before sending actions to Piper, it replaces `action[3]` with the current observed joint value.

## 2. Prepare the robot machine

The robot machine still needs the Piper stack from `lerobot.act`, including:

- `piper_sdk`
- CAN setup, for example `piper_scripts/can_activate.sh`
- the Piper robot runtime under `lerobot/common/robot_devices/...`

## 3. Run the Piper client on the robot machine

```bash
python scripts/piper_remote_client.py \
  --lerobot-root /mnt/hdd/sfy/lerobot.act \
  --server-host <gpu-server-ip> \
  --server-port 8000 \
  --task "your task instruction" \
  --fps 30 \
  --policy-fps 5
```

Running the client from the `lerobot` Python environment is usually the easiest path, because that environment already contains the Piper SDK and robot-side dependencies. The script injects the local `openpi-client` source tree automatically, so you do not need a full `openpi` install on the robot machine.

The client:

- reads `observation.images.one` and `observation.images.two`
- resizes them to `224 x 224`
- sends them together with the 7-D joint state to the openpi server
- receives the full returned action chunk from the server
- replays that chunk locally at `--fps`, with fresh server replans at `--policy-fps`
- linearly interpolates between chunk waypoints during local replay
- replaces action dimension `3` with the current joint value for `joint3mask` execution
- clamps per-step deltas before sending the command to Piper

Parameter meanings:

- `--fps`: local robot command rate on the robot machine
- `--policy-fps`: remote replan rate against the openpi policy server
- `--chunk-execute-steps`: optional cap on how many waypoints from each returned chunk are used during local replay; `0` uses the full chunk

## Notes

- The openpi side already supports Piper-shaped policy inputs/outputs through [`src/openpi/policies/piper_policy.py`](../../src/openpi/policies/piper_policy.py).
- The openpi websocket server is provided by [`scripts/serve_policy.py`](../../scripts/serve_policy.py).
- This workflow keeps hardware dependencies on the robot machine and keeps the model server on the GPU machine.
