#!/usr/bin/env bash
#
# Launcher for the MOTION2 UniMem ROS2 deploy node.
#
# The node runs under the SYSTEM ROS2 Python (rclpy, cv_bridge, upper_body_msgs), not
# under uv — the uv environment has JAX but no ROS.
#
#   MEMORY_MODE=text_keyframe ./custom_unimem/run_deploy_motion2.sh
#   HOST=192.168.1.5 PORT=8000 ./custom_unimem/run_deploy_motion2.sh
#
# MEMORY_MODE must match the trained checkpoint:
#   keyframe_* -> text_keyframe   event_* -> text   video_full -> video   plain -> none
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
M2_WS="${M2_WS:-$HOME/Documents/Work/m2_ws}"
PYTHON="${ROS_PYTHON:-/usr/bin/python3}"

if [[ ! -x "$PYTHON" ]]; then
  echo "[run_deploy] ERROR: python interpreter '$PYTHON' not found." >&2
  echo "[run_deploy]        Set ROS_PYTHON=/path/to/ros2/python and retry." >&2
  exit 1
fi
if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
  echo "[run_deploy] Note: conda env '$CONDA_DEFAULT_ENV' is active; using ROS2's $PYTHON"
  echo "[run_deploy]       ($("$PYTHON" --version 2>&1)) instead of the conda interpreter."
fi

if [[ ! -f "$ROS_SETUP" ]]; then
  echo "[run_deploy] ERROR: ROS2 setup not found at '$ROS_SETUP'." >&2
  echo "[run_deploy]        Set ROS_SETUP=/path/to/ros/setup.bash and retry." >&2
  exit 1
fi
source "$ROS_SETUP"

if [[ -f "$M2_WS/install/setup.bash" ]]; then
  source "$M2_WS/install/setup.bash"
else
  echo "[run_deploy] WARNING: '$M2_WS/install/setup.bash' not found." >&2
  echo "[run_deploy]          If upper_body_msgs is already on your ROS path this is fine;" >&2
  echo "[run_deploy]          otherwise build it and/or set M2_WS." >&2
fi

if ! "$PYTHON" -c "import openpi_client" 2>/dev/null; then
  echo "[run_deploy] Installing openpi-client into the ROS2 Python (user site) ..."
  "$PYTHON" -m pip install --user --no-warn-script-location -e "$REPO_ROOT/packages/openpi-client"
fi

ROS_ARGS=()
[[ -n "${HOST:-}" ]]            && ROS_ARGS+=(-p "host:=${HOST}")
[[ -n "${PORT:-}" ]]            && ROS_ARGS+=(-p "port:=${PORT}")
[[ -n "${PROMPT:-}" ]]          && ROS_ARGS+=(-p "prompt:=${PROMPT}")
[[ -n "${MEMORY_MODE:-}" ]]     && ROS_ARGS+=(-p "memory_mode:=${MEMORY_MODE}")
[[ -n "${ACTION_HORIZON:-}" ]]  && ROS_ARGS+=(-p "action_horizon:=${ACTION_HORIZON}")
[[ -n "${USE_COMPRESSED:-}" ]]  && ROS_ARGS+=(-p "use_compressed:=${USE_COMPRESSED}")
[[ -n "${VIDEO_NUM_FRAMES:-}" ]]    && ROS_ARGS+=(-p "video_num_frames:=${VIDEO_NUM_FRAMES}")
[[ -n "${VIDEO_STRIDE_FRAMES:-}" ]] && ROS_ARGS+=(-p "video_stride_frames:=${VIDEO_STRIDE_FRAMES}")
[[ -n "${EVENT_THRESHOLD:-}" ]] && ROS_ARGS+=(-p "event_confidence_threshold:=${EVENT_THRESHOLD}")

echo "[run_deploy] ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-<unset>}  memory_mode=${MEMORY_MODE:-text_keyframe}"
exec "$PYTHON" "$SCRIPT_DIR/deploy_motion2_ros2.py" --ros-args "${ROS_ARGS[@]}" "$@"
