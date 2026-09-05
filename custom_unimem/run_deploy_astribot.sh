#!/usr/bin/env bash
#
# Launcher for the Astribot UniMem ROS2 deploy node.
#
#   CAMERA_MODE=real MEMORY_MODE=text_keyframe ./custom_unimem/run_deploy_astribot.sh
#   CAMERA_MODE=sim ./custom_unimem/run_deploy_astribot.sh
#
# CAMERA_MODE picks the camera topics + transport: 'sim' publishes raw Image on the
# whole-body topics, 'real' publishes JPEG CompressedImage on the camera topics.
# MEMORY_MODE must match the trained checkpoint:
#   keyframe_* -> text_keyframe   event_* -> text   video_full -> video   plain -> none
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
ASTRIBOT_WS="${ASTRIBOT_WS:-$HOME/Documents/Work/vm_astribot}"
CAMERA_MODE="${CAMERA_MODE:-sim}"
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
  exit 1
fi
source "$ROS_SETUP"

if [[ -f "$ASTRIBOT_WS/install/setup.bash" ]]; then
  source "$ASTRIBOT_WS/install/setup.bash"
else
  echo "[run_deploy] WARNING: '$ASTRIBOT_WS/install/setup.bash' not found." >&2
  echo "[run_deploy]          If astribot_msgs is already on your ROS path this is fine;" >&2
  echo "[run_deploy]          otherwise build it in vm_astribot and/or set ASTRIBOT_WS." >&2
fi

if ! "$PYTHON" -c "import openpi_client" 2>/dev/null; then
  echo "[run_deploy] Installing openpi-client into the ROS2 Python (user site) ..."
  "$PYTHON" -m pip install --user --no-warn-script-location -e "$REPO_ROOT/packages/openpi-client"
fi

ROS_ARGS=()
case "$CAMERA_MODE" in
  sim)
    ROS_ARGS+=(-p "camera_transport:=raw")
    ROS_ARGS+=(-p "head_image_topic:=/astribot_whole_body/camera/head_rgbd/image_raw")
    ROS_ARGS+=(-p "left_wrist_image_topic:=/astribot_whole_body/camera/left_wrist_rgbd/image_raw")
    ROS_ARGS+=(-p "right_wrist_image_topic:=/astribot_whole_body/camera/right_wrist_rgbd/image_raw")
    ;;
  real)
    ROS_ARGS+=(-p "camera_transport:=compressed")
    ROS_ARGS+=(-p "head_image_topic:=/astribot_camera/head_rgbd/color_compress/compressed")
    ROS_ARGS+=(-p "left_wrist_image_topic:=/astribot_camera/left_wrist_rgbd/color_compress/compressed")
    ROS_ARGS+=(-p "right_wrist_image_topic:=/astribot_camera/right_wrist_rgbd/color_compress/compressed")
    ;;
  *)
    echo "[run_deploy] ERROR: CAMERA_MODE must be 'sim' or 'real' (got '$CAMERA_MODE')." >&2
    exit 1
    ;;
esac

[[ -n "${HOST:-}" ]]                && ROS_ARGS+=(-p "host:=${HOST}")
[[ -n "${PORT:-}" ]]                && ROS_ARGS+=(-p "port:=${PORT}")
[[ -n "${PROMPT:-}" ]]              && ROS_ARGS+=(-p "prompt:=${PROMPT}")
[[ -n "${MEMORY_MODE:-}" ]]         && ROS_ARGS+=(-p "memory_mode:=${MEMORY_MODE}")
[[ -n "${ACTION_HORIZON:-}" ]]      && ROS_ARGS+=(-p "action_horizon:=${ACTION_HORIZON}")
[[ -n "${VIDEO_NUM_FRAMES:-}" ]]    && ROS_ARGS+=(-p "video_num_frames:=${VIDEO_NUM_FRAMES}")
[[ -n "${VIDEO_STRIDE_FRAMES:-}" ]] && ROS_ARGS+=(-p "video_stride_frames:=${VIDEO_STRIDE_FRAMES}")
[[ -n "${EVENT_THRESHOLD:-}" ]]     && ROS_ARGS+=(-p "event_confidence_threshold:=${EVENT_THRESHOLD}")

echo "[run_deploy] CAMERA_MODE=$CAMERA_MODE  ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-<unset>}  memory_mode=${MEMORY_MODE:-text_keyframe}"
exec "$PYTHON" "$SCRIPT_DIR/deploy_astribot_ros2.py" --ros-args "${ROS_ARGS[@]}" "$@"
