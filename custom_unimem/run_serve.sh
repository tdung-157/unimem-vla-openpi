#!/usr/bin/env bash
#
# One-shot launcher for the UniMem policy SERVER.
#
# Usage:
#   ./custom_unimem/run_serve.sh
#   CONFIG=pi05_astribot_unimem_keyframe MODEL_DIR=checkpoints/.../100000 ./custom_unimem/run_serve.sh
#   PORT=8001 ./custom_unimem/run_serve.sh
#
# CONFIG must be the config the checkpoint was TRAINED with: create_trained_policy()
# compares it against the checkpoint's recorded shape (video_encoder / num_frames /
# event_tracking) and refuses a mismatch rather than silently serving the wrong history
# layout.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONFIG="${CONFIG:-pi05_astribot_unimem_keyframe}"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/checkpoints/$CONFIG/coffee_keyframe/100000}"
PORT="${PORT:-8000}"
# Serving loads the checkpoint from MODEL_DIR directly, but the PaliGemma tokenizer still
# resolves through the openpi cache (gs://big_vision/paligemma_tokenizer.model). Only set a
# default when that cluster cache exists; elsewhere openpi falls back to ~/.cache/openpi.
[[ -z "${OPENPI_DATA_HOME:-}" && -d /mnt/data/dungnt232_1/openpi_cache ]] \
  && export OPENPI_DATA_HOME=/mnt/data/dungnt232_1/openpi_cache

if [[ ! -d "$MODEL_DIR/params" ]]; then
  echo "[run_serve] ERROR: '$MODEL_DIR' has no params/ dir — not an openpi checkpoint." >&2
  echo "[run_serve]        Set MODEL_DIR=/path/to/checkpoint and retry." >&2
  exit 1
fi

echo "[run_serve] config=$CONFIG  port=$PORT"
echo "[run_serve] model=$MODEL_DIR"

cd "$REPO_ROOT"
exec uv run python custom_unimem/serve.py \
  --port "$PORT" \
  policy:checkpoint \
  --policy.config="$CONFIG" \
  --policy.dir="$MODEL_DIR" \
  "$@"
