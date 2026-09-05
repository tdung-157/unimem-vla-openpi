#!/bin/bash
#SBATCH --job-name=unimem_train
#SBATCH --nodelist=worker-0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-cpu=8192
#SBATCH --time=480:00:00
#SBATCH --output=logs/unimem_train_%j.out
#SBATCH --error=logs/unimem_train_%j.err
#
# Compute norm stats first, then submit:
#   CONFIG=pi05_astribot_unimem_event FAST=1 sbatch custom_unimem/sbatch_norm_stats.sh
#   CONFIG=pi05_astribot_unimem_keyframe sbatch custom_unimem/sbatch_train.sh --exp-name=coffee_keyframe
#
# Extra args pass through to train.py (--exp-name=..., --resume, --overwrite,
# --num-train-steps=..., --batch-size=...).
#
# GPU count: train.py rejects a run unless batch_size % gpus == 0, and make_mesh rejects
# it unless gpus % fsdp_devices == 0. FSDP_DEVICES below is derived from the allocation
# and passed on the command line, so it always overrides the config's fsdp_devices field.
#   *_keyframe       batch 44, LoRA   -> upstream's number; 4 frames x 3 cameras a sample
#   *_event          batch 44, LoRA   -> single frame
#   *_video          batch 44, LoRA   -> 4 frames x 3 cameras, no event head
#   *_keyframe_full  batch 16, 2 GPUs ->  8 samples x 12 images per device (full FT + EMA)
#   *_event_full     batch 32, 2 GPUs -> 16 samples x  3 images per device (full FT + EMA)
# The LoRA rows leave fsdp_devices=1 (upstream's default) and run data-parallel; this
# script passes --fsdp-devices=$SLURM_GPUS_ON_NODE, which is harmless for them.
# Halve batch_size (--batch-size=...) on OOM; keep it divisible by the GPU count.

set -euo pipefail

# Slurm copies this script to the node's spool dir, so BASH_SOURCE cannot locate the checkout.
cd "${SLURM_SUBMIT_DIR:-/mnt/data/dungnt232_1/repos/unimem-vla-openpi}"

NUM_WORKERS="$(( ${SLURM_CPUS_PER_TASK:-8} > 4 ? ${SLURM_CPUS_PER_TASK:-8} - 4 : 1 ))"
FSDP_DEVICES="${SLURM_GPUS_ON_NODE:-1}"

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

# train.py writes its loss lines with tqdm.write -> stdout. Slurm redirects stdout to a
# file, so Python block-buffers it at 8 KB and the metrics only appear every few hours.
# W&B is disabled (the compute nodes cannot reach api.wandb.ai), so this log is the only
# place action_loss / event_loss land.
# Where openpi resolves gs:// assets. gs://openpi-assets/checkpoints/pi05_base/params maps to
# $OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base/params, which is already populated on
# this cluster — the compute nodes have no internet, so without this the first step would
# hang trying to fetch ~10 GB of base weights into ~/.cache/openpi.
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/mnt/data/dungnt232_1/openpi_cache}"

export PYTHONUNBUFFERED=1

# One thread per worker. Torch defaults to one thread PER CORE in every worker process,
# so 26 workers on a 30-core node would fight over ~780 threads. The GPUs do the compute;
# the workers only decode video and collate.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "[train] $(hostname) $(date -Is) cwd=$PWD workers=$NUM_WORKERS fsdp=$FSDP_DEVICES"

uv run python custom_unimem/train.py "${CONFIG:-pi05_astribot_unimem_keyframe}" \
  --num-workers "$NUM_WORKERS" \
  --fsdp-devices "$FSDP_DEVICES" \
  --no-wandb-enabled \
  "$@"
