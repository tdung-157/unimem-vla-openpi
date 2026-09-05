#!/bin/bash
#SBATCH --job-name=unimem_norm_stats
#SBATCH --nodelist=worker-0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-cpu=8192
#SBATCH --time=08:00:00
#SBATCH --output=logs/unimem_norm_stats_%j.out
#SBATCH --error=logs/unimem_norm_stats_%j.err
#
#   CONFIG=pi05_astribot_unimem_event_full sbatch custom_unimem/sbatch_norm_stats.sh
#
# Norm stats depend only on state/actions, so ONE run per robot covers all five of that
# robot's configs — compute_norm_stats.py copies the result into the shared assets
# directory every config reads from.
#
# FAST=1 uses the parquet-only path (no video decoding), which is what you want on the
# 817k-frame MOTION2 set; otherwise the stock path runs with --max-frames.

set -euo pipefail

# Slurm copies this script to the node's spool dir, so BASH_SOURCE cannot locate the checkout.
cd "${SLURM_SUBMIT_DIR:-/mnt/data/dungnt232_1/repos/unimem-vla-openpi}"

CONFIG="${CONFIG:-pi05_astribot_unimem_event_full}"
echo "[norm_stats] $(hostname) $(date -Is) cwd=$PWD config=$CONFIG fast=${FAST:-0}"

export PYTHONUNBUFFERED=1

if [[ "${FAST:-0}" == "1" ]]; then
  uv run python custom_unimem/compute_norm_stats_fast.py "$CONFIG" --verify "$@"
else
  uv run python custom_unimem/compute_norm_stats.py "$CONFIG" --max-frames "${MAX_FRAMES:-200000}" "$@"
fi
