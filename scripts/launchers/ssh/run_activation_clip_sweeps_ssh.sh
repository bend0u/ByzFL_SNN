#!/usr/bin/env bash

# Gradient-preserving activation clipping (STE, linear ramp, adaptive
# per-coordinate quantile clip) + adaptive client-side gradient-norm clipping,
# all on MNIST cnn_mnist family. Meant to run on dclgpusrv over SSH, inside tmux.
#
# Usage (on dclgpusrv, from the repo root, inside tmux):
#   ./run_activation_clip_sweeps_ssh.sh [nb_jobs]

set -e

NB_JOBS="${1:-4}"

mkdir -p results/logs

CONFIGS=(
  "configs/activation_clip/cnn_mnist_clip_ste_1.json"
  "configs/activation_clip/cnn_mnist_clip_ste_2.json"
  "configs/activation_clip/cnn_mnist_clip_ramp_1.json"
  "configs/activation_clip/cnn_mnist_clip_ramp_2.json"
  "configs/activation_clip/cnn_mnist_clip_qcoord_plain_080.json"
  "configs/activation_clip/cnn_mnist_clip_qcoord_plain_090.json"
  "configs/activation_clip/cnn_mnist_qclip_070.json"
  # cnn_mnist_qclip_080 intentionally NOT here -- run separately/early via
  # run_qclip080_rcp.sh / run_qclip080_ssh.sh (its own job) to avoid duplicate
  # work / a write race on the same result folders.
)

echo "========================================="
echo "Activation-Clip Sweep (SSH / dclgpusrv)"
echo "  Parallel jobs per config: ${NB_JOBS}"
echo "========================================="

for config in "${CONFIGS[@]}"; do
  tag=$(basename "${config}" .json)
  log_file="results/logs/actclip_${tag}.log"
  echo "[$(date)] Starting ${config} -> ${log_file}"
  ./venv/bin/python3 scripts/experiments/run_activation_clip_sweep.py \
    --config "${config}" \
    --distribute_gpus --nb_jobs "${NB_JOBS}" \
    > "${log_file}" 2>&1 || echo "  ${config} encountered errors, see ${log_file}"
  echo "[$(date)] Finished ${config}"
done

echo "All activation-clip sweep configs finished."
