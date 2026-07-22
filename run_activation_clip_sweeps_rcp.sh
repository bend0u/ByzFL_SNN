#!/usr/bin/env bash

# RCP Sweep Runner: gradient-preserving activation clipping (STE, linear ramp,
# adaptive per-coordinate quantile clip) + adaptive client-side gradient-norm
# clipping, all on MNIST cnn_mnist family. Logs saved to results/logs/ so they
# persist on the dcl-scratch PVC. Meant to run inside the repo's Dockerfile
# container (see run_docker.sh).
#
# Usage (inside the RCP container, from the repo root):
#   ./run_activation_clip_sweeps_rcp.sh [nb_jobs]

set -e

NB_JOBS="${1:-50}"

mkdir -p results/logs

CONFIGS=(
  "configs/activation_clip/cnn_mnist_clip_ste_1.json"
  "configs/activation_clip/cnn_mnist_clip_ste_2.json"
  "configs/activation_clip/cnn_mnist_clip_ramp_1.json"
  "configs/activation_clip/cnn_mnist_clip_ramp_2.json"
  "configs/activation_clip/cnn_mnist_clip_qcoord_plain_080.json"
  "configs/activation_clip/cnn_mnist_clip_qcoord_plain_090.json"
  "configs/activation_clip/cnn_mnist_clip_qcoord_ste_080.json"
  "configs/activation_clip/cnn_mnist_clip_qcoord_ste_090.json"
  "configs/activation_clip/cnn_mnist_qclip_070.json"
  "configs/activation_clip/cnn_mnist_qclip_080.json"
)

echo "========================================="
echo "Starting Activation-Clip RCP Sweep"
echo "  - ${NB_JOBS} parallel jobs distributed across GPUs, per config"
echo "  - Logs saved to results/logs/"
echo "========================================="

for config in "${CONFIGS[@]}"; do
  tag=$(basename "${config}" .json)
  log_file="results/logs/actclip_${tag}.log"
  echo "[$(date)] Starting ${config} -> ${log_file}"
  python run_activation_clip_sweep.py \
    --config "${config}" \
    --distribute_gpus --nb_jobs "${NB_JOBS}" \
    > "${log_file}" 2>&1 || echo "  ${config} encountered errors, see ${log_file}"
  echo "[$(date)] Finished ${config}"
done

echo "========================================="
echo "[$(date)] All activation-clip RCP sweeps completed!"
echo "========================================="
