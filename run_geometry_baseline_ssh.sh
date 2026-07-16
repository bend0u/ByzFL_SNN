#!/usr/bin/env bash

# Gradient-geometry baseline (f=0, unattacked): 3 models (SNN atan, CNN ReLU,
# CNN Tanh) at their reference hyperparameters, gamma in {1.0,0.66,0.33,0.0},
# TrMean only (all aggregators see the same 10 honest vectors at f=0), 1 seed
# (f=0 is stable). Logs metrics_geometry.csv per run (online per-step
# consensus/dispersion/sign-agreement metrics) -- no raw vectors dumped.
# Meant to run on dclgpusrv over SSH.
#
# Usage (on dclgpusrv, from the repo root, inside tmux):
#   ./run_geometry_baseline_ssh.sh [nb_jobs]

set -e

NB_JOBS="${1:-3}"

mkdir -p results/logs

CONFIGS=(
  "configs/geometry_baseline/snn_atan12.json"
  "configs/geometry_baseline/cnn_relu.json"
  "configs/geometry_baseline/cnn_tanh.json"
)

echo "========================================="
echo "Gradient-Geometry Baseline (f=0, SNN atan / CNN ReLU / CNN Tanh)"
echo "  Parallel jobs per config: ${NB_JOBS}"
echo "========================================="

for config in "${CONFIGS[@]}"; do
  tag=$(basename "${config}" .json)
  log_file="results/logs/geometry_baseline_${tag}.log"
  echo "[$(date)] Starting ${config} -> ${log_file}"
  ./venv/bin/python3 run_snn_robust_sweeps.py \
    --config "${config}" \
    --distribute_gpus --nb_jobs "${NB_JOBS}" \
    > "${log_file}" 2>&1 || echo "  ${config} encountered errors, see ${log_file}"
  echo "[$(date)] Finished ${config}"
done

echo "All geometry baseline configs finished."
