#!/usr/bin/env bash

# Gradient-structure baseline: f=0, NoAttack, gamma in {1.0, 0.66, 0.33, 0.0},
# TrMean+NNM+ARC, 1 seed, 500 steps -- same reference hyperparameters as
# configs/geometry_baseline/*, but with store_client_vectors turned on so the
# raw per-client honest gradients (post-momentum, pre-pre-aggregation) get
# dumped every 100 steps for offline analysis (analysis/gradient_structure/).
#
# Meant to run on dclgpusrv over SSH (no RCP/docker needed).
#
# Usage (on dclgpusrv, from the repo root, inside tmux):
#   ./run_gradient_structure_ssh.sh [nb_jobs]

set -e

NB_JOBS="${1:-4}"

mkdir -p results/logs

CONFIGS=(
  "configs/gradient_structure/snn_atan12.json"
  "configs/gradient_structure/cnn_tanh.json"
  "configs/gradient_structure/cnn_relu.json"
)

echo "========================================="
echo "Gradient Structure Baseline (SNN atan12, CNN Tanh, CNN ReLU)"
echo "  Parallel jobs per config: ${NB_JOBS}"
echo "========================================="

for config in "${CONFIGS[@]}"; do
  tag=$(basename "${config}" .json)
  log_file="results/logs/gradient_structure_${tag}.log"
  echo "[$(date)] Starting ${config} -> ${log_file}"
  python run_snn_robust_sweeps.py \
    --config "${config}" \
    --distribute_gpus --nb_jobs "${NB_JOBS}" \
    > "${log_file}" 2>&1 || echo "  ${config} encountered errors, see ${log_file}"
  echo "[$(date)] Finished ${config}"
done

echo "All gradient structure configs finished."
