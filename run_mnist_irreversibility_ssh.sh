#!/usr/bin/env bash

# EXP 1 - Irreversibility test, meant to run on dclgpusrv over SSH.
# Condition: MNIST, gamma=0.66, f=5, Sign Flipping, TrMean + ARC/NNM -- the
# collapse cell confirmed by the cliff sweep (results/snn/mnist_cliff_smoketest).
# 300 steps total, Byzantine clients removed at step 150 (well past the point
# the SNN plateaus at chance accuracy).
#
# Runs 4 conditions, each its own results_directory (no folder collisions):
#   - SNN recovery  (attack removed at step 150)
#   - SNN baseline  (attack continues the whole 300 steps, for comparison)
#   - CNN recovery  (ReLU control, same protocol)
#   - CNN baseline  (ReLU control, continued attack)
#
# Usage (on dclgpusrv, from the repo root, inside tmux):
#   ./run_mnist_irreversibility_ssh.sh [nb_jobs]

set -e

NB_JOBS="${1:-4}"

mkdir -p results/logs

CONFIGS=(
  "configs/snn_robustness/mnist_irreversibility_snn_recovery.json"
  "configs/snn_robustness/mnist_irreversibility_snn_baseline.json"
  "configs/snn_robustness/mnist_irreversibility_cnn_recovery.json"
  "configs/snn_robustness/mnist_irreversibility_cnn_baseline.json"
)

echo "========================================="
echo "MNIST Irreversibility Test (EXP 1)"
echo "  4 conditions, ${NB_JOBS} parallel jobs each"
echo "  Log: results/logs/mnist_irreversibility.log"
echo "========================================="

for CONFIG in "${CONFIGS[@]}"; do
  echo "[$(date)] Starting ${CONFIG}..."
  python run_snn_robust_sweeps.py \
    --config "${CONFIG}" \
    --distribute_gpus --nb_jobs "${NB_JOBS}" \
    >> results/logs/mnist_irreversibility.log 2>&1 || echo "${CONFIG} encountered errors"
done

echo "[$(date)] MNIST irreversibility test finished."
