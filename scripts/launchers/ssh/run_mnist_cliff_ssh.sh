#!/usr/bin/env bash

# MNIST SNN Byzantine-cliff sweep, meant to run on dclgpusrv over SSH
# (plain venv + GPU, no docker/runai). Locates the (gamma, f) cell(s) where
# SNN Atan test accuracy collapses to chance under Sign Flipping + ARC/NNM +
# Trimmed Mean.
#
# Usage (on dclgpusrv, from the repo root, inside tmux):
#   ./run_mnist_cliff_ssh.sh [config] [nb_jobs]

set -e

CONFIG="${1:-configs/snn_robustness/mnist_cliff_smoketest.json}"
NB_JOBS="${2:-8}"

mkdir -p results/logs

echo "========================================="
echo "MNIST SNN Cliff Sweep"
echo "  Config: ${CONFIG}"
echo "  Parallel jobs: ${NB_JOBS}"
echo "  Log: results/logs/mnist_cliff_sweep.log"
echo "========================================="

echo "[$(date)] Starting MNIST cliff sweep..."
python run_snn_robust_sweeps.py \
  --config "${CONFIG}" \
  --distribute_gpus --nb_jobs "${NB_JOBS}" \
  > results/logs/mnist_cliff_sweep.log 2>&1 || echo "MNIST cliff sweep encountered errors"
echo "[$(date)] MNIST cliff sweep finished."
