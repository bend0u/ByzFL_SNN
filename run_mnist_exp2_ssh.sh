#!/usr/bin/env bash

# EXP 2 - Gain / firing-rate probe at the cliff, meant to run on dclgpusrv
# over SSH. Logs honest-client firing rate and the server's effective
# (post-attack, post-aggregation) gradient norm every step, for f=4 vs f=5
# at gamma=0.66 (the confirmed graceful-to-collapse boundary from the cliff
# sweep). Prediction (CLAUDE.md.txt): both metrics stay stable through f=4
# and crater at f=5, with the crater preceding the accuracy drop by 1-2
# rounds.
#
# Usage (on dclgpusrv, from the repo root, inside tmux):
#   ./run_mnist_exp2_ssh.sh [nb_jobs]

set -e

NB_JOBS="${1:-2}"

mkdir -p results/logs

echo "========================================="
echo "MNIST EXP2 Firing-Rate Probe (f=4 vs f=5, gamma=0.66)"
echo "  Parallel jobs: ${NB_JOBS}"
echo "  Log: results/logs/mnist_exp2_firing_rate_probe.log"
echo "========================================="

echo "[$(date)] Starting EXP2 firing-rate probe..."
python run_snn_robust_sweeps.py \
  --config configs/snn_robustness/mnist_exp2_firing_rate_probe.json \
  --distribute_gpus --nb_jobs "${NB_JOBS}" \
  > results/logs/mnist_exp2_firing_rate_probe.log 2>&1 || echo "EXP2 firing-rate probe encountered errors"
echo "[$(date)] EXP2 firing-rate probe finished."
