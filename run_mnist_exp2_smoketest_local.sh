#!/usr/bin/env bash

# Local CPU smoke test for EXP2's new instrumentation: honest-client firing
# rate (SNN spike hooks) and effective (post-aggregation) gradient norm.
# Tiny run (10 steps, f=4 and f=5 at gamma=0.66) -- just confirms the new
# honest_firing_rate_*.txt / effective_grad_norm_*.txt files are written
# with sane (non-crashing, non-all-NaN) values.
#
# Usage:
#   ./run_mnist_exp2_smoketest_local.sh

set -e

echo "========================================="
echo "MNIST EXP2 Firing-Rate Probe Smoke Test (local, CPU)"
echo "========================================="
python run_snn_robust_sweeps.py \
  --config configs/snn_robustness/mnist_exp2_firing_rate_probe_local.json \
  --nb_jobs 1

echo "Smoke test finished."
