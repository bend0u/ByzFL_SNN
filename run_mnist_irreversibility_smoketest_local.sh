#!/usr/bin/env bash

# Local CPU smoke test for the EXP1 irreversibility test's byzantine_removal_step
# code path. Two tiny runs:
#   1. f=4 (below TrMean's crash threshold even without the f-reset fix)
#      -- validates the config field is read and the mid-loop mutation fires once.
#   2. f=5 (2f >= n at removal for nb_honest_clients=10, the exact crash case)
#      -- validates the aggregator/pre-aggregator f-reset actually prevents it.
# Not expected to show a real recovery/collapse signal at this tiny scale.
#
# Usage:
#   ./run_mnist_irreversibility_smoketest_local.sh

set -e

echo "========================================="
echo "MNIST Irreversibility Smoke Test (local, CPU)"
echo "  Run 1/2: f=4 (plumbing check)"
echo "========================================="
python run_snn_robust_sweeps.py \
  --config configs/snn_robustness/mnist_irreversibility_snn_recovery_local_f4.json \
  --nb_jobs 1

echo "========================================="
echo "  Run 2/2: f=5 (TrMean crash-fix check)"
echo "========================================="
python run_snn_robust_sweeps.py \
  --config configs/snn_robustness/mnist_irreversibility_snn_recovery_local_f5.json \
  --nb_jobs 1

echo "Smoke test finished."
