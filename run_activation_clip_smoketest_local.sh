#!/usr/bin/env bash

# Local, CPU-only, end-to-end smoke test for the two new code paths before
# submitting the full sweeps remotely:
#   1. An activation variant that exercises the adaptive per-coordinate quantile
#      clip with the plain (true clamp) backward (cnn_mnist_clip_qcoord_plain_080).
#   2. The adaptive client-side gradient-norm clip on a plain cnn_mnist
#      (grad_clip_quantile=0.7, a tiny grad_clip_window=5 so clipping actually
#      kicks in within the smoke run's 20 steps).
# Tiny nb_steps/nb_training_seeds/f so this runs in well under a minute on CPU.
#
# Usage (from the repo root):
#   ./run_activation_clip_smoketest_local.sh

set -e

echo "========================================="
echo "Activation-clip smoke test (local, CPU)"
echo "========================================="

echo "[$(date)] Smoke test 1/2: adaptive per-coordinate quantile clip (plain, tau=0.8)"
python run_activation_clip_sweep.py \
  --config configs/activation_clip/_smoketest_qcoord_plain_local.json \
  --nb_jobs 1

echo "[$(date)] Smoke test 2/2: adaptive client-side gradient-norm clip (tau=0.7)"
python run_activation_clip_sweep.py \
  --config configs/activation_clip/_smoketest_qclip_local.json \
  --nb_jobs 1

echo "========================================="
echo "Smoke tests finished. Check results/activation_clip/_smoketest_*_local/"
echo "and plots/activation_clip/_smoketest_*_local/ for output."
echo "========================================="
