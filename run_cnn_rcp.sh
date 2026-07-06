#!/usr/bin/env bash

# RCP CNN Sweep Runner - Dedicated script to run CNN sweep inside Docker
# Logs are saved to results/logs/ so they persist on the dcl-scratch PVC

set -e

# Create log directory inside results (persisted on PVC)
mkdir -p results/logs

echo "========================================="
echo "Starting RCP CNN Robust Sweep on 4 GPUs"
echo "  - 50 parallel jobs distributed across GPUs"
echo "  - Logs saved to results/logs/"
echo "========================================="

echo "[$(date)] Starting CNN Robust Sweep..."
python run_cnn_robust_sweeps.py \
  --config configs/cnn_robust_comparison_sweep.json \
  --distribute_gpus --nb_jobs 50 \
  > results/logs/rcp_cnn_sweep.log 2>&1 || echo "CNN Sweep encountered errors"
echo "[$(date)] CNN Sweep finished."
