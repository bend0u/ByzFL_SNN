#!/usr/bin/env bash

# RCP CNN Learning Rate Sweep Runner
# Logs are saved to results/logs/ so they persist on the dcl-scratch PVC

set -e

# Create log directory inside results (persisted on PVC)
mkdir -p results/logs

echo "========================================="
echo "Starting RCP CNN Learning Rate Sweep on 4 GPUs"
echo "  - 40 parallel jobs distributed across GPUs"
echo "  - Logs saved to results/logs/"
echo "========================================="

echo "[$(date)] Starting CNN LR Sweep..."
python run_cnn_lr_sweeps.py \
  --config configs/cnn_robust_comparison_sweep.json \
  --distribute_gpus --nb_jobs 40 \
  > results/logs/rcp_cnn_lr_sweep.log 2>&1 || echo "CNN LR Sweep encountered errors"
echo "[$(date)] CNN LR Sweep finished."
