#!/usr/bin/env bash

# RCP Box Sweep Runner - Dedicated parallel script for Box sweep
# Logs are saved to results/logs/ so they persist on the dcl-scratch PVC

set -e

# Create log directory inside results (persisted on PVC)
mkdir -p results/logs

echo "========================================="
echo "Starting RCP Box Sweep on 8 GPUs"
echo "  - 50 parallel jobs distributed across GPUs"
echo "  - Logs saved to results/logs/"
echo "========================================="

echo "[$(date)] Starting Box Sweep..."
python run_snn_robust_sweeps.py \
  --config configs/snn_robust_new_box_sweep.json \
  --distribute_gpus --nb_jobs 50 \
  > results/logs/rcp_box_sweep.log 2>&1 || echo "Box Sweep encountered errors"
echo "[$(date)] Box Sweep finished."
