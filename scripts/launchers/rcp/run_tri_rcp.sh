#!/usr/bin/env bash

# RCP Tri Sweep Runner - Dedicated parallel script for Tri sweep
# Logs are saved to results/logs/ so they persist on the dcl-scratch PVC

set -e

# Create log directory inside results (persisted on PVC)
mkdir -p results/logs

echo "========================================="
echo "Starting RCP Tri Sweep on 8 GPUs"
echo "  - 50 parallel jobs distributed across GPUs"
echo "  - Logs saved to results/logs/"
echo "========================================="

echo "[$(date)] Starting Tri Sweep..."
python run_snn_robust_sweeps.py \
  --config configs/snn_robust_new_tri_sweep.json \
  --distribute_gpus --nb_jobs 50 \
  > results/logs/rcp_tri_sweep.log 2>&1 || echo "Tri Sweep encountered errors"
echo "[$(date)] Tri Sweep finished."
