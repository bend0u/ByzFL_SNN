#!/usr/bin/env bash

# RCP Full Sweep Runner for configs/full_sweep_thr04.json
# Logs are saved to results/logs/ so they persist on the dcl-scratch PVC

set -e

# Create log directory inside results (persisted on PVC)
mkdir -p results/logs

echo "========================================="
echo "Starting RCP Full Sweep (thr04) on 4 GPUs"
echo "  - 60 parallel jobs distributed across GPUs"
echo "  - Logs saved to results/logs/"
echo "========================================="

echo "[$(date)] Starting Full Sweep..."
python run_snn_robust_sweeps.py \
  --config configs/full_sweep_thr04.json \
  --distribute_gpus --nb_jobs 60 \
  > results/logs/rcp_full_sweep_thr04.log 2>&1 || echo "Full Sweep encountered errors"
echo "[$(date)] Full Sweep finished."
