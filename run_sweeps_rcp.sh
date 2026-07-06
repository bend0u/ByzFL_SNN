#!/usr/bin/env bash

# RCP Sweep Runner - Optimized for 10x GPUs (50 parallel jobs distributed across GPUs)
# Logs are saved to results/logs/ so they persist on the dcl-scratch PVC

set -e

# Create log directory inside results (persisted on PVC)
mkdir -p results/logs

echo "========================================="
echo "Starting RCP Sweep on 10 GPUs"
echo "  - 50 parallel jobs distributed across GPUs"
echo "  - Logs saved to results/logs/"
echo "========================================="

# 1. Atan Sweep
echo "[$(date)] Starting Atan Sweep..."
python run_snn_robust_sweeps.py \
  --config configs/snn_robust_new_atan_sweep.json \
  --distribute_gpus --nb_jobs 50 \
  > results/logs/rcp_atan_sweep.log 2>&1 || echo "Atan Sweep encountered errors"
echo "[$(date)] Atan Sweep finished."

# 2. Tri Sweep
echo "[$(date)] Starting Tri Sweep..."
python run_snn_robust_sweeps.py \
  --config configs/snn_robust_new_tri_sweep.json \
  --distribute_gpus --nb_jobs 50 \
  > results/logs/rcp_tri_sweep.log 2>&1 || echo "Tri Sweep encountered errors"
echo "[$(date)] Tri Sweep finished."

# 3. Box Sweep
echo "[$(date)] Starting Box Sweep..."
python run_snn_robust_sweeps.py \
  --config configs/snn_robust_new_box_sweep.json \
  --distribute_gpus --nb_jobs 50 \
  > results/logs/rcp_box_sweep.log 2>&1 || echo "Box Sweep encountered errors"
echo "[$(date)] Box Sweep finished."

echo "========================================="
echo "[$(date)] All RCP sweeps completed!"
echo "========================================="
