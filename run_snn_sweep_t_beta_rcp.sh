#!/usr/bin/env bash

# RCP SNN T and Beta Sweep Runner
# Logs and results persist on the dcl-scratch PVC under results/

set -e

mkdir -p results/logs

echo "========================================="
echo "Starting RCP SNN T & Beta Sweep on GPUs"
echo "  - 50 parallel jobs distributed across GPUs"
echo "  - Logs saved to results/logs/snn_sweep_t_beta.log"
echo "========================================="

echo "[$(date)] Starting SNN T & Beta Sweep..."
python run_snn_sweep_t_beta.py \
  --config configs/snn_robustness/snn_sweep_t_beta.json \
  --distribute_gpus --nb_jobs 50 \
  > results/logs/snn_sweep_t_beta.log 2>&1 || echo "SNN Sweep encountered errors"
echo "[$(date)] SNN Sweep finished."
