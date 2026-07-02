#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Ensure logs directory exists
mkdir -p logs

echo "========================================="
echo "Starting Sequential Sweep Experiment on GPU..."
echo "========================================="

# 1. Atan Sweep
echo "Starting New Atan Sweep..."
python run_snn_robust_sweeps.py --config configs/snn_robust_new_atan_sweep.json --gpu 0 --nb_jobs 4 > logs/new_atan_sweep.log 2>&1 || echo "Atan Sweep encountered errors"

# 2. Tri Sweep
echo "Starting New Tri Sweep..."
python run_snn_robust_sweeps.py --config configs/snn_robust_new_tri_sweep.json --gpu 0 --nb_jobs 4 > logs/new_tri_sweep.log 2>&1 || echo "Tri Sweep encountered errors"

# 3. Box Sweep
echo "Starting New Box Sweep..."
python run_snn_robust_sweeps.py --config configs/snn_robust_new_box_sweep.json --gpu 0 --nb_jobs 4 > logs/new_box_sweep.log 2>&1 || echo "Box Sweep encountered errors"

echo "========================================="
echo "All sweeps completed successfully!"
echo "========================================="
