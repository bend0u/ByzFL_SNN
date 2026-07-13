#!/usr/bin/env bash

# Dedicated parallel script for CNN Dense 93% Dropout Heatmap sweep
# Logs and plots are saved to results/ so they persist on the dcl-scratch PVC

set -e

# Create log and plots directory inside results (persisted on PVC)
mkdir -p results/logs
mkdir -p results/plots/dropout_93_heatmaps

echo "========================================="
echo "Starting Dense Dropout 93% Heatmap Sweep on GPUs"
echo "  - 60 parallel jobs distributed across GPUs"
echo "  - Logs saved to results/logs/dropout_heatmap.log"
echo "========================================="

echo "[$(date)] Starting Dropout Sweep..."
python run_dropout_experiment.py \
  > results/logs/dropout_heatmap.log 2>&1 || echo "Dropout Sweep encountered errors"
echo "[$(date)] Dropout Sweep finished."

echo "[$(date)] Generating Heatmap Plots..."
python plot_cnn_lr_robust_heatmaps.py results/cnn_dense_dropout_93_robustness results/plots/dropout_93_heatmaps \
  >> results/logs/dropout_heatmap.log 2>&1 || echo "Plotting encountered errors"
echo "[$(date)] Plotting finished. Heatmaps are available in results/plots/dropout_93_heatmaps/"

echo "========================================="
echo "All tasks completed."
echo "========================================="
