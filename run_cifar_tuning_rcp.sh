#!/usr/bin/env bash

# Dedicated parallel script for CIFAR-10 Tanh vs ReLU Baseline Tuning
# Logs and plots are saved to results/ so they persist on the dcl-scratch PVC

set -e

# Create log and plots directory inside results (persisted on PVC)
mkdir -p results/logs
mkdir -p results/plots/cifar_tuning

echo "========================================="
echo "Starting CIFAR-10 Activation Tuning (ReLU vs Tanh)"
echo "  - 6 parallel jobs distributed across 4 GPUs"
echo "  - Logs saved to results/logs/cifar_tuning.log"
echo "========================================="

echo "[$(date)] Starting CIFAR Tuning Sweep..."
python run_cifar_tuning.py \
  > results/logs/cifar_tuning.log 2>&1 || echo "Tuning Sweep encountered errors"
echo "[$(date)] Tuning Sweep finished."

echo "[$(date)] Generating Plot..."
python plot_cifar_tuning.py \
  >> results/logs/cifar_tuning.log 2>&1 || echo "Plotting encountered errors"
echo "[$(date)] Plotting finished. Plot is available in results/plots/cifar_tuning/"

echo "========================================="
echo "All tasks completed."
echo "========================================="
