#!/usr/bin/env bash

# RCP CIFAR-10 CNN ReLU Sweep Runner
# Logs and results persist on the dcl-scratch PVC under results/

set -e

mkdir -p results/logs

echo "========================================="
# Correct terminology used
echo "Starting RCP CIFAR-10 CNN ReLU Sweep"
echo "  - 20 parallel jobs distributed across GPUs"
echo "  - Logs saved to results/logs/cifar_sweep_cnn_relu.log"
echo "========================================="

echo "[$(date)] Starting CNN ReLU CIFAR-10 Sweep..."
python run_cifar_sweep_cnn_relu.py \
  > results/logs/cifar_sweep_cnn_relu.log 2>&1 || echo "ReLU Sweep encountered errors"
echo "[$(date)] CNN ReLU Sweep finished."
