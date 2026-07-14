#!/usr/bin/env bash

# RCP CIFAR-10 CNN Tanh Sweep Runner
# Logs and results persist on the dcl-scratch PVC under results/

set -e

mkdir -p results/logs

echo "========================================="
# Correct terminology used
echo "Starting RCP CIFAR-10 CNN Tanh Sweep"
echo "  - 20 parallel jobs distributed across GPUs"
echo "  - Logs saved to results/logs/cifar_sweep_cnn_tanh.log"
echo "========================================="

echo "[$(date)] Starting CNN Tanh CIFAR-10 Sweep..."
python run_cifar_sweep_cnn_tanh.py \
  > results/logs/cifar_sweep_cnn_tanh.log 2>&1 || echo "Tanh Sweep encountered errors"
echo "[$(date)] CNN Tanh Sweep finished."
