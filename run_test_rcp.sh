#!/usr/bin/env bash

# Exit on error
set -e

echo "========================================="
echo "Running RCP Verification Test on Multiple GPUs..."
echo "========================================="

# Execute the test configuration distributing jobs across available GPUs
python run_snn_robust_sweeps.py --config configs/snn_test_rcp.json --nb_jobs 2 --distribute_gpus

echo ""
echo "========================================="
echo "Verification Test completed successfully!"
echo "========================================="
