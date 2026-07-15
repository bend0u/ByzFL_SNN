#!/usr/bin/env bash

# Local CPU smoke test for the MNIST SNN Byzantine-cliff sweep.
# Confirms the config/pipeline runs end-to-end before launching the real
# sweep on the cluster. Tiny grid (f in {0,4}, gamma in {1.0,0.0}, 20 steps).
#
# Usage:
#   ./run_mnist_cliff_smoketest_local.sh [config] [results_dir]

set -e

CONFIG="${1:-configs/snn_robustness/mnist_cliff_smoketest_local.json}"
RESULTS_DIR="${2:-}"

if [ -n "$RESULTS_DIR" ]; then
    python - "$CONFIG" "$RESULTS_DIR" <<'PY'
import json, sys
config_path, results_dir = sys.argv[1], sys.argv[2]
with open(config_path) as f:
    cfg = json.load(f)
cfg["evaluation_and_results"]["results_directory"] = results_dir
with open(config_path, "w") as f:
    json.dump(cfg, f, indent=4)
PY
fi

echo "========================================="
echo "MNIST SNN Cliff Smoke Test (local, CPU)"
echo "  Config: ${CONFIG}"
echo "========================================="

python run_snn_robust_sweeps.py \
  --config "${CONFIG}" \
  --nb_jobs 1

echo "Smoke test finished."
