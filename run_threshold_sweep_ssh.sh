#!/usr/bin/env bash

# Threshold-only sweep (EXP3-adjacent, homeostasis pre-check): model.model_params.threshold
# in {1.0, 0.8, 0.6, 0.4} at fixed atan alpha=1.2, over the restricted grid
# f in {0,3,5} x gamma in {1.0,0.66}, all 4 aggregators, all 3 attacks. Meant
# to run on dclgpusrv over SSH.
#
# theta=1.0 is included here as the real baseline: no complete theta=1.0/
# alpha=1.2 run exists yet for this restricted grid (see CLAUDE.md.txt /
# prior investigation -- the older robust_new_atan_sweep results directory
# is at alpha=1.25 and only covers 2/4 aggregators, 1/3 attacks).
#
# Usage (on dclgpusrv, from the repo root, inside tmux):
#   ./run_threshold_sweep_ssh.sh [nb_jobs]

set -e

NB_JOBS="${1:-8}"

mkdir -p results/logs

CONFIGS=(
  "configs/threshold_sweep/config_thr08.json"
  "configs/threshold_sweep/config_thr06.json"
  "configs/threshold_sweep/config_thr04.json"
  "configs/threshold_sweep/config_thr10.json"
)

echo "========================================="
echo "Threshold Sweep (theta = 1.0, 0.8, 0.6, 0.4)"
echo "  Parallel jobs per config: ${NB_JOBS}"
echo "========================================="

for config in "${CONFIGS[@]}"; do
  tag=$(basename "${config}" .json)
  log_file="results/logs/${tag}.log"
  echo "[$(date)] Starting ${config} -> ${log_file}"
  python run_snn_robust_sweeps.py \
    --config "${config}" \
    --distribute_gpus --nb_jobs "${NB_JOBS}" \
    > "${log_file}" 2>&1 || echo "  ${config} encountered errors, see ${log_file}"
  echo "[$(date)] Finished ${config}"
done

echo "All threshold sweep configs finished."
