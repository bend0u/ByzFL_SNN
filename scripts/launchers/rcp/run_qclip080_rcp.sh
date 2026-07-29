#!/usr/bin/env bash

# Early, standalone RCP run of ONLY the adaptive client-side gradient-norm clip
# at tau=0.80 (model cnn_mnist, honest_clients.grad_clip_quantile=0.80, window=100),
# so its heatmaps are available early instead of waiting for the full sequential
# activation-clip sweep (run_activation_clip_sweeps_rcp.sh) to reach it.
#
# Submit as its OWN RunAI job (distinct --name) so it runs in parallel with the
# main sweep. qclip_080 has been removed from run_activation_clip_sweeps_rcp.sh,
# so this is the only thing that runs it -- no duplicate work / no write race.
#
# Runs inside the repo's Dockerfile container. Plots are written by
# run_activation_clip_sweep.py to results/activation_clip_plots/cnn_mnist_qclip_080/
# which is under results/ -- the PVC-mounted, persistent tree -- so they survive
# pod termination.
#
# Usage (inside the RCP container, from the repo root):
#   ./run_qclip080_rcp.sh [nb_jobs]

set -e

NB_JOBS="${1:-50}"

mkdir -p results/logs

CONFIG="configs/activation_clip/cnn_mnist_qclip_080.json"
tag=$(basename "${CONFIG}" .json)
log_file="results/logs/actclip_${tag}.log"

echo "========================================="
echo "Early norm-clip run (RCP): ${CONFIG}"
echo "  ${NB_JOBS} parallel jobs distributed across GPUs"
echo "  Log:      ${log_file}"
echo "  Plots ->  results/activation_clip_plots/${tag}/"
echo "========================================="

echo "[$(date)] Starting ${CONFIG}"
python run_activation_clip_sweep.py \
  --config "${CONFIG}" \
  --distribute_gpus --nb_jobs "${NB_JOBS}" \
  > "${log_file}" 2>&1 || { echo "  Run failed, see ${log_file}"; exit 1; }
echo "[$(date)] Finished ${CONFIG}. Plots in results/activation_clip_plots/${tag}/"
