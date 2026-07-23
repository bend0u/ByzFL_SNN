#!/usr/bin/env bash

# Standalone RCP run of the RAW-gradient adaptive norm clip at tau=0.80
# (model cnn_mnist, honest_clients.raw_grad_clip_quantile=0.80, window=100).
#
# This is the counterpart to run_qclip080_rcp.sh. Both use the same adaptive
# windowed-quantile mechanism at the same tau, on the same plain-ReLU cnn_mnist,
# with the same benchmark grid -- the ONLY difference is WHERE the clip is applied:
#
#   qclip_080    -> clips the POST-momentum vector sent to the server
#                   (momentum accumulator itself is left untouched)
#   rawqclip_080 -> clips the RAW gradient BEFORE it enters the momentum
#                   accumulator (same position as the fixed gradient_clip_val)
#
# so the two runs isolate exactly that one variable and are directly comparable.
#
# Submit as its OWN RunAI job (distinct --name). Its results_directory is
# distinct, so it does not collide with any other config.
#
# Runs inside the repo's Dockerfile container. Plots are written by
# run_activation_clip_sweep.py to results/activation_clip_plots/cnn_mnist_rawqclip_080/
# which is under results/ -- the PVC-mounted, persistent tree -- so they survive
# pod termination.
#
# Usage (inside the RCP container, from the repo root):
#   ./scripts/launchers/rcp/run_rawqclip080_rcp.sh [nb_jobs]

set -e

NB_JOBS="${1:-50}"

mkdir -p results/logs

CONFIG="configs/activation_clip/cnn_mnist_rawqclip_080.json"
tag=$(basename "${CONFIG}" .json)
log_file="results/logs/actclip_${tag}.log"

echo "========================================="
echo "Raw-gradient norm-clip run (RCP): ${CONFIG}"
echo "  ${NB_JOBS} parallel jobs distributed across GPUs"
echo "  Log:      ${log_file}"
echo "  Plots ->  results/activation_clip_plots/${tag}/"
echo "========================================="

echo "[$(date)] Starting ${CONFIG}"
python scripts/experiments/run_activation_clip_sweep.py \
  --config "${CONFIG}" \
  --distribute_gpus --nb_jobs "${NB_JOBS}" \
  > "${log_file}" 2>&1 || { echo "  Run failed, see ${log_file}"; exit 1; }
echo "[$(date)] Finished ${CONFIG}. Plots in results/activation_clip_plots/${tag}/"
