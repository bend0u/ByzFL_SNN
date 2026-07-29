#!/usr/bin/env bash

# Early, standalone run of ONLY the adaptive client-side gradient-norm clip at
# tau=0.80 (model cnn_mnist, honest_clients.grad_clip_quantile=0.80, window=100),
# so its heatmaps are available early without waiting for the full sequential
# activation-clip sweep (run_activation_clip_sweeps_rcp.sh) to reach it.
#
# Meant to run on dclgpusrv over SSH, inside tmux.
#
# Plots: run_activation_clip_sweep.py writes heatmaps to
#   results/activation_clip_plots/cnn_mnist_qclip_080/
# which is on the box's persistent local disk (no PVC / ephemeral-pod issue here),
# and is the same location the LaTeX report expects.
#
# Usage (on dclgpusrv, from the repo root, inside tmux):
#   git pull                       # make sure the box has the plot-path fix
#   ./run_qclip080_ssh.sh [nb_jobs]

set -e

NB_JOBS="${1:-4}"

mkdir -p results/logs

CONFIG="configs/activation_clip/cnn_mnist_qclip_080.json"
tag=$(basename "${CONFIG}" .json)
log_file="results/logs/actclip_${tag}.log"

echo "========================================="
echo "Early norm-clip run (SSH / dclgpusrv)"
echo "  Config:   ${CONFIG}"
echo "  Jobs:     ${NB_JOBS} (distributed across GPUs)"
echo "  Log:      ${log_file}"
echo "  Plots ->  results/activation_clip_plots/${tag}/"
echo "========================================="

echo "[$(date)] Starting ${CONFIG}"
./venv/bin/python3 run_activation_clip_sweep.py \
  --config "${CONFIG}" \
  --distribute_gpus --nb_jobs "${NB_JOBS}" \
  > "${log_file}" 2>&1 || { echo "  Run failed, see ${log_file}"; exit 1; }
echo "[$(date)] Finished ${CONFIG}. Plots in results/activation_clip_plots/${tag}/"
