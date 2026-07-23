#!/usr/bin/env bash

# Pilot run of the RAW-gradient adaptive norm clip at tau=0.80
# (model cnn_mnist, honest_clients.raw_grad_clip_quantile=0.80, window=100),
# at nb_training_seeds=2 for an early read.
#
# Counterpart to run_qclip080_ssh.sh. Same adaptive windowed-quantile mechanism,
# same tau, same plain-ReLU cnn_mnist, same grid -- the ONLY difference is WHERE
# the clip is applied:
#
#   qclip_080    -> clips the POST-momentum vector sent to the server
#   rawqclip_080 -> clips the RAW gradient BEFORE the momentum accumulator
#
# Seeds are training_seed + i, so this 2-seed run (42, 43) is a strict prefix of
# the full 5-seed run. Bumping nb_training_seeds to 5 in the config later reuses
# these cached seeds and only trains 44/45/46 -- this pilot costs nothing extra.
#
# Meant to run on a GPU box over SSH, inside tmux. Default nb_jobs=40 across
# 2x A10: cnn_mnist is tiny (~0.55 GiB reserved per job), so 40 jobs is ~22 GiB
# of the 48 GiB available -- the practical limit is CPU/dataloader contention
# rather than VRAM.
#
# Plots go to results/activation_clip_plots/cnn_mnist_rawqclip_080/ (persistent
# local disk here -- no PVC/ephemeral-pod concern as on RCP).
#
# Usage (from the repo root, inside tmux):
#   ./scripts/launchers/ssh/run_rawqclip080_ssh.sh [nb_jobs]

set -e

NB_JOBS="${1:-40}"

mkdir -p results/logs

CONFIG="configs/activation_clip/cnn_mnist_rawqclip_080.json"
tag=$(basename "${CONFIG}" .json)
log_file="results/logs/actclip_${tag}.log"

echo "========================================="
echo "Raw-gradient norm-clip pilot (SSH, 2 seeds)"
echo "  Config:   ${CONFIG}"
echo "  Jobs:     ${NB_JOBS} (distributed across GPUs)"
echo "  Log:      ${log_file}"
echo "  Plots ->  results/activation_clip_plots/${tag}/"
echo "========================================="

echo "[$(date)] Starting ${CONFIG}"
./venv/bin/python3 scripts/experiments/run_activation_clip_sweep.py \
  --config "${CONFIG}" \
  --distribute_gpus --nb_jobs "${NB_JOBS}" \
  > "${log_file}" 2>&1 || { echo "  Run failed, see ${log_file}"; exit 1; }
echo "[$(date)] Finished ${CONFIG}. Plots in results/activation_clip_plots/${tag}/"
