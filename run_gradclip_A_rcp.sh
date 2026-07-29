#!/usr/bin/env bash
#
# POD A of 2 (fixed grad-norm clip study). Runs ALL 5 clip configs but only on the
# LESS-heterogeneous half of the grid (gamma in {1.0, 0.66}). POD B does the same
# 5 configs on gamma in {0.33, 0.0}. Each pod = 5 mechanisms x 2 gammas = exactly
# half the total work, so two 4-GPU pods finish at ~the same time.
#
# Both pods write into the SAME results_directory per config (per-setting folders
# are uniquely named by gamma, so the halves merge). Heatmaps are SKIPPED here
# (--no_plots) because each pod holds only half the gammas; regenerate them once
# after BOTH pods finish with:  bash run_gradclip_plots.sh
#
# Run inside the RCP container from the repo root. Requires a REBUILT+PUSHED image.
# Usage:  bash run_gradclip_A_rcp.sh [nb_jobs]

set -e
NB_JOBS="${1:-80}"
GAMMAS="1.0 0.66"
mkdir -p results/logs

CONFIGS=(
  "configs/activation_clip/cnn_mnist_gradclip21.json"
  "configs/activation_clip/cnn_mnist_gradclip_calib.json"
  "configs/activation_clip/cnn_mnist_gradclip_qlow.json"
  "configs/activation_clip/cnn_mnist_gradclip_qhigh.json"
  "configs/activation_clip/cnn_mnist_layerclip.json"
)

echo "========================================="
echo "[$(date)] Grad-clip POD A -- gammas {${GAMMAS}} (nb_jobs=${NB_JOBS})"
echo "========================================="
for config in "${CONFIGS[@]}"; do
  tag=$(basename "${config}" .json)
  log_file="results/logs/gradclip_${tag}_A.log"
  echo "[$(date)] Starting ${config} [gammas ${GAMMAS}] -> ${log_file}"
  python run_activation_clip_sweep.py \
    --config "${config}" --gammas ${GAMMAS} --no_plots \
    --distribute_gpus --nb_jobs "${NB_JOBS}" \
    > "${log_file}" 2>&1 || echo "  ${config} encountered errors, see ${log_file}"
  echo "[$(date)] Finished ${config}"
done
echo "[$(date)] POD A complete. (Run run_gradclip_plots.sh after POD B also finishes.)"
