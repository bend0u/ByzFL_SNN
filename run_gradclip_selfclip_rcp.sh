#!/usr/bin/env bash
#
# Self-calibrated warmup clip study -- SEPARATE from tonight's gradclip A/B pods
# (a different mechanism, not required for those). Each client freezes its OWN
# absolute cap from its own first N raw grad-norms (max seen * margin), no offline
# probe or SNN needed. See reports/gradclip_experiment_plan.md.
#
#   1. cnn_mnist_selfclip_w75 -- warmup=75, margin=1.1 (data-justified: converges
#      to ~0% honest clipping, matching the offline-calibrated ceiling).
#   2. cnn_mnist_selfclip_w1  -- warmup=1, margin=1.0 (literally "clip to the
#      first gradient" -- deliberate ablation, expected to UNDER-perform: a
#      single-sample cap sits 4.5x-8x below the true honest peak and should clip
#      34-95% of honest steps, same failure mode as the adaptive quantile/STE).
# Both full 4-aggregator sweeps, comparable to the rest of the activation_clip
# family. Fits in one 4-GPU pod (2 sweeps).
#
# Run inside the RCP container from the repo root. Requires a REBUILT+PUSHED image
# (client/managers/train changed for self_grad_clip_warmup/self_grad_clip_margin).
# Usage:  bash run_gradclip_selfclip_rcp.sh [nb_jobs]

set -e
NB_JOBS="${1:-80}"
mkdir -p results/logs

CONFIGS=(
  "configs/activation_clip/cnn_mnist_selfclip_w75.json"
  "configs/activation_clip/cnn_mnist_selfclip_w1.json"
)

echo "========================================="
echo "[$(date)] Self-calibrated warmup clip study (nb_jobs=${NB_JOBS})"
echo "========================================="
for config in "${CONFIGS[@]}"; do
  tag=$(basename "${config}" .json)
  log_file="results/logs/gradclip_${tag}.log"
  echo "[$(date)] Starting ${config} -> ${log_file}"
  python run_activation_clip_sweep.py \
    --config "${config}" \
    --distribute_gpus --nb_jobs "${NB_JOBS}" \
    > "${log_file}" 2>&1 || echo "  ${config} encountered errors, see ${log_file}"
  echo "[$(date)] Finished ${config}"
done
echo "[$(date)] Self-calibrated clip study complete."
