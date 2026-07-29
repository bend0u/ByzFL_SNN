#!/usr/bin/env bash
# One-shot SCHEDULED launcher: waits 1 hour, then runs the self-calibrated warmup
# clip pilot with warmup=50 (GeometricMedian only, 3 seeds) on this SSH box's 2x A10.
# Fully detached (nohup setsid) so it survives session/terminal close.
# Separate results_directory from the w1 pilot -> no collision.
set -e
REPO=/localhome/bendouro/ByzFL_snn/byzfl
cd "$REPO"
LOG="$REPO/results/logs/selfclip_w50_gm3_ssh.log"
mkdir -p "$REPO/results/logs"

echo "[$(date)] scheduled: sleeping 3600s before starting w50 run" > "$LOG"
sleep 3600
echo "[$(date)] waking up, starting w50 sweep" >> "$LOG"

rm -rf "$REPO/results/activation_clip/cnn_mnist_selfclip_w50_gm3"
./venv/bin/python3 run_activation_clip_sweep.py \
  --config configs/activation_clip/cnn_mnist_selfclip_w50_gm3.json \
  --distribute_gpus --nb_jobs 30 >> "$LOG" 2>&1
echo "[$(date)] w50 sweep finished" >> "$LOG"
