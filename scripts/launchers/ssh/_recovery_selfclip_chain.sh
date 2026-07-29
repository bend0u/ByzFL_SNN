#!/usr/bin/env bash
# Recovery: run the two self-calibrated-clip pilots SEQUENTIALLY at safe nb_jobs,
# after the earlier concurrent nb_jobs=30 runs OOM-thrashed. w1 RESUMES (skips its
# already-done trainings); then w50 runs. Fully detached. Plots at end of each.
set -e
REPO=/localhome/bendouro/ByzFL_snn/byzfl
cd "$REPO"
NB_JOBS=12
echo "[$(date)] RECOVERY start: w1 (resume) then w50, nb_jobs=$NB_JOBS" > results/logs/selfclip_recovery.log
echo "[$(date)] --- w1 resume ---" >> results/logs/selfclip_recovery.log
./venv/bin/python3 run_activation_clip_sweep.py \
  --config configs/activation_clip/cnn_mnist_selfclip_w1_gm3.json \
  --distribute_gpus --nb_jobs "$NB_JOBS" >> results/logs/selfclip_w1_gm3_ssh.log 2>&1
echo "[$(date)] w1 done" >> results/logs/selfclip_recovery.log
echo "[$(date)] --- w50 ---" >> results/logs/selfclip_recovery.log
./venv/bin/python3 run_activation_clip_sweep.py \
  --config configs/activation_clip/cnn_mnist_selfclip_w50_gm3.json \
  --distribute_gpus --nb_jobs "$NB_JOBS" >> results/logs/selfclip_w50_gm3_ssh.log 2>&1
echo "[$(date)] w50 done -- RECOVERY complete" >> results/logs/selfclip_recovery.log
