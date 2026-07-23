#!/usr/bin/env python3
"""
Pilot run of the RAW-gradient adaptive norm clip at tau=0.80
(model cnn_mnist, honest_clients.raw_grad_clip_quantile=0.80, window=100),
at nb_training_seeds=2 for an early read.

Counterpart to the qclip_080 run. Same adaptive windowed-quantile mechanism,
same tau, same plain-ReLU cnn_mnist, same grid -- the ONLY difference is WHERE
the clip is applied:

    qclip_080    -> clips the POST-momentum vector sent to the server
    rawqclip_080 -> clips the RAW gradient BEFORE the momentum accumulator

Seeds are training_seed + i, so this 2-seed run (42, 43) is a strict prefix of
the full 5-seed run. Bumping nb_training_seeds to 5 in the generator later reuses
these cached seeds and only trains 44/45/46 -- this pilot costs nothing extra.

Defaults to 40 parallel jobs distributed across GPUs, sized for 2x A10:
cnn_mnist is tiny (~0.55 GiB reserved per job), so 40 jobs is ~22 GiB of the
48 GiB available -- the practical limit is CPU/dataloader contention, not VRAM.

Plots go to results/activation_clip_plots/cnn_mnist_rawqclip_080/.

Usage (from anywhere; the script cd's to the repo root itself):
    python scripts/launchers/ssh/run_rawqclip080.py
    python scripts/launchers/ssh/run_rawqclip080.py --nb_jobs 20
    python scripts/launchers/ssh/run_rawqclip080.py --gpu 0        # pin to one GPU
"""
import os
import sys
import argparse
import datetime

# This file lives at <repo>/scripts/launchers/ssh/, so the repo root is 3 levels up.
REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
# Relative paths inside the config (configs/, results/, ./data) are resolved
# against the CWD, so run from the repo root regardless of where this was invoked.
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "experiments"))

from run_activation_clip_sweep import run_sweep  # noqa: E402

CONFIG = "configs/activation_clip/cnn_mnist_rawqclip_080.json"


def main():
    parser = argparse.ArgumentParser(
        description="Pilot run: raw-gradient adaptive norm clip (tau=0.80, 2 seeds)"
    )
    parser.add_argument("--nb_jobs", type=int, default=40,
                        help="Number of parallel jobs (default: 40, sized for 2x A10)")
    parser.add_argument("--gpu", type=str, default=None,
                        help="Pin to a single GPU index instead of distributing across all")
    args = parser.parse_args()

    distribute_gpus = args.gpu is None

    print("=" * 60)
    print("Raw-gradient norm-clip pilot (2 seeds)")
    print(f"  Config:   {CONFIG}")
    print(f"  Jobs:     {args.nb_jobs}"
          f"{' (distributed across GPUs)' if distribute_gpus else f' (pinned to GPU {args.gpu})'}")
    print(f"  Plots ->  results/activation_clip_plots/cnn_mnist_rawqclip_080/")
    print("=" * 60)
    print(f"[{datetime.datetime.now()}] Starting {CONFIG}")

    run_sweep(
        CONFIG,
        nb_jobs=args.nb_jobs,
        distribute_gpus=distribute_gpus,
        gpu=args.gpu if args.gpu is not None else "0",
    )

    print(f"[{datetime.datetime.now()}] Finished {CONFIG}.")
    print("Plots in results/activation_clip_plots/cnn_mnist_rawqclip_080/")


if __name__ == "__main__":
    main()
