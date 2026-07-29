#!/usr/bin/env python3
"""
Read-only heatmap generator for the activation-clip / adaptive-clip sweep.

Unlike run_activation_clip_sweep.py, this does NOT call run_benchmark - it only
reads whatever results already exist on disk (per config's results_directory)
and (re)generates heatmaps from them. It never writes into results_directory
itself, so it is safe to run concurrently with a live training job: you can
start analyzing configs that have already finished without waiting for the
rest of the (sequential) sweep to complete.

Plots are written to <results_directory>/plots/, nested under results/ rather
than the sibling plots/activation_clip/ tree, so they land on whatever
persistent storage results/ is already mounted on (e.g. a PVC-backed
--pvc dcl-scratch:/home/bendouro/results mount).

Configs with no results yet (results_directory doesn't exist) are skipped.

Usage:
    python plot_activation_clip_results.py                     # all found configs, parallel
    python plot_activation_clip_results.py --workers 8
    python plot_activation_clip_results.py --configs cnn_mnist_clip_ste_1 cnn_mnist_clip_ramp_1
"""
import os
import sys
import json
import glob
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from plot_robust_heatmaps import plot_all

CONFIG_DIR = "configs/activation_clip"


def discover_configs():
    paths = sorted(glob.glob(os.path.join(CONFIG_DIR, "*.json")))
    # Skip internal smoketest configs (prefixed with "_")
    return [p for p in paths if not os.path.basename(p).startswith("_")]


def process_one(config_path):
    name = os.path.splitext(os.path.basename(config_path))[0]
    with open(config_path, "r") as f:
        cfg = json.load(f)
    results_dir = cfg["evaluation_and_results"]["results_directory"]

    if not os.path.isdir(results_dir):
        return name, "skipped (no results directory yet)"

    variant = os.path.basename(results_dir.rstrip("/\\"))
    plots_dir = os.path.join("results", "activation_clip_plots", variant)
    try:
        plot_all(results_dir, plots_dir)
        return name, f"ok -> {plots_dir}"
    except Exception as e:
        return name, f"error: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Read-only heatmap (re)generation for the activation-clip sweep, "
                    "safe to run alongside a live training job."
    )
    parser.add_argument(
        "--configs", nargs="*", default=None,
        help="Config basenames (without .json) to process; default = every config "
             "found in configs/activation_clip/ (excluding _smoketest_*)."
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker processes")
    args = parser.parse_args()

    all_paths = discover_configs()
    if args.configs:
        wanted = set(args.configs)
        all_paths = [p for p in all_paths if os.path.splitext(os.path.basename(p))[0] in wanted]

    if not all_paths:
        print("No matching configs found.")
        sys.exit(1)

    print(f"Processing {len(all_paths)} config(s) with {args.workers} parallel worker(s)...")
    print("(Read-only: no training, no GPU, safe alongside a live sweep job.)")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one, p): p for p in all_paths}
        for future in as_completed(futures):
            path = futures[future]
            name = os.path.splitext(os.path.basename(path))[0]
            try:
                name, status = future.result()
            except Exception as e:
                status = f"error: {e}"
            print(f"[{name}] {status}")


if __name__ == "__main__":
    main()
