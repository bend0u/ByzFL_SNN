#!/usr/bin/env python3
"""Find/delete result folders whose saved config.json doesn't match an expected nb_steps.

run_benchmark()'s resume logic (eliminate_experiments_done) only checks that a
result folder + train_time marker exist -- it never checks that the config
inside the folder matches the config you're currently running. If an earlier
pass wrote results for a combo with a different nb_steps, later sweeps will
silently skip retraining it. This script finds (and optionally deletes) any
folder under a results directory whose config.json nb_steps doesn't match
what you expect, so a rerun of the sweep will regenerate just those combos.
"""
import argparse
import json
import os
import shutil
import sys


def find_stale_dirs(results_dir, expected_nb_steps):
    stale = []
    for name in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, name)
        config_path = os.path.join(path, "config.json")
        if not os.path.isfile(config_path):
            continue
        try:
            with open(config_path) as f:
                config = json.load(f)
            nb_steps = config["benchmark_config"]["nb_steps"]
        except Exception as e:
            print(f"  [WARN] could not read nb_steps from {config_path}: {e}", file=sys.stderr)
            continue
        if nb_steps != expected_nb_steps:
            stale.append((path, nb_steps))
    return stale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", help="Results directory to scan (e.g. results/cifar_sweep_cnn_tanh)")
    parser.add_argument("--expected-nb-steps", type=int, default=5000,
                         help="nb_steps a folder must have to be kept (default: 5000)")
    parser.add_argument("--delete", action="store_true",
                         help="Actually delete stale folders (default: dry-run, list only)")
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"Not a directory: {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    stale = find_stale_dirs(args.results_dir, args.expected_nb_steps)

    if not stale:
        print("No stale folders found.")
        return

    print(f"Found {len(stale)} folder(s) with nb_steps != {args.expected_nb_steps}:")
    for path, nb_steps in stale:
        print(f"  nb_steps={nb_steps:<6} {path}")

    if args.delete:
        for path, _ in stale:
            shutil.rmtree(path)
        print(f"\nDeleted {len(stale)} folder(s). Rerun the sweep script to regenerate them.")
    else:
        print(f"\nDry run only -- rerun with --delete to actually remove these {len(stale)} folder(s).")


if __name__ == "__main__":
    main()
