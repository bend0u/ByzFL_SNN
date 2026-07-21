#!/usr/bin/env python3
"""Gradient-structure baseline sweep runner: f=0, NoAttack, gamma in
{1.0, 0.66, 0.33, 0.0}, TrMean+NNM+ARC, 1 seed, 500 steps, with
store_gradient_structure_metrics on (same reference hyperparameters as
configs/geometry_baseline/*, see configs/gradient_structure/*.json).

Loops the 3 model configs (SNN atan12, CNN Tanh, CNN ReLU) through the
existing generic run_snn_robust_sweeps.py entrypoint -- no new training
logic here, just a python-native alternative to a shell script (avoids
needing chmod +x after a fresh clone).

Usage (on dclgpusrv, from the repo root, inside tmux):
    python run_gradient_structure_sweep.py [nb_jobs]
"""
import os
import subprocess
import sys
from datetime import datetime

CONFIGS = [
    "configs/gradient_structure/snn_atan12.json",
    "configs/gradient_structure/cnn_tanh.json",
    "configs/gradient_structure/cnn_relu.json",
]


def main():
    nb_jobs = sys.argv[1] if len(sys.argv) > 1 else "4"
    os.makedirs("results/logs", exist_ok=True)

    print("=========================================")
    print("Gradient Structure Baseline (SNN atan12, CNN Tanh, CNN ReLU)")
    print(f"  Parallel jobs per config: {nb_jobs}")
    print("=========================================")

    for config in CONFIGS:
        tag = os.path.splitext(os.path.basename(config))[0]
        log_file = f"results/logs/gradient_structure_{tag}.log"
        print(f"[{datetime.now()}] Starting {config} -> {log_file}")
        with open(log_file, "w") as f:
            result = subprocess.run(
                [sys.executable, "run_snn_robust_sweeps.py",
                 "--config", config, "--distribute_gpus", "--nb_jobs", str(nb_jobs)],
                stdout=f, stderr=subprocess.STDOUT,
            )
        if result.returncode != 0:
            print(f"  {config} encountered errors, see {log_file}")
        print(f"[{datetime.now()}] Finished {config}")

    print("All gradient structure configs finished.")


if __name__ == "__main__":
    main()
