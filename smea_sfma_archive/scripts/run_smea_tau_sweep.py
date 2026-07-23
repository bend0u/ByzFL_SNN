#!/usr/bin/env python3
import sys
from byzfl.benchmark.benchmark import run_benchmark

def main():
    print("Starting SMEA Tau Sweep on SNN Atan 1.2 (f=4, gamma=0.33)")
    config_path = "configs/smea_tau_sweep.json"
    try:
        run_benchmark(config_path, nb_jobs=15, distribute_gpus=True)
        print("SMEA Tau Sweep completed successfully!")
    except Exception as e:
        print(f"Experiment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
