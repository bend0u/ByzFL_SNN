#!/usr/bin/env python3
import sys
from byzfl.benchmark.benchmark import run_benchmark

def main():
    print("Starting targeted comparison sweep for GM and CC against SF and IPM...")
    config_path = "configs/robust_targeted_comparison.json"
    try:
        run_benchmark(config_path, nb_jobs=10, distribute_gpus=True)
        print("Targeted comparison sweep completed successfully!")
    except Exception as e:
        print(f"Experiment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
