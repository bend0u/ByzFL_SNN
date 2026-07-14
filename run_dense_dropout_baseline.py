#!/usr/bin/env python3
import sys
from byzfl.benchmark.benchmark import run_benchmark
from torchvision import datasets

def main():
    print("Pre-downloading MNIST to prevent multiprocessing race conditions...")
    datasets.MNIST(root='./data', train=True, download=True)
    datasets.MNIST(root='./data', train=False, download=True)

    print("Starting Dense Dropout Baseline (CNN Tanh with 93% dropout on dense layer only)...")
    config_path = "configs/dense_dropout_baseline.json"
    try:
        run_benchmark(config_path, nb_jobs=20, distribute_gpus=True)
        print("Dense Dropout Baseline completed successfully!")
    except Exception as e:
        print(f"Experiment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
