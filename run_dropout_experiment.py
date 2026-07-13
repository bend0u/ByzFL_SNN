#!/usr/bin/env python3
import sys
from byzfl.benchmark.benchmark import run_benchmark
from torchvision import datasets

def main():
    print("Pre-downloading MNIST to prevent multiprocessing race conditions...")
    datasets.MNIST(root='./data', train=True, download=True)
    datasets.MNIST(root='./data', train=False, download=True)

    print("Starting dropout sparsity experiment (CNN Tanh with 40% dropout)...")
    config_path = "configs/robust_dropout_experiment.json"
    try:
        # Massive heatmap sweep: scaled up to 60 parallel jobs for 4x GPUs on RunAI
        run_benchmark(config_path, nb_jobs=60, distribute_gpus=True)
        print("Dropout sparsity experiment completed successfully!")
    except Exception as e:
        print(f"Experiment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

