#!/usr/bin/env python3
import os
from byzfl.benchmark.benchmark import run_benchmark
from torchvision import datasets

def main():
    print("Pre-downloading CIFAR-10 to prevent multiprocessing race conditions...")
    datasets.CIFAR10(root='./data', train=True, download=True)
    datasets.CIFAR10(root='./data', train=False, download=True)

    config_path = "configs/cifar_sweep_cnn_relu.json"

    print("=" * 60)
    print("CIFAR-10 Sweep: CNN (ReLU)")
    print("  Running on 4 GPUs (V100)...")
    print("=" * 60)

    # Use nb_jobs=20 to distribute across the 4 GPUs (5 jobs per GPU)
    run_benchmark(config_path, nb_jobs=20, distribute_gpus=True)
    print("Sweep completed! Results are in results/cifar_sweep_cnn_relu")

if __name__ == "__main__":
    main()
