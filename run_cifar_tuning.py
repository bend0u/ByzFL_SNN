#!/usr/bin/env python3
import json
import os
import shutil
from byzfl.benchmark.benchmark import run_benchmark
from torchvision import datasets

def main():
    print("Pre-downloading CIFAR-10 to prevent multiprocessing race conditions...")
    datasets.CIFAR10(root='./data', train=True, download=True)
    datasets.CIFAR10(root='./data', train=False, download=True)

    config_path = "configs/cnn_baselines/tune_cifar_activations.json"

    # Clean previous results if they exist to avoid mixing
    if os.path.exists("./results/cnn_cifar_tuning"):
        shutil.rmtree("./results/cnn_cifar_tuning")

    print(f"--- Running tuning configurations in PARALLEL via Ray (Distributing GPUs) ---")
    run_benchmark(config_path, nb_jobs=6, distribute_gpus=True)
    print("Tuning completed! Results are in results/cnn_cifar_tuning")

if __name__ == "__main__":
    main()
