#!/usr/bin/env python3
import os
from byzfl.benchmark.benchmark import run_benchmark
from torchvision import datasets

def main():
    print("Pre-downloading MNIST to prevent multiprocessing race conditions...")
    datasets.MNIST(root='./data', train=True, download=True)
    datasets.MNIST(root='./data', train=False, download=True)

    print("Starting CNN Tanh Dropout Calibration Sweep...")
    config_path = "configs/dropout_calibration.json"
    
    try:
        run_benchmark(config_path, nb_jobs=15, distribute_gpus=True)
        print("Dropout calibration completed successfully!")
    except Exception as e:
        print(f"Error during dropout calibration: {e}")

if __name__ == "__main__":
    main()
