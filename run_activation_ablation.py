#!/usr/bin/env python3
"""
Run CNN activation function ablation study (Sigmoid vs Tanh vs ReLU).
None of the models use gradient clipping.
All use GeometricMedian + NNM_ARC, f=1, gamma=0.0, lr=0.15.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from byzfl.benchmark.benchmark import run_benchmark


def run_experiments():
    print("=" * 60)
    print("Running CNN Activation Ablation Sweep (Sigmoid & Tanh)")
    print("=" * 60)
    run_benchmark('configs/cnn_activation_ablation.json', nb_jobs=10, distribute_gpus=True)
    print("\nExperiments done!")


if __name__ == "__main__":
    run_experiments()

