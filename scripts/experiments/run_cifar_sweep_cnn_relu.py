#!/usr/bin/env python3
import os
import json
from byzfl.benchmark.benchmark import run_benchmark
from torchvision import datasets
from byzfl.benchmark.evaluate_results import test_heatmap, aggregated_test_heatmap

def main():
    print("Pre-downloading CIFAR-10 to prevent multiprocessing race conditions...")
    datasets.CIFAR10(root='./data', train=True, download=True)
    datasets.CIFAR10(root='./data', train=False, download=True)

    config_path = os.environ.get("CONFIG_PATH", "configs/cifar_sweep_cnn_relu.json")

    print("=" * 60)
    print("CIFAR-10 Sweep: CNN (ReLU)")
    print("  Running on 4 GPUs (V100)...")
    print("=" * 60)

    # Use nb_jobs=20 to distribute across the 4 GPUs (5 jobs per GPU)
    run_benchmark(config_path, nb_jobs=20, distribute_gpus=True)
    print("Sweep completed!")

    # Read config to find results dir
    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)
        results_dir = config_data["evaluation_and_results"]["results_directory"]
    except Exception as e:
        print(f"Error reading config {config_path} for results dir: {e}")
        results_dir = "results/cifar_sweep_cnn_relu"

    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print(f"Generating heatmaps in {plots_dir}...")
    print("=" * 60)
    
    attacks = [None, "SignFlipping", "Optimal_InnerProductManipulation", "Optimal_ALittleIsEnough_neg1"]
    for attack in attacks:
        attack_label = attack if attack else "merged"
        print(f"\n--> Generating heatmaps for attack: {attack_label}")
        try:
            test_heatmap(results_dir, plots_dir, target_attack=attack)
            print("    - Saved line plots")
        except Exception as e:
            print(f"    - Error generating test line plots: {e}")
            
        try:
            aggregated_test_heatmap(results_dir, plots_dir, target_attack=attack)
            print("    - Saved aggregated test heatmaps")
        except Exception as e:
            print(f"    - Error generating aggregated test heatmaps: {e}")
            
    print(f"\nAll plots saved in {plots_dir}/")

if __name__ == "__main__":
    main()
