#!/usr/bin/env python3
import os
import argparse
import sys
import json
from byzfl.benchmark.benchmark import run_benchmark
from byzfl.benchmark.evaluate_results import test_heatmap, aggregated_test_heatmap
from torchvision import datasets

def main():
    parser = argparse.ArgumentParser(description="Run CIFAR-10 Sweep with Custom Config")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON configuration file")
    parser.add_argument("--gpu", type=str, default="0", help="GPU index to use (e.g. 0 or 1)")
    parser.add_argument("--nb_jobs", type=int, default=20, help="Number of parallel jobs to run")
    parser.add_argument("--distribute_gpus", action="store_true", help="Distribute jobs across all GPUs")
    
    args = parser.parse_args()
    
    config_path = args.config
    gpu_idx = args.gpu
    nb_jobs = args.nb_jobs
    distribute = args.distribute_gpus
    
    if not os.path.exists(config_path):
        print(f"Error: configuration file '{config_path}' does not exist.")
        sys.exit(1)
        
    print(f"=======================================================")
    if distribute:
        print(f"Starting CIFAR Sweep using {config_path} distributed across all available GPUs with {nb_jobs} jobs")
    else:
        print(f"Starting CIFAR Sweep using {config_path} on GPU {gpu_idx} with {nb_jobs} jobs")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx

    print("Pre-downloading CIFAR-10 to prevent multiprocessing race conditions...")
    datasets.CIFAR10(root='./data', train=True, download=True)
    datasets.CIFAR10(root='./data', train=False, download=True)

    # Run the benchmark
    run_benchmark(config_path, nb_jobs=nb_jobs, distribute_gpus=distribute)
    print("Sweep completed!")

    # Read config to find results dir
    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)
        results_dir = config_data["evaluation_and_results"]["results_directory"]
    except Exception as e:
        print(f"Error reading config {config_path} for results dir: {e}")
        return

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
