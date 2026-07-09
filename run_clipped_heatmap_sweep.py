#!/usr/bin/env python3
import sys
import os
from byzfl.benchmark.benchmark import run_benchmark
from byzfl.benchmark.evaluate_results import test_heatmap, loss_heatmap, aggregated_test_heatmap

def main():
    config_path = "configs/cnn_clipped_heatmap_sweep.json"
    results_dir = "results/cnn/clipped_heatmap_sweep"
    plots_dir = "plots/cnn_clipped_heatmaps"

    print("=" * 60)
    print("Starting CNN Clipped Heatmap Sweep (GM + NNM_ARC, f=0..5, γ=0..1)")
    print("=" * 60)
    
    # Run the benchmark with 10 jobs, distributing across GPUs
    run_benchmark(config_path, nb_jobs=20, distribute_gpus=True)
    
    print("\n" + "=" * 60)
    print("Benchmark complete! Generating heatmaps...")
    print("=" * 60)
    
    os.makedirs(plots_dir, exist_ok=True)
    
    # We have 3 attacks in the config
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
            loss_heatmap(results_dir, plots_dir, target_attack=attack)
            print("    - Saved loss heatmaps")
        except Exception as e:
            print(f"    - Error generating loss heatmaps: {e}")
            
        try:
            aggregated_test_heatmap(results_dir, plots_dir, target_attack=attack)
            print("    - Saved aggregated test heatmaps")
        except Exception as e:
            print(f"    - Error generating aggregated test heatmaps: {e}")
            
    print(f"\nAll plots saved in {plots_dir}/")

if __name__ == "__main__":
    main()
