#!/usr/bin/env python3
"""
Generic runner for the gradient-preserving activation-clipping / adaptive
client-norm-clipping sweep configs under configs/activation_clip/.

Heatmaps are written to results/activation_clip_plots/<variant>/ -- i.e. UNDER
results/, which on RunAI is the only PVC-mounted (persistent) tree, so plots
survive pod termination. This is also the location the LaTeX report expects.

Can be used two ways:
  * as a CLI:      python scripts/experiments/run_activation_clip_sweep.py \
                       --config configs/activation_clip/cnn_mnist_clip_ste_1.json \
                       --distribute_gpus --nb_jobs 50
  * as a library:  from run_activation_clip_sweep import run_sweep
                   run_sweep("configs/.../x.json", nb_jobs=40, distribute_gpus=True)
"""
import os
import sys
import json
import argparse

from byzfl.benchmark.benchmark import run_benchmark
from byzfl.benchmark.evaluate_results import test_heatmap, loss_heatmap, aggregated_test_heatmap


def run_sweep(config_path, nb_jobs=20, distribute_gpus=False, gpu="0"):
    """
    Run one sweep config end-to-end: pre-download the dataset, run the benchmark,
    then generate heatmaps into results/activation_clip_plots/<variant>/.

    Parameters
    ----------
    config_path : str
        Path to the JSON sweep configuration.
    nb_jobs : int
        Number of parallel jobs.
    distribute_gpus : bool
        Distribute jobs across all visible GPUs. If False, pin to `gpu`.
    gpu : str
        GPU index to pin to when distribute_gpus is False.
    """
    if not os.path.exists(config_path):
        print(f"Error: configuration file '{config_path}' does not exist.")
        sys.exit(1)

    with open(config_path, "r") as f:
        cfg = json.load(f)

    dataset_name = cfg.get("model", {}).get("dataset_name", "mnist").lower()
    data_folder = cfg.get("evaluation_and_results", {}).get("data_folder", "./data")
    if dataset_name == "mnist":
        try:
            print("Pre-downloading MNIST dataset sequentially to avoid parallel race conditions...")
            from torchvision import datasets
            datasets.MNIST(root=data_folder, train=True, download=True)
            datasets.MNIST(root=data_folder, train=False, download=True)
            print("MNIST dataset is ready!")
        except Exception as e:
            print(f"Warning: could not pre-download dataset: {e}")

    if not distribute_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    print("=" * 60)
    print(f"Starting sweep: {config_path}")
    print(f"  model.name = {cfg.get('model', {}).get('name')}")
    print(f"  nb_jobs = {nb_jobs}, distribute_gpus = {distribute_gpus}")
    print("=" * 60)

    run_benchmark(config_path, nb_jobs=nb_jobs, distribute_gpus=distribute_gpus)

    results_dir = cfg["evaluation_and_results"]["results_directory"]
    # Write plots under results/ (the only PVC-mounted tree on RunAI) so they
    # persist after the pod terminates, into results/activation_clip_plots/<variant>
    # -- the same location manual copies + the LaTeX report already expect.
    variant = os.path.basename(results_dir.rstrip("/\\"))
    plots_dir = os.path.join("results", "activation_clip_plots", variant)
    os.makedirs(plots_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("Benchmark complete! Generating heatmaps...")
    print(f"  results_dir = {results_dir}")
    print(f"  plots_dir   = {plots_dir}")
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


def main():
    parser = argparse.ArgumentParser(description="Run one activation-clip / adaptive-clip sweep config")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON configuration file")
    parser.add_argument("--gpu", type=str, default="0", help="GPU index to use (e.g. 0 or 1), ignored if --distribute_gpus")
    parser.add_argument("--nb_jobs", type=int, default=20, help="Number of parallel jobs to run")
    parser.add_argument("--distribute_gpus", action="store_true", help="Distribute jobs across all available GPUs")
    args = parser.parse_args()

    run_sweep(
        args.config,
        nb_jobs=args.nb_jobs,
        distribute_gpus=args.distribute_gpus,
        gpu=args.gpu,
    )


if __name__ == "__main__":
    main()
