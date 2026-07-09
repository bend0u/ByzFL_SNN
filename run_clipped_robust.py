#!/usr/bin/env python3
"""
Run CNN clipped robust experiment and generate comparison plots.
Compares: CNN (no clip) vs CNN (clip=21) vs SNN — all under Byzantine attack.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from byzfl.benchmark.benchmark import run_benchmark


def run_experiments():
    print("=" * 50)
    print("Running CNN Clipped Robust Experiment")
    print("=" * 50)
    run_benchmark('configs/cnn_clipped_robust.json', nb_jobs=20, distribute_gpus=True)
    print("\nAll experiments done!")


def find_experiment_folders(results_dir, attack_name, agg_name):
    """Find all seed folders for a given attack+aggregator in clean_directory_structure."""
    matches = []
    if not os.path.isdir(results_dir):
        return matches
    # With clean_directory_structure, the layout is:
    # results_dir/mnist_direct/<Attack>_<Agg>_f_1_gamma_similarity_niid_0.0/
    for parent in os.listdir(results_dir):
        parent_path = os.path.join(results_dir, parent)
        if not os.path.isdir(parent_path):
            continue
        for folder in os.listdir(parent_path):
            folder_path = os.path.join(parent_path, folder)
            if not os.path.isdir(folder_path):
                continue
            # Check if this folder matches our attack+aggregator
            if attack_name in folder and agg_name in folder:
                matches.append(folder_path)
    return matches


def read_metric_clean(folder_path, metric_name):
    """Read a metric file from a clean_directory_structure experiment folder."""
    # With clean structure, metrics are saved as e.g. honest_max_deviation.txt (one per seed)
    # But multiple seeds create separate folders, so each folder has one file
    filepath = os.path.join(folder_path, f"{metric_name}.txt")
    if os.path.exists(filepath):
        try:
            return np.loadtxt(filepath, delimiter=",")
        except Exception:
            pass
    # Also try seed-based naming
    arrays = []
    for f in os.listdir(folder_path):
        if f.startswith(metric_name) and f.endswith(".txt"):
            try:
                arrays.append(np.loadtxt(os.path.join(folder_path, f), delimiter=","))
            except Exception:
                pass
    if arrays:
        return np.mean(arrays, axis=0)
    return None


def generate_plots():
    print("\n" + "=" * 50)
    print("Generating Comparison Plots")
    print("=" * 50)

    clipped_dir = "results/cnn/clipped_robust"
    output_dir = "plots/clipped_comparison"
    os.makedirs(output_dir, exist_ok=True)

    attacks = [
        ("SignFlipping", "SignFlipping"),
        ("Optimal_InnerProductManipulation", "OptIPM"),
        ("Optimal_ALittleIsEnough_neg1", "OptALIE"),
    ]
    aggregators = [
        ("GeometricMedian", "GeoMed"),
        ("CenteredClipping", "CC"),
        ("TrMean", "TrMean"),
        ("MultiKrum", "MKrum"),
    ]

    metrics = [
        ("honest_max_deviation", "Max Deviation from Mean"),
        ("honest_grad_norm_max", "Max Gradient Norm"),
        ("honest_grad_norm_std", "Std Dev of Gradient Norms"),
        ("honest_mean_cos_sim", "Mean Cosine Similarity"),
        ("honest_max_abs_grad", "Max Absolute Gradient Coordinate"),
        ("test_accuracy", "Test Accuracy"),
    ]

    colors_model = {
        "CNN Clipped": "#1f77b4",
    }

    # For each attack x aggregator, plot each metric
    for attack_name, attack_label in attacks:
        for agg_name, agg_label in aggregators:
            folders = find_experiment_folders(clipped_dir, attack_name, agg_name)
            if not folders:
                print(f"  No results for {attack_label} + {agg_label}, skipping...")
                continue

            for metric_name, metric_title in metrics:
                fig, ax = plt.subplots(figsize=(10, 6))

                # Read from all seed folders
                arrays = []
                for folder in folders:
                    data = read_metric_clean(folder, metric_name)
                    if data is not None:
                        arrays.append(data)

                if not arrays:
                    plt.close()
                    continue

                all_data = np.array(arrays)
                mean_curve = all_data.mean(axis=0)
                std_curve = all_data.std(axis=0)
                steps = np.arange(len(mean_curve))

                ax.plot(steps, mean_curve, color=colors_model["CNN Clipped"],
                        label="CNN Clipped (21.0)", linewidth=2)
                ax.fill_between(steps, mean_curve - std_curve, mean_curve + std_curve,
                                color=colors_model["CNN Clipped"], alpha=0.2)

                ax.set_xlabel("Training Step", fontsize=12)
                ax.set_ylabel(metric_title, fontsize=12)
                ax.set_title(f"{metric_title}\n{attack_label} + {agg_label} (f=1, γ=0.0)",
                             fontsize=13)
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3)

                if metric_name != "test_accuracy" and metric_name != "honest_mean_cos_sim":
                    ax.set_yscale('log')

                fname = f"{metric_name}_{attack_label}_{agg_label}.pdf"
                fig.tight_layout()
                fig.savefig(os.path.join(output_dir, fname), dpi=150)
                plt.close()
                print(f"  Saved {fname}")

    print(f"\nAll plots saved in {output_dir}/")


if __name__ == "__main__":
    run_experiments()
    generate_plots()
