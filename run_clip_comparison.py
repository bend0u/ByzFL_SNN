#!/usr/bin/env python3
"""
Compare CNN with clipping vs CNN without clipping.
Both use GeometricMedian + NNM_ARC, f=1, gamma=0, lr=0.15.
3 attacks: SignFlipping, OptIPM, OptALIE.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from byzfl.benchmark.benchmark import run_benchmark


def run_experiments():
    print("=" * 60)
    print("Step 1: CNN WITHOUT clipping")
    print("=" * 60)
    run_benchmark('configs/cnn_noclip_robust.json', nb_jobs=15, distribute_gpus=True)

    print("\n" + "=" * 60)
    print("Step 2: CNN WITH clipping (clip=21.0)")
    print("=" * 60)
    run_benchmark('configs/cnn_clipped_robust_nnmarc.json', nb_jobs=15, distribute_gpus=True)

    print("\nAll experiments done!")


def read_metric_clean(folder_path, metric_name):
    filepath = os.path.join(folder_path, f"{metric_name}.txt")
    if os.path.exists(filepath):
        try:
            return np.loadtxt(filepath, delimiter=",")
        except Exception:
            pass
    return None


def find_attack_folders(results_dir, attack_name):
    matches = []
    if not os.path.isdir(results_dir):
        return matches
    for parent in os.listdir(results_dir):
        parent_path = os.path.join(results_dir, parent)
        if not os.path.isdir(parent_path):
            continue
        for folder in os.listdir(parent_path):
            folder_path = os.path.join(parent_path, folder)
            if not os.path.isdir(folder_path) or attack_name not in folder:
                continue
            matches.append(folder_path)
    return matches


def collect_metric(folders, metric_name):
    arrays = []
    for folder in folders:
        data = read_metric_clean(folder, metric_name)
        if data is not None:
            arrays.append(data)
    if not arrays:
        return None, None
    all_data = np.array(arrays)
    return all_data.mean(axis=0), all_data.std(axis=0)


def generate_plots():
    print("\n" + "=" * 60)
    print("Generating Comparison Plots")
    print("=" * 60)

    noclip_dir = "results/cnn/noclip_robust"
    clipped_dir = "results/cnn/clipped_robust_nnmarc"
    output_dir = "plots/clip_vs_noclip"
    os.makedirs(output_dir, exist_ok=True)

    attacks = [
        ("SignFlipping", "SignFlipping"),
        ("Optimal_InnerProductManipulation", "OptIPM"),
        ("Optimal_ALittleIsEnough_neg1", "OptALIE"),
    ]

    metrics = [
        ("test_accuracy", "Test Accuracy"),
        ("honest_max_deviation", "Max Deviation from Mean"),
        ("honest_grad_norm_max", "Max Gradient Norm"),
        ("honest_grad_norm_std", "Std Dev of Gradient Norms"),
        ("honest_mean_cos_sim", "Mean Cosine Similarity"),
        ("honest_max_abs_grad", "Max Absolute Gradient Coordinate"),
    ]

    color_noclip = "#d62728"   # red
    color_clipped = "#1f77b4"  # blue

    for attack_name, attack_label in attacks:
        noclip_folders = find_attack_folders(noclip_dir, attack_name)
        clipped_folders = find_attack_folders(clipped_dir, attack_name)

        if not noclip_folders and not clipped_folders:
            print(f"  No results for {attack_label}, skipping...")
            continue

        for metric_name, metric_title in metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            has_data = False

            # CNN no clip
            mean_nc, std_nc = collect_metric(noclip_folders, metric_name)
            if mean_nc is not None:
                steps = np.arange(len(mean_nc))
                ax.plot(steps, mean_nc, color=color_noclip, label="CNN (no clip)",
                        linewidth=2, linestyle='-')
                ax.fill_between(steps, mean_nc - std_nc, mean_nc + std_nc,
                                color=color_noclip, alpha=0.15)
                has_data = True

            # CNN clipped
            mean_cl, std_cl = collect_metric(clipped_folders, metric_name)
            if mean_cl is not None:
                steps = np.arange(len(mean_cl))
                ax.plot(steps, mean_cl, color=color_clipped, label="CNN (clip=21.0)",
                        linewidth=2, linestyle='--')
                ax.fill_between(steps, mean_cl - std_cl, mean_cl + std_cl,
                                color=color_clipped, alpha=0.15)
                has_data = True

            if not has_data:
                plt.close()
                continue

            ax.set_xlabel("Training Step", fontsize=12)
            ax.set_ylabel(metric_title, fontsize=12)
            ax.set_title(f"{metric_title}\n{attack_label} | GeoMed + NNM_ARC | f=1, γ=0.0",
                         fontsize=13)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)

            if metric_name not in ("test_accuracy", "honest_mean_cos_sim"):
                ax.set_yscale('log')

            fname = f"{metric_name}_{attack_label}.pdf"
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, fname), dpi=150)
            plt.close()
            print(f"  Saved {fname}")

    print(f"\nAll plots saved in {output_dir}/")


if __name__ == "__main__":
    run_experiments()
    generate_plots()
