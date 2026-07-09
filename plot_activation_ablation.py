#!/usr/bin/env python3
"""
Plot the results of the CNN activation function ablation study (Sigmoid vs Tanh vs ReLU).
Reads results and plots accuracy and gradient metrics.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read_metric_clean(folder_path, metric_name):
    filepath = os.path.join(folder_path, f"{metric_name}.txt")
    if os.path.exists(filepath):
        try:
            return np.loadtxt(filepath, delimiter=",")
        except Exception:
            pass
    return None


def find_folders(results_dir, model_name, attack_name):
    import json
    
    # Detect if the results directory was run with clean_directory_structure=True
    is_clean = False
    config_path = os.path.join(results_dir, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                data = json.load(f)
                is_clean = data.get("clean_directory_structure", False) or \
                           data.get("evaluation_and_results", {}).get("clean_directory_structure", False) or \
                           data.get("benchmark_config", {}).get("clean_directory_structure", False)
        except:
            pass
            
    matches = []
    if not os.path.isdir(results_dir):
        return matches
        
    for root, dirs, files in os.walk(results_dir):
        for d in dirs:
            folder_path = os.path.join(root, d)
            # Check if this folder has result text files (leaves)
            try:
                folder_files = os.listdir(folder_path)
                has_txt = any(f.endswith('.txt') for f in folder_files)
                if not has_txt:
                    continue
            except Exception:
                continue
                
            if is_clean:
                if attack_name in d:
                    matches.append(folder_path)
            else:
                if model_name in d and attack_name in d:
                    matches.append(folder_path)
    return list(set(matches))


def collect_metric(folders, metric_name):
    arrays = []
    for folder in folders:
        try:
            files = os.listdir(folder)
        except Exception:
            continue
        for f in files:
            if f == f"{metric_name}.txt" or (f.startswith(metric_name) and f.endswith(".txt")):
                filepath = os.path.join(folder, f)
                try:
                    data = np.loadtxt(filepath, delimiter=",")
                    # If it's a 1D array or has content, collect it
                    if data.ndim > 0 and len(data) > 0:
                        arrays.append(data)
                except Exception:
                    pass
    if not arrays:
        return None, None
    all_data = np.array(arrays)
    return all_data.mean(axis=0), all_data.std(axis=0)


def generate_plots():
    print("=" * 60)
    print("Generating Ablation Comparison Plots")
    print("=" * 60)

    noclip_relu_dir = "results/cnn/noclip_robust"
    clipped_relu_dir = "results/cnn/clipped_robust_nnmarc"
    ablation_dir = "results/cnn/activation_ablation"
    output_dir = "plots/activation_ablation"
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
    ]

    models = [
        ("cnn_mnist", "ReLU (No Clip)", "#d62728", noclip_relu_dir),            # red
        ("cnn_mnist", "ReLU (Clip=21.0)", "#1f77b4", clipped_relu_dir),         # blue
        ("cnn_mnist_sigmoid", "Sigmoid (No Clip)", "#2ca02c", ablation_dir),    # green
        ("cnn_mnist_tanh", "Tanh (No Clip)", "#ff7f0e", ablation_dir),          # orange
    ]

    for attack_name, attack_label in attacks:
        for metric_name, metric_title in metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            has_data = False

            for model_key, model_label, color, results_dir in models:
                folders = find_folders(results_dir, model_key, attack_name)
                if not folders:
                    continue

                mean_val, std_val = collect_metric(folders, metric_name)
                if mean_val is not None:
                    steps = np.arange(len(mean_val))
                    ax.plot(steps, mean_val, color=color, label=f"CNN {model_label}",
                            linewidth=2)
                    ax.fill_between(steps, mean_val - std_val, mean_val + std_val,
                                    color=color, alpha=0.15)
                    has_data = True

            if not has_data:
                plt.close()
                continue

            ax.set_xlabel("Training Step", fontsize=12)
            ax.set_ylabel(metric_title, fontsize=12)
            ax.set_title(f"{metric_title}\n{attack_label} | GM + NNM_ARC | f=1, γ=0.0 (No Clip)",
                         fontsize=13)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)

            if metric_name not in ("test_accuracy", "honest_mean_cos_sim"):
                ax.set_yscale('log')

            fname_pdf = f"{metric_name}_{attack_label}.pdf"
            fname_png = f"{metric_name}_{attack_label}.png"
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, fname_pdf), dpi=150)
            fig.savefig(os.path.join(output_dir, fname_png), dpi=150)
            plt.close()
            print(f"  Saved {fname_pdf} and {fname_png}")

    print(f"\nAll plots saved in {output_dir}/")


if __name__ == "__main__":
    generate_plots()
