import os
import json
import numpy as np
import matplotlib.pyplot as plt

def find_folders_for_gamma(results_dir, gamma):
    matches = []
    if not os.path.isdir(results_dir):
        return matches
    for folder in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        config_path = os.path.join(folder_path, "config.json")
        if not os.path.exists(config_path):
            continue
        with open(config_path) as f:
            config = json.load(f)
        try:
            g = config["benchmark_config"]["data_distribution"]["distribution_parameter"]
            if abs(g - gamma) < 1e-6:
                matches.append(folder_path)
        except KeyError:
            pass
    return matches

def read_metric_all_seeds(folder_path, metric_prefix):
    arrays = []
    for file in sorted(os.listdir(folder_path)):
        if file.startswith(metric_prefix) and file.endswith(".txt"):
            filepath = os.path.join(folder_path, file)
            try:
                data = np.loadtxt(filepath, delimiter=",")
                arrays.append(data)
            except Exception as e:
                pass
    if arrays:
        return np.array(arrays)
    return None

def main():
    cnn_dir = "results/cnn/variance_sweep"
    snn_dir = "results/snn/variance_sweep"
    output_dir = "plots/robustness_metrics"
    os.makedirs(output_dir, exist_ok=True)
    
    gammas = [1.0, 0.33, 0.2, 0.1, 0.0]
    
    # Metrics to plot
    metrics = [
        ("honest_max_deviation", "Max Deviation from Mean (max ||g_i - E[g]||)"),
        ("honest_grad_norm_max", "Max Gradient Norm (max ||g_i||)"),
        ("honest_grad_norm_min", "Min Gradient Norm (min ||g_i||)"),
        ("honest_grad_norm_std", "Std Dev of Gradient Norms"),
        ("honest_mean_cos_sim", "Mean Cosine Similarity to Mean Gradient"),
        ("honest_max_abs_grad", "Max Absolute Coordinate Value (max |g_i,j|)")
    ]
    
    colors = {1.0: "#2ca02c", 0.33: "#ff7f0e", 0.2: "#9467bd", 0.1: "#e377c2", 0.0: "#d62728"}
    
    for metric_prefix, metric_title in metrics:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        has_data = False
        
        for gamma in gammas:
            # CNN
            cnn_folders = find_folders_for_gamma(cnn_dir, gamma)
            cnn_arrays = []
            for folder in cnn_folders:
                data = read_metric_all_seeds(folder, metric_prefix)
                if data is not None:
                    cnn_arrays.append(data)
            if cnn_arrays:
                has_data = True
                all_data = np.concatenate(cnn_arrays, axis=0)
                mean_curve = all_data.mean(axis=0)
                std_curve = all_data.std(axis=0)
                steps = np.arange(len(mean_curve))
                ax.plot(steps, mean_curve, linewidth=2, color=colors[gamma],
                        label=f"CNN γ={gamma}")
                ax.fill_between(steps, mean_curve - std_curve, mean_curve + std_curve,
                                alpha=0.15, color=colors[gamma])
            
            # SNN
            snn_folders = find_folders_for_gamma(snn_dir, gamma)
            snn_arrays = []
            for folder in snn_folders:
                data = read_metric_all_seeds(folder, metric_prefix)
                if data is not None:
                    snn_arrays.append(data)
            if snn_arrays:
                has_data = True
                all_data = np.concatenate(snn_arrays, axis=0)
                mean_curve = all_data.mean(axis=0)
                std_curve = all_data.std(axis=0)
                steps = np.arange(len(mean_curve))
                ax.plot(steps, mean_curve, linewidth=2, linestyle="--",
                        color=colors[gamma],
                        label=f"SNN γ={gamma}")
                ax.fill_between(steps, mean_curve - std_curve, mean_curve + std_curve,
                                alpha=0.15, color=colors[gamma])
        
        if not has_data:
            plt.close(fig)
            continue
            
        ax.set_xlabel("Training Step", fontsize=13)
        ax.set_ylabel(metric_title, fontsize=13)
        ax.set_title(f"{metric_title} vs Training Step (CNN solid, SNN dashed)", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        
        if "max" in metric_prefix or "std" in metric_prefix or "min" in metric_prefix:
            ax.set_yscale('log')
        
        safe_name = metric_prefix.replace("honest_", "")
        save_path = os.path.join(output_dir, f"{safe_name}_vs_step.pdf")
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {save_path}")
    
    print(f"Done! Plots saved to {output_dir}")

if __name__ == "__main__":
    main()
