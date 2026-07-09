import os
import json
import matplotlib.pyplot as plt
import numpy as np
import sys

def parse_variance_results(results_dir):
    """
    Parses all subdirectories in results_dir, reads the variance metrics,
    and returns a dictionary mapping gamma -> mean_metric_value.
    """
    gamma_to_metrics = {}
    
    if not os.path.isdir(results_dir):
        print(f"Directory {results_dir} not found.")
        return gamma_to_metrics

    for folder in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder)
        if not os.path.isdir(folder_path):
            continue
            
        config_path = os.path.join(folder_path, "config.json")
        if not os.path.exists(config_path):
            continue
            
        with open(config_path, "r") as f:
            try:
                config = json.load(f)
                gamma = config["benchmark_config"]["data_distribution"]["distribution_parameter"]
            except Exception as e:
                print(f"Error parsing {config_path}: {e}")
                continue
                
        # Find all seed combinations for the metrics
        # The files are named like honest_var_trace_tr_seed_X_dd_seed_Y.txt
        metric_sums = {
            "var_trace": 0.0,
            "var_norm": 0.0,
            "mean_grad_norm": 0.0,
            "normalized_trace_var": 0.0,
            "normalized_norm_var": 0.0
        }
        metric_counts = {k: 0 for k in metric_sums}

        for file in os.listdir(folder_path):
            if not file.endswith(".txt"):
                continue
            
            for metric in metric_sums.keys():
                if file.startswith(f"honest_{metric}_tr_seed_"):
                    filepath = os.path.join(folder_path, file)
                    try:
                        data = np.loadtxt(filepath, delimiter=",")
                        # Average over time steps (axis 0)
                        mean_val = np.mean(data)
                        metric_sums[metric] += mean_val
                        metric_counts[metric] += 1
                    except Exception as e:
                        print(f"Error reading {filepath}: {e}")
        
        # Average across seeds
        avg_metrics = {}
        valid = True
        for metric in metric_sums:
            if metric_counts[metric] == 0:
                valid = False
                break
            avg_metrics[metric] = metric_sums[metric] / metric_counts[metric]
            
        if valid:
            gamma_to_metrics[gamma] = avg_metrics
            
    return gamma_to_metrics

def plot_comparisons(cnn_dir, snn_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    cnn_metrics = parse_variance_results(cnn_dir)
    snn_metrics = parse_variance_results(snn_dir)
    
    if not cnn_metrics and not snn_metrics:
        print("No metrics found to plot.")
        return
        
    gammas_cnn = sorted(list(cnn_metrics.keys()), reverse=True)
    gammas_snn = sorted(list(snn_metrics.keys()), reverse=True)
    
    # We will plot using gammas on X axis. Since 1.0 is IID and 0.0 is Non-IID,
    # it's usually plotted with 1.0 on left and 0.0 on right, or we can just plot as is.
    
    metrics_to_plot = [
        "var_trace", 
        "var_norm", 
        "mean_grad_norm", 
        "normalized_trace_var", 
        "normalized_norm_var"
    ]
    
    metric_titles = {
        "var_trace": "Raw Trace Variance (Sum of Coordinate Variances)",
        "var_norm": "Raw L2 Norm of Variance Vector",
        "mean_grad_norm": "Mean Honest Gradient L2 Norm",
        "normalized_trace_var": "Normalized Trace Variance (Trace Var / ||Mean Grad||^2)",
        "normalized_norm_var": "Normalized Norm Variance (Norm Var / ||Mean Grad||^2)"
    }

    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        
        if cnn_metrics:
            cnn_vals = [cnn_metrics[g][metric] for g in gammas_cnn]
            plt.plot(gammas_cnn, cnn_vals, marker='o', label='CNN (lr=0.15)', linewidth=2)
            
        if snn_metrics:
            snn_vals = [snn_metrics[g][metric] for g in gammas_snn]
            plt.plot(gammas_snn, snn_vals, marker='s', label='SNN (lr=0.10, Atan alpha=1.2)', linewidth=2)
            
        plt.gca().invert_xaxis() # 1.0 (IID) to 0.0 (Extreme Non-IID)
        plt.title(metric_titles[metric])
        plt.xlabel("Data Heterogeneity: Gamma (1.0 = IID, 0.0 = Non-IID)")
        plt.ylabel(metric_titles[metric])
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.yscale('log') # Use log scale because variance can explode
        
        save_path = os.path.join(output_dir, f"{metric}_comparison.pdf")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved {save_path}")
        
    print(f"All plots saved to {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python plot_variance_comparison.py <cnn_results_dir> <snn_results_dir> <output_dir>")
        sys.exit(1)
        
    plot_comparisons(sys.argv[1], sys.argv[2], sys.argv[3])
