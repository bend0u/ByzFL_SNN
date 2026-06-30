import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt

def generate_plots(results_dir, plots_dir):
    print(f"Scanning results directory: {results_dir}")
    if not os.path.exists(results_dir):
        print(f"Error: results directory '{results_dir}' does not exist.")
        return

    os.makedirs(plots_dir, exist_ok=True)

    # 1. Discover subfolders
    subfolders = [f for f in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, f)) and f != "best_hyperparameters"]
    if not subfolders:
        print("No result subfolders found.")
        return

    # 2. Extract parameters by splitting on underscores
    folder_params = []
    for folder in subfolders:
        params_dict = {}
        parts = folder.split('_')
        for idx in range(len(parts) - 1):
            key = parts[idx]
            val_str = parts[idx + 1]
            try:
                # Exclude strings that look like hex IDs or non-parameter numbers
                if len(val_str) > 0 and (val_str[0].isdigit() or val_str.startswith('.')):
                    val = float(val_str)
                    params_dict[key] = val
            except ValueError:
                pass
        if params_dict:
            folder_params.append((folder, params_dict))

    if not folder_params:
        print("No numeric parameters found in folder names.")
        return

    # Find which parameter varies across folders
    all_keys = set()
    for _, p_dict in folder_params:
        all_keys.update(p_dict.keys())
        
    varying_param = None
    # Exclude standard FL settings to find the actual surrogate parameter
    excluded_keys = ["lr", "mom", "wd", "n", "f", "d", "ts", "seed", "delta"]
    for k in sorted(all_keys):
        if k in excluded_keys:
            continue
        unique_vals = set(p_dict[k] for _, p_dict in folder_params if k in p_dict)
        if len(unique_vals) > 1:
            varying_param = k
            break
            
    # Default fallback if nothing else found
    if not varying_param:
        for k in sorted(all_keys):
            if k not in excluded_keys:
                varying_param = k
                break
                
    if not varying_param:
        varying_param = list(all_keys)[0]

    print(f"Detected varying parameter for plotting: '{varying_param}'")

    param_data = {}
    for folder, p_dict in folder_params:
        if varying_param not in p_dict:
            continue
        param_val = p_dict[varying_param]
        
        folder_path = os.path.join(results_dir, folder)
        acc_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.startswith("test_accuracy_") and f.endswith(".txt")]
        if not acc_files:
            continue
            
        all_curves = []
        for f in acc_files:
            try:
                data = np.loadtxt(f, delimiter=',')
                all_curves.append(data)
            except Exception as e:
                print(f"Warning: could not read {f}: {e}")
                
        if all_curves:
            # Ensure all loaded curves are at least 1D arrays
            all_curves = [c if (isinstance(c, np.ndarray) and c.ndim > 0) else np.array([c]) for c in all_curves]
            
            # Read config to get step info
            try:
                with open(os.path.join(results_dir, 'config.json'), 'r') as f_cfg:
                    import json
                    cfg_data = json.load(f_cfg)
                    evaluation_delta = cfg_data["evaluation_and_results"]["evaluation_delta"]
                    nb_steps = cfg_data["benchmark_config"]["nb_steps"]
            except:
                evaluation_delta = 50
                nb_steps = 500
            
            target_len = 1 + (nb_steps // evaluation_delta)
            
            # If at least one seed has finished, filter out incomplete seeds
            finished_curves = [c for c in all_curves if len(c) >= target_len]
            if finished_curves:
                all_curves = finished_curves
            
            # Find the minimum length among remaining curves
            min_len = min(len(c) for c in all_curves)
            all_curves = [c[:min_len] for c in all_curves]
            
            all_curves = np.array(all_curves)
            mean_curve = np.mean(all_curves, axis=0)
            std_curve = np.std(all_curves, axis=0)
            
            steps = np.arange(0, nb_steps + evaluation_delta, evaluation_delta)
            
            # Ensure step count aligns with data
            if len(steps) > len(mean_curve):
                steps = steps[:len(mean_curve)]
            elif len(steps) < len(mean_curve):
                mean_curve = mean_curve[:len(steps)]
                std_curve = std_curve[:len(steps)]
                
            param_data[param_val] = {
                "name": varying_param,
                "steps": steps,
                "mean": mean_curve,
                "std": std_curve,
                "final_acc": mean_curve[-1]
            }

    if not param_data:
        print("No valid parameter sweeps found with test accuracy logs.")
        return

    # Sort parameters
    sorted_params = sorted(param_data.keys())

    # 3. Create Accuracy Plot
    plt.figure(figsize=(10, 6), dpi=150)
    
    # Choose a nice colormap
    colors = plt.cm.plasma(np.linspace(0, 0.85, len(sorted_params)))

    for i, val in enumerate(sorted_params):
        data = param_data[val]
        lbl = f"{varying_param} = {val} (Final: {data['final_acc']:.4f})"
        plt.plot(data["steps"], data["mean"], label=lbl, color=colors[i], linewidth=2.0)
        plt.fill_between(data["steps"], data["mean"] - data["std"], data["mean"] + data["std"], color=colors[i], alpha=0.1)

    plt.title(f"SNN Baseline Convergence Sweep over surrogate gradient {varying_param.upper()} (f=0)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Training Steps", fontsize=12, labelpad=10)
    plt.ylabel("Test Accuracy", fontsize=12, labelpad=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10)
    plt.tight_layout()

    plot_path = os.path.join(plots_dir, f"baseline_{varying_param}_convergence.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"--> Saved convergence plot: {plot_path}")

    # 4. Create final accuracy summary plot
    final_accs = [param_data[val]["final_acc"] for val in sorted_params]
    
    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(sorted_params, final_accs, marker='o', color='#3f51b5', linewidth=2.0, markersize=8)
    plt.title(f"Final Test Accuracy vs Surrogate Gradient {varying_param.upper()}", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel(f"{varying_param.upper()} parameter value", fontsize=11, labelpad=10)
    plt.ylabel("Final Test Accuracy (Step 500)", fontsize=11, labelpad=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    summary_path = os.path.join(plots_dir, f"final_accuracy_vs_{varying_param}.png")
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"--> Saved final accuracy summary: {summary_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_snn_baseline_convergence.py <results_dir> <plots_dir>")
        sys.exit(1)
    
    generate_plots(sys.argv[1], sys.argv[2])
