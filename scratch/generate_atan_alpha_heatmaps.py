import os
import sys
import json
import shutil

# Ensure workspace root is in Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from byzfl.benchmark.evaluate_results import aggregated_test_heatmap

def main():
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(workspace_dir, "results", "snn", "robust_new_atan_sweep")
    plots_dir = os.path.join(workspace_dir, "plots", "snn", "robust_new_atan_sweep_f5")

    config_path = os.path.join(results_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        sys.exit(1)

    # Backup original config
    backup_config_path = config_path + ".bak"
    shutil.copyfile(config_path, backup_config_path)

    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)

        # Set varying parameter (alpha) and values
        varying_param = "alpha"
        param_values = [1.0, 1.25, 1.5, 2.0, 3.0]

        # Limit f to [0, 1, 2, 3, 4, 5]
        config_data["benchmark_config"]["f"] = [0, 1, 2, 3, 4, 5]

        for val in param_values:
            print(f"\nGenerating aggregated heatmap for alpha = {val}...")
            
            # Create a temporary config with the single alpha value
            temp_config = json.loads(json.dumps(config_data)) # deep copy
            temp_config["model"]["model_params"]["surrogate_params"][varying_param] = val
            
            with open(config_path, "w") as f:
                json.dump(temp_config, f, indent=4)
                
            val_plots_dir = os.path.join(plots_dir, f"alpha_{val}")
            os.makedirs(val_plots_dir, exist_ok=True)
            
            try:
                aggregated_test_heatmap(results_dir, val_plots_dir, metric="best_step")
                print(f"--> Saved aggregated test heatmap in {val_plots_dir}")
            except Exception as e:
                print(f"--> Error generating aggregated test heatmap for alpha={val}: {e}")

    finally:
        # Restore original config
        if os.path.exists(backup_config_path):
            shutil.move(backup_config_path, config_path)
            print("\nRestored original config.json successfully.")

if __name__ == "__main__":
    main()
