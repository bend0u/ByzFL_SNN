import os
import json
import shutil
from byzfl.benchmark.evaluate_results import test_heatmap, loss_heatmap, aggregated_test_heatmap

def generate_range_heatmaps(results_dir, plots_dir):
    config_path = os.path.join(results_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return

    # Backup original config
    backup_config_path = config_path + ".bak"
    shutil.copyfile(config_path, backup_config_path)

    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)

        # Detect varying surrogate parameter
        model_params = config_data["model"]["model_params"]
        surrogate_params = model_params["surrogate_params"]
        
        varying_param = None
        for key, val in surrogate_params.items():
            if isinstance(val, list):
                varying_param = key
                param_values = val
                break
        attacks_list = config_data.get("attack", [])
        if isinstance(attacks_list, dict):
            attacks_list = [attacks_list]
        attack_names = [None] + [att["name"] for att in attacks_list if isinstance(att, dict) and "name" in att]

        if not varying_param:
            print("No varying surrogate parameter list found in config.json. Plotting normally...")
            for attack_name in attack_names:
                suffix = f"_{attack_name}" if attack_name else "_merged"
                print(f"Generating normal plots for attack: {attack_name if attack_name else 'merged'}...")
                test_heatmap(results_dir, plots_dir, target_attack=attack_name)
                loss_heatmap(results_dir, plots_dir, target_attack=attack_name)
                aggregated_test_heatmap(results_dir, plots_dir, target_attack=attack_name)
            return

        print(f"Detected varying parameter: '{varying_param}' with values {param_values}")

        for val in param_values:
            print(f"\nGenerating plots for {varying_param} = {val}...")
            
            # Create a temporary config with the single parameter value
            temp_config = json.loads(json.dumps(config_data)) # deep copy
            temp_config["model"]["model_params"]["surrogate_params"][varying_param] = val
            
            with open(config_path, "w") as f:
                json.dump(temp_config, f, indent=4)
                
            # Run library plotting functions
            val_plots_dir = os.path.join(plots_dir, f"{varying_param}_{val}")
            os.makedirs(val_plots_dir, exist_ok=True)
            
            for attack_name in attack_names:
                attack_label = attack_name if attack_name else "merged"
                print(f"--> Target Attack: {attack_label}")
                
                try:
                    test_heatmap(results_dir, val_plots_dir, target_attack=attack_name)
                    print(f"    - Saved line plots")
                except Exception as e:
                    print(f"    - Error generating test line plots: {e}")

                try:
                    loss_heatmap(results_dir, val_plots_dir, target_attack=attack_name)
                    print(f"    - Saved loss heatmaps")
                except Exception as e:
                    print(f"    - Error generating loss heatmaps: {e}")

                try:
                    aggregated_test_heatmap(results_dir, val_plots_dir, target_attack=attack_name)
                    print(f"    - Saved aggregated test heatmaps")
                except Exception as e:
                    print(f"    - Error generating aggregated test heatmaps: {e}")
    finally:
        # Restore original config
        if os.path.exists(backup_config_path):
            shutil.move(backup_config_path, config_path)
            print("\nRestored original config.json successfully.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python plot_snn_robust_heatmaps.py <results_dir> <plots_dir>")
        sys.exit(1)
    generate_range_heatmaps(sys.argv[1], sys.argv[2])
