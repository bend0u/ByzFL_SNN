import os
import json
import shutil
import sys
from byzfl.benchmark.evaluate_results import test_heatmap, loss_heatmap, aggregated_test_heatmap

def generate_lr_heatmaps(results_dir, plots_dir):
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

        # Detect varying learning rate parameter
        lr_val = config_data["model"]["learning_rate"]
        
        varying_param = None
        if isinstance(lr_val, list):
            varying_param = "learning_rate"
            param_values = lr_val
        
        attacks_list = config_data.get("attack", [])
        if isinstance(attacks_list, dict):
            attacks_list = [attacks_list]
        attack_names = [None] + [att["name"] for att in attacks_list if isinstance(att, dict) and "name" in att]

        if not varying_param:
            print("No varying learning rate list found in config.json. Plotting normally...")
            for attack_name in attack_names:
                suffix = f"_{attack_name}" if attack_name else "_merged"
                print(f"Generating normal plots for attack: {attack_name if attack_name else 'merged'}...")
                test_heatmap(results_dir, plots_dir, target_attack=attack_name)
                loss_heatmap(results_dir, plots_dir, target_attack=attack_name)
                aggregated_test_heatmap(results_dir, plots_dir, target_attack=attack_name)
            return

        print(f"Detected varying parameter: '{varying_param}' with values {param_values}")

        # Temporarily hide best_hyperparameters directory if it exists,
        # so the library plotting functions fall back to using the single learning rate in the config.
        best_hyperparameters_dir = os.path.join(results_dir, "best_hyperparameters")
        hidden_hyperparameters_dir = os.path.join(results_dir, "best_hyperparameters.bak")
        has_hyperparameters = os.path.exists(best_hyperparameters_dir)
        
        if has_hyperparameters:
            print("Temporarily hiding best_hyperparameters directory to force plotting specific learning rates...")
            shutil.move(best_hyperparameters_dir, hidden_hyperparameters_dir)

        try:
            for val in param_values:
                print(f"\nGenerating plots for {varying_param} = {val}...")
                
                # Create a temporary config with the single parameter value
                temp_config = json.loads(json.dumps(config_data)) # deep copy
                temp_config["model"]["learning_rate"] = val
                
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
            # Restore best_hyperparameters directory if it was hidden
            if has_hyperparameters and os.path.exists(hidden_hyperparameters_dir):
                shutil.move(hidden_hyperparameters_dir, best_hyperparameters_dir)
                print("\nRestored best_hyperparameters directory.")

    finally:
        # Restore original config
        if os.path.exists(backup_config_path):
            shutil.move(backup_config_path, config_path)
            print("\nRestored original config.json successfully.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_cnn_lr_robust_heatmaps.py <results_dir> <plots_dir>")
        sys.exit(1)
    generate_lr_heatmaps(sys.argv[1], sys.argv[2])
