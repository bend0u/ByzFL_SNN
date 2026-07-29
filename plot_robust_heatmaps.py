import os
import sys
import json
from byzfl.benchmark.evaluate_results import test_heatmap, loss_heatmap, aggregated_test_heatmap

def plot_all(results_dir, plots_dir):
    config_path = os.path.join(results_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return

    os.makedirs(plots_dir, exist_ok=True)
    
    with open(config_path, "r") as f:
        config_data = json.load(f)

    attacks_list = config_data.get("attack", [])
    if isinstance(attacks_list, dict):
        attacks_list = [attacks_list]
        
    attack_names = [None] + [att["name"] for att in attacks_list if isinstance(att, dict) and "name" in att]

    for attack_name in attack_names:
        suffix = f"_{attack_name}" if attack_name else "_merged"
        print(f"Generating plots for attack: {attack_name if attack_name else 'merged'}...")
        
        try:
            test_heatmap(results_dir, plots_dir, target_attack=attack_name)
            print("  - Saved test heatmaps")
        except Exception as e:
            print(f"  - Error test heatmaps: {e}")
            
        try:
            loss_heatmap(results_dir, plots_dir, target_attack=attack_name)
            print("  - Saved loss heatmaps")
        except Exception as e:
            print(f"  - Error loss heatmaps: {e}")
            
        try:
            aggregated_test_heatmap(results_dir, plots_dir, target_attack=attack_name)
            print("  - Saved aggregated test heatmaps")
        except Exception as e:
            print(f"  - Error aggregated test heatmaps: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_robust_heatmaps.py <results_dir> <plots_dir>")
        sys.exit(1)
    plot_all(sys.argv[1], sys.argv[2])
