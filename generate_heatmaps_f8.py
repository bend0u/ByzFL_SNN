import os
import json
import traceback
from byzfl.benchmark.evaluate_results import test_heatmap, loss_heatmap, aggregated_test_heatmap

def main():
    experiments = [
        ("cnn_complete_results", "cnn_complete_plots_extended_f8"),
        ("snn_complete_results_direct", "snn_complete_plots_direct_extended_f8"),
        ("snn_complete_results_nmnist", "snn_complete_plots_nmnist_extended_f8"),
        ("snn_complete_results_rate", "snn_complete_plots_rate_extended_f8")
    ]
    
    attacks = [
        "Optimal_ALittleIsEnough", 
        "Optimal_InnerProductManipulation", 
        "SignFlipping"
    ]
    
    for results_dir, plots_dir in experiments:
        config_path = os.path.join(results_dir, 'config.json')
        if not os.path.exists(config_path):
            print(f"Config file '{config_path}' does not exist. Skipping.")
            continue
            
        print(f"\n=======================================================")
        print(f"Generating heatmaps up to f=8 for: {results_dir}")
        print(f"=======================================================")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Load original config
        with open(config_path, 'r') as file:
            original_config = json.load(file)
            
        try:
            # Modify config temporarily
            modified_config = original_config.copy()
            f_list = modified_config["benchmark_config"]["f"]
            modified_config["benchmark_config"]["f"] = [item for item in f_list if item <= 8]
            
            with open(config_path, 'w') as file:
                json.dump(modified_config, file, indent=4)
                
            # 1. Generate general heatmaps
            print("Generating general heatmaps...")
            test_heatmap(results_dir, plots_dir)
            try:
                loss_heatmap(results_dir, plots_dir)
            except Exception as e:
                print(f"Could not generate loss heatmap: {e}")
            aggregated_test_heatmap(results_dir, plots_dir)
            
            # 2. Generate fixed-attack heatmaps
            for attack in attacks:
                print(f"Generating plots for fixed attack: {attack}...")
                test_heatmap(results_dir, plots_dir, target_attack=attack)
                aggregated_test_heatmap(results_dir, plots_dir, target_attack=attack)
                
        except Exception as e:
            print(f"Error processing {results_dir}: {e}")
            traceback.print_exc()
        finally:
            # Always restore the original config
            with open(config_path, 'w') as file:
                json.dump(original_config, file, indent=4)
            print(f"Restored original config.json for {results_dir}")

    print("\nAll heatmaps up to f=8 successfully generated!")

if __name__ == "__main__":
    main()
