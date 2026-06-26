import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import copy
import traceback
from byzfl.benchmark.evaluate_results import test_heatmap, loss_heatmap, aggregated_test_heatmap

def main():
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_direct = os.path.join(script_dir, "snn_complete_results_direct")
    results_rate = os.path.join(script_dir, "snn_complete_results_rate")
    
    plots_base_dir = os.path.join(script_dir, "plots_direct_vs_rate_f10")
    plots_direct = os.path.join(plots_base_dir, "snn_direct")
    plots_rate = os.path.join(plots_base_dir, "snn_rate")
    
    os.makedirs(plots_direct, exist_ok=True)
    os.makedirs(plots_rate, exist_ok=True)
    
    experiments = [
        (results_direct, plots_direct),
        (results_rate, plots_rate)
    ]
    
    attacks = [
        "Optimal_ALittleIsEnough", 
        "Optimal_InnerProductManipulation", 
        "SignFlipping"
    ]
    
    # We want to force the config during plotting to use only seed 42,
    # and only the original three attacks (without delta=1 or delta=-10).
    PLOT_CONFIG_OVERRIDES = {
        "benchmark_config": {
            "training_seed": 42,
            "nb_training_seeds": 1,
            "f": [0, 2, 4, 6, 8, 10],
            "data_distribution": [
                {
                    "name": "gamma_similarity_niid",
                    "distribution_parameter": [1.0, 0.66, 0.33, 0.0]
                }
            ]
        },
        "aggregator": [
            {"name": "TrMean", "parameters": {}},
            {"name": "GeometricMedian", "parameters": {"nu": 0.1, "T": 3}},
            {"name": "MultiKrum", "parameters": {}},
            {"name": "CenteredClipping", "parameters": {}}
        ],
        "attack": [
            {"name": "SignFlipping", "parameters": {}},
            {"name": "Optimal_InnerProductManipulation", "parameters": {}},
            {"name": "Optimal_ALittleIsEnough", "parameters": {}}
        ]
    }
    
    for results_dir, plots_dir in experiments:
        config_path = os.path.join(results_dir, 'config.json')
        if not os.path.exists(config_path):
            print(f"Config file '{config_path}' does not exist. Skipping.")
            continue
            
        print(f"\n=======================================================")
        print(f"Generating f=10 heatmaps for: {results_dir}")
        print(f"=======================================================")
        
        # Load current config to restore later
        with open(config_path, 'r') as file:
            current_config = json.load(file)
            
        try:
            # Build the clean plotting config
            plot_config = copy.deepcopy(current_config)
            
            # Apply standard overrides for clean plotting config
            for key, value in PLOT_CONFIG_OVERRIDES.items():
                if isinstance(value, dict) and key in plot_config:
                    plot_config[key].update(value)
                else:
                    plot_config[key] = value
            
            # Temporarily write the clean plotting config to results folder
            with open(config_path, 'w') as file:
                json.dump(plot_config, file, indent=4)
            
            print("\n--- Metric: best_step (validation) ---")
            
            # 1. Generate general heatmaps
            print("Generating general heatmaps...")
            test_heatmap(results_dir, plots_dir, metric="best_step")
            try:
                loss_heatmap(results_dir, plots_dir)
            except Exception as e:
                print(f"Could not generate loss heatmap: {e}")
            aggregated_test_heatmap(results_dir, plots_dir, metric="best_step")
            
            # 2. Generate fixed-attack heatmaps
            for attack in attacks:
                print(f"Generating plots for fixed attack: {attack}...")
                test_heatmap(results_dir, plots_dir, target_attack=attack, metric="best_step")
                aggregated_test_heatmap(results_dir, plots_dir, target_attack=attack, metric="best_step")
                    
        except Exception as e:
            print(f"Error processing {results_dir}: {e}")
            traceback.print_exc()
        finally:
            # Restore original config.json
            with open(config_path, 'w') as file:
                json.dump(current_config, file, indent=4)
            print(f"Restored original config.json for {results_dir}")
            
    print("\nHeatmap generation for SNN Direct and Rate complete!")

if __name__ == "__main__":
    main()
