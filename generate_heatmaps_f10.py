import os
import json
import copy
import traceback
from byzfl.benchmark.evaluate_results import test_heatmap, loss_heatmap, aggregated_test_heatmap

# The ORIGINAL config values that CNN and SNN direct had before the user modified them
# for the new multi-seed alpha=0 experiments. These match the nmnist/rate configs.
ORIGINAL_OVERRIDES = {
    "cnn_complete_results": {
        "benchmark_config": {
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
    },
    "snn_complete_results_direct": {
        "benchmark_config": {
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
}

def main():
    experiments = [
        ("cnn_complete_results", "cnn_complete_plots_extended_f10"),
        ("snn_complete_results_direct", "snn_complete_plots_direct_extended_f10"),
        ("snn_complete_results_nmnist", "snn_complete_plots_nmnist_extended_f10"),
        ("snn_complete_results_rate", "snn_complete_plots_rate_extended_f10")
    ]
    
    attacks = [
        "Optimal_ALittleIsEnough", 
        "Optimal_InnerProductManipulation", 
        "SignFlipping"
    ]
    
    metrics = ["best_step", "max_test"]
    
    for results_dir, plots_dir in experiments:
        config_path = os.path.join(results_dir, 'config.json')
        if not os.path.exists(config_path):
            print(f"Config file '{config_path}' does not exist. Skipping.")
            continue
            
        print(f"\n=======================================================")
        print(f"Generating heatmaps up to f=10 for: {results_dir}")
        print(f"=======================================================")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Load current (possibly modified) config
        with open(config_path, 'r') as file:
            current_config = json.load(file)
            
        try:
            # Deep copy to build the plotting config
            plot_config = copy.deepcopy(current_config)
            
            # Apply original overrides for CNN and SNN direct
            # (their configs were modified for new multi-seed experiments)
            if results_dir in ORIGINAL_OVERRIDES:
                overrides = ORIGINAL_OVERRIDES[results_dir]
                print(f"  Applying original config overrides for {results_dir}")
                for key, value in overrides.items():
                    if isinstance(value, dict) and key in plot_config:
                        plot_config[key].update(value)
                    else:
                        plot_config[key] = value
            else:
                # For nmnist/rate, just ensure f includes 10
                plot_config["benchmark_config"]["f"] = [0, 2, 4, 6, 8, 10]
            
            # Write the plotting config temporarily
            with open(config_path, 'w') as file:
                json.dump(plot_config, file, indent=4)
            
            for metric in metrics:
                metric_label = "best_step (validation)" if metric == "best_step" else "max_test"
                print(f"\n--- Metric: {metric_label} ---")
                
                # 1. Generate general heatmaps
                print("Generating general heatmaps...")
                test_heatmap(results_dir, plots_dir, metric=metric)
                if metric == "best_step":
                    try:
                        loss_heatmap(results_dir, plots_dir)
                    except Exception as e:
                        print(f"Could not generate loss heatmap: {e}")
                aggregated_test_heatmap(results_dir, plots_dir, metric=metric)
                
                # 2. Generate fixed-attack heatmaps
                for attack in attacks:
                    print(f"Generating plots for fixed attack: {attack}...")
                    test_heatmap(results_dir, plots_dir, target_attack=attack, metric=metric)
                    aggregated_test_heatmap(results_dir, plots_dir, target_attack=attack, metric=metric)
                
        except Exception as e:
            print(f"Error processing {results_dir}: {e}")
            traceback.print_exc()
        finally:
            # Always restore the CURRENT config (the user's modified one)
            with open(config_path, 'w') as file:
                json.dump(current_config, file, indent=4)
            print(f"Restored config.json for {results_dir}")

    print("\nAll heatmaps up to f=10 successfully generated!")

if __name__ == "__main__":
    main()
