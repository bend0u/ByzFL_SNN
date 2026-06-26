import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import copy
from byzfl.benchmark.evaluate_results import test_heatmap, loss_heatmap, aggregated_test_heatmap

# Config overrides to ensure only seed 42, f up to 10, and original aggregators/attacks are evaluated
OVERRIDES = {
    "benchmark_config": {
        "training_seed": 42,
        "nb_training_seeds": 1,
        "f": [0, 2, 4, 6, 8, 10],
        "data_distribution": [{"name": "gamma_similarity_niid", "distribution_parameter": [1.0, 0.66, 0.33, 0.0]}]
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

def run_plots(results_dir, plots_dir):
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_path = os.path.join(script_dir, results_dir)
    plots_path = os.path.join(script_dir, plots_dir)
    
    os.makedirs(plots_path, exist_ok=True)
    config_path = os.path.join(results_path, 'config.json')
    
    with open(config_path, 'r') as f:
        current_config = json.load(f)
        
    try:
        plot_config = copy.deepcopy(current_config)
        # Apply standard overrides for clean plotting config
        for key, value in OVERRIDES.items():
            if isinstance(value, dict) and key in plot_config:
                plot_config[key].update(value)
            else:
                plot_config[key] = value
                
        with open(config_path, 'w') as f:
            json.dump(plot_config, f, indent=4)
            
        print(f"\nGenerating heatmaps for {results_dir} (better step only)...")
        # 1. Generate general heatmaps
        test_heatmap(results_path, plots_path, metric="best_step")
        try:
            loss_heatmap(results_path, plots_path)
        except Exception as e:
            print(f"Could not generate loss heatmap: {e}")
        aggregated_test_heatmap(results_path, plots_path, metric="best_step")
        
        # 2. Generate fixed-attack heatmaps
        for attack in ["Optimal_ALittleIsEnough", "Optimal_InnerProductManipulation", "SignFlipping"]:
            test_heatmap(results_path, plots_path, target_attack=attack, metric="best_step")
            aggregated_test_heatmap(results_path, plots_path, target_attack=attack, metric="best_step")
    finally:
        # Restore original config.json
        with open(config_path, 'w') as f:
            json.dump(current_config, f, indent=4)

if __name__ == "__main__":
    run_plots("cnn_complete_results", "plots_cnn_vs_snn_direct_f10/cnn")
    run_plots("snn_complete_results_direct", "plots_cnn_vs_snn_direct_f10/snn_direct")
    print("\nHeatmap generation completed successfully!")
