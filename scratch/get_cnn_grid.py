import os
import sys
import json
import numpy as np

# Ensure workspace root is in Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from byzfl.benchmark.evaluate_results import (
    get_accuracy_at_best_step,
    get_dist_param_val,
    custom_dict_to_str,
    ensure_list
)

def get_cnn_heatmap_data(results_dir):
    config_path = os.path.join(results_dir, "config.json")
    with open(config_path, "r") as file:
        data = json.load(file)

    # Force the aggregators and attacks we want to compare with SNN
    aggregators = [
        {"name": "GeometricMedian"},
        {"name": "CenteredClipping"}
    ]
    attacks = [
        {"name": "Optimal_ALittleIsEnough_neg1"}
    ]
    data_distributions = data["benchmark_config"]["data_distribution"]
    nb_honest_clients = data["benchmark_config"]["nb_honest_clients"]
    nb_byz = [0, 1, 2, 3, 4, 5]
    nb_declared = [5]

    training_seed = data["benchmark_config"]["training_seed"]
    nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
    data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
    nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
    evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]

    model_name = data["model"]["name"]
    dataset_name = data["model"]["dataset_name"]
    lr_list = ensure_list(data["model"]["learning_rate"])
    
    # Suffix for CNN is empty
    snn_suffix = ""
    clean = False

    nb_honest = nb_honest_clients[0]
    nb_decl = nb_declared[0]
    actual_nb_byz = [item for item in nb_byz if item <= nb_decl]
    data_dist = data_distributions[0]
    distribution_parameter_list = ensure_list(data_dist["distribution_parameter"])

    heat_map_cube = np.zeros((len(aggregators), len(distribution_parameter_list), len(actual_nb_byz)))

    # Best hyperparameters are not loaded for CNN, use standard ones from config
    lr = lr_list[0]
    momentum = 0.9
    wd = 0.0001

    pre_agg_names = "NNM_ARC"

    for z, agg in enumerate(aggregators):
        heat_map_table = np.zeros((len(distribution_parameter_list), len(actual_nb_byz)))

        for y, nb_byzantine in enumerate(actual_nb_byz):
            nb_decl = nb_byzantine  # declared_equal_real is True
            nb_nodes = nb_honest + nb_byzantine

            for x, dist_param in enumerate(distribution_parameter_list):
                dist_param_val = get_dist_param_val(data_dist['name'], dist_param)

                worst_accuracy = np.inf
                for attack in attacks:
                    accuracy = get_accuracy_at_best_step(
                        results_dir, clean, dataset_name, model_name, nb_nodes, nb_byzantine, nb_decl,
                        data_dist['name'], dist_param_val, agg['name'], pre_agg_names, attack['name'],
                        lr, momentum, wd, snn_suffix, None,
                        nb_data_distribution_seeds, nb_training_seeds,
                        training_seed, data_distribution_seed, evaluation_delta
                    )
                    if accuracy < worst_accuracy:
                        worst_accuracy = accuracy

                heat_map_table[len(heat_map_table)-1-x][y] = worst_accuracy

        heat_map_cube[z] = heat_map_table

    final_heat_map = np.max(heat_map_cube, axis=0)
    return final_heat_map, distribution_parameter_list, actual_nb_byz

def main():
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(workspace_dir, "results", "cnn", "weekend")
    
    grid, gammas, f_vals = get_cnn_heatmap_data(results_dir)
    
    print("\n=== CNN Baseline Results Grid ===")
    print("f_vals:", f_vals)
    print("gammas:", gammas)
    
    gammas_list = [1.0, 0.66, 0.33, 0.0]
    for idx, gamma in enumerate(gammas_list):
        row_idx = 3 - idx
        vals = grid[row_idx] * 100
        row_str = " | ".join(f"f={f}: {val:.2f}%" for f, val in zip(f_vals, vals))
        print(f"gamma={gamma} => {row_str}")
        
    print("\nMean Accuracy:", np.mean(grid) * 100)
    print("Worst-case Accuracy:", np.min(grid) * 100)

if __name__ == "__main__":
    main()
