import math
import json
import os

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import seaborn as sns

from byzfl.benchmark.managers import ParamsManager, get_snn_suffix


def custom_dict_to_str(dictionary):
    """
    Safely convert a dictionary to a string.
    Returns an empty string if the dictionary is empty.
    """
    return '' if not dictionary else str(dictionary)


def ensure_list(value):
    """
    Ensure the given value is returned as a list.
    If it is not, wrap it in a list.
    """
    if not isinstance(value, list):
        value = [value]
    return value


def get_dist_param_val(dist_name, dist_parameter):
    """
    Helper function to return None if dist_name is iid or extreme_niid.
    """
    name = custom_dict_to_str(dist_name)
    if name in ["iid", "extreme_niid"]:
        return None
    return dist_parameter


def get_accuracy_file_name(dataset_name, model_name, nb_nodes, nb_byzantine, nb_decl, 
                           dist_name, dist_param_val, agg_name, pre_agg_names, 
                           attack_name, lr, momentum, wd, snn_suffix):
    """
    Helper function to build directory names uniformly.
    """
    return (
        f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_"
        f"d_{nb_decl}_{custom_dict_to_str(dist_name)}_"
        f"{dist_param_val}_{custom_dict_to_str(agg_name)}_"
        f"{pre_agg_names}_{custom_dict_to_str(attack_name)}_"
        f"lr_{lr}_mom_{momentum}_wd_{wd}"
        f"{snn_suffix}"
    )


def get_path_and_filename(path_to_results, clean, dataset_name, model_name, nb_nodes, nb_byzantine, nb_decl, 
                          data_dist_name, dist_param_val, agg_name, pre_agg_names, attack_name, 
                          lr, momentum, wd, snn_suffix, run, training_seed, run_dd, data_distribution_seed, 
                          file_type, pm=None):
    if clean:
        encoding = pm.get_encoding_type() if pm else "constant"
        enc_name = "direct" if encoding == "constant" else encoding
        parent_dir = f"{dataset_name}_{enc_name}"
        
        pre_agg_names_list = pre_agg_names.split("_") if pre_agg_names else []
        preaggs_aggregator = '_'.join(pre_agg_names_list + [agg_name])
        
        d_name = custom_dict_to_str(data_dist_name)
        if d_name in ["iid", "extreme_niid"] or dist_param_val is None:
            dist_part = d_name
        else:
            dist_part = f"{d_name}_{dist_param_val}"

        folder_name = (
            f"{custom_dict_to_str(attack_name)}_"
            f"{preaggs_aggregator}_"
            f"f_{nb_byzantine}_"
            f"{dist_part}"
        )
        folder_path = os.path.join(path_to_results, parent_dir, folder_name)
        filename = f"{file_type}.txt"
        return os.path.join(folder_path, filename)
    else:
        file_name = get_accuracy_file_name(
            dataset_name, model_name, nb_nodes, nb_byzantine, nb_decl, 
            data_dist_name, dist_param_val, agg_name, pre_agg_names, 
            attack_name, lr, momentum, wd, snn_suffix
        )
        filename = f"{file_type}_tr_seed_{run + training_seed}_dd_seed_{run_dd + data_distribution_seed}.txt"
        return os.path.join(path_to_results, file_name, filename)


def get_config_file_path(path_to_results, clean, dataset_name, model_name, nb_nodes, nb_byzantine, nb_decl, 
                         data_dist_name, dist_param_val, agg_name, pre_agg_names, attack_name, 
                         lr, momentum, wd, snn_suffix, pm=None):
    if clean:
        encoding = pm.get_encoding_type() if pm else "constant"
        enc_name = "direct" if encoding == "constant" else encoding
        parent_dir = f"{dataset_name}_{enc_name}"
        
        pre_agg_names_list = pre_agg_names.split("_") if pre_agg_names else []
        preaggs_aggregator = '_'.join(pre_agg_names_list + [agg_name])
        
        d_name = custom_dict_to_str(data_dist_name)
        if d_name in ["iid", "extreme_niid"] or dist_param_val is None:
            dist_part = d_name
        else:
            dist_part = f"{d_name}_{dist_part}" if 'dist_part' in locals() else f"{d_name}_{dist_param_val}"

        folder_name = (
            f"{custom_dict_to_str(attack_name)}_"
            f"{preaggs_aggregator}_"
            f"f_{nb_byzantine}_"
            f"{dist_part}"
        )
        return os.path.join(path_to_results, parent_dir, folder_name, "config.json")
    else:
        file_name = get_accuracy_file_name(
            dataset_name, model_name, nb_nodes, nb_byzantine, nb_decl, 
            data_dist_name, dist_param_val, agg_name, pre_agg_names, 
            attack_name, lr, momentum, wd, snn_suffix
        )
        return os.path.join(path_to_results, file_name, "config.json")


def get_accuracy_at_best_step(path_to_results, clean, dataset_name, model_name, nb_nodes, nb_byzantine, nb_decl,
                              dist_name, dist_param_val, agg_name, pre_agg_names, attack_name,
                              lr, momentum, wd, snn_suffix, pm,
                              nb_data_distribution_seeds, nb_training_seeds,
                              training_seed, data_distribution_seed, evaluation_delta):
    """
    Load test accuracies across all seeds, average them, and return the accuracy at the step 
    maximizing validation (loaded from pre-computed 'better_step'). Falls back to the maximum 
    test accuracy if the 'better_step' file is not found.
    """
    # 1. Resolve nb_accuracies from config file
    config_file_path = get_config_file_path(
        path_to_results, clean, dataset_name, model_name, nb_nodes, nb_byzantine, nb_decl, 
        dist_name, dist_param_val, agg_name, pre_agg_names, 
        attack_name, lr, momentum, wd, snn_suffix, pm
    )
    
    nb_steps = 100
    try:
        with open(config_file_path, 'r') as file:
            sub_data = json.load(file)
            nb_steps = sub_data["benchmark_config"]["nb_steps"]
    except Exception:
        try:
            with open(os.path.join(path_to_results, 'config.json'), 'r') as file:
                main_data = json.load(file)
                nb_steps = main_data["benchmark_config"]["nb_steps"]
        except Exception:
            pass

    nb_accuracies = int(1 + math.ceil(nb_steps / evaluation_delta))

    # 2. Load test accuracies
    tab_acc = np.zeros((nb_data_distribution_seeds, nb_training_seeds, nb_accuracies))
    
    try:
        if clean:
            acc_path = get_path_and_filename(
                path_to_results, clean, dataset_name, model_name, nb_nodes, nb_byzantine, nb_decl, 
                dist_name, dist_param_val, agg_name, pre_agg_names, 
                attack_name, lr, momentum, wd, snn_suffix, 0, training_seed, 0, data_distribution_seed, 
                "test_accuracy", pm
            )
            val = genfromtxt(acc_path, delimiter=',')
            for run_dd in range(nb_data_distribution_seeds):
                for run in range(nb_training_seeds):
                    tab_acc[run_dd][run] = val
        else:
            for run_dd in range(nb_data_distribution_seeds):
                for run in range(nb_training_seeds):
                    acc_path = get_path_and_filename(
                        path_to_results, clean, dataset_name, model_name, nb_nodes, nb_byzantine, nb_decl, 
                        dist_name, dist_param_val, agg_name, pre_agg_names, 
                        attack_name, lr, momentum, wd, snn_suffix, run, training_seed, run_dd, data_distribution_seed, 
                        "test_accuracy", pm
                    )
                    tab_acc[run_dd][run] = genfromtxt(acc_path, delimiter=',')
    except Exception as e:
        print(f"Warning: could not load accuracy files for f={nb_byzantine}, param={dist_param_val}. Details: {e}")
        return 0.0

    tab_acc = tab_acc.reshape(nb_data_distribution_seeds * nb_training_seeds, nb_accuracies)
    tab_acc = tab_acc.mean(axis=0)

    # 3. Load best step from 'better_step' folder
    best_idx = None
    path_to_hyperparameters = os.path.join(path_to_results, "best_hyperparameters")
    step_file_name = (
        f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_"
        f"d_{nb_decl}_{custom_dict_to_str(dist_name)}_"
        f"{dist_param_val}_{pre_agg_names}_{agg_name}_"
        f"{custom_dict_to_str(attack_name)}.txt"
    )
    step_path = os.path.join(path_to_hyperparameters, "better_step", step_file_name)
    
    if os.path.exists(step_path):
        try:
            step_val = float(np.loadtxt(step_path))
            best_idx = int(round(step_val / evaluation_delta))
        except Exception as e:
            print(f"Error reading step path {step_path}: {e}")
            
    # Fallback 1: If 'better_step' is missing (e.g. hyperparameter search was not run),
    # dynamically load the validation accuracy files and locate the step where validation is maximized.
    if best_idx is None:
        try:
            tab_val = np.zeros((nb_data_distribution_seeds, nb_training_seeds, nb_accuracies))
            if clean:
                val_path = get_path_and_filename(
                    path_to_results, clean, dataset_name, model_name, nb_nodes, nb_byzantine, nb_decl, 
                    dist_name, dist_param_val, agg_name, pre_agg_names, 
                    attack_name, lr, momentum, wd, snn_suffix, 0, training_seed, 0, data_distribution_seed, 
                    "val_accuracy", pm
                )
                val_v = genfromtxt(val_path, delimiter=',')
                for run_dd in range(nb_data_distribution_seeds):
                    for run in range(nb_training_seeds):
                        tab_val[run_dd][run] = val_v
            else:
                for run_dd in range(nb_data_distribution_seeds):
                    for run in range(nb_training_seeds):
                        val_path = get_path_and_filename(
                            path_to_results, clean, dataset_name, model_name, nb_nodes, nb_byzantine, nb_decl, 
                            dist_name, dist_param_val, agg_name, pre_agg_names, 
                            attack_name, lr, momentum, wd, snn_suffix, run, training_seed, run_dd, data_distribution_seed, 
                            "val_accuracy", pm
                        )
                        tab_val[run_dd][run] = genfromtxt(val_path, delimiter=',')
            
            tab_val = tab_val.reshape(nb_data_distribution_seeds * nb_training_seeds, nb_accuracies)
            tab_val = tab_val.mean(axis=0)
            best_idx = np.argmax(tab_val)
        except Exception:
            # If loading validation files fails, leave best_idx as None to trigger Fallback 2.
            pass

    # If we successfully determined the best step index (from better_step or validation curves)
    if best_idx is not None:
        best_idx = min(max(0, best_idx), len(tab_acc) - 1)
        return tab_acc[best_idx]
            
    # Fallback 2: If both better_step and validation files are missing, default to the peak average test accuracy
    return np.max(tab_acc)


def find_best_hyperparameters(path_to_results):
    """
    Find the best hyperparameters (learning rate, momentum, weight decay) 
    that maximize the minimum accuracy across different attacks.

    Reads a configuration file (config.json) in `path_to_results` 
    and writes out the best hyperparameters and the corresponding 
    step at which maximum accuracy was reached for each aggregator 
    and each attack.
    """
    try:
        with open(os.path.join(path_to_results, 'config.json'), 'r') as file:
            data = json.load(file)
    except Exception as e:
        print(f"ERROR reading config.json: {e}")
        return
    
    pm = ParamsManager(data)
    snn_suffix = get_snn_suffix(pm)
    clean = (
        data.get("clean_directory_structure", False) or
        data.get("evaluation_and_results", {}).get("clean_directory_structure", False) or
        data.get("benchmark_config", {}).get("clean_directory_structure", False)
    )
    
    path_hyperparameters = path_to_results + "/best_hyperparameters"

    # <-------------- Benchmark Config ------------->
    training_seed = data["benchmark_config"]["training_seed"]
    nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
    nb_honest_clients = data["benchmark_config"]["nb_honest_clients"]
    nb_byz = data["benchmark_config"]["f"]
    nb_declared = data["benchmark_config"].get("tolerated_f", None)
    data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
    nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
    data_distributions = data["benchmark_config"]["data_distribution"]
    set_honest_clients_as_clients = data["benchmark_config"]["set_honest_clients_as_clients"]
    nb_steps = data["benchmark_config"]["nb_steps"]


    # <-------------- Evaluation and Results ------------->
    evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]

    # <-------------- Model Config ------------->
    model_name = data["model"]["name"]
    dataset_name = data["model"]["dataset_name"]
    lr_list = data["model"]["learning_rate"]

    # <-------------- Honest Nodes Config ------------->
    momentum_list = data["honest_clients"]["momentum"]
    wd_list = data["honest_clients"]["weight_decay"]

    # <-------------- Aggregators Config ------------->
    aggregators = data["aggregator"]
    pre_aggregators = data["pre_aggregators"]

    # <-------------- Attacks Config ------------->
    attacks = data["attack"]

    # Ensure certain configurations are always lists
    nb_honest_clients = ensure_list(nb_honest_clients)
    nb_byz = ensure_list(nb_byz)
    nb_declared = ensure_list(nb_declared)
    data_distributions = ensure_list(data_distributions)
    aggregators = ensure_list(aggregators)

    # Pre-aggregators can be multiple or single dict; unify them
    if not pre_aggregators or isinstance(pre_aggregators[0], dict):
        pre_aggregators = [pre_aggregators]

    attacks = ensure_list(attacks)
    lr_list = ensure_list(lr_list)
    momentum_list = ensure_list(momentum_list)
    wd_list = ensure_list(wd_list)

    # Number of accuracy checkpoints
    nb_accuracies = 1 + math.ceil(nb_steps / evaluation_delta)

    # Main nested loops to explore configurations
    for nb_honest in nb_honest_clients:
        for nb_byzantine in nb_byz:
            
            if nb_declared[0] is None:
                nb_declared_list = [nb_byzantine]
            else:
                nb_declared_list = nb_declared.copy()
                nb_declared_list = [item for item in nb_declared_list if item >= nb_byzantine]

            for nb_decl in nb_declared_list:
                if set_honest_clients_as_clients:
                    nb_nodes = nb_honest
                else:
                    nb_nodes = nb_honest + nb_byzantine

                for data_dist in data_distributions:
                    distribution_parameter_list = ensure_list(data_dist["distribution_parameter"])
                    for distribution_parameter in distribution_parameter_list:
                        dist_param_val = get_dist_param_val(data_dist['name'], distribution_parameter)
                        for pre_agg in pre_aggregators:
                            # Build a single name from all pre-aggregators
                            pre_agg_names_list = [p["name"] for p in pre_agg]
                            pre_agg_names = "_".join(pre_agg_names_list)

                            # Prepare arrays to store final best hyperparams & steps
                            real_hyper_parameters = np.zeros((len(aggregators), 3))
                            real_steps = np.zeros((len(aggregators), len(attacks)))

                            for k, agg in enumerate(aggregators):
                                # We'll store max accuracy for each (lr, momentum, wd) across attacks
                                num_combinations = len(lr_list) * len(momentum_list) * len(wd_list)
                                max_acc_config = np.zeros((num_combinations, len(attacks)))
                                hyper_parameters = np.zeros((num_combinations, 3))
                                steps_max_reached = np.zeros((num_combinations, len(attacks)))

                                index_combination = 0
                                for lr in lr_list:
                                    for momentum in momentum_list:
                                        for wd in wd_list:
                                            # tab_acc shape: (len(attacks), nb_dd_seeds, nb_training_seeds, nb_accuracies)
                                            tab_acc = np.zeros(
                                                (
                                                    len(attacks),
                                                    nb_data_distribution_seeds,
                                                    nb_training_seeds,
                                                    nb_accuracies
                                                )
                                            )

                                            # Fill tab_acc with loaded accuracy files
                                            for i, attack in enumerate(attacks):
                                                for run_dd in range(nb_data_distribution_seeds):
                                                    for run in range(nb_training_seeds):
                                                        acc_path = get_path_and_filename(
                                                            path_to_results, clean, dataset_name, model_name, nb_nodes, nb_byzantine, nb_decl, 
                                                            data_dist['name'], dist_param_val, agg['name'], pre_agg_names, 
                                                            attack['name'], lr, momentum, wd, snn_suffix, run, training_seed, run_dd, data_distribution_seed, 
                                                            "val_accuracy", pm
                                                        )
                                                        tab_acc[i, run_dd, run] = genfromtxt(acc_path, delimiter=',')

                                            tab_acc = tab_acc.reshape(
                                                len(attacks),
                                                nb_data_distribution_seeds * nb_training_seeds,
                                                nb_accuracies
                                            )
                                            
                                            # Compute average accuracy across seeds, find max
                                            for i in range(len(attacks)):
                                                avg_accuracy = np.mean(tab_acc[i], axis=0)
                                                idx_max = np.argmax(avg_accuracy)
                                                max_acc_config[index_combination, i] = avg_accuracy[idx_max]
                                                steps_max_reached[index_combination, i] = idx_max * evaluation_delta

                                            hyper_parameters[index_combination] = [lr, momentum, wd]
                                            index_combination += 1

                                # Create path if needed
                                if not os.path.exists(path_hyperparameters):
                                    try:
                                        os.makedirs(path_hyperparameters)
                                    except OSError as error:
                                        print(f"Error creating directory: {error}")

                                # Find the combination that maximizes the minimum accuracy across attacks
                                max_minimum_idx = -1
                                max_minimum_val = -1
                                for i in range(num_combinations):
                                    current_min = np.min(max_acc_config[i])
                                    if current_min > max_minimum_val:
                                        max_minimum_idx = i
                                        max_minimum_val = current_min

                                real_hyper_parameters[k] = hyper_parameters[max_minimum_idx]
                                real_steps[k] = steps_max_reached[max_minimum_idx]

                            # Save results to folder
                            hyper_parameters_folder = os.path.join(path_hyperparameters, "hyperparameters")
                            steps_folder = os.path.join(path_hyperparameters, "better_step")

                            os.makedirs(hyper_parameters_folder, exist_ok=True)
                            os.makedirs(steps_folder, exist_ok=True)

                            for i, agg in enumerate(aggregators):
                                # Save best hyperparameters
                                file_name_hparams = (
                                    f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_"
                                    f"d_{nb_decl}_{custom_dict_to_str(data_dist['name'])}_"
                                    f"{dist_param_val}_{pre_agg_names}_{agg['name']}.txt"
                                )
                                np.savetxt(
                                    os.path.join(hyper_parameters_folder, file_name_hparams),
                                    real_hyper_parameters[i]
                                )

                                # Save step at which max accuracy occurs for each attack
                                for j, attack in enumerate(attacks):
                                    file_name_steps = (
                                        f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_"
                                        f"d_{nb_decl}_{custom_dict_to_str(data_dist['name'])}_"
                                        f"{dist_param_val}_{pre_agg_names}_{agg['name']}_"
                                        f"{custom_dict_to_str(attack['name'])}.txt"
                                    )
                                    step_val = np.array([real_steps[i, j]])
                                    np.savetxt(os.path.join(steps_folder, file_name_steps), step_val)

colors = [(0, 0.4470, 0.7410), (0.8500, 0.3250, 0.0980), (0.4660, 0.6740, 0.1880), (120/255,120/255, 120/255), (0.7, 0.2, 0.5)]
tab_sign = ['-', '--', '-.', ':', 'solid']
markers = ['^','s', '<', 'o', '*']

def test_accuracy_curve(path_to_results, path_to_plot, colors=colors, tab_sign=tab_sign, markers=markers):
        
        try:
            with open(os.path.join(path_to_results, 'config.json'), 'r') as file:
                data = json.load(file)
        except Exception as e:
            print(f"ERROR reading config.json: {e}")
            return
        
        pm = ParamsManager(data)
        snn_suffix = get_snn_suffix(pm)
        clean = (
            data.get("clean_directory_structure", False) or
            data.get("evaluation_and_results", {}).get("clean_directory_structure", False) or
            data.get("benchmark_config", {}).get("clean_directory_structure", False)
        )
        
        try:
            os.makedirs(path_to_plot, exist_ok=True)
        except OSError as error:
            print(f"Error creating directory: {error}")
        
        path_to_hyperparameters = path_to_results + "/best_hyperparameters"
        

        # <-------------- Benchmark Config ------------->
        training_seed = data["benchmark_config"]["training_seed"]
        nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
        nb_honest_clients = data["benchmark_config"]["nb_honest_clients"]
        nb_byz = data["benchmark_config"]["f"]
        nb_declared = data["benchmark_config"].get("tolerated_f", None)
        data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
        nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
        data_distributions = data["benchmark_config"]["data_distribution"]
        set_honest_clients_as_clients = data["benchmark_config"]["set_honest_clients_as_clients"]
        nb_steps = data["benchmark_config"]["nb_steps"]


        # <-------------- Evaluation and Results ------------->
        evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]

        # <-------------- Model Config ------------->
        model_name = data["model"]["name"]
        dataset_name = data["model"]["dataset_name"]
        lr_list = data["model"]["learning_rate"]

        # <-------------- Honest Nodes Config ------------->
        momentum_list = data["honest_clients"]["momentum"]
        wd_list = data["honest_clients"]["weight_decay"]

        # <-------------- Aggregators Config ------------->
        aggregators = data["aggregator"]
        pre_aggregators = data["pre_aggregators"]

        # <-------------- Attacks Config ------------->
        attacks = data["attack"]

        # Ensure certain configurations are always lists
        nb_honest_clients = ensure_list(nb_honest_clients)
        nb_byz = ensure_list(nb_byz)
        nb_declared = ensure_list(nb_declared)
        data_distributions = ensure_list(data_distributions)
        aggregators = ensure_list(aggregators)

        # Pre-aggregators can be multiple or single dict; unify them
        if not pre_aggregators or isinstance(pre_aggregators[0], dict):
            pre_aggregators = [pre_aggregators]

        attacks = ensure_list(attacks)
        lr_list = ensure_list(lr_list)
        momentum_list = ensure_list(momentum_list)
        wd_list = ensure_list(wd_list)

        nb_accuracies = int(1+math.ceil(nb_steps/evaluation_delta))

        for nb_honest in nb_honest_clients:
            for nb_byzantine in nb_byz:

                if nb_declared[0] is None:
                    nb_declared_list = [nb_byzantine]
                else:
                    nb_declared_list = nb_declared.copy()
                    nb_declared_list = [item for item in nb_declared_list if item >= nb_byzantine]
                
                for nb_decl in nb_declared_list:

                    if set_honest_clients_as_clients:
                        nb_nodes = nb_honest
                    else:
                        nb_nodes = nb_honest + nb_byzantine
                    
                    for data_dist in data_distributions:
                        dist_parameter_list = ensure_list(data_dist["distribution_parameter"])
                        for dist_parameter in dist_parameter_list:
                            dist_param_val = get_dist_param_val(data_dist['name'], dist_parameter)
                            for pre_agg in pre_aggregators:
                                pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
                                pre_agg_names = "_".join(pre_agg_list_names)
                                for agg in aggregators:

                                    hyper_file_name = (
                                    f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                    f"{custom_dict_to_str(data_dist['name'])}_{dist_param_val}_{pre_agg_names}_{agg['name']}.txt"
                                    )


                                    full_path = os.path.join(path_to_hyperparameters, "hyperparameters", hyper_file_name)

                                    if os.path.exists(full_path):
                                        hyperparameters = np.loadtxt(full_path)
                                        lr = hyperparameters[0]
                                        momentum = hyperparameters[1]
                                        wd = hyperparameters[2]
                                    else:
                                        lr = lr_list[0]
                                        momentum = momentum_list[0]
                                        wd = wd_list[0]

                                    tab_acc = np.zeros((
                                        len(attacks), 
                                        nb_data_distribution_seeds,
                                        nb_training_seeds,
                                        nb_accuracies
                                    ))

                                    for i, attack in enumerate(attacks):
                                        for run_dd in range(nb_data_distribution_seeds):
                                            for run in range(nb_training_seeds):
                                                acc_path = get_path_and_filename(
                                                    path_to_results, clean, dataset_name, model_name, nb_nodes, nb_byzantine, nb_decl, 
                                                    data_dist['name'], dist_param_val, agg['name'], pre_agg_names, 
                                                    attack['name'], lr, momentum, wd, snn_suffix, run, training_seed, run_dd, data_distribution_seed, 
                                                    "test_accuracy", pm
                                                )
                                                tab_acc[i, run_dd, run] = genfromtxt(acc_path, delimiter=',')

                                    tab_acc = tab_acc.reshape(
                                        len(attacks),
                                        nb_data_distribution_seeds * nb_training_seeds,
                                        nb_accuracies
                                    )
                                    
                                    err = np.zeros((len(attacks), nb_accuracies))
                                    for i in range(len(err)):
                                        err[i] = (1.96*np.std(tab_acc[i], axis = 0))/math.sqrt(nb_training_seeds*nb_data_distribution_seeds)
                                    
                                    plt.rcParams.update({'font.size': 12})

                                    
                                    for i, attack in enumerate(attacks):
                                        attack = attack["name"]
                                        plt.plot(np.arange(nb_accuracies)*evaluation_delta, np.mean(tab_acc[i], axis = 0), label = attack, color = colors[i], linestyle = tab_sign[i], marker = markers[i], markevery = 1)
                                        plt.fill_between(np.arange(nb_accuracies)*evaluation_delta, np.mean(tab_acc[i], axis = 0) - err[i], np.mean(tab_acc[i], axis = 0) + err[i], alpha = 0.25)

                                    plt.xlabel('Round')
                                    plt.ylabel('Accuracy')
                                    plt.xlim(0,(nb_accuracies-1)*evaluation_delta)
                                    plt.ylim(0,1)
                                    plt.grid()
                                    plt.legend()

                                    plot_name = (
                                        f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                        f"{custom_dict_to_str(data_dist['name'])}_{dist_param_val}_"
                                        f"{custom_dict_to_str(agg['name'])}_{pre_agg_names}_lr_{lr}_mom_{momentum}_wd_{wd}"
                                    )
                                    
                                    plt.savefig(path_to_plot+"/"+plot_name+'_plot.pdf')
                                    plt.close()


def loss_heatmap(path_to_results, path_to_plot, target_attack=None):
    """
    Creates a heatmap where the axis are the number of 
    byzantine nodes and the distribution parameter.
    Each number is the mean of the best training losses reached 
    by the model across seeds, using a specific aggregation.
    """
    try:
        with open(os.path.join(path_to_results, 'config.json'), 'r') as file:
            data = json.load(file)
    except Exception as e:
        print(f"ERROR reading config.json: {e}")
        return
    
    pm = ParamsManager(data)
    snn_suffix = get_snn_suffix(pm)
    clean = (
        data.get("clean_directory_structure", False) or
        data.get("evaluation_and_results", {}).get("clean_directory_structure", False) or
        data.get("benchmark_config", {}).get("clean_directory_structure", False)
    )

    try:
        os.makedirs(path_to_plot, exist_ok=True)
    except OSError as error:
        print(f"Error creating directory: {error}")
    
    path_to_hyperparameters = path_to_results + "/best_hyperparameters" 

    # <-------------- Benchmark Config ------------->
    training_seed = data["benchmark_config"]["training_seed"]
    nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
    nb_honest_clients = data["benchmark_config"]["nb_honest_clients"]
    nb_byz = data["benchmark_config"]["f"]
    nb_declared = data["benchmark_config"].get("tolerated_f", None)
    data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
    nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
    data_distributions = data["benchmark_config"]["data_distribution"]
    set_honest_clients_as_clients = data["benchmark_config"]["set_honest_clients_as_clients"]
    nb_steps = data["benchmark_config"]["nb_steps"]


    # <-------------- Evaluation and Results ------------->
    evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]

    # <-------------- Model Config ------------->
    model_name = data["model"]["name"]
    dataset_name = data["model"]["dataset_name"]
    lr_list = data["model"]["learning_rate"]

    # <-------------- Honest Nodes Config ------------->
    momentum_list = data["honest_clients"]["momentum"]
    wd_list = data["honest_clients"]["weight_decay"]

    # <-------------- Aggregators Config ------------->
    aggregators = data["aggregator"]
    pre_aggregators = data["pre_aggregators"]

    # <-------------- Attacks Config ------------->
    attacks = data["attack"]

    # Ensure certain configurations are always lists
    nb_honest_clients = ensure_list(nb_honest_clients)
    nb_byz = ensure_list(nb_byz)
    nb_declared = ensure_list(nb_declared)
    data_distributions = ensure_list(data_distributions)
    aggregators = ensure_list(aggregators)

    # Pre-aggregators can be multiple or single dict; unify them
    if not pre_aggregators or isinstance(pre_aggregators[0], dict):
        pre_aggregators = [pre_aggregators]

    attacks = ensure_list(attacks)
    if target_attack is not None:
        attacks = [att for att in attacks if att["name"] == target_attack]
    lr_list = ensure_list(lr_list)
    momentum_list = ensure_list(momentum_list)
    wd_list = ensure_list(wd_list)

    if nb_declared[0] is None:
        declared_equal_real = True
        nb_declared = [nb_byz[-1]]
    else:
        declared_equal_real = False

    for pre_agg in pre_aggregators:

        pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
        pre_agg_names = "_".join(pre_agg_list_names)

        for agg in aggregators:

            for data_dist in data_distributions:

                distribution_parameter_list = data_dist["distribution_parameter"]
                distribution_parameter_list = ensure_list(distribution_parameter_list)

                for nb_honest in nb_honest_clients:

                    for nb_decl in nb_declared:
                        actual_nb_byz = [item for item in nb_byz if item <= nb_decl]
                        heat_map_table = np.zeros((len(distribution_parameter_list), len(actual_nb_byz)))

                        for y, nb_byzantine in enumerate(actual_nb_byz):

                            if declared_equal_real:
                                nb_decl = nb_byzantine

                            if set_honest_clients_as_clients:
                                nb_nodes = nb_honest
                                nb_honest = nb_nodes - nb_byzantine
                            else:
                                nb_nodes = nb_honest + nb_byzantine

                            for x, dist_param in enumerate(distribution_parameter_list):
                                dist_param_val = get_dist_param_val(data_dist['name'], dist_param)

                                hyper_file_name = (
                                    f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                    f"{custom_dict_to_str(data_dist['name'])}_{dist_param_val}_{pre_agg_names}_{agg['name']}.txt"
                                )

                                
                                full_path = os.path.join(path_to_hyperparameters, "hyperparameters", hyper_file_name)

                                if os.path.exists(full_path):
                                    hyperparameters = np.loadtxt(full_path)
                                    lr = hyperparameters[0]
                                    momentum = hyperparameters[1]
                                    wd = hyperparameters[2]
                                else:
                                    lr = lr_list[0]
                                    momentum = momentum_list[0]
                                    wd = wd_list[0]

                                
                                lowest_loss = 0
                                for attack in attacks:

                                    config_file_path = get_config_file_path(
                                        path_to_results, clean, dataset_name, model_name, nb_nodes, nb_byzantine, nb_decl, 
                                        data_dist['name'], dist_param_val, agg['name'], pre_agg_names, 
                                        attack['name'], lr, momentum, wd, snn_suffix, pm
                                    )

                                    try:
                                        with open(config_file_path, 'r') as file:
                                            sub_data = json.load(file)
                                    except Exception as e:
                                        print("ERROR: "+ str(e))
                                        sub_data = data

                                    nb_steps = sub_data["benchmark_config"]["nb_steps"]

                                    losses = np.zeros(
                                        (
                                            nb_data_distribution_seeds,
                                            nb_training_seeds,
                                            nb_steps
                                        )
                                    )

                                    if clean:
                                        loss_path = get_path_and_filename(
                                            path_to_results, clean, dataset_name, model_name, nb_nodes, nb_byzantine, nb_decl, 
                                            data_dist['name'], dist_param_val, agg['name'], pre_agg_names, 
                                            attack['name'], lr, momentum, wd, snn_suffix, 0, training_seed, 0, data_distribution_seed, 
                                            "train_loss", pm
                                        )
                                        val = genfromtxt(loss_path, delimiter=',')
                                        for run_dd in range(nb_data_distribution_seeds):
                                            for run in range(nb_training_seeds):
                                                losses[run_dd][run] = val
                                    else:
                                        for run_dd in range(nb_data_distribution_seeds):
                                            for run in range(nb_training_seeds):
                                                loss_path = get_path_and_filename(
                                                    path_to_results, clean, dataset_name, model_name, nb_nodes, nb_byzantine, nb_decl, 
                                                    data_dist['name'], dist_param_val, agg['name'], pre_agg_names, 
                                                    attack['name'], lr, momentum, wd, snn_suffix, run, training_seed, run_dd, data_distribution_seed, 
                                                    "train_loss", pm
                                                )
                                                losses[run_dd][run] = genfromtxt(loss_path, delimiter=',')

                                    losses = losses.reshape(
                                        nb_data_distribution_seeds * nb_training_seeds,
                                        nb_steps
                                    )

                                    losses = np.mean(losses, axis=0)

                                    temp_lowest_loss = np.min(losses)

                                    if temp_lowest_loss > lowest_loss:
                                        lowest_loss = temp_lowest_loss
                                    
                                heat_map_table[len(heat_map_table)-1-x][y] = lowest_loss

                        if declared_equal_real:
                            end_file_name = "tolerated_f_equal_real.pdf"
                        else:
                            end_file_name = f"tolerated_f_{nb_decl}.pdf"

                        file_prefix = f"train_loss_{target_attack}_" if target_attack else "train_loss_"
                        file_name = (
                            f"{file_prefix}{dataset_name}_"
                            f"{model_name}_"
                            f"{custom_dict_to_str(data_dist['name'])}_"
                            f"{pre_agg_names}_"
                            f"{agg['name']}_"
                            f"nb_honest_clients_{nb_honest}_"
                            + end_file_name
                        )

                    
                        column_names = [str(dist_param) for dist_param in distribution_parameter_list]
                        row_names = [str(nb_byzantine) for nb_byzantine in actual_nb_byz]
                        column_names.reverse()

                        try:
                            os.makedirs(path_to_plot, exist_ok=True)
                        except OSError as error:
                            print(f"Error creating directory: {error}")

                        sns.heatmap(heat_map_table, xticklabels=row_names, yticklabels=column_names, cmap=sns.cm.rocket_r, annot=True)
                        if target_attack:
                            plt.title(f"Attack: {target_attack}")
                        plt.xlabel("Number of Byzantine clients")
                        plt.ylabel("Data heterogeneity level")
                        plt.tight_layout()
                        plt.savefig(path_to_plot +"/"+ file_name)
                        plt.close()


def test_heatmap(path_to_results, path_to_plot, target_attack=None):
    """
    Creates a heatmap where the axis are the number of 
    byzantine nodes and the distribution parameter.
    Each number is the mean of the best accuracy reached 
    by the model across seeds, using a specific aggregation.
    """
    try:
        with open(os.path.join(path_to_results, 'config.json'), 'r') as file:
            data = json.load(file)
    except Exception as e:
        print(f"ERROR reading config.json: {e}")
        return
    
    pm = ParamsManager(data)
    snn_suffix = get_snn_suffix(pm)
    clean = (
        data.get("clean_directory_structure", False) or
        data.get("evaluation_and_results", {}).get("clean_directory_structure", False) or
        data.get("benchmark_config", {}).get("clean_directory_structure", False)
    )
    
    try:
        os.makedirs(path_to_plot, exist_ok=True)
    except OSError as error:
        print(f"Error creating directory: {error}")
    
    path_to_hyperparameters = path_to_results + "/best_hyperparameters"

    # <-------------- Benchmark Config ------------->
    training_seed = data["benchmark_config"]["training_seed"]
    nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
    nb_honest_clients = data["benchmark_config"]["nb_honest_clients"]
    nb_byz = data["benchmark_config"]["f"]
    nb_declared = data["benchmark_config"].get("tolerated_f", None)
    data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
    nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
    data_distributions = data["benchmark_config"]["data_distribution"]
    set_honest_clients_as_clients = data["benchmark_config"]["set_honest_clients_as_clients"]
    nb_steps = data["benchmark_config"]["nb_steps"]


    # <-------------- Evaluation and Results ------------->
    evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]

    # <-------------- Model Config ------------->
    model_name = data["model"]["name"]
    dataset_name = data["model"]["dataset_name"]
    lr_list = data["model"]["learning_rate"]

    # <-------------- Honest Nodes Config ------------->
    momentum_list = data["honest_clients"]["momentum"]
    wd_list = data["honest_clients"]["weight_decay"]

    # <-------------- Aggregators Config ------------->
    aggregators = data["aggregator"]
    pre_aggregators = data["pre_aggregators"]

    # <-------------- Attacks Config ------------->
    attacks = data["attack"]

    # Ensure certain configurations are always lists
    nb_honest_clients = ensure_list(nb_honest_clients)
    nb_byz = ensure_list(nb_byz)
    nb_declared = ensure_list(nb_declared)
    data_distributions = ensure_list(data_distributions)
    aggregators = ensure_list(aggregators)

    # Pre-aggregators can be multiple or single dict; unify them
    if not pre_aggregators or isinstance(pre_aggregators[0], dict):
        pre_aggregators = [pre_aggregators]

    attacks = ensure_list(attacks)
    if target_attack is not None:
        attacks = [att for att in attacks if att["name"] == target_attack]
    lr_list = ensure_list(lr_list)
    momentum_list = ensure_list(momentum_list)
    wd_list = ensure_list(wd_list)

    if nb_declared[0] is None:
        declared_equal_real = True
        nb_declared = [nb_byz[-1]]
    else:
        declared_equal_real = False

    for pre_agg in pre_aggregators:

        pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
        pre_agg_names = "_".join(pre_agg_list_names)

        for agg in aggregators:

            for data_dist in data_distributions:

                distribution_parameter_list = data_dist["distribution_parameter"]
                distribution_parameter_list = ensure_list(distribution_parameter_list)

                for nb_honest in nb_honest_clients:

                    for nb_decl in nb_declared:
                        actual_nb_byz = [item for item in nb_byz if item <= nb_decl]
                        heat_map_table = np.zeros((len(distribution_parameter_list), len(actual_nb_byz)))

                        for y, nb_byzantine in enumerate(actual_nb_byz):

                            if declared_equal_real:
                                nb_decl = nb_byzantine

                            if set_honest_clients_as_clients:
                                nb_nodes = nb_honest
                                nb_honest = nb_nodes - nb_byzantine
                            else:
                                nb_nodes = nb_honest + nb_byzantine

                            for x, dist_param in enumerate(distribution_parameter_list):
                                dist_param_val = get_dist_param_val(data_dist['name'], dist_param)

                                hyper_file_name = (
                                    f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                    f"{custom_dict_to_str(data_dist['name'])}_{dist_param_val}_{pre_agg_names}_{agg['name']}.txt"
                                )

                                full_path = os.path.join(path_to_hyperparameters, "hyperparameters", hyper_file_name)

                                if os.path.exists(full_path):
                                    hyperparameters = np.loadtxt(full_path)
                                    lr = hyperparameters[0]
                                    momentum = hyperparameters[1]
                                    wd = hyperparameters[2]
                                else:
                                    lr = lr_list[0]
                                    momentum = momentum_list[0]
                                    wd = wd_list[0]

                                
                                worst_accuracy = np.inf
                                for attack in attacks:

                                    accuracy = get_accuracy_at_best_step(
                                        path_to_results, clean, dataset_name, model_name, nb_nodes, nb_byzantine, nb_decl,
                                        data_dist['name'], dist_param_val, agg['name'], pre_agg_names, attack['name'],
                                        lr, momentum, wd, snn_suffix, pm,
                                        nb_data_distribution_seeds, nb_training_seeds,
                                        training_seed, data_distribution_seed, evaluation_delta
                                    )

                                    if accuracy < worst_accuracy:
                                        worst_accuracy = accuracy
                                    
                                heat_map_table[len(heat_map_table)-1-x][y] = worst_accuracy
                    
                        if declared_equal_real:
                            end_file_name = "tolerated_f_equal_real.pdf"
                        else:
                            end_file_name = f"tolerated_f_{nb_decl}.pdf"

                        file_prefix = f"test_{target_attack}_" if target_attack else "test_"
                        file_name = (
                            f"{file_prefix}{dataset_name}_"
                            f"{model_name}_"
                            f"{custom_dict_to_str(data_dist['name'])}_"
                            f"{pre_agg_names}_"
                            f"{agg['name']}_"
                            f"nb_honest_clients_{nb_honest}_"
                            + end_file_name
                        )

                    
                        column_names = [str(dist_param) for dist_param in distribution_parameter_list]
                        row_names = [str(nb_byzantine) for nb_byzantine in actual_nb_byz]
                        column_names.reverse()

                        try:
                            os.makedirs(path_to_plot, exist_ok=True)
                        except OSError as error:
                            print(f"Error creating directory: {error}")

                        sns.heatmap(heat_map_table, xticklabels=row_names, yticklabels=column_names, annot=True)
                        if target_attack:
                            plt.title(f"Attack: {target_attack}")
                        plt.xlabel("Number of Byzantine clients")
                        plt.ylabel("Data heterogeneity level")
                        plt.tight_layout()
                        plt.savefig(path_to_plot +"/"+ file_name)
                        plt.close()


def aggregated_test_heatmap(path_to_results, path_to_plot, target_attack=None):
    """
    Heatmap with the aggregated info of all aggregators, 
    for every region in the heatmap, it shows the aggregation 
    with the best accuracy.
    """
    try:
        with open(path_to_results+'/config.json', 'r') as file:
            data = json.load(file)
    except Exception as e:
        print("ERROR: "+ str(e))

    pm = ParamsManager(data)
    snn_suffix = get_snn_suffix(pm)
    clean = (
        data.get("clean_directory_structure", False) or
        data.get("evaluation_and_results", {}).get("clean_directory_structure", False) or
        data.get("benchmark_config", {}).get("clean_directory_structure", False)
    )

    try:
        os.makedirs(path_to_plot, exist_ok=True)
    except OSError as error:
        print(f"Error creating directory: {error}")
    
    path_to_hyperparameters = path_to_results + "/best_hyperparameters"
    
    # <-------------- Benchmark Config ------------->
    training_seed = data["benchmark_config"]["training_seed"]
    nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
    nb_honest_clients = data["benchmark_config"]["nb_honest_clients"]
    nb_byz = data["benchmark_config"]["f"]
    nb_declared = data["benchmark_config"].get("tolerated_f", None)
    data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
    nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
    data_distributions = data["benchmark_config"]["data_distribution"]
    set_honest_clients_as_clients = data["benchmark_config"]["set_honest_clients_as_clients"]
    nb_steps = data["benchmark_config"]["nb_steps"]


    # <-------------- Evaluation and Results ------------->
    evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]

    # <-------------- Model Config ------------->
    model_name = data["model"]["name"]
    dataset_name = data["model"]["dataset_name"]
    lr_list = data["model"]["learning_rate"]

    # <-------------- Honest Nodes Config ------------->
    momentum_list = data["honest_clients"]["momentum"]
    wd_list = data["honest_clients"]["weight_decay"]

    # <-------------- Aggregators Config ------------->
    aggregators = data["aggregator"]
    pre_aggregators = data["pre_aggregators"]

    # <-------------- Attacks Config ------------->
    attacks = data["attack"]

    # Ensure certain configurations are always lists
    nb_honest_clients = ensure_list(nb_honest_clients)
    nb_byz = ensure_list(nb_byz)
    nb_declared = ensure_list(nb_declared)
    data_distributions = ensure_list(data_distributions)
    aggregators = ensure_list(aggregators)

    # Pre-aggregators can be multiple or single dict; unify them
    if not pre_aggregators or isinstance(pre_aggregators[0], dict):
        pre_aggregators = [pre_aggregators]

    attacks = ensure_list(attacks)
    if target_attack is not None:
        attacks = [att for att in attacks if att["name"] == target_attack]
    lr_list = ensure_list(lr_list)
    momentum_list = ensure_list(momentum_list)
    wd_list = ensure_list(wd_list)

    if nb_declared[0] is None:
        declared_equal_real = True
        nb_declared = [nb_byz[-1]]
    else:
        declared_equal_real = False
    
    for pre_agg in pre_aggregators:

        for nb_honest in nb_honest_clients:

            for nb_decl in nb_declared:
                actual_nb_byz = [item for item in nb_byz if item <= nb_decl]
                pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
                pre_agg_names = "_".join(pre_agg_list_names)

                for data_dist in data_distributions:

                    data_dist["distribution_parameter"] = ensure_list(data_dist["distribution_parameter"])
                    distribution_parameter_list = data_dist["distribution_parameter"]
                    heat_map_cube = np.zeros((len(aggregators), len(distribution_parameter_list), len(actual_nb_byz)))

                    for z, agg in enumerate(aggregators):

                        heat_map_table = np.zeros((len(distribution_parameter_list), len(actual_nb_byz)))

                        for y, nb_byzantine in enumerate(actual_nb_byz):

                            if declared_equal_real:
                                nb_decl = nb_byzantine

                            if set_honest_clients_as_clients:
                                nb_nodes = nb_honest
                                nb_honest = nb_nodes - nb_byzantine
                            else:
                                nb_nodes = nb_honest + nb_byzantine

                            for x, dist_param in enumerate(distribution_parameter_list):
                                dist_param_val = get_dist_param_val(data_dist['name'], dist_param)

                                hyper_file_name = (
                                    f"{dataset_name}_"
                                    f"{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                    f"{custom_dict_to_str(data_dist['name'])}_{dist_param_val}_"
                                    f"{pre_agg_names}_{agg['name']}.txt"
                                )

                                
                                full_path = os.path.join(path_to_hyperparameters, "hyperparameters", hyper_file_name)

                                if os.path.exists(full_path):
                                    hyperparameters = np.loadtxt(full_path)
                                    lr = hyperparameters[0]
                                    momentum = hyperparameters[1]
                                    wd = hyperparameters[2]
                                else:
                                    lr = lr_list[0]
                                    momentum = momentum_list[0]
                                    wd = wd_list[0]

                                
                                worst_accuracy = np.inf
                                for attack in attacks:
                                    accuracy = get_accuracy_at_best_step(
                                        path_to_results, clean, dataset_name, model_name, nb_nodes, nb_byzantine, nb_decl,
                                        data_dist['name'], dist_param_val, agg['name'], pre_agg_names, attack['name'],
                                        lr, momentum, wd, snn_suffix, pm,
                                        nb_data_distribution_seeds, nb_training_seeds,
                                        training_seed, data_distribution_seed, evaluation_delta
                                    )

                                    if accuracy < worst_accuracy:
                                        worst_accuracy = accuracy
                                    
                                heat_map_table[len(heat_map_table)-1-x][y] = worst_accuracy

                        heat_map_cube[z] = heat_map_table
                    

                    if declared_equal_real:
                        end_file_name = "tolerated_f_equal_real.pdf"
                    else:
                        end_file_name = f"tolerated_f_{nb_decl}.pdf"

                    file_prefix = f"best_test_{target_attack}_" if target_attack else "best_test_"
                    file_name = (
                        f"{file_prefix}{dataset_name}_"
                        f"{model_name}_"
                        f"{custom_dict_to_str(data_dist['name'])}_"
                        f"{pre_agg_names}_"
                        f"nb_honest_clients_{nb_honest}_"
                        + end_file_name
                    )
                    
                    column_names = [str(dist_param) for dist_param in distribution_parameter_list]
                    row_names = [str(nb_byzantine) for nb_byzantine in actual_nb_byz]
                    column_names.reverse()

                    heat_map_table = np.max(heat_map_cube, axis=0)
                    sns.heatmap(heat_map_table, xticklabels=row_names, yticklabels=column_names, annot=True)
                    if target_attack:
                        plt.title(f"Attack: {target_attack}")
                    plt.xlabel("Number of Byzantine clients")
                    plt.ylabel("Data heterogeneity level")
                    plt.tight_layout()
                    plt.savefig(path_to_plot +"/"+ file_name)
                    plt.close()