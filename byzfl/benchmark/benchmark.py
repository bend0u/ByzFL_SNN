import json
from multiprocessing import Pool, Value
import os
import copy

from byzfl.benchmark.train import start_training
from byzfl.benchmark.evaluate_results import find_best_hyperparameters
from byzfl.benchmark.managers import ParamsManager, get_snn_suffix

default_config = {
    "benchmark_config": {
        "training_algorithm": {
            "name": "DSGD",
            "parameters": {}
        },
        "nb_steps": 800,
        "device": "cuda",
        "training_seed": 0,
        "nb_training_seeds": 3,
        "nb_honest_clients": 10,
        "f": [1, 2, 3, 4],
        "data_distribution_seed": 0,
        "nb_data_distribution_seeds": 1,
        "data_distribution": [
            {
                "name": "gamma_similarity_niid",
                "distribution_parameter": [1.0, 0.66, 0.33, 0.0]
            }
        ],
    },
    "model": {
        "name": "cnn_mnist",
        "dataset_name": "mnist",
        "nb_labels": 10,
        "loss": "NLLLoss",
        "learning_rate": 0.1,
        "learning_rate_decay": 1.0,
        "milestones": []
    },
    "aggregator": [
        {
            "name": "GeometricMedian",
            "parameters": {
                "nu": 0.1,
                "T": 3
            }
        },
        {
            "name": "TrMean",
            "parameters": {}
        }
    ],
    "pre_aggregators": [
        {
            "name": "Clipping",
            "parameters": {}
        },
        {
            "name": "NNM",
            "parameters": {}
        }
    ],
    "honest_clients": {
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "batch_size": 25
    },
    "attack": [
        {
            "name": "SignFlipping",
            "parameters": {}
        },
        {
            "name": "Optimal_InnerProductManipulation",
            "parameters": {}
        },
        {
            "name": "Optimal_ALittleIsEnough",
            "parameters": {}
        }
    ],
    "evaluation_and_results": {
        "evaluation_delta": 50,
        "batch_size_evaluation": 128,
        "evaluate_on_test": True,
        "store_per_client_metrics": True,
        "store_models": False,
        "data_folder": "./data",
        "results_directory": "./results"
    }
}

def generate_all_combinations_aux(list_dict, orig_dict, aux_dict, rest_list):
    """
    Recursively builds all combinations of key-value pairs from a nested dictionary.

    This helper function iterates over the keys in `orig_dict` and, depending on the type of the corresponding value,
    it recursively constructs combinations of values:
      - If a value is a list:
          - When the list is empty or the key is in `rest_list` (and its first element is not a list),
            the entire list is assigned to the key.
          - Otherwise, the function iterates over each item in the list. If an item is a dictionary, the function
            recursively generates combinations for that dictionary; if not, it treats the item as a single-element list.
      - If a value is a dictionary, it recursively generates combinations for that sub-dictionary.
      - For other types, the value is directly assigned.

    When the auxiliary dictionary `aux_dict` has entries for all keys in `orig_dict`, it is considered a complete
    combination and is appended to `list_dict`.

    Parameters:
        list_dict (list): A list that accumulates the resulting combinations. Each element is a dictionary representing
            one combination.
        orig_dict (dict): The original dictionary from which combinations are to be generated. Its values may be lists,
            dictionaries, or atomic values.
        aux_dict (dict): An auxiliary dictionary used to build up a single combination during the recursive process.
        rest_list (list): A list of keys for which list values in `orig_dict` should be treated as atomic (i.e., not iterated
            over), even if they contain list elements.

    Returns:
        None: The function appends complete combinations to `list_dict` as a side effect.
    """
    if len(aux_dict) < len(orig_dict):
        key = list(orig_dict)[len(aux_dict)]
        if isinstance(orig_dict[key], list):
            if not orig_dict[key] or key in rest_list:
                aux_dict[key] = orig_dict[key]
                generate_all_combinations_aux(list_dict, 
                                              orig_dict, 
                                              aux_dict, 
                                              rest_list)
            else:
                for item in orig_dict[key]:
                    if isinstance(item, dict):
                        new_list_dict = []
                        new_aux_dict = {}
                        generate_all_combinations_aux(new_list_dict, 
                                                    item, 
                                                    new_aux_dict, 
                                                    rest_list)
                    else:
                        new_list_dict = [item]
                    for new_dict in new_list_dict:
                        new_aux_dict = copy.deepcopy(aux_dict)
                        new_aux_dict[key] = new_dict
                        
                        generate_all_combinations_aux(list_dict,
                                                    orig_dict, 
                                                    new_aux_dict, 
                                                    rest_list)
        elif isinstance(orig_dict[key], dict):
            new_list_dict = []
            new_aux_dict = {}
            generate_all_combinations_aux(new_list_dict, 
                                          orig_dict[key], 
                                          new_aux_dict, 
                                          rest_list)
            for dictionary in new_list_dict:
                new_aux_dict = aux_dict.copy()
                new_aux_dict[key] = dictionary
                generate_all_combinations_aux(list_dict, 
                                              orig_dict, 
                                              new_aux_dict, 
                                              rest_list)
        else:
            aux_dict[key] = orig_dict[key]
            generate_all_combinations_aux(list_dict, 
                                          orig_dict, 
                                          aux_dict, 
                                          rest_list)
    else:
        list_dict.append(aux_dict)

def generate_all_combinations(original_dict, restriction_list):
    """
    Generates all possible combinations from a nested dictionary structure.

    This function acts as the entry point for generating combinations from `original_dict`. It handles
    nested structures where values can be lists or dictionaries, and uses the helper function
    `generate_all_combinations_aux` to recursively construct every possible combination of key-value pairs.
    For keys specified in `restriction_list`, list values are treated as atomic (i.e., the list is not iterated
    over) and is directly assigned as the value.

    Parameters:
        original_dict (dict): The dictionary from which to generate combinations. Its values may include lists,
            nested dictionaries, or simple values.
        restriction_list (list): A list of keys whose list values should be treated as atomic, meaning the list
            is used as is without iterating over its items.

    Returns:
        list: A list of dictionaries. Each dictionary represents one complete combination of key-value pairs
            generated from `original_dict`.
    """
    list_dict = []
    aux_dict = {}
    generate_all_combinations_aux(list_dict, original_dict, aux_dict, restriction_list)
    return list_dict

# Global variable to keep track of training progress
counter = None

def init_pool_processes(shared_value):
    """
    Initialize a global counter variable for multiprocess tracking.

    Parameters
    ----------
    shared_value : multiprocessing.Value
        A shared memory integer used to track the number of finished trainings.
    """
    global counter
    counter = shared_value


def run_training(params):
    """
    Run a single training job, then increment the global training counter.

    Parameters
    ----------
    params : dict
        A dictionary containing all necessary parameters for the training job.
    """
    try:
        start_training(params)
    except Exception as e:
        import traceback
        attack_name = params.get("attack", {}).get("name", "Unknown") if isinstance(params.get("attack"), dict) else "Unknown"
        agg_name = params.get("aggregator", {}).get("name", "Unknown") if isinstance(params.get("aggregator"), dict) else "Unknown"
        f_val = params.get("benchmark_config", {}).get("f", "Unknown")
        print(f"\n[ERROR] Training failed for config: attack={attack_name}, aggregator={agg_name}, f={f_val}. Error: {e}")
        traceback.print_exc()
    finally:
        with counter.get_lock():
            print(f"Training {counter.value} done")
            counter.value += 1

def eliminate_experiments_done(dict_list):
    """
    Remove any configurations (experiments) that have already been completed.

    Parameters
    ----------
    dict_list : list of dict
        A list of configuration dictionaries for each experiment.

    Returns
    -------
    list of dict
        The filtered list of configurations for which experiments are not yet done.
    """
    if not dict_list:
        return dict_list

    directory = dict_list[0]["evaluation_and_results"]["results_directory"]
    if not os.path.isdir(directory):
        return dict_list

    new_dict_list = []
    for setting in dict_list:
        clean = (
            setting.get("clean_directory_structure", False) or
            setting.get("evaluation_and_results", {}).get("clean_directory_structure", False) or
            setting.get("benchmark_config", {}).get("clean_directory_structure", False)
        )

        pre_aggregation_names = [
            agg['name'] for agg in setting["pre_aggregators"]
        ]
        
        pm = ParamsManager(setting)
        
        if clean:
            encoding = pm.get_encoding_type()
            enc_name = "direct" if encoding == "constant" else encoding
            parent_dir = f"{setting['model']['dataset_name']}_{enc_name}"
            
            preaggs_aggregator = '_'.join(pre_aggregation_names + [setting['aggregator']['name']])
            
            dist_name = setting['benchmark_config']['data_distribution']['name']
            dist_param = setting['benchmark_config']['data_distribution']['distribution_parameter']
            if dist_name in ["iid", "extreme_niid"]:
                dist_part = dist_name
            else:
                dist_part = f"{dist_name}_{dist_param}"

            folder_name = (
                f"{setting['attack']['name']}_"
                f"{preaggs_aggregator}_"
                f"f_{setting['benchmark_config']['f']}_"
                f"{dist_part}"
            )
            target_folder_path = os.path.join(directory, parent_dir, folder_name)
        else:
            folder_name = (
                f"{setting['model']['dataset_name']}_"
                f"{setting['model']['name']}_"
                f"n_{setting['benchmark_config']['nb_workers']}_"
                f"f_{setting['benchmark_config']['f']}_"
                f"d_{setting['benchmark_config']['tolerated_f']}_"
                f"{setting['benchmark_config']['data_distribution']['name']}_"
                f"{setting['benchmark_config']['data_distribution']['distribution_parameter']}_"
                f"{setting['aggregator']['name']}_"
                f"{'_'.join(pre_aggregation_names)}_"
                f"{setting['attack']['name']}_"
                f"lr_{setting['model']['learning_rate']}_"
                f"mom_{setting['honest_clients']['momentum']}_"
                f"wd_{setting['honest_clients']['weight_decay']}"
            )
            folder_name += get_snn_suffix(pm)
            target_folder_path = os.path.join(directory, folder_name)

        if os.path.isdir(target_folder_path):
            # Check if a particular seed combination is already done
            training_seed = setting["benchmark_config"]["training_seed"]
            data_distribution_seed = setting["benchmark_config"]["data_distribution_seed"]

            file_name = (
                f"train_time_tr_seed_{training_seed}"
                f"_dd_seed_{data_distribution_seed}.txt"
            )
            try:
                files_in_folder = os.listdir(target_folder_path)
            except Exception:
                files_in_folder = []
            if file_name not in files_in_folder:
                new_dict_list.append(setting)
        else:
            new_dict_list.append(setting)

    return new_dict_list


def delegate_training_seeds(dict_list):
    """
    For each configuration, generate new configurations for each specified training seed.

    Parameters
    ----------
    dict_list : list of dict
        A list of configuration dictionaries (each containing a base training_seed).

    Returns
    -------
    list of dict
        A new list of configurations, each with a unique training_seed.
    """
    new_dict_list = []
    for setting in dict_list:
        original_seed = setting["benchmark_config"]["training_seed"]
        nb_seeds = setting["benchmark_config"]["nb_training_seeds"]
        for i in range(nb_seeds):
            new_setting = copy.deepcopy(setting)
            new_setting["benchmark_config"]["training_seed"] = original_seed + i
            new_dict_list.append(new_setting)
    return new_dict_list


def delegate_data_distribution_seeds(dict_list):
    """
    For each configuration, generate new configurations for each specified data distribution seed.

    Parameters
    ----------
    dict_list : list of dict
        A list of configuration dictionaries (each containing a base data_distribution_seed).

    Returns
    -------
    list of dict
        A new list of configurations, each with a unique data_distribution_seed.
    """
    new_dict_list = []
    for setting in dict_list:
        original_seed = setting["benchmark_config"]["data_distribution_seed"]
        nb_seeds = setting["benchmark_config"]["nb_data_distribution_seeds"]
        for i in range(nb_seeds):
            new_setting = copy.deepcopy(setting)
            new_setting["benchmark_config"]["data_distribution_seed"] = original_seed + i
            new_dict_list.append(new_setting)
    return new_dict_list


def remove_real_greater_declared(dict_list):
    """
    Filter out configurations where the real number of Byzantine workers
    exceeds the declared number.

    Parameters
    ----------
    dict_list : list of dict
        A list of configuration dictionaries.

    Returns
    -------
    list of dict
        The filtered list where tolerated_f >= f.
    """
    new_dict_list = []
    for setting in dict_list:
        real_byz = setting["benchmark_config"]["f"]
        declared_byz = setting["benchmark_config"]["tolerated_f"]
        if declared_byz >= real_byz:
            new_dict_list.append(setting)
    return new_dict_list


def set_tolerated_f_equal_to_real_f(dict_list):
    """
    Set the 'tolerated_f' parameter equal to 'f' for each configuration.

    Parameters
    ----------
    dict_list : list of dict
        A list of configuration dictionaries.

    Returns
    -------
    list of dict
        The modified list with 'tolerated_f' set to 'f'.
    """
    new_dict_list = []
    for setting in dict_list:
        setting["benchmark_config"]["tolerated_f"] = setting["benchmark_config"]["f"]
        new_dict_list.append(setting)
    return new_dict_list

def set_declared_as_aggregation_parameter(dict_list):
    """
    For each configuration, set the aggregator and preaggregator parameter 'f' to the declared number of Byzantine workers.

    Parameters
    ----------
    dict_list : list of dict
        A list of configuration dictionaries.

    Returns
    -------
    list of dict
        The modified list with aggregator parameters updated.
    """
    for setting in dict_list:
        declared_byz = setting["benchmark_config"]["tolerated_f"]
        setting["aggregator"]["parameters"]["f"] = declared_byz

        for pre_agg in setting["pre_aggregators"]:
                pre_agg["parameters"]["f"] = declared_byz
                
    return dict_list

def compute_number_of_workers(dict_list):
    for setting in dict_list:
        # Adjust the number of workers if needed
        if setting["benchmark_config"]["set_honest_clients_as_clients"]:
            setting["benchmark_config"]["nb_workers"] = setting["benchmark_config"]["nb_honest_clients"]
            setting["benchmark_config"]["nb_honest_clients"] = (
                setting["benchmark_config"]["nb_workers"]
                - setting["benchmark_config"]["f"]
            )
        else:
            setting["benchmark_config"]["nb_workers"] = (
                setting["benchmark_config"]["nb_honest_clients"]
                + setting["benchmark_config"]["f"]
            )
    return dict_list

def ensure_key_parameters(dict_list):
    """
    Ensures that each dictionary in dict_list contains a "parameters" key within 
    "aggregator", "pre_aggregators", and "attack" dictionaries. If the "parameters" 
    key is missing, it is initialized as an empty dictionary.
    """
    for setting in dict_list:
        if "parameters" not in setting["aggregator"].keys():
            setting["aggregator"]["parameters"] = {}
        
        if "pre_aggregators" not in setting.keys():
            setting["pre_aggregators"] = []

        for pre_agg in setting["pre_aggregators"]:
            if "parameters" not in pre_agg.keys():
                pre_agg["parameters"] = {}
        
        if "parameters" not in setting["attack"].keys():
            setting["attack"]["parameters"] = {}
        
        if "training_algorithm" not in setting["benchmark_config"].keys():
            setting["benchmark_config"]["training_algorithm"] = {
                "name": "DSGD",
                "parameters": {}
            }
        
        if setting["benchmark_config"]["training_algorithm"]["name"] == "FedAvg":
            ta_params =  setting["benchmark_config"]["training_algorithm"].get("parameters", {})

            # Check for 'proportion_selected_clients'
            if "proportion_selected_clients" not in ta_params:
                raise ValueError("Missing 'proportion_selected_clients' in training algorithm parameters for FedAvg")

            proportion_selected_clients = ta_params["proportion_selected_clients"]
            if proportion_selected_clients <= 0 or proportion_selected_clients > 1:
                raise ValueError(f"Proportion of selected clients must be in (0, 1], but got {proportion_selected_clients}")

            # Check for 'local_steps_per_client'
            if "local_steps_per_client" not in ta_params:
                raise ValueError("Missing 'local_steps_per_client' in training algorithm parameters for FedAvg")

            local_steps_per_client = ta_params["local_steps_per_client"]
            if not isinstance(local_steps_per_client, int) or local_steps_per_client <= 0:
                raise ValueError(f"Number of training rounds per step must be a positive integer, but got {local_steps_per_client}")

                
    return dict_list


def ensure_optional_config_parameters(data):

    if "nb_honest_clients" not in data["benchmark_config"].keys():
        data["benchmark_config"]["nb_honest_clients"] = 10

    if "set_honest_clients_as_clients" not in data["benchmark_config"].keys():
        data["benchmark_config"]["set_honest_clients_as_clients"] = False
    
    if "training_seed" not in data["benchmark_config"].keys():
        data["benchmark_config"]["training_seed"] = 0

    if "nb_training_seeds" not in data["benchmark_config"].keys():
        data["benchmark_config"]["nb_training_seeds"] = 1
    
    if "data_distribution_seed" not in data["benchmark_config"].keys():
        data["benchmark_config"]["data_distribution_seed"] = 0

    if "nb_data_distribution_seeds" not in data["benchmark_config"].keys():
        data["benchmark_config"]["nb_data_distribution_seeds"] = 1
    
    if "results_directory" not in data["evaluation_and_results"].keys():
        data["evaluation_and_results"]["results_directory"] = "./results"

    if "size_train_set" not in data["benchmark_config"].keys():
        data["benchmark_config"]["size_train_set"] = 0.8

    return data


def run_benchmark(config_file="config.json", nb_jobs=1, distribute_gpus=False):
    """
    Run benchmark experiments in parallel, based on configurations defined
    in the specified config file.
    """
    # Support backward compatibility when run_benchmark was called as run_benchmark(nb_jobs)
    if isinstance(config_file, int):
        nb_jobs = config_file
        config_file = "config.json"
    # Attempt to load the configuration or create one if not found
    try:
        with open(config_file, 'r') as file:
            data = json.load(file)
        
        data = ensure_optional_config_parameters(data)
        if float(data["benchmark_config"]["size_train_set"]) == 1.0:
            print("WARNING: NO VALIDATION DATASET USED FOR HYPERPARAMETER EXPLORATION (Learning Rate, Momentum, Weight Decay)")

    except FileNotFoundError:
        if config_file == "config.json":
            print("'config.json' not found. Creating a default one...")
            with open('config.json', 'w') as f:
                json.dump(default_config, f, indent=4)
            print("'config.json' created successfully.")
            print("Please configure the experiment you want to run and re-run.")
        else:
            print(f"Error: configuration file '{config_file}' not found.")

        return

    # Determine the results directory (default to ./results)
    results_directory = data["evaluation_and_results"]["results_directory"]
    os.makedirs(results_directory, exist_ok=True)

    # Save the current config inside the results directory
    config_path = os.path.join(results_directory, "config.json")
    with open(config_path, 'w') as json_file:
        json.dump(data, json_file, indent=4, separators=(',', ': '))

    # Generate all combination dictionaries
    restriction_list = ["pre_aggregators", "milestones"]
    dict_list = generate_all_combinations(data, restriction_list)



    # Ensure that the key parameters are present in the dictionaries
    # even if they are not in the config file
    dict_list = ensure_key_parameters(dict_list)

    # Filter combinations based on f vs. tolerated f
    if "tolerated_f" not in data["benchmark_config"]:
        dict_list = set_tolerated_f_equal_to_real_f(dict_list)
    else:
        dict_list = remove_real_greater_declared(dict_list)

    # Set declared parameters in the dictionaries where necessary
    dict_list = set_declared_as_aggregation_parameter(dict_list)

    # Compute the number of workers
    dict_list = compute_number_of_workers(dict_list)

    # Assign seeds
    dict_list = delegate_training_seeds(dict_list)
    dict_list = delegate_data_distribution_seeds(dict_list)

    # Remove already completed experiments
    dict_list = eliminate_experiments_done(dict_list)

    # Sort dict_list by learning rate to ensure sequential completion of each LR
    if dict_list and "model" in dict_list[0] and "learning_rate" in dict_list[0]["model"]:
        dict_list = sorted(dict_list, key=lambda x: x["model"]["learning_rate"])

    # Distribute tasks across available GPUs if requested and target device is "cuda"
    device_setting = data["benchmark_config"].get("device", "cuda")
    if distribute_gpus and device_setting == "cuda":
        import torch
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                print(f"Distributing tasks across {num_gpus} GPUs...")
                for idx, setting in enumerate(dict_list):
                    gpu_id = idx % num_gpus
                    setting["benchmark_config"]["device"] = f"cuda:{gpu_id}"

    print(f"Total trainings to do: {len(dict_list)}")
    print(f"Running {nb_jobs} trainings in parallel...")

    import multiprocessing
    ctx = multiprocessing.get_context("spawn")
    counter = ctx.Value('i', 0)
    with ctx.Pool(initializer=init_pool_processes, initargs=(counter,), processes=nb_jobs, maxtasksperchild=1) as pool:
        pool.map(run_training, dict_list, chunksize=1)

    print("All trainings finished.")

    if float(data["benchmark_config"]["size_train_set"]) == 1.0:
        print("No hyperparameter exploration done.")
    else:
        try:
            find_best_hyperparameters(results_directory)
            print("Done")
        except Exception as e:
            print(f"Warning: Best hyperparameter selection skipped or failed: {e}")