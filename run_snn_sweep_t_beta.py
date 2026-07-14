import os
import argparse
import sys
import copy
import json
import multiprocessing

# Import necessary functions from byzfl.benchmark.benchmark
from byzfl.benchmark.benchmark import (
    ensure_optional_config_parameters,
    generate_all_combinations,
    ensure_key_parameters,
    set_tolerated_f_equal_to_real_f,
    remove_real_greater_declared,
    set_declared_as_aggregation_parameter,
    compute_number_of_workers,
    delegate_training_seeds,
    delegate_data_distribution_seeds,
    eliminate_experiments_done,
    init_pool_processes,
    run_training
)

def run_custom_benchmark(config_file, nb_jobs=1, distribute_gpus=False):
    # Load configuration
    with open(config_file, 'r') as file:
        data = json.load(file)
    
    data = ensure_optional_config_parameters(data)
    
    # Determine the results directory
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

    # ================= CUSTOM FILTERING FOR SPECIFIC EXPERIMENTS =================
    # We only want:
    # 1. f == 0 and gamma == 1.0 (baseline)
    # 2. f in [2, 3, 4, 5] and gamma == 0.33
    filtered_dict_list = []
    for setting in dict_list:
        f_val = setting["benchmark_config"]["f"]
        dist_param = setting["benchmark_config"]["data_distribution"]["distribution_parameter"]
        
        is_baseline = (f_val == 0 and abs(dist_param - 1.0) < 1e-3)
        is_niid_sweep = (f_val in [2, 3, 4, 5] and abs(dist_param - 0.33) < 1e-3)
        
        if is_baseline or is_niid_sweep:
            filtered_dict_list.append(setting)
            
    dict_list = filtered_dict_list
    # =============================================================================

    # Remove already completed experiments
    dict_list = eliminate_experiments_done(dict_list)

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

    print(f"Total custom SNN sweep trainings to do: {len(dict_list)}")
    if len(dict_list) == 0:
        print("All experiments have already been completed or no valid configurations generated.")
        return
        
    print(f"Running {nb_jobs} trainings in parallel...")

    ctx = multiprocessing.get_context("spawn")
    counter = ctx.Value('i', 0)
    with ctx.Pool(initializer=init_pool_processes, initargs=(counter,), processes=nb_jobs, maxtasksperchild=1) as pool:
        pool.map(run_training, dict_list, chunksize=1)

    print("All custom SNN sweep trainings finished.")

def main():
    parser = argparse.ArgumentParser(description="Run SNN T & Beta Sweep")
    parser.add_argument("--config", type=str, default="configs/snn_robustness/snn_sweep_t_beta.json", help="Path to configuration file")
    parser.add_argument("--nb_jobs", type=int, default=20, help="Number of parallel jobs to run")
    parser.add_argument("--distribute_gpus", action="store_true", help="Distribute jobs across all GPUs")
    
    args = parser.parse_args()
    
    # Pre-download MNIST sequentially to prevent parallel download conflicts
    try:
        from torchvision import datasets
        datasets.MNIST(root="./data", train=True, download=True)
        datasets.MNIST(root="./data", train=False, download=True)
    except Exception as e:
        print(f"Dataset pre-download warning: {e}")
        
    run_custom_benchmark(args.config, nb_jobs=args.nb_jobs, distribute_gpus=args.distribute_gpus)

if __name__ == "__main__":
    main()
