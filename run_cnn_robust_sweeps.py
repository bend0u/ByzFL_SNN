import os
import argparse
import sys
import json
from byzfl import run_benchmark
from byzfl.benchmark.evaluate_results import test_heatmap, loss_heatmap, aggregated_test_heatmap

def main():
    parser = argparse.ArgumentParser(description="Run CNN Robust Sweep")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON configuration file")
    parser.add_argument("--gpu", type=str, default="0", help="GPU index to use (e.g. 0 or 1)")
    parser.add_argument("--nb_jobs", type=int, default=8, help="Number of parallel jobs to run")
    parser.add_argument("--distribute_gpus", action="store_true", help="Distribute jobs across both GPUs on this server")
    
    args = parser.parse_args()
    
    config_file = args.config
    gpu_idx = args.gpu
    nb_jobs = args.nb_jobs
    distribute = args.distribute_gpus
    
    if not os.path.exists(config_file):
        print(f"Error: configuration file '{config_file}' does not exist.")
        sys.exit(1)
        
    print(f"=======================================================")
    if distribute:
        print(f"Starting CNN Robust Sweep using {config_file} distributed across all available GPUs with {nb_jobs} jobs")
        # Do not overwrite CUDA_VISIBLE_DEVICES if already set by environment
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            pass
    else:
        print(f"Starting CNN Robust Sweep using {config_file} on GPU {gpu_idx} with {nb_jobs} jobs")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx

    # Pre-download dataset sequentially to avoid race conditions in parallel jobs
    try:
        with open(config_file, "r") as f:
            cfg = json.load(f)
        dataset_name = cfg.get("model", {}).get("dataset_name", "mnist").lower()
        data_folder = cfg.get("evaluation_and_results", {}).get("data_folder", "./data")
        if dataset_name == "mnist":
            print("Pre-downloading MNIST dataset sequentially to avoid parallel race conditions...")
            from torchvision import datasets
            datasets.MNIST(root=data_folder, train=True, download=True)
            datasets.MNIST(root=data_folder, train=False, download=True)
            print("MNIST dataset is ready!")
    except Exception as e:
        print(f"Warning: could not pre-download dataset: {e}")

    # Run the benchmark
    run_benchmark(config_file, nb_jobs=nb_jobs, distribute_gpus=distribute)
    
    # Read the results and plots folders from config to generate heatmaps
    try:
        with open(config_file, "r") as f:
            config_data = json.load(f)
        results_dir = config_data["evaluation_and_results"]["results_directory"]
        plots_dir = results_dir + "/plots"
        
        print("\nSweep complete. Generating robust heatmaps...")
        
        attacks_list = config_data.get("attack", [])
        if isinstance(attacks_list, dict):
            attacks_list = [attacks_list]
        attack_names = [None] + [att["name"] for att in attacks_list if isinstance(att, dict) and "name" in att]

        os.makedirs(plots_dir, exist_ok=True)
        
        for attack_name in attack_names:
            attack_label = attack_name if attack_name else "merged"
            print(f"--> Generating plots for attack: {attack_label}")
            
            try:
                test_heatmap(results_dir, plots_dir, target_attack=attack_name)
                print(f"    - Saved line plots")
            except Exception as e:
                print(f"    - Error generating test line plots: {e}")

            try:
                loss_heatmap(results_dir, plots_dir, target_attack=attack_name)
                print(f"    - Saved loss heatmaps")
            except Exception as e:
                print(f"    - Error generating loss heatmaps: {e}")

            try:
                aggregated_test_heatmap(results_dir, plots_dir, target_attack=attack_name)
                print(f"    - Saved aggregated test heatmaps")
            except Exception as e:
                print(f"    - Error generating aggregated test heatmaps: {e}")

        print("Done!")
    except Exception as e:
        print(f"Error resolving results/plots directory or generating heatmaps: {e}")

if __name__ == "__main__":
    main()
