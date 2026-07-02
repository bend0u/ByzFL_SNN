import os
import argparse
import sys
from byzfl import run_benchmark
from plot_snn_robust_heatmaps import generate_range_heatmaps

def main():
    parser = argparse.ArgumentParser(description="Run SNN Robust Sweep over Surrogate Ranges")
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
        print(f"Starting Robust Sweep using {config_file} distributed across all available GPUs with {nb_jobs} jobs")
        # Do not overwrite CUDA_VISIBLE_DEVICES if already set by Kubernetes/Docker
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            # Fallback for local run to see all GPUs if not specified
            pass
    else:
        print(f"Starting Robust Sweep using {config_file} on GPU {gpu_idx} with {nb_jobs} jobs")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
    # Pre-download dataset sequentially to avoid race conditions in parallel jobs
    try:
        import json
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
    
    # Read the results and plots folders from config
    try:
        import json
        with open(config_file, "r") as f:
            cfg = json.load(f)
        results_dir = cfg["evaluation_and_results"]["results_directory"]
        
        # Determine plots directory
        # e.g., results_dir = "./results/snn/robust_atan_sweep" -> plots_dir = "./plots/snn/robust_atan_sweep"
        plots_dir = results_dir.replace("results", "plots")
        
        print("\nSweep complete. Generating robust heatmaps...")
        generate_range_heatmaps(results_dir, plots_dir)
        print("Done!")
    except Exception as e:
        print(f"Error resolving results/plots directory or generating heatmaps: {e}")

if __name__ == "__main__":
    main()
