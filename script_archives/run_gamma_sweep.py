import os
import json
import traceback
from byzfl import run_benchmark
from byzfl.benchmark.evaluate_results import test_heatmap, aggregated_test_heatmap

TEMPLATE_PATH = "snn_mnist_direct.json"
SWEEP_CONFIG_PATH = "gamma_sweep_config.json"
RESULTS_DIR = "./test_snn_benchmark_results_gamma"
PLOTS_DIR = "./test_snn_benchmark_plots_gamma"

def main():
    print("=== SNN BYZANTINE BENCHMARK SWEEP ===")
    print(f"Loading template configuration from: {TEMPLATE_PATH}")
    
    if not os.path.exists(TEMPLATE_PATH):
        print(f"Error: Template config '{TEMPLATE_PATH}' not found.")
        return
        
    with open(TEMPLATE_PATH, "r") as f:
        config = json.load(f)
        
    # 1. Update config parameters for the sweep
    config["benchmark_config"]["device"] = "cuda"
    config["benchmark_config"]["nb_steps"] = 100
    config["benchmark_config"]["evaluation_delta"] = 10
    config["benchmark_config"]["f"] = [1, 2, 3, 4]
    
    # Remove tolerated_f so the runner sets tolerated_f dynamically equal to real_f
    if "tolerated_f" in config["benchmark_config"]:
        del config["benchmark_config"]["tolerated_f"]
        
    # Data distribution parameters: gamma_similarity_niid with 0.0, 0.33, 0.66, 1.0
    config["benchmark_config"]["data_distribution"] = [
        {
            "name": "gamma_similarity_niid",
            "distribution_parameter": [0.0, 0.33, 0.66, 1.0]
        }
    ]
    
    # Aggregators under evaluation
    config["aggregator"] = [
        {"name": "TrMean", "parameters": {}},
        {"name": "GeometricMedian", "parameters": {"nu": 0.1, "T": 3}}
    ]
    config["pre_aggregators"] = []
    
    # Attacks under evaluation
    config["attack"] = [
        {"name": "SignFlipping", "parameters": {}},
        {"name": "Optimal_InnerProductManipulation", "parameters": {}}
    ]
    
    # Directories setup
    config["evaluation_and_results"]["results_directory"] = RESULTS_DIR
    config["evaluation_and_results"]["clean_directory_structure"] = True

    # Save the config file temporarily
    print(f"Saving temporary sweep configuration to: {SWEEP_CONFIG_PATH}")
    with open(SWEEP_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)
        
    # 2. Run the sweep using 8 parallel jobs on cuda
    print("\nStarting sweep benchmark execution (8 parallel jobs)...")
    try:
        run_benchmark(SWEEP_CONFIG_PATH, nb_jobs=4)
        print("Sweep benchmark execution completed successfully!")
    except Exception as e:
        print(f"Error executing run_benchmark: {e}")
        traceback.print_exc()
        return
    finally:
        # Clean up temporary configuration file
        if os.path.exists(SWEEP_CONFIG_PATH):
            os.remove(SWEEP_CONFIG_PATH)
            print(f"Removed temporary configuration file: {SWEEP_CONFIG_PATH}")
            
    # 3. Generate heatmap plots
    print("\nGenerating heatmap plots...")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    try:
        print("1. Generating individual worst-case accuracy heatmaps...")
        test_heatmap(RESULTS_DIR, PLOTS_DIR)
        
        print("2. Generating aggregated comparison heatmap...")
        aggregated_test_heatmap(RESULTS_DIR, PLOTS_DIR)
        
        print(f"Successfully generated all heatmaps! Saved in: {PLOTS_DIR}")
        
        # Print list of generated files
        print("Generated files:")
        for fn in sorted(os.listdir(PLOTS_DIR)):
            print(f"  - {os.path.join(PLOTS_DIR, fn)}")
    except Exception as e:
        print(f"Error during heatmap generation: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
