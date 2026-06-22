import os
from byzfl import run_benchmark
from byzfl.benchmark.evaluate_results import test_heatmap, loss_heatmap, aggregated_test_heatmap

def run_sweep(config_path):
    print(f"\n=======================================================")
    print(f"Running SNN Benchmark Sweep: {config_path}")
    print(f"=======================================================")
    
    # Run the sweep with nb_jobs = 2
    run_benchmark(config_path, nb_jobs=2)
    
    # Define directories
    results_dir = config_path.replace(".json", "").replace("snn_complete_", "snn_complete_results_")
    plots_dir = results_dir.replace("results", "plots_extended")
    
    # Generate heatmap plots
    print(f"Generating heatmaps for {config_path}...")
    os.makedirs(plots_dir, exist_ok=True)
    test_heatmap(results_dir, plots_dir)
    loss_heatmap(results_dir, plots_dir)
    aggregated_test_heatmap(results_dir, plots_dir)
    print(f"Plots successfully saved in: {plots_dir}")

if __name__ == "__main__":
    run_sweep("snn_complete_direct.json")
