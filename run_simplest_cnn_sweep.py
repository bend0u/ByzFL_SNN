import os
from byzfl import run_benchmark
from byzfl.benchmark.evaluate_results import test_heatmap, loss_heatmap, aggregated_test_heatmap

def run_sweep(config_path):
    print(f"\n=======================================================")
    print(f"Running CNN Simplest Sweep: {config_path}")
    print(f"=======================================================")
    
    # Run the sweep with nb_jobs = 4 (or adjust as needed)
    run_benchmark(config_path, nb_jobs=4)
    
    # Define directories
    results_dir = "./cnn_simplest_results"
    plots_dir = "./cnn_simplest_plots"
    
    # Generate heatmap plots
    print(f"Generating heatmaps for {config_path}...")
    os.makedirs(plots_dir, exist_ok=True)
    test_heatmap(results_dir, plots_dir)
    loss_heatmap(results_dir, plots_dir)
    aggregated_test_heatmap(results_dir, plots_dir)
    print(f"Plots successfully saved in: {plots_dir}")

if __name__ == "__main__":
    run_sweep("cnn_simplest_sweep.json")
