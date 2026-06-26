import os
from byzfl import run_benchmark
from byzfl.benchmark.evaluate_results import test_heatmap, loss_heatmap, aggregated_test_heatmap

def run_sweep(config_path, results_dir, plots_dir):
    print(f"\n=======================================================")
    print(f"Running SNN Benchmark Sweep: {config_path}")
    print(f"=======================================================")
    
    # Run the sweep
    run_benchmark(config_path, nb_jobs=2)
    
    # Generate heatmap plots
    print(f"Generating heatmaps for {config_path}...")
    os.makedirs(plots_dir, exist_ok=True)
    test_heatmap(results_dir, plots_dir)
    loss_heatmap(results_dir, plots_dir)
    aggregated_test_heatmap(results_dir, plots_dir)
    print(f"Plots successfully saved in: {plots_dir}")

if __name__ == "__main__":
    run_sweep("configs/snn_complete_direct.json", "./results/snn/complete_direct", "./plots/snn/complete_direct")
    run_sweep("configs/snn_complete_rate.json", "./results/snn/complete_rate", "./plots/snn/complete_rate")
