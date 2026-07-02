import os
from byzfl import run_benchmark
from byzfl.benchmark.evaluate_results import test_heatmap, loss_heatmap, aggregated_test_heatmap

def run_sweep(config_path, results_dir, plots_dir):
    print(f"\n=======================================================")
    print(f"Running N-MNIST SNN Benchmark Sweep: {config_path}")
    print(f"=======================================================")
    
    run_benchmark(config_path, nb_jobs=1)
    
    # Generate heatmap plots
    print(f"Generating heatmaps for {config_path}...")
    os.makedirs(plots_dir, exist_ok=True)
    test_heatmap(results_dir, plots_dir)
    loss_heatmap(results_dir, plots_dir)
    aggregated_test_heatmap(results_dir, plots_dir)
    print(f"Plots successfully saved in: {plots_dir}")

if __name__ == "__main__":
    run_sweep("configs/snn_complete_nmnist.json", "./results/snn/complete_nmnist", "./plots/snn/complete_nmnist")
