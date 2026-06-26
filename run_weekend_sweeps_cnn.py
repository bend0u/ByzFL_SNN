import os
from byzfl import run_benchmark
from byzfl.benchmark.evaluate_results import test_heatmap, loss_heatmap, aggregated_test_heatmap

def run_cnn_weekend_sweep():
    print("\n=======================================================")
    print("Running CNN Weekend Sweep (5 seeds, 10 honest, 0-5 Byzantine, 16 jobs, GPUs distributed)")
    print("=======================================================")
    run_benchmark("configs/cnn_weekend_sweep.json", nb_jobs=8, distribute_gpus=True)
    
    # Generate CNN heatmaps
    print("\nGenerating heatmaps for CNN Weekend Sweep...")
    cnn_results_dir = "./results/cnn/weekend"
    cnn_plots_dir = "./plots/cnn/weekend"
    os.makedirs(cnn_plots_dir, exist_ok=True)
    try:
        test_heatmap(cnn_results_dir, cnn_plots_dir)
        loss_heatmap(cnn_results_dir, cnn_plots_dir)
        aggregated_test_heatmap(cnn_results_dir, cnn_plots_dir)
        print(f"--> CNN plots successfully saved in: {cnn_plots_dir}")
    except Exception as e:
        print(f"--> Error generating CNN heatmaps: {e}")

if __name__ == "__main__":
    run_cnn_weekend_sweep()
