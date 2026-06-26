import os
from byzfl import run_benchmark
from byzfl.benchmark.evaluate_results import test_heatmap, loss_heatmap, aggregated_test_heatmap

def run_snn_weekend_sweep():
    print("\n=======================================================")
    print("Running SNN Weekend Sweep (5 seeds, 10 honest, 0-5 Byzantine, 16 jobs, GPUs distributed)")
    print("=======================================================")
    run_benchmark("configs/snn_weekend_sweep.json", nb_jobs=16, distribute_gpus=True)
    
    # Generate SNN heatmaps
    print("\nGenerating heatmaps for SNN Weekend Sweep...")
    snn_results_dir = "./results/snn/weekend"
    snn_plots_dir = "./plots/snn/weekend"
    os.makedirs(snn_plots_dir, exist_ok=True)
    try:
        test_heatmap(snn_results_dir, snn_plots_dir)
        loss_heatmap(snn_results_dir, snn_plots_dir)
        aggregated_test_heatmap(snn_results_dir, snn_plots_dir)
        print(f"--> SNN plots successfully saved in: {snn_plots_dir}")
    except Exception as e:
        print(f"--> Error generating SNN heatmaps: {e}")

if __name__ == "__main__":
    run_snn_weekend_sweep()
