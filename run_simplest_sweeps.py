import os
from byzfl import run_benchmark
from byzfl.benchmark.evaluate_results import test_heatmap, loss_heatmap, aggregated_test_heatmap

def run_all_sweeps():
    # 1. Run CNN Sweep
    print("\n=======================================================")
    print("Running CNN Simplest Sweep (500 steps, 10 honest clients, 2 GPUs)")
    print("=======================================================")
    run_benchmark("cnn_simplest_sweep.json", nb_jobs=4, distribute_gpus=True)
    
    # Generate CNN heatmaps
    print("\nGenerating heatmaps for CNN Simplest Sweep...")
    cnn_results_dir = "./cnn_simplest_results"
    cnn_plots_dir = "./cnn_simplest_plots"
    os.makedirs(cnn_plots_dir, exist_ok=True)
    try:
        test_heatmap(cnn_results_dir, cnn_plots_dir)
        loss_heatmap(cnn_results_dir, cnn_plots_dir)
        aggregated_test_heatmap(cnn_results_dir, cnn_plots_dir)
        print(f"--> CNN plots successfully saved in: {cnn_plots_dir}")
    except Exception as e:
        print(f"--> Error generating CNN heatmaps: {e}")

    # 2. Run SNN Sweep
    print("\n=======================================================")
    print("Running SNN Simplest Sweep (500 steps, 10 honest clients, 2 GPUs)")
    print("=======================================================")
    run_benchmark("snn_simplest_sweep.json", nb_jobs=4, distribute_gpus=True)
    
    # Generate SNN heatmaps
    print("\nGenerating heatmaps for SNN Simplest Sweep...")
    snn_results_dir = "./snn_simplest_results"
    snn_plots_dir = "./snn_simplest_plots"
    os.makedirs(snn_plots_dir, exist_ok=True)
    try:
        test_heatmap(snn_results_dir, snn_plots_dir)
        loss_heatmap(snn_results_dir, snn_plots_dir)
        aggregated_test_heatmap(snn_results_dir, snn_plots_dir)
        print(f"--> SNN plots successfully saved in: {snn_plots_dir}")
    except Exception as e:
        print(f"--> Error generating SNN heatmaps: {e}")

    print("\nAll sweeps and heatmap generations are complete!")

if __name__ == "__main__":
    run_all_sweeps()
