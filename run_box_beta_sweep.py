import os
from byzfl import run_benchmark
from byzfl.benchmark.evaluate_results import test_heatmap, loss_heatmap, aggregated_test_heatmap

def run_box_beta_sweep():
    print("\n=======================================================")
    print("Running SNN Box Beta Sweep (7 betas, 5 seeds, f=0, GPUs distributed)")
    print("=======================================================")
    run_benchmark("configs/snn_box_beta_sweep.json", nb_jobs=5, distribute_gpus=False)
    
    # Generate heatmaps
    print("\nGenerating heatmaps for Box Beta Sweep...")
    results_dir = "./results/snn/box_beta_sweep"
    plots_dir = "./plots/snn/box_beta_sweep"
    os.makedirs(plots_dir, exist_ok=True)
    try:
        test_heatmap(results_dir, plots_dir)
        loss_heatmap(results_dir, plots_dir)
        aggregated_test_heatmap(results_dir, plots_dir)
        print(f"--> Plots successfully saved in: {plots_dir}")
    except Exception as e:
        print(f"--> Error generating heatmaps: {e}")

if __name__ == "__main__":
    run_box_beta_sweep()
