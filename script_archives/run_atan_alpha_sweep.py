import os
from byzfl import run_benchmark
from byzfl.benchmark.evaluate_results import test_heatmap, loss_heatmap, aggregated_test_heatmap

def run_atan_alpha_sweep():
    print("\n=======================================================")
    print("Running SNN Atan Alpha Sweep (9 alphas, 5 seeds, f=0, GPUs distributed)")
    print("=======================================================")
    run_benchmark("configs/snn_atan_alpha_sweep.json", nb_jobs=9, distribute_gpus=True)
    
    # Generate convergence plots
    print("\nGenerating baseline convergence plots for Atan Alpha Sweep...")
    results_dir = "./results/snn/atan_alpha_sweep"
    plots_dir = "./plots/snn/atan_alpha_sweep"
    try:
        from plot_snn_baseline_convergence import generate_plots
        generate_plots(results_dir, plots_dir)
    except Exception as e:
        print(f"--> Error generating plots: {e}")

if __name__ == "__main__":
    run_atan_alpha_sweep()
