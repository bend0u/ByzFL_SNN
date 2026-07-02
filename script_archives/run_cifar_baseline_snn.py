import os
from byzfl import run_benchmark
from byzfl.benchmark.evaluate_results import test_heatmap, loss_heatmap

def run_snn_baseline():
    print("\n=======================================================")
    print("Running SNN CIFAR-10 Baseline Sweep (500 steps, f=0)")
    print("=======================================================")
    run_benchmark("configs/snn_cifar_baseline.json", nb_jobs=1, distribute_gpus=False)
    
    # Generate SNN heatmaps
    print("\nGenerating heatmaps for SNN CIFAR-10 Baseline...")
    snn_results_dir = "./results/snn/cifar_baseline"
    snn_plots_dir = "./plots/snn/cifar_baseline"
    os.makedirs(snn_plots_dir, exist_ok=True)
    try:
        test_heatmap(snn_results_dir, snn_plots_dir)
        loss_heatmap(snn_results_dir, snn_plots_dir)
        print(f"--> SNN plots successfully saved in: {snn_plots_dir}")
    except Exception as e:
        print(f"--> Error generating SNN heatmaps: {e}")

if __name__ == "__main__":
    run_snn_baseline()
