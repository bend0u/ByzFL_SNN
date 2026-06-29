import os
from byzfl import run_benchmark
from byzfl.benchmark.evaluate_results import test_heatmap, loss_heatmap

def run_cnn_baseline():
    print("\n=======================================================")
    print("Running CNN CIFAR-10 Baseline Sweep (4000 steps)")
    print("=======================================================")
    run_benchmark("configs/cnn_cifar_baseline.json", nb_jobs=8, distribute_gpus=True)
    
    # Generate CNN heatmaps
    print("\nGenerating heatmaps for CNN CIFAR-10 Baseline...")
    cnn_results_dir = "./results/cnn/cifar_baseline"
    cnn_plots_dir = "./plots/cnn/cifar_baseline"
    os.makedirs(cnn_plots_dir, exist_ok=True)
    try:
        test_heatmap(cnn_results_dir, cnn_plots_dir)
        loss_heatmap(cnn_results_dir, cnn_plots_dir)
        print(f"--> CNN plots successfully saved in: {cnn_plots_dir}")
    except Exception as e:
        print(f"--> Error generating CNN heatmaps: {e}")

if __name__ == "__main__":
    run_cnn_baseline()
