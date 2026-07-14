"""
Run the sparsity measurement experiment for SNN ATAN α=1.2, CNN (ReLU), and CNN Tanh.
Trains with f=0, Average aggregator, 10 honest clients, across all γ levels.
Saves Hoyer, Gini, and 8 other sparsity metrics per training step.
"""
import subprocess
import sys

def main():
    print("=" * 60)
    print("Sparsity Measurement Experiment: SNN & CNN Comparison")
    print("  f=0, Average aggregator, 10 honest clients")
    print("  γ ∈ {1.0, 0.66, 0.33, 0.0}")
    print("  5 training seeds × 4 γ levels = 20 jobs per model")
    print("=" * 60)

    configs = [
        'configs/sparsity_measure_atan12.json',
        'configs/sparsity_measure_cnn.json',
        'configs/sparsity_measure_cnn_tanh.json'
    ]

    print("\nStarting training for all configurations...")
    for config in configs:
        print(f"\n>>> Running benchmark for config: {config}")
        cmd = [
            sys.executable, "-c",
            f"from byzfl.benchmark.benchmark import run_benchmark; "
            f"run_benchmark('{config}', nb_jobs=5, distribute_gpus=True)"
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Training failed for {config} with return code {result.returncode}")
            sys.exit(1)

    print("\nAll training jobs completed!")
    print("Results saved to: ./results/sparsity_measure/")
    print("Run 'python plot_sparsity_measure.py' to generate plots.")

if __name__ == "__main__":
    main()

