"""
Run the inter-architecture geometry metrics experiment.

Trains all 9 configurations (4 SNN θ-sweep + 4 CNN clip-sweep + 1 Tanh)
at f=0, Average aggregator, 10 honest clients, 500 steps, 5 seeds,
γ ∈ {1.0, 0.66, 0.33, 0.0}.

Each config → 4 γ × 5 seeds = 20 jobs.
Total: 9 configs × 20 = 180 jobs.
Uses distribute_gpus=True with nb_jobs=5 on 2× A10.
"""
import subprocess
import sys
import os

def main():
    print("=" * 70)
    print("Inter-Architecture Geometry Metrics Experiment")
    print("  f=0, Average aggregator, 10 honest clients, 500 steps, 5 seeds")
    print("  γ ∈ {1.0, 0.66, 0.33, 0.0}")
    print("  SNN θ ∈ {1.0, 0.8, 0.6, 0.4} (lr=0.1)")
    print("  CNN ReLU clip ∈ {∞, 21, 10, 5} (lr=0.15)")
    print("  CNN Tanh (lr=0.15)")
    print("  9 configs × 20 jobs = 180 total jobs")
    print("=" * 70)

    configs = [
        # SNN θ-sweep
        'configs/interarch/config_snn_thr10.json',
        'configs/interarch/config_snn_thr08.json',
        'configs/interarch/config_snn_thr06.json',
        'configs/interarch/config_snn_thr04.json',
        # CNN ReLU: nu + clipped
        'configs/interarch/config_cnn_relu.json',
        'configs/interarch/config_cnn_relu_clip21.json',
        'configs/interarch/config_cnn_relu_clip10.json',
        'configs/interarch/config_cnn_relu_clip5.json',
        # CNN Tanh
        'configs/interarch/config_cnn_tanh.json',
    ]

    print(f"\nStarting training for {len(configs)} configurations...")
    for i, config in enumerate(configs, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(configs)}] Running benchmark for: {config}")
        print(f"{'='*70}")
        cmd = [
            sys.executable, "-c",
            f"from byzfl.benchmark.benchmark import run_benchmark; "
            f"run_benchmark('{config}', nb_jobs=8, distribute_gpus=True)"
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"WARNING: Training failed for {config} with return code {result.returncode}")
            # Continue with other configs instead of stopping
            continue

    print("\n" + "=" * 70)
    print("All training jobs completed!")
    print("Results saved to: ./results/interarch_metrics/")
    print("Run 'python plot_interarch_metrics.py' to generate plots.")
    print("=" * 70)

if __name__ == "__main__":
    main()
