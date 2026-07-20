import sys
from byzfl.benchmark.benchmark import run_benchmark

def main():
    config_file = 'configs/full_sweep_thr04.json'
    print(f"Running benchmark for: {config_file}")
    # 4 V100 GPUs means distribute_gpus=True
    # To utilize them well, we can run 4, 8, 12, or 16 jobs in parallel.
    # 16 jobs in parallel (4 per GPU) is a good balance for V100.
    run_benchmark(config_file, nb_jobs=16, distribute_gpus=True)

if __name__ == "__main__":
    main()
