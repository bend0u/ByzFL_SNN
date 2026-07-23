import json
from byzfl.benchmark.benchmark import run_benchmark

if __name__ == '__main__':
    config_path = "configs/snn_robustness/snn_sfma_test.json"
    run_benchmark(config_path, nb_jobs=20, distribute_gpus=True)
