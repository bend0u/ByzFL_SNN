from byzfl import run_benchmark

if __name__ == "__main__":
    run_benchmark("cnn_extra_seeds.json", nb_jobs=2)
    run_benchmark("snn_extra_seeds_direct.json", nb_jobs=2)
