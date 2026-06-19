import os
import json
import shutil
from byzfl import run_benchmark
from byzfl.benchmark.evaluate_results import test_heatmap, loss_heatmap, aggregated_test_heatmap

TEST_CONFIG = "temp_test_config.json"
TEST_RESULTS = "./test_verify_results"
TEST_PLOTS = "./test_verify_plots"

def test_pipeline():
    print("=== STARTING PIPELINE INTEGRATION TEST ===")
    
    # 1. Load complete direct config
    with open("snn_complete_direct.json", "r") as f:
        config = json.load(f)
        
    # 2. Modify config for a fast 4-run test (10 steps, 2x2 grid)
    config["benchmark_config"]["nb_steps"] = 10
    config["evaluation_and_results"]["evaluation_delta"] = 5
    config["benchmark_config"]["f"] = [2, 4]
    config["benchmark_config"]["data_distribution"] = [{
        "name": "gamma_similarity_niid",
        "distribution_parameter": [0.33, 0.66]
    }]
    config["aggregator"] = [{"name": "TrMean", "parameters": {}}]
    config["attack"] = [{"name": "SignFlipping", "parameters": {}}]
    config["evaluation_and_results"]["results_directory"] = TEST_RESULTS

    # Save temporary test config
    with open(TEST_CONFIG, "w") as f:
        json.dump(config, f, indent=4)

    # 3. Clean any existing test directories
    if os.path.exists(TEST_RESULTS):
        shutil.rmtree(TEST_RESULTS)
    if os.path.exists(TEST_PLOTS):
        shutil.rmtree(TEST_PLOTS)

    # 4. Run the benchmark
    try:
        print("\n--> Running 10-step SNN benchmark training...")
        run_benchmark(TEST_CONFIG, nb_jobs=4)
        print("--> Training completed successfully!")
    except Exception as e:
        print(f"--> [FAIL] Training crashed: {e}")
        return
    finally:
        if os.path.exists(TEST_CONFIG):
            os.remove(TEST_CONFIG)

    # 5. Generate heatmaps
    try:
        print("\n--> Generating heatmaps...")
        os.makedirs(TEST_PLOTS, exist_ok=True)
        test_heatmap(TEST_RESULTS, TEST_PLOTS)
        loss_heatmap(TEST_RESULTS, TEST_PLOTS)
        aggregated_test_heatmap(TEST_RESULTS, TEST_PLOTS)
        
        print("\n--> [SUCCESS] Heatmaps successfully generated!")
        print("Generated files:")
        for fn in sorted(os.listdir(TEST_PLOTS)):
            print(f"  - {os.path.join(TEST_PLOTS, fn)}")
    except Exception as e:
        print(f"--> [FAIL] Heatmap generation crashed: {e}")

if __name__ == "__main__":
    test_pipeline()
