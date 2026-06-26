import os
import sys
import json
import argparse
import time
from byzfl import run_benchmark

def main():
    parser = argparse.ArgumentParser(description="Train SNN on MNIST with Latency Encoding.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use for training (cuda or cpu).")
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "snn_mnist_latency.json")
    
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        sys.exit(1)
        
    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)
        
    # Update device based on argument
    config["benchmark_config"]["device"] = args.device
    
    # Save a temporary config file for execution
    temp_config_path = os.path.join(script_dir, "temp_snn_mnist_latency_run.json")
    with open(temp_config_path, "w") as f:
        json.dump(config, f, indent=4)
        
    print(f"Starting SNN latency encoding training on device: {args.device}...")
    start_time = time.time()
    try:
        run_benchmark(temp_config_path, nb_jobs=1)
        elapsed_time = time.time() - start_time
        mins, secs = divmod(elapsed_time, 60)
        print("Training completed successfully!")
        print(f"Total execution time: {int(mins)}m {secs:.2f}s ({elapsed_time:.2f} seconds)")
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"Error during benchmark run after {elapsed_time:.2f} seconds: {e}")
    finally:
        # Clean up temporary config file
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

if __name__ == "__main__":
    main()
