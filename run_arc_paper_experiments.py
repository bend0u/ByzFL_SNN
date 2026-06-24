import os
import sys
import json
import argparse
import traceback
import multiprocessing
import numpy as np
import torch

# Ensure byzfl is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from byzfl.benchmark.train import start_training
from byzfl.benchmark.managers import ParamsManager

def run_training_job(params):
    """
    Runner wrapper for each training configuration to execute in the pool.
    """
    try:
        agg_name = params["aggregator"]["name"]
        pre_aggs = [p["name"] for p in params["pre_aggregators"]]
        pre_aggs_str = "+".join(pre_aggs) if pre_aggs else "None"
        f = params["benchmark_config"]["f"]
        attack_name = params["attack"]["name"]
        delta = params["attack"]["parameters"].get("delta", 10.0)
        mode = "Corrected" if delta < 0 else "Default"
        
        print(f"[RUNNING] Mode: {mode} ALIE | Aggregator: {agg_name} | Pre-Aggs: {pre_aggs_str} | f: {f} on {params['benchmark_config']['device']}")
        
        start_training(params)
        
        print(f"[COMPLETED] Mode: {mode} ALIE | Aggregator: {agg_name} | Pre-Aggs: {pre_aggs_str} | f: {f}")
    except Exception as e:
        print(f"[ERROR] Failed configuration: {e}")
        traceback.print_exc()

def make_config_list(nb_steps, eval_delta):
    configs = []
    
    # Check GPU availability
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    # Attack modes: Default ALIE (delta = 10) vs Corrected ALIE (delta = -10)
    modes = [
        ("default_alie", 10.0, "./arc_mnist_results_default_alie"),
        ("corrected_alie", -10.0, "./arc_mnist_results_corrected_alie")
    ]
    
    # Aggregators to sweep
    aggregators = [
        {"name": "TrMean", "parameters": {}},
        {"name": "GeometricMedian", "parameters": {"nu": 0.1, "T": 3}}
    ]
    
    # Pre-aggregators to sweep
    pre_agg_settings = [
        "NNM_only",
        "NNM_ARC"
    ]
    
    # Sweep f from 1 to 5 (Honest workers n=10, total workers = 10+f)
    f_values = [1, 2, 3, 4, 5]
    
    idx = 0
    for mode_name, delta_val, result_dir in modes:
        for agg in aggregators:
            for pre_agg_set in pre_agg_settings:
                for f in f_values:
                    # Target GPU
                    device = f"cuda:{idx % num_gpus}" if torch.cuda.is_available() else "cpu"
                    idx += 1
                    
                    # Base configuration
                    config = {
                        "benchmark_config": {
                            "device": device,
                            "training_seed": 42,
                            "nb_training_seeds": 1,
                            "nb_honest_clients": 10,
                            "f": f,
                            "tolerated_f": f,
                            "size_train_set": 0.8,
                            "data_distribution_seed": 42,
                            "nb_data_distribution_seeds": 1,
                            "data_distribution": {
                                "name": "extreme_niid",
                                "distribution_parameter": 0.0
                            },
                            "training_algorithm": {
                                "name": "DSGD",
                                "parameters": {}
                            },
                            "nb_steps": nb_steps
                        },
                        "model": {
                            "name": "cnn_mnist",
                            "dataset_name": "mnist",
                            "nb_labels": 10,
                            "loss": "NLLLoss",
                            "learning_rate": 0.1,
                            "learning_rate_decay": 1.0,
                            "milestones": []
                        },
                        "honest_clients": {
                            "momentum": 0.9,
                            "weight_decay": 0.0001,
                            "batch_size": 25
                        },
                        "evaluation_and_results": {
                            "evaluation_delta": eval_delta,
                            "batch_size_evaluation": 128,
                            "evaluate_on_test": True,
                            "store_per_client_metrics": True,
                            "store_models": False,
                            "data_folder": "./data",
                            "results_directory": result_dir,
                            "clean_directory_structure": True
                        }
                    }
                    
                    # Set aggregator and its parameters
                    agg_config = {"name": agg["name"], "parameters": agg["parameters"].copy()}
                    if agg["name"] == "TrMean":
                        agg_config["parameters"]["f"] = f
                    config["aggregator"] = agg_config
                    
                    # Set pre-aggregators
                    if pre_agg_set == "NNM_only":
                        config["pre_aggregators"] = [
                            {"name": "NNM", "parameters": {"f": f}}
                        ]
                    else:
                        config["pre_aggregators"] = [
                            {"name": "NNM", "parameters": {"f": f}},
                            {"name": "ARC", "parameters": {"f": f}}
                        ]
                        
                    # Set attack
                    config["attack"] = {
                        "name": "Optimal_ALittleIsEnough",
                        "parameters": {
                            "delta": delta_val
                        }
                    }
                    
                    configs.append(config)
                    
    return configs

def get_folder_path(result_dir, attack_name, pre_aggs, agg, f):
    pre_aggs_str = "_".join(pre_aggs)
    preaggs_aggregator = f"{pre_aggs_str}_{agg}"
    folder_name = f"{attack_name}_{preaggs_aggregator}_f_{f}_extreme_niid"
    return os.path.join(result_dir, "mnist_direct", folder_name)

def generate_report(nb_steps):
    print("\n" + "=" * 80)
    print("MNIST ARC SWEEP REPORT")
    print("=" * 80)
    
    modes = [
        ("Default ALIE (delta = 10.0)", "./arc_mnist_results_default_alie"),
        ("Corrected ALIE (delta = -10.0)", "./arc_mnist_results_corrected_alie")
    ]
    
    f_values = [1, 2, 3, 4, 5]
    
    for mode_title, result_dir in modes:
        print(f"\n>>> {mode_title} <<<")
        print(f"{'Aggregator':<16} | {'Pre-Agg':<12} | " + " | ".join([f"f={f:<7}" for f in f_values]))
        print("-" * 90)
        
        for agg in ["TrMean", "GeometricMedian"]:
            for pre_agg_label, pre_aggs in [("NNM", ["NNM"]), ("NNM+ARC", ["NNM", "ARC"])]:
                acc_list = []
                peak_list = []
                for f in f_values:
                    path = get_folder_path(result_dir, "Optimal_ALittleIsEnough", pre_aggs, agg, f)
                    acc_file = os.path.join(path, "test_accuracy.txt")
                    if os.path.exists(acc_file):
                        try:
                            data = np.loadtxt(acc_file, delimiter=",")
                            if data.ndim == 0:
                                acc_list.append(f"{data:.4f}")
                                peak_list.append(f"{data:.4f}")
                            else:
                                acc_list.append(f"{data[-1]:.4f}")
                                peak_list.append(f"{np.max(data):.4f}")
                        except Exception:
                            acc_list.append("Error ")
                            peak_list.append("Error ")
                    else:
                        acc_list.append("N/A   ")
                        peak_list.append("N/A   ")
                
                print(f"{agg:<16} | {pre_agg_label:<12} | " + " | ".join([f"{a} (Pk:{p})" for a, p in zip(acc_list, peak_list)]))

def main():
    parser = argparse.ArgumentParser(description="Sweep configurations from ARC (ICLR 2025) paper under Default and Corrected ALIE.")
    parser.add_argument("--test", action="store_true", help="Run a fast test with 2 steps to verify functionality.")
    parser.add_argument("--nb_jobs", type=int, default=4, help="Number of parallel training processes.")
    args = parser.parse_args()
    
    nb_steps = 2 if args.test else 800
    eval_delta = 1 if args.test else 50
    
    configs = make_config_list(nb_steps, eval_delta)
    
    print(f"Total configurations generated: {len(configs)}")
    print(f"Running sweep with processes={args.nb_jobs}...")
    
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=args.nb_jobs) as pool:
        pool.map(run_training_job, configs)
        
    print("\nSweep execution completed!")
    generate_report(nb_steps)

if __name__ == "__main__":
    main()
