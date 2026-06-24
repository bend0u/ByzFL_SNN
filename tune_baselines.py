import os
import json
import shutil
import multiprocessing
import numpy as np
import torch
from byzfl.benchmark.train import start_training

RESULTS_DIR = "./tune_baseline_results"

def clean_results():
    if os.path.exists(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)

def run_job(params):
    try:
        model_name = params["model"]["name"]
        lr = params["model"]["learning_rate"]
        steps = params["benchmark_config"]["nb_steps"]
        print(f"[START] Model: {model_name} | LR: {lr} | Steps: {steps}")
        start_training(params)
        print(f"[COMPLETE] Model: {model_name} | LR: {lr} | Steps: {steps}")
    except Exception as e:
        print(f"[ERROR] Model: {params['model']['name']} | LR: {params['model']['learning_rate']} | Steps: {params['benchmark_config']['nb_steps']} - Error: {e}")

def make_configs():
    configs = []
    lrs = [0.01, 0.05, 0.1]
    steps_list = [500, 1000, 1500]
    
    device = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cuda:0"
    
    # 1. CNN Configs
    for lr in lrs:
        for steps in steps_list:
            config = {
                "benchmark_config": {
                    "device": device,
                    "training_seed": 42,
                    "nb_training_seeds": 1,
                    "nb_honest_clients": 10,
                    "f": 0,
                    "tolerated_f": 0,
                    "size_train_set": 0.8,
                    "data_distribution_seed": 42,
                    "nb_data_distribution_seeds": 1,
                    "data_distribution": {
                        "name": "iid",
                        "distribution_parameter": None
                    },
                    "training_algorithm": {
                        "name": "DSGD",
                        "parameters": {}
                    },
                    "nb_steps": steps
                },
                "model": {
                    "name": "cnn_mnist",
                    "is_snn": False,
                    "dataset_name": "mnist",
                    "nb_labels": 10,
                    "loss": "NLLLoss",
                    "learning_rate": lr,
                    "learning_rate_decay": 1.0,
                    "milestones": []
                },
                "aggregator": {
                    "name": "TrMean",
                    "parameters": {"f": 0}
                },
                "pre_aggregators": [
                    {"name": "NNM", "parameters": {"f": 0}},
                    {"name": "ARC", "parameters": {"f": 0}}
                ],
                "honest_clients": {
                    "momentum": 0.9,
                    "weight_decay": 0.0001,
                    "batch_size": 128
                },
                "attack": {
                    "name": "SignFlipping",
                    "parameters": {}
                },
                "evaluation_and_results": {
                    "evaluation_delta": 50,
                    "batch_size_evaluation": 128,
                    "evaluate_on_test": True,
                    "store_per_client_metrics": True,
                    "store_models": False,
                    "data_folder": "./data",
                    "results_directory": RESULTS_DIR
                }
            }
            configs.append(config)
            
    # 2. SNN Configs
    for lr in lrs:
        for steps in steps_list:
            config = {
                "benchmark_config": {
                    "device": device,
                    "training_seed": 42,
                    "nb_training_seeds": 1,
                    "nb_honest_clients": 10,
                    "f": 0,
                    "tolerated_f": 0,
                    "size_train_set": 0.8,
                    "data_distribution_seed": 42,
                    "nb_data_distribution_seeds": 1,
                    "data_distribution": {
                        "name": "iid",
                        "distribution_parameter": None
                    },
                    "training_algorithm": {
                        "name": "DSGD",
                        "parameters": {}
                    },
                    "nb_steps": steps
                },
                "model": {
                    "name": "cnn_mnist_snn",
                    "is_snn": True,
                    "dataset_name": "mnist",
                    "nb_labels": 10,
                    "loss": "ce_rate_loss",
                    "accuracy_name": "accuracy_rate",
                    "optimizer_name": "SGD",
                    "encoding": {
                        "type": "constant",
                        "time_steps": 25
                    },
                    "learning_rate": lr,
                    "learning_rate_decay": 1.0,
                    "milestones": [],
                    "model_params": {
                        "beta": 0.95,
                        "surrogate_gradient": "atan",
                        "threshold": 1.0,
                        "learn_threshold": False
                    }
                },
                "aggregator": {
                    "name": "TrMean",
                    "parameters": {"f": 0}
                },
                "pre_aggregators": [
                    {"name": "NNM", "parameters": {"f": 0}},
                    {"name": "ARC", "parameters": {"f": 0}}
                ],
                "honest_clients": {
                    "momentum": 0.9,
                    "weight_decay": 0.0001,
                    "batch_size": 128
                },
                "attack": {
                    "name": "SignFlipping",
                    "parameters": {}
                },
                "evaluation_and_results": {
                    "evaluation_delta": 50,
                    "batch_size_evaluation": 128,
                    "evaluate_on_test": True,
                    "store_per_client_metrics": True,
                    "store_models": False,
                    "data_folder": "./data",
                    "results_directory": RESULTS_DIR
                }
            }
            configs.append(config)
            
    return configs

def report_results(configs):
    print("\n" + "=" * 60)
    print("BASELINE TUNING SUMMARY REPORT")
    print("=" * 60)
    print(f"{'Model':<16} | {'LR':<6} | {'Steps':<6} | {'Max Test Acc':<14} | {'Final Test Acc':<14}")
    print("-" * 65)
    
    for cfg in configs:
        model_name = cfg["model"]["name"]
        lr = cfg["model"]["learning_rate"]
        steps = cfg["benchmark_config"]["nb_steps"]
        
        # Build path to find folder
        # We can find directories inside RESULTS_DIR that contain the model name and lr
        folder_found = None
        for dir_name in os.listdir(RESULTS_DIR):
            if model_name in dir_name and f"lr_{lr}" in dir_name:
                # Need to check if it matches steps
                # Read config.json inside dir_name to verify steps
                config_path = os.path.join(RESULTS_DIR, dir_name, "config.json")
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        saved_cfg = json.load(f)
                        if saved_cfg["benchmark_config"]["nb_steps"] == steps:
                            folder_found = dir_name
                            break
                            
        if folder_found:
            acc_path = os.path.join(RESULTS_DIR, folder_found, "test_accuracy_tr_seed_42_dd_seed_42.txt")
            if os.path.exists(acc_path):
                try:
                    data = np.loadtxt(acc_path, delimiter=",")
                    if data.ndim == 0:
                        max_acc = data.item()
                        final_acc = data.item()
                    else:
                        max_acc = np.max(data)
                        final_acc = data[-1]
                    print(f"{model_name:<16} | {lr:<6} | {steps:<6} | {max_acc:.4%}(step {np.argmax(data)*50})      | {final_acc:.4%}")
                except Exception as e:
                    print(f"{model_name:<16} | {lr:<6} | {steps:<6} | Error loading file: {e}")
            else:
                print(f"{model_name:<16} | {lr:<6} | {steps:<6} | File not found")
        else:
            print(f"{model_name:<16} | {lr:<6} | {steps:<6} | Folder not found")

def main():
    clean_results()
    configs = make_configs()
    print(f"Generated {len(configs)} configuration jobs.")
    
    # Run in parallel using multiprocessing (4 jobs)
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=4) as pool:
        pool.map(run_job, configs)
        
    print("\nTuning completed!")
    report_results(configs)

if __name__ == "__main__":
    main()
