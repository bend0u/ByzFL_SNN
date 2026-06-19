import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import traceback
import torch

from byzfl import run_benchmark
from byzfl.benchmark.managers import ParamsManager

RESULTS_DIR = "./snn_benchmark_results_11"
PLOTS_DIR = "./snn_benchmark_plots_11"

def get_run_path(setting, results_dir):
    pm = ParamsManager(setting)
    encoding = pm.get_encoding_type()
    enc_name = "direct" if encoding == "constant" else encoding
    parent_dir = f"{setting['model']['dataset_name']}_{enc_name}"
    
    pre_aggregation_names = [
        agg['name'] for agg in setting["pre_aggregators"]
    ]
    preaggs_aggregator = '_'.join(pre_aggregation_names + [setting['aggregator']['name']])
    
    dist_name = setting['benchmark_config']['data_distribution']['name']
    dist_param = setting['benchmark_config']['data_distribution']['distribution_parameter']
    if dist_name in ["iid", "extreme_niid"]:
        dist_part = dist_name
    else:
        dist_part = f"{dist_name}_{dist_param}"

    folder_name = (
        f"{setting['attack']['name']}_"
        f"{preaggs_aggregator}_"
        f"f_{setting['benchmark_config']['f']}_"
        f"{dist_part}"
    )
    return os.path.join(results_dir, parent_dir, folder_name)

def make_sweep_configs(tmpl_path, nb_steps, evaluation_delta):
    # Load user's default template config
    with open(tmpl_path, "r") as f:
        tmpl = json.load(f)
        
    device = "cuda"
    
    # 1. Update basic params
    tmpl["benchmark_config"]["device"] = device
    tmpl["benchmark_config"]["nb_steps"] = nb_steps
    tmpl["benchmark_config"]["nb_honest_clients"] = 16
    tmpl["benchmark_config"]["training_seed"] = 42
    tmpl["benchmark_config"]["data_distribution_seed"] = 42
    tmpl["benchmark_config"]["nb_training_seeds"] = 1
    tmpl["benchmark_config"]["nb_data_distribution_seeds"] = 1
    tmpl["benchmark_config"]["data_distribution"] = [
        {"name": "iid", "distribution_parameter": 1.0},
        {"name": "dirichlet_niid", "distribution_parameter": 0.5}
    ]
    tmpl["evaluation_and_results"]["evaluation_delta"] = evaluation_delta
    tmpl["evaluation_and_results"]["batch_size_evaluation"] = 128
    tmpl["evaluation_and_results"]["results_directory"] = RESULTS_DIR
    tmpl["evaluation_and_results"]["clean_directory_structure"] = True
    
    # 2. Build baseline config (f=0)
    config_f0 = json.loads(json.dumps(tmpl))
    config_f0["benchmark_config"]["f"] = [0]
    config_f0["attack"] = [{"name": "NoAttack", "parameters": {}}]
    config_f0["aggregator"] = [{"name": "Average", "parameters": {}}]
    
    # 3. Build attack sweeps config (f=1..4)
    config_attacks = json.loads(json.dumps(tmpl))
    config_attacks["benchmark_config"]["f"] = [1, 2, 3, 4]
    config_attacks["attack"] = [
        {"name": "SignFlipping", "parameters": {}},
        {"name": "Optimal_InnerProductManipulation", "parameters": {}}
    ]
    config_attacks["aggregator"] = [
        {"name": "Average", "parameters": {}},
        {"name": "TrMean", "parameters": {}}
    ]
    
    return config_f0, config_attacks

def generate_comparison_plots(nb_steps, evaluation_delta):
    print("\n==========================================================")
    print("GENERATING SWEEP COMPARISON PLOTS")
    print("==========================================================")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    encodings = ["direct", "rate"]
    distributions = ["iid", "dirichlet_niid"]
    
    # Reconstruct the list of combinations we expect to parse
    for enc in encodings:
        enc_type = "constant" if enc == "direct" else "rate"
        time_steps = 10 if enc == "direct" else 25
        for dist in distributions:
            dist_param = 1.0 if dist == "iid" else 0.5
            
            matching_runs = []
            
            # 1. Baseline (f=0)
            baseline_run = {
                "benchmark_config": {
                    "f": 0,
                    "data_distribution": {"name": dist, "distribution_parameter": dist_param}
                },
                "model": {
                    "dataset_name": "mnist",
                    "encoding": {"type": enc_type, "time_steps": time_steps}
                },
                "aggregator": {"name": "Average"},
                "pre_aggregators": [],
                "attack": {"name": "NoAttack"}
            }
            matching_runs.append(baseline_run)
            
            # 2. Attacks f=1..4
            for f in [1, 2, 3, 4]:
                for attack in ["SignFlipping", "Optimal_InnerProductManipulation"]:
                    for agg in ["Average", "TrMean"]:
                        run = {
                            "benchmark_config": {
                                "f": f,
                                "data_distribution": {"name": dist, "distribution_parameter": dist_param}
                            },
                            "model": {
                                "dataset_name": "mnist",
                                "encoding": {"type": enc_type, "time_steps": time_steps}
                            },
                            "aggregator": {"name": agg},
                            "pre_aggregators": [],
                            "attack": {"name": attack}
                        }
                        matching_runs.append(run)
            
            f_values = [0, 1, 2, 3, 4]
            results = {
                "SignFlipping_Average": [],
                "SignFlipping_TrMean": [],
                "Optimal_InnerProductManipulation_Average": [],
                "Optimal_InnerProductManipulation_TrMean": []
            }
            
            # Load baseline (f=0) accuracy
            baseline_path = get_run_path(baseline_run, RESULTS_DIR)
            baseline_accs = load_accuracy(baseline_path, "test_accuracy.txt")
            baseline_final = baseline_accs[-1] if baseline_accs is not None else 0.0
            
            for key in results:
                results[key].append((0, baseline_final))
                
            for f in [1, 2, 3, 4]:
                for attack in ["SignFlipping", "Optimal_InnerProductManipulation"]:
                    for agg in ["Average", "TrMean"]:
                        run = [
                            c for c in matching_runs
                            if c["benchmark_config"]["f"] == f
                            and c["attack"]["name"] == attack
                            and c["aggregator"]["name"] == agg
                        ][0]
                        
                        path = get_run_path(run, RESULTS_DIR)
                        accs = load_accuracy(path, "test_accuracy.txt")
                        final_acc = accs[-1] if accs is not None else 0.0
                        results[f"{attack}_{agg}"].append((f, final_acc))
            
            # Plot 1: Final test accuracy vs. f
            fig, ax = plt.subplots(figsize=(8, 5.5))
            
            for key, val in results.items():
                val_sorted = sorted(val, key=lambda x: x[0])
                fs = [x[0] for x in val_sorted]
                accs = [x[1] for x in val_sorted]
                
                label = key.replace("_", " + ")
                if "SignFlipping" in key:
                    color = "#dc2626" if "TrMean" in key else "#f87171"
                    marker = "o" if "TrMean" in key else "x"
                else:
                    color = "#2563eb" if "TrMean" in key else "#60a5fa"
                    marker = "^" if "TrMean" in key else "v"
                
                ax.plot(fs, accs, label=label, color=color, marker=marker, linewidth=2.5, markersize=8)
                
            ax.set_title(f"Robustness Sweep - MNIST {enc.capitalize()} coding ({dist.upper()})", fontweight="bold", pad=15)
            ax.set_xlabel("Number of Byzantine Clients (f)")
            ax.set_ylabel("Final Test Accuracy")
            ax.set_ylim(-0.02, 1.02)
            ax.set_xticks(f_values)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="lower left", frameon=True)
            
            filename = f"mnist_{enc}_{dist}_accuracy_vs_f.png"
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
            plt.close()
            print(f"Saved accuracy vs f plot: {filename}")
            
            # Plot 2: Dynamic training curves for f=4
            fig, ax = plt.subplots(figsize=(9, 6))
            
            if baseline_accs is not None:
                steps = [i * evaluation_delta for i in range(len(baseline_accs))]
                steps = [min(s, nb_steps) for s in steps]
                ax.plot(steps, baseline_accs, label="Baseline (f=0)", color="#10b981", linewidth=3, linestyle="--")
                
            for attack in ["SignFlipping", "Optimal_InnerProductManipulation"]:
                for agg in ["Average", "TrMean"]:
                    run = [
                        c for c in matching_runs
                        if c["benchmark_config"]["f"] == 4
                        and c["attack"]["name"] == attack
                        and c["aggregator"]["name"] == agg
                    ][0]
                    
                    path = get_run_path(run, RESULTS_DIR)
                    accs = load_accuracy(path, "test_accuracy.txt")
                    if accs is not None:
                        steps = [i * evaluation_delta for i in range(len(accs))]
                        steps = [min(s, nb_steps) for s in steps]
                        label = f"{attack} + {agg}"
                        if "SignFlipping" in label:
                            color = "#dc2626" if "TrMean" in label else "#f87171"
                        else:
                            color = "#2563eb" if "TrMean" in label else "#60a5fa"
                        ax.plot(steps, accs, label=label, color=color, linewidth=2)
            
            ax.set_title(f"Training Curves (f=4) - MNIST {enc.capitalize()} coding ({dist.upper()})", fontweight="bold", pad=15)
            ax.set_xlabel("Communication Step")
            ax.set_ylabel("Test Accuracy")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="lower right", frameon=True)
            
            filename = f"mnist_{enc}_{dist}_learning_curves_f_4.png"
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
            plt.close()
            print(f"Saved learning curves plot: {filename}")

def load_accuracy(path, filename):
    file_path = os.path.join(path, filename)
    if not os.path.exists(file_path):
        return None
    try:
        data = np.loadtxt(file_path, delimiter=",")
        if data.ndim == 0:
            data = np.array([data])
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run SNN Byzantine benchmark sweeps.")
    parser.add_argument("--test", action="store_true", help="Run a quick test of the pipeline with 2 steps.")
    args = parser.parse_args()
    
    nb_steps = 2 if args.test else 500
    evaluation_delta = 1 if args.test else 10
    
    # Generate the config dictionaries using the template files as bases
    direct_f0, direct_attacks = make_sweep_configs("snn_mnist_direct.json", nb_steps, evaluation_delta)
    rate_f0, rate_attacks = make_sweep_configs("snn_mnist_rate.json", nb_steps, evaluation_delta)
    
    # Save the 4 config files temporarily
    configs_to_run = [
        ("snn_mnist_sweep_direct_f0.json", direct_f0),
        ("snn_mnist_sweep_direct_attacks.json", direct_attacks),
        ("snn_mnist_sweep_rate_f0.json", rate_f0),
        ("snn_mnist_sweep_rate_attacks.json", rate_attacks)
    ]
    
    for fn, cfg in configs_to_run:
        with open(fn, "w") as f:
            json.dump(cfg, f, indent=4)
            
    print("Starting Sweep execution...")
    try:
        for fn, _ in configs_to_run:
            print(f"\n--- Running Config: {fn} ---")
            run_benchmark(fn, nb_jobs=1)
    except Exception as e:
        print(f"Error executing run_benchmark: {e}")
        traceback.print_exc()
    finally:
        # Clean up temporary JSON files
        for fn, _ in configs_to_run:
            if os.path.exists(fn):
                os.remove(fn)
                
    # Generate plots
    generate_comparison_plots(nb_steps, evaluation_delta)
    print("\nDone! Sweeps completed and comparison plots generated.")

if __name__ == "__main__":
    main()
