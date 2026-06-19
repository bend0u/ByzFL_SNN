import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import traceback
import torch

from byzfl import run_benchmark
from byzfl.benchmark.managers import ParamsManager
from byzfl.attacks.attacks import Gaussian
from byzfl.utils.misc import check_vectors_type, random_tool

# Monkey patch Gaussian call to ensure the generated noise is on the same device as the honest vectors
def patched_gaussian_call(self, honest_vectors):
    _, hv = check_vectors_type(honest_vectors)
    random = random_tool(hv)
    shape = hv.shape[1]
    res = random.normal(loc=self.mu, scale=self.sigma, size=shape)
    if isinstance(hv, torch.Tensor):
        res = res.to(hv.device)
    return res

Gaussian.__call__ = patched_gaussian_call

RESULTS_DIR = "./snn_benchmark_results_11"
PLOTS_DIR = "./snn_benchmark_plots_cc_krum_rate"

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

def make_sweep_configs(tmpl_path, aggregator_name, nb_steps, evaluation_delta):
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
    config_f0["aggregator"] = [{"name": aggregator_name, "parameters": {}}]
    
    # 3. Build attack sweeps config (f=3, 5)
    config_attacks = json.loads(json.dumps(tmpl))
    config_attacks["benchmark_config"]["f"] = [3, 5]
    config_attacks["attack"] = [
        {"name": "SignFlipping", "parameters": {}},
        {"name": "Optimal_InnerProductManipulation", "parameters": {}},
        {"name": "Optimal_ALittleIsEnough", "parameters": {}},
        {"name": "Gaussian", "parameters": {}},
        {"name": "Mimic", "parameters": {}}
    ]
    config_attacks["aggregator"] = [
        {"name": aggregator_name, "parameters": {}}
    ]
    
    return config_f0, config_attacks

def generate_comparison_plots(nb_steps, evaluation_delta):
    print("\n==========================================================")
    print("GENERATING SWEEP COMPARISON PLOTS (CenteredClipping vs Krum - RATE)")
    print("==========================================================")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    distributions = ["iid", "dirichlet_niid"]
    attacks = ["SignFlipping", "Optimal_InnerProductManipulation", "Optimal_ALittleIsEnough", "Gaussian", "Mimic"]
    aggregators = ["CenteredClipping", "Krum"]
    
    colors = {
        "SignFlipping": "#2563eb",                  # Blue
        "Optimal_InnerProductManipulation": "#dc2626", # Red
        "Optimal_ALittleIsEnough": "#f59e0b",          # Orange
        "Gaussian": "#10b981",                         # Green
        "Mimic": "#8b5cf6"                             # Purple
    }
    
    for dist in distributions:
        dist_param = 1.0 if dist == "iid" else 0.5
        
        # 1. Reconstruct all matching configurations to query their directories
        matching_runs = []
        
        # Baselines
        for agg in aggregators:
            baseline_run = {
                "benchmark_config": {
                    "f": 0,
                    "data_distribution": {"name": dist, "distribution_parameter": dist_param}
                },
                "model": {
                    "dataset_name": "mnist",
                    "encoding": {"type": "rate", "time_steps": 25}
                },
                "aggregator": {"name": agg},
                "pre_aggregators": [],
                "attack": {"name": "NoAttack"}
            }
            matching_runs.append(baseline_run)
            
        # Attacks f=3, 5
        for f in [3, 5]:
            for attack in attacks:
                for agg in aggregators:
                    run = {
                        "benchmark_config": {
                            "f": f,
                            "data_distribution": {"name": dist, "distribution_parameter": dist_param}
                        },
                        "model": {
                            "dataset_name": "mnist",
                            "encoding": {"type": "rate", "time_steps": 25}
                        },
                        "aggregator": {"name": agg},
                        "pre_aggregators": [],
                        "attack": {"name": attack}
                    }
                    matching_runs.append(run)
                    
        # Load Baseline Accuracy Values
        baseline_path_cc = get_run_path(matching_runs[0], RESULTS_DIR)
        baseline_accs_cc = load_accuracy(baseline_path_cc, "test_accuracy.txt")
        baseline_final_cc = baseline_accs_cc[-1] if baseline_accs_cc is not None else 0.0
        
        baseline_path_krum = get_run_path(matching_runs[1], RESULTS_DIR)
        baseline_accs_krum = load_accuracy(baseline_path_krum, "test_accuracy.txt")
        baseline_final_krum = baseline_accs_krum[-1] if baseline_accs_krum is not None else 0.0
        
        results = {}
        for attack in attacks:
            for agg in aggregators:
                results[f"{attack}_{agg}"] = []
                
        # Inject baselines at f=0
        for key in results:
            if "CenteredClipping" in key:
                results[key].append((0, baseline_final_cc))
            else:
                results[key].append((0, baseline_final_krum))
                
        # Load f=3 and f=5 final accuracies
        for f in [3, 5]:
            for attack in attacks:
                for agg in aggregators:
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
        fig, ax = plt.subplots(figsize=(10, 6.5))
        
        for key, val in results.items():
            val_sorted = sorted(val, key=lambda x: x[0])
            fs = [x[0] for x in val_sorted]
            accs = [x[1] for x in val_sorted]
            
            parts = key.split("_")
            agg = parts[-1]
            attack = "_".join(parts[:-1])
            
            linestyle = "-" if agg == "CenteredClipping" else "--"
            marker = "o" if agg == "CenteredClipping" else "x"
            c = colors.get(attack, "#4b5563")
            
            label = f"{attack} + {agg}"
            ax.plot(fs, accs, label=label, color=c, marker=marker, linestyle=linestyle, linewidth=2.0, markersize=7)
            
        title_dist = "IID" if dist == "iid" else "Dirichlet NIID (0.5)"
        ax.set_title(f"CenteredClipping vs Krum Sweep - MNIST Rate Encoding ({title_dist})", fontweight="bold", pad=15)
        ax.set_xlabel("Number of Byzantine Clients (f)")
        ax.set_ylabel("Final Test Accuracy")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks([0, 3, 5])
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower left", frameon=True, fontsize=9)
        
        filename = f"mnist_cc_vs_krum_{dist}_accuracy_vs_f.png"
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
        plt.close()
        print(f"Saved accuracy vs f plot: {filename}")
        
        # Plot 2: Dynamic learning curves at f=5
        fig, ax = plt.subplots(figsize=(11, 7))
        
        if baseline_accs_cc is not None:
            steps = [i * evaluation_delta for i in range(len(baseline_accs_cc))]
            steps = [min(s, nb_steps) for s in steps]
            ax.plot(steps, baseline_accs_cc, label="CC Baseline (f=0)", color="#10b981", linewidth=3, linestyle="--")
            
        if baseline_accs_krum is not None:
            steps = [i * evaluation_delta for i in range(len(baseline_accs_krum))]
            steps = [min(s, nb_steps) for s in steps]
            ax.plot(steps, baseline_accs_krum, label="Krum Baseline (f=0)", color="#6b7280", linewidth=3, linestyle=":")
            
        for attack in attacks:
            for agg in aggregators:
                run = [
                    c for c in matching_runs
                    if c["benchmark_config"]["f"] == 5
                    and c["attack"]["name"] == attack
                    and c["aggregator"]["name"] == agg
                ][0]
                
                path = get_run_path(run, RESULTS_DIR)
                accs = load_accuracy(path, "test_accuracy.txt")
                if accs is not None:
                    steps = [i * evaluation_delta for i in range(len(accs))]
                    steps = [min(s, nb_steps) for s in steps]
                    label = f"{attack} + {agg} (f=5)"
                    c = colors.get(attack, "#4b5563")
                    linestyle = "-" if agg == "CenteredClipping" else "--"
                    ax.plot(steps, accs, label=label, color=c, linestyle=linestyle, linewidth=2.0)
                    
        ax.set_title(f"CC vs Krum Learning Curves (f=5) - MNIST Rate Encoding ({title_dist})", fontweight="bold", pad=15)
        ax.set_xlabel("Communication Step")
        ax.set_ylabel("Test Accuracy")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right", frameon=True, fontsize=8)
        
        filename = f"mnist_cc_vs_krum_{dist}_learning_curves_f_5.png"
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
    parser = argparse.ArgumentParser(description="Run SNN Byzantine benchmark sweeps with CenteredClipping and Krum using Rate Encoding.")
    parser.add_argument("--test", action="store_true", help="Run a quick test of the pipeline with 2 steps.")
    args = parser.parse_args()
    
    nb_steps = 2 if args.test else 500
    evaluation_delta = 1 if args.test else 10
    
    # Generate configurations for CenteredClipping and Krum (Rate Encoding)
    cc_f0, cc_attacks = make_sweep_configs("snn_mnist_rate.json", "CenteredClipping", nb_steps, evaluation_delta)
    krum_f0, krum_attacks = make_sweep_configs("snn_mnist_rate.json", "Krum", nb_steps, evaluation_delta)
    
    configs_to_run = [
        ("snn_mnist_sweep_cc_f0_rate.json", cc_f0),
        ("snn_mnist_sweep_cc_attacks_rate.json", cc_attacks),
        ("snn_mnist_sweep_krum_f0_rate.json", krum_f0),
        ("snn_mnist_sweep_krum_attacks_rate.json", krum_attacks)
    ]
    
    for fn, cfg in configs_to_run:
        with open(fn, "w") as f:
            json.dump(cfg, f, indent=4)
            
    print("Starting Sweep execution for Rate Encoding (CenteredClipping vs Krum)...")
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
                
    # Generate comparison plots
    generate_comparison_plots(nb_steps, evaluation_delta)
    print("\nDone! Sweeps completed and comparison plots generated.")

if __name__ == "__main__":
    main()
