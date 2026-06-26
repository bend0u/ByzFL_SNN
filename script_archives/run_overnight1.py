import os
import json
import numpy as np
import matplotlib.pyplot as plt
from byzfl import run_benchmark
import traceback


# 1. List of configuration files to run sequentially
CONFIG_FILES = [
    #"snn_mnist_rate.json",
    #"snn_mnist_latency.json",
    #"snn_mnist_direct.json",
    "snn_nmnist.json",
    "snn_nmnist_1.json"
]

RESULTS_DIR = "./snn_benchmark_results"
PLOTS_DIR = "./snn_benchmark_plots"

def run_all_benchmarks():
    print("==========================================================")
    print("STARTING OVERNIGHT SNN BENCHMARK RUNS")
    print("==========================================================")
    for config_file in CONFIG_FILES:
        if not os.path.exists(config_file):
            print(f"Warning: Configuration file {config_file} not found. Skipping.")
            continue
        print(f"\n---> Running benchmark for: {config_file}")
        try:
            # We run sequentially (nb_jobs=1) to prevent CPU core/memory thrashing on CPU runs
            run_benchmark(config_file, nb_jobs=1)
            print(f"---> Successfully completed: {config_file}")
        except Exception as e:
            print(f"Error executing benchmark {config_file}: {e}")
            traceback.print_exc()

def parse_benchmark_results():
    """
    Scans the results directory, reads the config.json inside each run folder,
    and loads the corresponding test/validation accuracy txt files.
    """
    if not os.path.isdir(RESULTS_DIR):
        print(f"Results directory '{RESULTS_DIR}' does not exist.")
        return []

    runs_data = []
    # Scan subdirectories
    for item in os.listdir(RESULTS_DIR):
        item_path = os.path.join(RESULTS_DIR, item)
        if not os.path.isdir(item_path) or item == "best_hyperparameters":
            continue

        config_path = os.path.join(item_path, "config.json")
        if not os.path.exists(config_path):
            continue

        # Load run config
        with open(config_path, "r") as f:
            config = json.load(f)

        # Retrieve accuracy logs
        test_acc_file = None
        val_acc_file = None
        for f_name in os.listdir(item_path):
            if f_name.startswith("test_accuracy_") and f_name.endswith(".txt"):
                test_acc_file = os.path.join(item_path, f_name)
            elif f_name.startswith("val_accuracy_") and f_name.endswith(".txt"):
                val_acc_file = os.path.join(item_path, f_name)

        if not test_acc_file or not val_acc_file:
            continue

        try:
            test_accs = np.loadtxt(test_acc_file, delimiter=",")
            val_accs = np.loadtxt(val_acc_file, delimiter=",")
        except Exception as e:
            print(f"Error loading accuracy files from {item}: {e}")
            continue

        # Convert 0-d arrays to 1-d arrays
        if test_accs.ndim == 0:
            test_accs = np.array([test_accs])
        if val_accs.ndim == 0:
            val_accs = np.array([val_accs])

        # Extract parameters
        model_cfg = config.get("model", {})
        bench_cfg = config.get("benchmark_config", {})
        
        # Get encoding type
        encoding_info = model_cfg.get("encoding", {})
        encoding_type = encoding_info.get("type", "constant") if encoding_info else "none"
        if model_cfg.get("dataset_name", "").lower() == "nmnist":
            encoding_type = "neuromorphic"

        runs_data.append({
            "folder": item,
            "dataset": model_cfg.get("dataset_name", "mnist"),
            "model": model_cfg.get("name", "convnet_snn"),
            "encoding": encoding_type,
            "nb_clients": bench_cfg.get("nb_honest_clients", 1),
            "nb_steps": bench_cfg.get("nb_steps", 1500),
            "evaluation_delta": config.get("evaluation_and_results", {}).get("evaluation_delta", 375),
            "test_accuracies": test_accs,
            "val_accuracies": val_accs
        })
    
    return runs_data

def generate_plots(runs_data):
    if not runs_data:
        print("No runs data found to plot.")
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'figure.titlesize': 14})
    
    print("\n==========================================================")
    print("GENERATING ACCURACY PLOTS")
    print("==========================================================")

    # -------------------------------------------------------------
    # 1. Individual Plots for each experiment
    # -------------------------------------------------------------
    for run in runs_data:
        title = f"{run['dataset'].upper()} - {run['model']} ({run['encoding']} encoding)\nClients: {run['nb_clients']}"
        fig, ax = plt.subplots(figsize=(8, 5))
        
        steps = [i * run['evaluation_delta'] for i in range(len(run['test_accuracies']))]
        # cap steps at nb_steps if evaluation output contains final step beyond boundary
        steps = [min(s, run['nb_steps']) for s in steps]
        
        ax.plot(steps, run['val_accuracies'], label="Validation Accuracy", color="#3b82f6", linewidth=2.5, marker="o")
        ax.plot(steps, run['test_accuracies'], label="Test Accuracy", color="#10b981", linewidth=2.5, marker="s")
        
        ax.set_title(title, fontweight="bold", pad=15)
        ax.set_xlabel("Communication Step")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="lower right")
        
        filename = f"run_{run['dataset']}_{run['encoding']}_clients_{run['nb_clients']}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
        plt.close()
        print(f"Saved individual plot: {filename}")

    # -------------------------------------------------------------
    # 2. Comparison Plots: 1 vs 16 Clients for each encoding
    # -------------------------------------------------------------
    # Group by (dataset, encoding)
    grouped = {}
    for run in runs_data:
        key = (run['dataset'], run['encoding'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(run)

    for (dataset, encoding), runs in grouped.items():
        if len(runs) < 2:
            continue
        fig, ax = plt.subplots(figsize=(9, 5.5))
        
        # Sort so client=1 is plotted first, then 16
        runs_sorted = sorted(runs, key=lambda r: r['nb_clients'])
        colors = {1: "#2563eb", 16: "#dc2626"}
        markers = {1: "o", 16: "^"}
        
        for run in runs_sorted:
            steps = [i * run['evaluation_delta'] for i in range(len(run['test_accuracies']))]
            steps = [min(s, run['nb_steps']) for s in steps]
            label = f"{run['nb_clients']} Client(s) (Test Acc)"
            
            c = colors.get(run['nb_clients'], "#4b5563")
            m = markers.get(run['nb_clients'], "o")
            
            ax.plot(steps, run['test_accuracies'], label=label, color=c, linewidth=2.5, marker=m)

        ax.set_title(f"Client Scaling Comparison - {dataset.upper()} ({encoding} encoding)", fontweight="bold", pad=15)
        ax.set_xlabel("Communication Step")
        ax.set_ylabel("Test Accuracy")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="lower right")
        
        filename = f"compare_1_vs_16_{dataset}_{encoding}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
        plt.close()
        print(f"Saved 1-vs-16 comparison plot: {filename}")

    # -------------------------------------------------------------
    # 3. Combined Comparison of all 16-client runs
    # -------------------------------------------------------------
    runs_16 = [run for run in runs_data if run['nb_clients'] == 16]
    if runs_16:
        fig, ax = plt.subplots(figsize=(10, 6.5))
        
        # Define rich distinct color scheme
        color_scheme = {
            "constant": "#10b981",      # Emerald Green
            "rate": "#3b82f6",          # Blue
            "latency": "#f59e0b",       # Amber Orange
            "neuromorphic": "#ec4899"   # Pink
        }
        
        for run in runs_16:
            steps = [i * run['evaluation_delta'] for i in range(len(run['test_accuracies']))]
            steps = [min(s, run['nb_steps']) for s in steps]
            label = f"{run['dataset'].upper()} - {run['encoding']} coding"
            c = color_scheme.get(run['encoding'], "#4b5563")
            
            ax.plot(steps, run['test_accuracies'], label=label, color=c, linewidth=2.5, marker="d")

        ax.set_title("SNN Multi-Client (16 Clients) Performance Comparison", fontweight="bold", pad=15)
        ax.set_xlabel("Communication Step")
        ax.set_ylabel("Test Accuracy")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="lower right")
        
        filename = "compare_all_16_clients.png"
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
        plt.close()
        print(f"Saved all 16-client comparison plot: {filename}")

def main():
    # Run the benchmarks
    run_all_benchmarks()
    
    # Parse results and plot
    runs_data = parse_benchmark_results()
    generate_plots(runs_data)
    print("\nOvernight runs and plotting completed successfully!")

if __name__ == "__main__":
    main()
