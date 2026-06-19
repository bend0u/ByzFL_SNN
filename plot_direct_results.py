import os
import json
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "./snn_benchmark_results_11/mnist_direct"
OUTPUT_DIR = "./snn_benchmark_results_11/plots_comparison"

ATTACKS = ["SignFlipping", "Optimal_InnerProductManipulation"]
DISTRIBUTIONS = ["iid", "dirichlet_niid_0.5"]
AGGREGATORS = ["Average", "TrMean"]
F_VALUES = [1, 2, 3, 4]

# Premium styling colors
COLORS = {
    "baseline": "#10b981",    # Emerald Green
    "Average": "#3b82f6",     # Blue
    "TrMean": "#ef4444",      # Red
}

def load_accuracy(folder_name):
    path = os.path.join(RESULTS_DIR, folder_name, "test_accuracy.txt")
    if not os.path.exists(path):
        return None
    try:
        data = np.loadtxt(path, delimiter=",")
        if data.ndim == 0:
            data = np.array([data])
        return data
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def plot_learning_curves(attack, dist):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    axes = axes.flatten()
    
    # Load baseline
    baseline_folder = f"NoAttack_Average_f_0_{dist}"
    baseline_acc = load_accuracy(baseline_folder)
    
    plotted_any = False
    
    for idx, f in enumerate(F_VALUES):
        ax = axes[idx]
        
        # Plot baseline
        if baseline_acc is not None:
            steps_base = np.arange(len(baseline_acc)) * 10
            ax.plot(steps_base, baseline_acc, label="Baseline (No Attack)", color=COLORS["baseline"], linewidth=2.5, linestyle="--")
            plotted_any = True
            
        # Plot each aggregator
        for agg in AGGREGATORS:
            run_folder = f"{attack}_{agg}_f_{f}_{dist}"
            acc = load_accuracy(run_folder)
            if acc is not None:
                steps = np.arange(len(acc)) * 10
                ax.plot(steps, acc, label=f"{agg} (f={f})", color=COLORS[agg], linewidth=2.0)
                plotted_any = True
                
        ax.set_title(f"Byzantine Clients f = {f}", fontweight="bold", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim(-0.02, 1.02)
        if idx >= 2:
            ax.set_xlabel("Communication Round")
        if idx % 2 == 0:
            ax.set_ylabel("Test Accuracy")
            
        # Place legend in the first subplot or a suitable one
        if idx == 0:
            ax.legend(loc="lower left", frameon=True, fontsize=9)
            
    if plotted_any:
        title_dist = "IID" if dist == "iid" else "Dirichlet NIID (0.5)"
        fig.suptitle(f"MNIST Direct coding - {attack} - {title_dist}\nLearning Curves Comparison", fontweight="bold", fontsize=14, y=0.98)
        plt.tight_layout()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, f"learning_curves_{attack}_{dist}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved learning curves comparison plot to {out_path}")
    else:
        plt.close()

def plot_accuracy_vs_f(attack, dist):
    # Load baseline
    baseline_folder = f"NoAttack_Average_f_0_{dist}"
    baseline_acc = load_accuracy(baseline_folder)
    baseline_final = baseline_acc[-1] if baseline_acc is not None else None
    
    fig, ax = plt.subplots(figsize=(8, 5.5))
    plotted_any = False
    
    for agg in AGGREGATORS:
        fs = [0]
        accs = [baseline_final] if baseline_final is not None else []
        
        for f in F_VALUES:
            run_folder = f"{attack}_{agg}_f_{f}_{dist}"
            acc = load_accuracy(run_folder)
            if acc is not None:
                fs.append(f)
                accs.append(acc[-1])
                
        # If we have data beyond baseline f=0
        if len(fs) > 1 and len(accs) == len(fs):
            ax.plot(fs, accs, label=agg, color=COLORS[agg], marker="o", linewidth=2.5, markersize=8)
            plotted_any = True
            
    if plotted_any:
        # Plot baseline point
        if baseline_final is not None:
            ax.plot([0], [baseline_final], color=COLORS["baseline"], marker="s", markersize=10, label="Baseline (No Attack)", linestyle="")
            
        title_dist = "IID" if dist == "iid" else "Dirichlet NIID (0.5)"
        ax.set_title(f"MNIST Direct coding - {attack} ({title_dist})\nFinal Test Accuracy vs. Number of Byzantine Clients (f)", fontweight="bold", pad=15)
        ax.set_xlabel("Number of Byzantine Clients (f)")
        ax.set_ylabel("Final Test Accuracy")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower left", frameon=True)
        
        plt.tight_layout()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, f"accuracy_vs_f_{attack}_{dist}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved accuracy vs f comparison plot to {out_path}")
    else:
        plt.close()

def main():
    print("==========================================================")
    print("GENERATING CUSTOM SWEET COMPARISON PLOTS (DIRECT)")
    print("==========================================================")
    
    if not os.path.isdir(RESULTS_DIR):
        print(f"Error: results directory '{RESULTS_DIR}' does not exist.")
        return
        
    for attack in ATTACKS:
        for dist in DISTRIBUTIONS:
            plot_learning_curves(attack, dist)
            plot_accuracy_vs_f(attack, dist)
            
    print("\nCustom comparisons generated successfully in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
