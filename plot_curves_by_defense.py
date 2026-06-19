import os
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "./snn_benchmark_results_11/mnist_direct"
OUTPUT_DIR = "./snn_benchmark_results_11/plots_comparison"

ATTACKS = ["SignFlipping", "Optimal_InnerProductManipulation"]
DISTRIBUTIONS = ["iid", "dirichlet_niid_0.5"]
DEFENSES = ["Average", "TrMean"]
F_VALUES = [1, 2, 3, 4]

# Define color palettes
PALETTES = {
    "Average": {
        0: "#10b981", # Emerald Green for baseline
        1: "#93c5fd", # Light blue
        2: "#60a5fa", # Medium blue
        3: "#2563eb", # Strong blue
        4: "#1e3a8a"  # Dark blue
    },
    "TrMean": {
        0: "#10b981", # Emerald Green for baseline
        1: "#fca5a5", # Light red
        2: "#f87171", # Medium red
        3: "#dc2626", # Strong red
        4: "#7f1d1d"  # Dark red
    }
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

def plot_curves(attack, dist, defense):
    plt.rcParams.update({'font.size': 11})
    fig, ax = plt.subplots(figsize=(9, 6.5))
    
    # Load baseline f=0
    baseline_folder = f"NoAttack_Average_f_0_{dist}"
    baseline_acc = load_accuracy(baseline_folder)
    
    palette = PALETTES[defense]
    plotted_any = False
    
    # 1. Plot baseline f=0
    if baseline_acc is not None:
        steps_base = np.arange(len(baseline_acc)) * 10
        ax.plot(steps_base, baseline_acc, label="Baseline (No Attack, f=0)", 
                color=palette[0], linewidth=3.0, linestyle="--")
        plotted_any = True
        
    # 2. Plot f = 1..4 curves
    for f in F_VALUES:
        run_folder = f"{attack}_{defense}_f_{f}_{dist}"
        acc = load_accuracy(run_folder)
        if acc is not None:
            steps = np.arange(len(acc)) * 10
            ax.plot(steps, acc, label=f"Attack f={f}", 
                    color=palette[f], linewidth=2.0)
            plotted_any = True
            
    if plotted_any:
        title_dist = "IID" if dist == "iid" else "Dirichlet NIID (0.5)"
        title = (
            f"MNIST Direct - {attack} ({title_dist})\n"
            f"Defense: {defense} Aggregator"
        )
        ax.set_title(title, fontweight="bold", pad=15, fontsize=13)
        ax.set_xlabel("Communication Round")
        ax.set_ylabel("Test Accuracy")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right", frameon=True)
        
        plt.tight_layout()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_name = f"curves_{attack}_{dist}_{defense}.png"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_name}")
    else:
        plt.close()

def main():
    print("==========================================================")
    print("GENERATING ACCURACY CURVES BY DEFENSE (DIRECT)")
    print("==========================================================")
    
    if not os.path.isdir(RESULTS_DIR):
        print(f"Error: results directory '{RESULTS_DIR}' does not exist.")
        return
        
    for attack in ATTACKS:
        for dist in DISTRIBUTIONS:
            for defense in DEFENSES:
                plot_curves(attack, dist, defense)
                
    print("\nAll plots generated successfully in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
