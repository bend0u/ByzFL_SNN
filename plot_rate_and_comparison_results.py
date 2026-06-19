import os
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR_DIRECT = os.path.join(script_dir, "snn_benchmark_results_11/mnist_direct")
RESULTS_DIR_RATE = os.path.join(script_dir, "snn_benchmark_results_11/mnist_rate")
OUTPUT_DIR = os.path.join(script_dir, "snn_benchmark_results_11/plots_comparison")

ATTACKS = ["SignFlipping", "Optimal_InnerProductManipulation"]
DISTRIBUTIONS = ["iid", "dirichlet_niid_0.5"]
AGGREGATORS = ["Average", "TrMean"]
F_VALUES = [1, 2, 3, 4]

# Premium styling colors
COLORS = {
    # Direct encoding
    "direct_baseline": "#059669",      # Vibrant Emerald Green (dark)
    "direct_Average": "#1d4ed8",       # Deep Royal Blue
    "direct_TrMean": "#b91c1c",        # Crimson Red
    
    # Rate encoding
    "rate_baseline": "#34d399",        # Minty Green (light)
    "rate_Average": "#60a5fa",         # Sky Blue (light)
    "rate_TrMean": "#fb923c",          # Orange (light)
}

# Apply customized premium plot configuration
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 15,
    'legend.fontsize': 9,
    'grid.alpha': 0.5,
    'grid.linestyle': '--'
})

def load_accuracy(results_dir, folder_name):
    path = os.path.join(results_dir, folder_name, "test_accuracy.txt")
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

def plot_rate_learning_curves(attack, dist):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    axes = axes.flatten()
    
    # Load baseline
    baseline_folder = f"NoAttack_Average_f_0_{dist}"
    baseline_acc = load_accuracy(RESULTS_DIR_RATE, baseline_folder)
    
    plotted_any = False
    
    for idx, f in enumerate(F_VALUES):
        ax = axes[idx]
        
        # Plot baseline
        if baseline_acc is not None:
            steps_base = np.arange(len(baseline_acc)) * 10
            ax.plot(steps_base, baseline_acc, label="Baseline (No Attack)", color=COLORS["rate_baseline"], linewidth=2.5, linestyle="--")
            plotted_any = True
            
        # Plot each aggregator
        for agg in AGGREGATORS:
            run_folder = f"{attack}_{agg}_f_{f}_{dist}"
            acc = load_accuracy(RESULTS_DIR_RATE, run_folder)
            if acc is not None:
                steps = np.arange(len(acc)) * 10
                ax.plot(steps, acc, label=f"{agg} (f={f})", color=COLORS[f"rate_{agg}"], linewidth=2.0)
                plotted_any = True
                
        ax.set_title(f"Byzantine Clients f = {f}", fontweight="bold", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim(-0.02, 1.02)
        if idx >= 2:
            ax.set_xlabel("Communication Round")
        if idx % 2 == 0:
            ax.set_ylabel("Test Accuracy")
            
        # Place legend in the first subplot
        if idx == 0:
            ax.legend(loc="lower left", frameon=True, fontsize=9)
            
    if plotted_any:
        title_dist = "IID" if dist == "iid" else "Dirichlet NIID (0.5)"
        fig.suptitle(f"MNIST Rate coding - {attack} - {title_dist}\nLearning Curves Comparison", fontweight="bold", fontsize=14, y=0.98)
        plt.tight_layout()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, f"rate_learning_curves_{attack}_{dist}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved rate learning curves comparison plot to {out_path}")
    else:
        plt.close()

def plot_rate_accuracy_vs_f(attack, dist):
    # Load baseline
    baseline_folder = f"NoAttack_Average_f_0_{dist}"
    baseline_acc = load_accuracy(RESULTS_DIR_RATE, baseline_folder)
    baseline_final = baseline_acc[-1] if baseline_acc is not None else None
    
    fig, ax = plt.subplots(figsize=(8, 5.5))
    plotted_any = False
    
    for agg in AGGREGATORS:
        fs = [0]
        accs = [baseline_final] if baseline_final is not None else []
        
        for f in F_VALUES:
            run_folder = f"{attack}_{agg}_f_{f}_{dist}"
            acc = load_accuracy(RESULTS_DIR_RATE, run_folder)
            if acc is not None:
                fs.append(f)
                accs.append(acc[-1])
                
        # If we have data beyond baseline f=0
        if len(fs) > 1 and len(accs) == len(fs):
            ax.plot(fs, accs, label=agg, color=COLORS[f"rate_{agg}"], marker="s", linewidth=2.5, markersize=8)
            plotted_any = True
            
    if plotted_any:
        # Plot baseline point
        if baseline_final is not None:
            ax.plot([0], [baseline_final], color=COLORS["rate_baseline"], marker="o", markersize=10, label="Baseline (No Attack)", linestyle="")
            
        title_dist = "IID" if dist == "iid" else "Dirichlet NIID (0.5)"
        ax.set_title(f"MNIST Rate coding - {attack} ({title_dist})\nFinal Test Accuracy vs. Number of Byzantine Clients (f)", fontweight="bold", pad=15)
        ax.set_xlabel("Number of Byzantine Clients (f)")
        ax.set_ylabel("Final Test Accuracy")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower left", frameon=True)
        
        plt.tight_layout()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, f"rate_accuracy_vs_f_{attack}_{dist}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved rate accuracy vs f comparison plot to {out_path}")
    else:
        plt.close()

def plot_comparison_accuracy_vs_f(attack, dist):
    # Load baselines
    baseline_folder = f"NoAttack_Average_f_0_{dist}"
    base_direct = load_accuracy(RESULTS_DIR_DIRECT, baseline_folder)
    base_rate = load_accuracy(RESULTS_DIR_RATE, baseline_folder)
    
    base_direct_final = base_direct[-1] if base_direct is not None else None
    base_rate_final = base_rate[-1] if base_rate is not None else None
    
    fig, ax = plt.subplots(figsize=(9, 6))
    plotted_any = False
    
    # Plot Direct curves
    for agg in AGGREGATORS:
        fs = [0]
        accs = [base_direct_final] if base_direct_final is not None else []
        for f in F_VALUES:
            run_folder = f"{attack}_{agg}_f_{f}_{dist}"
            acc = load_accuracy(RESULTS_DIR_DIRECT, run_folder)
            if acc is not None:
                fs.append(f)
                accs.append(acc[-1])
        if len(fs) > 1 and len(accs) == len(fs):
            ax.plot(fs, accs, label=f"Direct - {agg}", color=COLORS[f"direct_{agg}"], 
                    marker="o", linewidth=2.5, markersize=8, linestyle="-")
            plotted_any = True
            
    # Plot Rate curves
    for agg in AGGREGATORS:
        fs = [0]
        accs = [base_rate_final] if base_rate_final is not None else []
        for f in F_VALUES:
            run_folder = f"{attack}_{agg}_f_{f}_{dist}"
            acc = load_accuracy(RESULTS_DIR_RATE, run_folder)
            if acc is not None:
                fs.append(f)
                accs.append(acc[-1])
        if len(fs) > 1 and len(accs) == len(fs):
            ax.plot(fs, accs, label=f"Rate - {agg}", color=COLORS[f"rate_{agg}"], 
                    marker="s", linewidth=2.5, markersize=8, linestyle="--")
            plotted_any = True
            
    if plotted_any:
        # Plot baseline points at f=0
        if base_direct_final is not None:
            ax.plot([0], [base_direct_final], color=COLORS["direct_baseline"], 
                    marker="o", markersize=10, label="Baseline Direct (No Attack)", linestyle="")
        if base_rate_final is not None:
            ax.plot([0], [base_rate_final], color=COLORS["rate_baseline"], 
                    marker="s", markersize=10, label="Baseline Rate (No Attack)", linestyle="")
            
        title_dist = "IID" if dist == "iid" else "Dirichlet NIID (0.5)"
        ax.set_title(f"MNIST Encoding Comparison - {attack} ({title_dist})\nFinal Test Accuracy vs. Number of Byzantine Clients (f)", fontweight="bold", pad=15)
        ax.set_xlabel("Number of Byzantine Clients (f)")
        ax.set_ylabel("Final Test Accuracy")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower left", frameon=True)
        
        plt.tight_layout()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, f"comparison_accuracy_vs_f_{attack}_{dist}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved comparison accuracy vs f plot to {out_path}")
    else:
        plt.close()

def plot_comparison_learning_curves(attack, dist):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    # Load baselines
    baseline_folder = f"NoAttack_Average_f_0_{dist}"
    base_direct = load_accuracy(RESULTS_DIR_DIRECT, baseline_folder)
    base_rate = load_accuracy(RESULTS_DIR_RATE, baseline_folder)
    
    plotted_any = False
    
    for idx, f in enumerate(F_VALUES):
        ax = axes[idx]
        
        # Plot direct baseline
        if base_direct is not None:
            steps_base = np.arange(len(base_direct)) * 10
            ax.plot(steps_base, base_direct, label="Baseline Direct", color=COLORS["direct_baseline"], linewidth=2.0, linestyle=":")
            plotted_any = True
            
        # Plot rate baseline
        if base_rate is not None:
            steps_base = np.arange(len(base_rate)) * 10
            ax.plot(steps_base, base_rate, label="Baseline Rate", color=COLORS["rate_baseline"], linewidth=2.0, linestyle="--")
            plotted_any = True
            
        # Plot Direct aggregators
        for agg in AGGREGATORS:
            run_folder = f"{attack}_{agg}_f_{f}_{dist}"
            acc = load_accuracy(RESULTS_DIR_DIRECT, run_folder)
            if acc is not None:
                steps = np.arange(len(acc)) * 10
                ax.plot(steps, acc, label=f"Direct - {agg}", color=COLORS[f"direct_{agg}"], linewidth=2.0, linestyle="-")
                plotted_any = True
                
        # Plot Rate aggregators
        for agg in AGGREGATORS:
            run_folder = f"{attack}_{agg}_f_{f}_{dist}"
            acc = load_accuracy(RESULTS_DIR_RATE, run_folder)
            if acc is not None:
                steps = np.arange(len(acc)) * 10
                ax.plot(steps, acc, label=f"Rate - {agg}", color=COLORS[f"rate_{agg}"], linewidth=2.0, linestyle="--")
                plotted_any = True
                
        ax.set_title(f"Byzantine Clients f = {f}", fontweight="bold", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim(-0.02, 1.02)
        if idx >= 2:
            ax.set_xlabel("Communication Round")
        if idx % 2 == 0:
            ax.set_ylabel("Test Accuracy")
            
        # Legend only in first subplot
        if idx == 0:
            ax.legend(loc="lower left", frameon=True, fontsize=8)
            
    if plotted_any:
        title_dist = "IID" if dist == "iid" else "Dirichlet NIID (0.5)"
        fig.suptitle(f"MNIST Encoding Comparison - {attack} - {title_dist}\nLearning Curves Comparison (Direct vs. Rate)", fontweight="bold", fontsize=14, y=0.98)
        plt.tight_layout()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, f"comparison_learning_curves_{attack}_{dist}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved comparison learning curves plot to {out_path}")
    else:
        plt.close()

def main():
    print("==========================================================")
    print("GENERATING ACCURACY PLOTS FOR RATE AND ENCODING COMPARISON")
    print("==========================================================")
    
    if not os.path.isdir(RESULTS_DIR_RATE):
        print(f"Error: results directory for rate '{RESULTS_DIR_RATE}' does not exist.")
        return
    if not os.path.isdir(RESULTS_DIR_DIRECT):
        print(f"Error: results directory for direct '{RESULTS_DIR_DIRECT}' does not exist.")
        return
        
    for attack in ATTACKS:
        for dist in DISTRIBUTIONS:
            print(f"\nProcessing Attack: {attack}, Distribution: {dist}...")
            # 1. Plot rate results (learning curves and final accuracy vs f)
            plot_rate_learning_curves(attack, dist)
            plot_rate_accuracy_vs_f(attack, dist)
            
            # 2. Plot comparison results
            plot_comparison_accuracy_vs_f(attack, dist)
            plot_comparison_learning_curves(attack, dist)
            
    print("\nAll plots generated successfully in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
