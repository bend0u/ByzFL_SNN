import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def get_accuracy_metrics(results_dir, alpha, f_val, agg_name, attack_name, gamma_val):
    # Pattern to match the directory
    pattern = os.path.join(
        results_dir, 
        f"*_f_{f_val}_*_gamma_similarity_niid_{gamma_val}_{agg_name}_NNM_ARC_{attack_name}_*_surrogate_gradient_atan_alpha_{alpha}_*"
    )
    matching_dirs = glob.glob(pattern)
    
    accuracies = []
    for directory in matching_dirs:
        acc_files = glob.glob(os.path.join(directory, "test_accuracy_tr_seed_*.txt"))
        for acc_file in acc_files:
            try:
                data = np.genfromtxt(acc_file, delimiter=',')
                if data.ndim == 0:
                    continue
                accuracies.append(data[-1])
            except:
                pass
    if not accuracies:
        return None
    return np.mean(accuracies), np.std(accuracies)

def main():
    results_dir = "./results/snn/robust_atan_sweep"
    plots_dir = "./plots/snn"
    os.makedirs(plots_dir, exist_ok=True)
    
    alphas = [1.0, 1.5, 2.0, 3.0, 4.0]
    gammas = [1.0, 0.66, 0.33, 0.0]
    aggregators = ["GeometricMedian", "CenteredClipping"]
    attacks = ["SignFlipping", "Optimal_ALittleIsEnough_neg1"]
    
    # 2x2 Plot grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    
    colors = {1.0: "blue", 0.66: "cyan", 0.33: "orange", 0.0: "red"}
    markers = {1.0: "o", 0.66: "s", 0.33: "^", 0.0: "x"}
    
    for row, agg in enumerate(aggregators):
        for col, attack in enumerate(attacks):
            ax = axes[row, col]
            
            # 1. Plot f=0 baseline (mean over all gammas, since there is no attack, they behave similarly)
            f0_accs = []
            for alpha in alphas:
                # Average over all gammas for f0
                f0_vals = []
                for gamma in gammas:
                    res = get_accuracy_metrics(results_dir, alpha, 0, agg, attack, gamma)
                    if res:
                        f0_vals.append(res[0])
                if f0_vals:
                    f0_accs.append(np.mean(f0_vals))
                else:
                    f0_accs.append(None)
                    
            valid_f0 = [i for i, v in enumerate(f0_accs) if v is not None]
            if valid_f0:
                ax.plot([alphas[i] for i in valid_f0], [f0_accs[i] for i in valid_f0], 
                        '--k', linewidth=2, label="f=0 (Baseline)")
            
            # 2. Plot f=1 curves for each gamma
            for gamma in gammas:
                f1_accs = []
                f1_stds = []
                for alpha in alphas:
                    res = get_accuracy_metrics(results_dir, alpha, 1, agg, attack, gamma)
                    if res:
                        f1_accs.append(res[0])
                        f1_stds.append(res[1])
                    else:
                        f1_accs.append(None)
                        f1_stds.append(None)
                
                valid_f1 = [i for i, v in enumerate(f1_accs) if v is not None]
                if valid_f1:
                    x = [alphas[i] for i in valid_f1]
                    y = [f1_accs[i] for i in valid_f1]
                    yerr = [f1_stds[i] for i in valid_f1]
                    ax.errorbar(x, y, yerr=yerr, fmt='-' + markers[gamma], color=colors[gamma],
                                capsize=4, label=f"f=1, gamma={gamma}")
            
            ax.set_title(f"{agg} under {attack}")
            ax.set_xlabel("Alpha (Stiffness)")
            if col == 0:
                ax.set_ylabel("Final Test Accuracy")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(fontsize=9)
            ax.set_ylim(0.0, 1.05)
            
    plt.suptitle("Atan Surrogate Stiffness vs. Robustness under Attacks (T=10, LR=0.10)", fontsize=16, y=0.98)
    plt.tight_layout()
    
    plot_path = os.path.join(plots_dir, "atan_robustness_comparison.png")
    plt.savefig(plot_path, dpi=300)
    print(f"--> Saved Atan robustness comparison plot: {plot_path}")

if __name__ == "__main__":
    main()
