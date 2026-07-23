"""
Plot sparsity comparison between SNN ATAN 1.2, CNN (ReLU), and CNN (Tanh).
Generates comparison plots for Hoyer Sparsity, Gini Index, and other metrics
across all γ levels (1.0, 0.66, 0.33, 0.0).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ───────────────────── Configuration ─────────────────────

RESULTS_DIR = "results/sparsity_measure"
PLOTS_DIR = "plots/sparsity_measure"
NB_STEPS = 500

GAMMAS = [1.0, 0.66, 0.33, 0.0]
GAMMA_LABELS = {1.0: "γ = 1.0 (IID)", 0.66: "γ = 0.66",
                0.33: "γ = 0.33", 0.0: "γ = 0.0 (extreme non-IID)"}

MODELS = {
    'cnn_mnist': {
        'label': 'CNN (ReLU)',
        'color': '#2196F3', # Blue
        'pattern': 'mnist_cnn_mnist_n_10_f_0_d_0_gamma_similarity_niid_'
    },
    'cnn_mnist_tanh': {
        'label': 'CNN (Tanh)',
        'color': '#4CAF50', # Green
        'pattern': 'mnist_cnn_mnist_tanh_n_10_f_0_d_0_gamma_similarity_niid_'
    },
    'cnn_mnist_snn': {
        'label': 'SNN (ATAN 1.2)',
        'color': '#F44336', # Red
        'pattern': 'mnist_cnn_mnist_snn_n_10_f_0_d_0_gamma_similarity_niid_'
    }
}

METRICS = [
    ("hoyer", "Hoyer Sparsity", "Hoyer Index"),
    ("gini", "Gini Index", "Gini Index"),
    ("l1_l2_ratio", "Normalized L1/L2 Ratio", "L1/(L2·√d)"),
    ("near_zero_1e3", "Near-Zero Fraction (|g|<1e-3)", "Fraction"),
    ("near_zero_1e5", "Near-Zero Fraction (|g|<1e-5)", "Fraction"),
    ("kurtosis", "Excess Kurtosis", "Kurtosis"),
    ("top1_concentration", "Top-1% L1 Concentration", "L1 Share"),
    ("top10_concentration", "Top-10% L1 Concentration", "L1 Share"),
    ("entropy", "Normalized Entropy", "Entropy"),
]

# ───────────────────── Helpers ─────────────────────

def find_experiment_dirs(model_name, gamma):
    """Find all experiment directories for a given model and gamma value."""
    dirs = []
    if not os.path.isdir(RESULTS_DIR):
        return dirs
    pattern = MODELS[model_name]['pattern'] + f"{gamma}_"
    for entry in os.listdir(RESULTS_DIR):
        full = os.path.join(RESULTS_DIR, entry)
        if not os.path.isdir(full):
            continue
        if pattern in entry or (entry.startswith(MODELS[model_name]['pattern']) and entry.endswith(f"{gamma}")):
            dirs.append(full)
    return dirs


def load_metric_across_seeds(exp_dirs, metric_key, stat="mean"):
    """Load a sparsity metric file across seeds, return (nb_seeds, nb_steps) array."""
    filename_pattern = f"sparsity_{metric_key}_{stat}"
    all_data = []
    for exp_dir in exp_dirs:
        for f in os.listdir(exp_dir):
            if f.startswith(filename_pattern) and f.endswith(".txt"):
                filepath = os.path.join(exp_dir, f)
                try:
                    data = np.loadtxt(filepath, delimiter=',')
                    if data.ndim == 1:
                        all_data.append(data)
                    elif data.ndim == 2:
                        all_data.append(data[0])
                except Exception as e:
                    pass
    return np.array(all_data) if all_data else None


def plot_comparison_grid(metric_key, display_name, y_label):
    """Plot comparison between the three models in a 2x2 grid of γ levels."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes_flat = axes.flatten()

    has_any_data = False
    for idx, gamma in enumerate(GAMMAS):
        ax = axes_flat[idx]
        has_gamma_data = False
        
        for model_name, info in MODELS.items():
            exp_dirs = find_experiment_dirs(model_name, gamma)
            if not exp_dirs:
                continue
            
            data = load_metric_across_seeds(exp_dirs, metric_key, "mean")
            if data is None or len(data) == 0:
                continue
            
            has_gamma_data = True
            has_any_data = True
            steps = np.arange(data.shape[1])
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            
            ax.plot(steps, mean, label=info['label'], color=info['color'], linewidth=2)
            ax.fill_between(steps, mean - std, mean + std, alpha=0.15, color=info['color'])
            
        ax.set_xlabel("Training Step", fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title(GAMMA_LABELS[gamma], fontsize=12, fontweight='semibold')
        ax.grid(True, linestyle='--', alpha=0.5)
        if has_gamma_data:
            ax.legend(fontsize=10)

    if not has_any_data:
        plt.close()
        return False

    fig.suptitle(f"Sparsity Comparison: {display_name} (f=0, Average, n=10)", fontsize=16, y=0.96, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    safe_name = metric_key.replace("/", "_")
    fig.savefig(os.path.join(PLOTS_DIR, f"comparison_{safe_name}.pdf"), dpi=150)
    fig.savefig(os.path.join(PLOTS_DIR, f"comparison_{safe_name}.png"), dpi=150)
    plt.close()
    return True


# ───────────────────── Main ─────────────────────

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("=" * 60)
    print("Plotting Model Sparsity Comparisons")
    print(f"  Results dir: {RESULTS_DIR}")
    print(f"  Output dir:  {PLOTS_DIR}")
    print("=" * 60)

    for metric_key, display_name, y_label in METRICS:
        success = plot_comparison_grid(metric_key, display_name, y_label)
        status = "✓" if success else "✗ (no data)"
        print(f"  {status} {display_name} Comparison Grid")

    print(f"\nAll comparison plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
