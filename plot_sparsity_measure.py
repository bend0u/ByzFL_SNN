"""
Plot sparsity metrics from the sparsity measurement experiment.
Generates plots of Hoyer, Gini, and all other sparsity metrics
as a function of training steps, across all γ levels.
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
GAMMA_LABELS = {1.0: r"$\gamma=1.0$ (IID)", 0.66: r"$\gamma=0.66$",
                0.33: r"$\gamma=0.33$", 0.0: r"$\gamma=0.0$ (extreme non-IID)"}
GAMMA_COLORS = {1.0: "#2196F3", 0.66: "#4CAF50", 0.33: "#FF9800", 0.0: "#F44336"}

SEEDS = [42, 43, 44, 45, 46]

# Metric display info: (file_key, display_name, y_label, description)
METRICS = [
    ("hoyer", "Hoyer Sparsity", "Hoyer Index",
     r"$(\\sqrt{d} - \\|x\\|_1/\\|x\\|_2) / (\\sqrt{d} - 1)$. 0=uniform, 1=sparse"),
    ("gini", "Gini Index", "Gini Index",
     "Lorenz-curve inequality measure. 0=uniform, 1=sparse"),
    ("l1_l2_ratio", "Normalized L1/L2 Ratio", "L1/(L2·√d)",
     "1=uniform, 0=sparse (inverse of Hoyer direction)"),
    ("near_zero_1e5", "Near-Zero Fraction (|g|<1e-5)", "Fraction",
     "Fraction of gradient components with |g| < 1e-5"),
    ("near_zero_1e3", "Near-Zero Fraction (|g|<1e-3)", "Fraction",
     "Fraction of gradient components with |g| < 1e-3"),
    ("kurtosis", "Excess Kurtosis", "Kurtosis",
     "Peakedness measure. Higher = heavier tails / more sparse"),
    ("top1_concentration", "Top-1% L1 Concentration", "L1 Share",
     "Fraction of total L1 norm in top 1% of components"),
    ("top5_concentration", "Top-5% L1 Concentration", "L1 Share",
     "Fraction of total L1 norm in top 5% of components"),
    ("top10_concentration", "Top-10% L1 Concentration", "L1 Share",
     "Fraction of total L1 norm in top 10% of components"),
    ("entropy", "Normalized Entropy", "Entropy",
     "Shannon entropy of |g| distribution (100 bins). 1=uniform, 0=concentrated"),
]


# ───────────────────── Helpers ─────────────────────

def find_experiment_dirs(gamma):
    """Find all experiment directories for a given gamma value."""
    dirs = []
    if not os.path.isdir(RESULTS_DIR):
        return dirs
    for entry in os.listdir(RESULTS_DIR):
        full = os.path.join(RESULTS_DIR, entry)
        if not os.path.isdir(full):
            continue
        # Match pattern: *_gamma_similarity_niid_{gamma}_*
        if f"gamma_similarity_niid_{gamma}_" in entry or entry.endswith(f"gamma_similarity_niid_{gamma}"):
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
                        all_data.append(data[0])  # write_array_in_file wraps in extra dim
                except Exception as e:
                    print(f"  Warning: Could not load {filepath}: {e}")
    return np.array(all_data) if all_data else None


def plot_metric(metric_key, display_name, y_label, description):
    """Plot a single sparsity metric across all gammas."""
    fig, ax = plt.subplots(figsize=(10, 6))

    has_data = False
    for gamma in GAMMAS:
        exp_dirs = find_experiment_dirs(gamma)
        if not exp_dirs:
            continue

        data = load_metric_across_seeds(exp_dirs, metric_key, "mean")
        if data is None or len(data) == 0:
            continue

        has_data = True
        steps = np.arange(data.shape[1])
        mean = data.mean(axis=0)
        std = data.std(axis=0)

        color = GAMMA_COLORS[gamma]
        label = GAMMA_LABELS[gamma]
        ax.plot(steps, mean, label=label, color=color, linewidth=2)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.15, color=color)

    if not has_data:
        plt.close()
        return False

    ax.set_xlabel("Training Step", fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(f"{display_name}\n(SNN ATAN α=1.2, f=0, Average, n=10)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    safe_name = metric_key.replace("/", "_")
    fig.savefig(os.path.join(PLOTS_DIR, f"{safe_name}.pdf"), dpi=150)
    fig.savefig(os.path.join(PLOTS_DIR, f"{safe_name}.png"), dpi=150)
    plt.close()
    return True


def plot_combined_main_metrics():
    """Plot Hoyer and Gini side-by-side (the two main metrics)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, (metric_key, display_name, y_label, _) in zip(
        axes, [METRICS[0], METRICS[1]]  # Hoyer and Gini
    ):
        for gamma in GAMMAS:
            exp_dirs = find_experiment_dirs(gamma)
            if not exp_dirs:
                continue
            data = load_metric_across_seeds(exp_dirs, metric_key, "mean")
            if data is None or len(data) == 0:
                continue
            steps = np.arange(data.shape[1])
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            color = GAMMA_COLORS[gamma]
            label = GAMMA_LABELS[gamma]
            ax.plot(steps, mean, label=label, color=color, linewidth=2)
            ax.fill_between(steps, mean - std, mean + std, alpha=0.15, color=color)

        ax.set_xlabel("Training Step", fontsize=13)
        ax.set_ylabel(y_label, fontsize=13)
        ax.set_title(display_name, fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)

    fig.suptitle("SNN ATAN α=1.2 — Gradient Sparsity (f=0, Average, n=10)", fontsize=15, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "hoyer_gini_combined.pdf"), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(PLOTS_DIR, "hoyer_gini_combined.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_all_metrics_grid():
    """Plot all 10 metrics in a 2x5 grid for a comprehensive overview."""
    fig, axes = plt.subplots(2, 5, figsize=(28, 10))
    axes_flat = axes.flatten()

    for ax, (metric_key, display_name, y_label, desc) in zip(axes_flat, METRICS):
        for gamma in GAMMAS:
            exp_dirs = find_experiment_dirs(gamma)
            if not exp_dirs:
                continue
            data = load_metric_across_seeds(exp_dirs, metric_key, "mean")
            if data is None or len(data) == 0:
                continue
            steps = np.arange(data.shape[1])
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            color = GAMMA_COLORS[gamma]
            ax.plot(steps, mean, color=color, linewidth=1.5)
            ax.fill_between(steps, mean - std, mean + std, alpha=0.1, color=color)

        ax.set_xlabel("Step", fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.set_title(display_name, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.tick_params(labelsize=8)

    # Shared legend
    legend_handles = [
        Line2D([0], [0], color=GAMMA_COLORS[g], linewidth=2, label=GAMMA_LABELS[g])
        for g in GAMMAS
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=4, fontsize=11,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("SNN ATAN α=1.2 — All Gradient Sparsity Metrics (f=0, Average, n=10)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "all_metrics_grid.pdf"), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(PLOTS_DIR, "all_metrics_grid.png"), dpi=150, bbox_inches='tight')
    plt.close()


# ───────────────────── Main ─────────────────────

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("=" * 60)
    print("Plotting Sparsity Metrics")
    print(f"  Results dir: {RESULTS_DIR}")
    print(f"  Output dir:  {PLOTS_DIR}")
    print("=" * 60)

    # Individual metric plots
    for metric_key, display_name, y_label, description in METRICS:
        success = plot_metric(metric_key, display_name, y_label, description)
        status = "✓" if success else "✗ (no data)"
        print(f"  {status} {display_name}")

    # Combined main metrics plot
    print("\n  Generating combined Hoyer+Gini plot...")
    plot_combined_main_metrics()
    print("  ✓ hoyer_gini_combined")

    # All metrics grid
    print("  Generating all-metrics grid...")
    plot_all_metrics_grid()
    print("  ✓ all_metrics_grid")

    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
