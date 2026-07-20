"""
Plot inter-architecture geometry metrics from the metrics_interarch.csv files.

Reads results from ./results/interarch_metrics/ and generates comparison plots
across architectures (SNN θ-sweep, CNN clip-sweep, Tanh).

IMPORTANT: scale and sigma_H are NOT comparable across loss functions
(NLLLoss vs ce_rate_loss). These plots are generated intra-model only.
"""
import os
import glob
import re
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Config ────────────────────────────────────────────────────────────────
RESULTS_DIR = "./results/interarch_metrics"
OUTPUT_DIR = "./plots/interarch_metrics"
STABLE_RANGE = (200, 400)  # steps for summary table

# Styling
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
})

# Color scheme
ARCH_COLORS = {
    'SNN θ=1.0': '#e63946',
    'SNN θ=0.8': '#f4845f',
    'SNN θ=0.6': '#f7b267',
    'SNN θ=0.4': '#f7d08a',
    'ReLU (no clip)': '#457b9d',
    'ReLU clip=21': '#1d3557',
    'ReLU clip=10': '#264653',
    'ReLU clip=5': '#2a9d8f',
    'Tanh': '#6a4c93',
}

GAMMA_LINESTYLES = {
    1.0: '-',
    0.66: '--',
    0.33: '-.',
    0.0: ':',
}


def identify_run(exp_dir):
    """Identify architecture label from config.json in experiment directory."""
    config_path = os.path.join(exp_dir, "config.json")
    if not os.path.exists(config_path):
        return None, None
    with open(config_path) as f:
        cfg = json.load(f)

    model_name = cfg.get("model", cfg.get("benchmark_config", {})).get("name", "")
    if isinstance(cfg.get("model"), dict):
        model_name = cfg["model"].get("name", "")

    # Get gamma
    gamma = None
    dd = cfg.get("benchmark_config", {}).get("data_distribution", {})
    if isinstance(dd, dict):
        gamma = dd.get("distribution_parameter")
    elif isinstance(dd, list) and len(dd) > 0:
        gamma = dd[0].get("distribution_parameter")
    if isinstance(gamma, list):
        gamma = gamma[0] if len(gamma) == 1 else gamma

    # Get threshold / clip
    model_params = cfg.get("model", {}).get("model_params", {})
    threshold = model_params.get("threshold")
    clip_val = cfg.get("honest_clients", {}).get("gradient_clip_val", 0)

    if "snn" in model_name.lower():
        label = f"SNN θ={threshold}"
    elif "tanh" in model_name.lower():
        label = "Tanh"
    elif clip_val and clip_val > 0:
        label = f"ReLU clip={int(clip_val)}"
    else:
        label = "ReLU (no clip)"

    return label, gamma


def load_all_interarch_csvs(results_dir):
    """Walk result directories and load all metrics_interarch CSVs."""
    records = []
    for root, dirs, files in os.walk(results_dir):
        csv_files = [f for f in files if f.startswith("metrics_interarch") and f.endswith(".csv")]
        if not csv_files:
            continue
        label, gamma = identify_run(root)
        if label is None:
            continue
        for csv_file in csv_files:
            try:
                df = pd.read_csv(os.path.join(root, csv_file))
                df["arch"] = label
                df["gamma"] = gamma if not isinstance(gamma, list) else gamma
                # Extract seed from filename
                seed_match = re.search(r"tr_seed_(\d+)", csv_file)
                df["seed"] = int(seed_match.group(1)) if seed_match else 42
                records.append(df)
            except Exception as e:
                print(f"  Warning: failed to load {os.path.join(root, csv_file)}: {e}")
    if not records:
        print("No metrics_interarch CSVs found!")
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def plot_metric_vs_step(df, metric, ylabel, title, filename, prefix=""):
    """Plot a single metric vs step, one line per architecture, faceted by gamma."""
    col = f"{prefix}{metric}" if prefix else metric
    if col not in df.columns:
        print(f"  Skipping {filename}: column '{col}' not found")
        return

    gammas = sorted(df["gamma"].unique())
    fig, axes = plt.subplots(1, len(gammas), figsize=(5 * len(gammas), 4), sharey=True)
    if len(gammas) == 1:
        axes = [axes]

    for ax, gamma in zip(axes, gammas):
        sub = df[df["gamma"] == gamma]
        for arch in sorted(sub["arch"].unique()):
            arch_data = sub[sub["arch"] == arch]
            mean = arch_data.groupby("step")[col].mean()
            color = ARCH_COLORS.get(arch, 'gray')
            ax.plot(mean.index, mean.values, color=color, label=arch, linewidth=1.5)
        ax.set_title(f"γ={gamma}")
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"  Saved {filename}")


def plot_cveff_vs_cveff_act(df, filename="cveff_decomposition.png"):
    """Scatter CV_eff vs CV_eff_act per architecture to diagnose values vs participation."""
    if "CV_eff" not in df.columns or "CV_eff_act" not in df.columns:
        print(f"  Skipping {filename}: columns not found")
        return

    gammas = sorted(df["gamma"].unique())
    fig, axes = plt.subplots(1, len(gammas), figsize=(5 * len(gammas), 4.5))
    if len(gammas) == 1:
        axes = [axes]

    for ax, gamma in zip(axes, gammas):
        sub = df[df["gamma"] == gamma]
        for arch in sorted(sub["arch"].unique()):
            arch_data = sub[sub["arch"] == arch]
            color = ARCH_COLORS.get(arch, 'gray')
            ax.scatter(arch_data["CV_eff_act"], arch_data["CV_eff"],
                      s=5, alpha=0.4, color=color, label=arch)
        # Diagonal reference
        lims = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=0.8)
        ax.set_xlabel("CV_eff_act (active clients only)")
        ax.set_title(f"γ={gamma}")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("CV_eff (all clients)")
    axes[-1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.suptitle("CV_eff decomposition: values vs participation\n"
                 "(above diagonal → dispersion from support, on diagonal → from values)",
                 fontsize=12, y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"  Saved {filename}")


def plot_scale_intramodel(df, filename="scale_intramodel.png"):
    """Plot scale vs step, separated by model family (not cross-model)."""
    if "scale" not in df.columns:
        return

    families = {
        "SNN": [a for a in df["arch"].unique() if "SNN" in a],
        "CNN": [a for a in df["arch"].unique() if "SNN" not in a],
    }
    gammas = sorted(df["gamma"].unique())

    for family_name, archs in families.items():
        if not archs:
            continue
        fig, axes = plt.subplots(1, len(gammas), figsize=(5 * len(gammas), 4), sharey=True)
        if len(gammas) == 1:
            axes = [axes]
        for ax, gamma in zip(axes, gammas):
            sub = df[(df["gamma"] == gamma) & (df["arch"].isin(archs))]
            for arch in sorted(archs):
                arch_data = sub[sub["arch"] == arch]
                if arch_data.empty:
                    continue
                mean = arch_data.groupby("step")["scale"].mean()
                color = ARCH_COLORS.get(arch, 'gray')
                ax.plot(mean.index, mean.values, color=color, label=arch, linewidth=1.5)
            ax.set_title(f"γ={gamma}")
            ax.set_xlabel("Step")
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("scale (mean ‖m^(k)‖₂)")
        axes[-1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
        fname = f"scale_{family_name.lower()}.png"
        fig.suptitle(f"Scale — {family_name} only (intra-model, not cross-comparable)", fontsize=13, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, fname))
        plt.close()
        print(f"  Saved {fname}")


def plot_pre_vs_post_momentum(df, metric="CV_eff", filename="pre_vs_post_momentum.png"):
    """Overlay pre-momentum (g_) vs post-momentum metric."""
    post_col = metric
    pre_col = f"g_{metric}"
    if post_col not in df.columns or pre_col not in df.columns:
        print(f"  Skipping {filename}: columns not found")
        return

    gammas = sorted(df["gamma"].unique())
    fig, axes = plt.subplots(1, len(gammas), figsize=(5 * len(gammas), 4), sharey=True)
    if len(gammas) == 1:
        axes = [axes]

    for ax, gamma in zip(axes, gammas):
        sub = df[df["gamma"] == gamma]
        for arch in sorted(sub["arch"].unique()):
            arch_data = sub[sub["arch"] == arch]
            color = ARCH_COLORS.get(arch, 'gray')
            mean_post = arch_data.groupby("step")[post_col].mean()
            mean_pre = arch_data.groupby("step")[pre_col].mean()
            ax.plot(mean_post.index, mean_post.values, color=color, linewidth=1.5, label=f"{arch} (post-mom)")
            ax.plot(mean_pre.index, mean_pre.values, color=color, linewidth=1.0, linestyle=':', alpha=0.7, label=f"{arch} (pre-mom)")
        ax.set_title(f"γ={gamma}")
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(metric)
    axes[-1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=7)
    fig.suptitle(f"{metric}: pre-momentum (dotted) vs post-momentum (solid)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"  Saved {filename}")


def generate_summary_table(df, filename="summary_table.csv"):
    """Generate summary table with means over stable range."""
    lo, hi = STABLE_RANGE
    stable = df[(df["step"] >= lo) & (df["step"] <= hi)]
    if stable.empty:
        print(f"  No data in stable range [{lo}, {hi}]")
        return

    metrics = ["N", "CV_eff", "CV_eff_act", "Pi", "cos_mean", "scale", "sigma_H",
               "g_N", "g_CV_eff", "g_CV_eff_act", "g_Pi", "g_cos_mean", "g_scale"]
    metrics = [m for m in metrics if m in stable.columns]

    summary = stable.groupby(["arch", "gamma"])[metrics].mean()
    summary_path = os.path.join(OUTPUT_DIR, filename)
    summary.to_csv(summary_path, float_format="%.6f")
    print(f"  Saved {filename}")

    # Print to console
    print("\n  Summary (steps {}-{}):".format(lo, hi))
    print(summary.to_string())


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading inter-architecture metrics CSVs...")
    df = load_all_interarch_csvs(RESULTS_DIR)
    if df.empty:
        print("No data found. Run run_interarch_experiment.py first.")
        return

    print(f"Loaded {len(df)} rows from {df['arch'].nunique()} architectures, "
          f"{df['gamma'].nunique()} gamma levels, {df['seed'].nunique()} seeds")

    # 1. Core metrics vs step
    print("\nGenerating core metric plots...")
    plot_metric_vs_step(df, "N", "N (norm dispersion)", "N — Dispersion of Client Norms", "N_vs_step.png")
    plot_metric_vs_step(df, "CV_eff", "CV_eff", "CV_eff — Effective Coefficient of Variation", "CV_eff_vs_step.png")
    plot_metric_vs_step(df, "cos_mean", "cos_mean", "Mean Pairwise Cosine Similarity", "cos_mean_vs_step.png")
    plot_metric_vs_step(df, "Pi", "Π", "Π — Signal-Weighted Participation", "Pi_vs_step.png")

    # 2. CV_eff decomposition
    print("\nGenerating CV_eff decomposition scatter...")
    plot_cveff_vs_cveff_act(df)

    # 3. Scale (intra-model only)
    print("\nGenerating scale plots (intra-model)...")
    plot_scale_intramodel(df)

    # 4. Pre vs post momentum comparison
    print("\nGenerating pre vs post momentum comparison...")
    plot_pre_vs_post_momentum(df, "CV_eff", "pre_vs_post_CV_eff.png")
    plot_pre_vs_post_momentum(df, "N", "pre_vs_post_N.png")

    # 5. Pre-momentum metrics
    print("\nGenerating pre-momentum metric plots...")
    plot_metric_vs_step(df, "N", "g_N (norm dispersion)", "g_N — Pre-Momentum Norm Dispersion", "g_N_vs_step.png", prefix="g_")
    plot_metric_vs_step(df, "CV_eff", "g_CV_eff", "g_CV_eff — Pre-Momentum CV_eff", "g_CV_eff_vs_step.png", prefix="g_")

    # 6. Summary table
    print("\nGenerating summary table...")
    generate_summary_table(df)

    print(f"\nAll plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
