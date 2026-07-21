#!/usr/bin/env python3
"""Plot gradient-structure statistics collected by collect_gradient_stats.py.

Reads gradient_stats_combined.csv (one row per config_tag/gamma/seed/step,
columns already computed online during training -- see
byzfl/utils/gradient_structure_metrics.py and gradient_geometry.py) and
produces one PNG set per config_tag:
  - per-client norm spread across gamma (strip plot, from norm_0..norm_{n-1})
  - PCA effective-rank curve across gamma (# components for 90% variance)
  - PCA 2D scatter of the honest clients, small multiples across gamma
  - subset coordinate-value repartition (histogram), across gamma
  - active-coordinate mean magnitude vs threshold, across gamma

Colors follow the project's validated categorical palette (fixed hue per
gamma value, never re-cycled): gamma=1.0 blue, 0.66 green, 0.33 magenta,
0.0 yellow.
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GAMMA_COLORS = {
    1.0: "#2a78d6",   # blue
    0.66: "#008300",  # green
    0.33: "#e87ba4",  # magenta
    0.0: "#eda100",   # yellow
}
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
MUTED = "#898781"
INK = "#0b0b0b"
SUBSET_COLORS = ["#2a78d6", "#008300", "#e87ba4"]

NORM_RE = re.compile(r"^norm_(\d+)$")
PCA_PROJ_RE = re.compile(r"^pca_proj_c(\d+)_pc(\d+)$")
ACTIVE_MEAN_RE = re.compile(r"^active_mean_thr_(.+)$")
SUBSET_HIST_RE = re.compile(r"^subset(\d+)_client(\d+)_hist_(\d+)$")

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})


def _gamma_color(gamma):
    return GAMMA_COLORS.get(round(float(gamma), 2), "#898781")


def _style_axes(ax):
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)


def _gammas_sorted(df):
    return sorted(df["gamma"].unique(), reverse=True)


def _matching_columns(df, pattern):
    return [c for c in df.columns if pattern.match(c)]


def plot_norm_spread(df, config_tag, out_dir):
    sub = df[df["config_tag"] == config_tag]
    norm_cols = _matching_columns(sub, NORM_RE)
    if sub.empty or not norm_cols:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    gammas = _gammas_sorted(sub)
    rng = np.random.default_rng(0)
    for i, gamma in enumerate(gammas):
        rows = sub[sub["gamma"] == gamma]
        norms = rows[norm_cols].to_numpy().ravel()
        norms = norms[~np.isnan(norms)]
        if norms.size == 0:
            continue
        color = _gamma_color(gamma)
        jitter = (rng.random(len(norms)) - 0.5) * 0.3
        ax.scatter(np.full(len(norms), i) + jitter, norms, s=36, color=color,
                   alpha=0.75, edgecolor="white", linewidth=0.5, zorder=3,
                   label=f"gamma={gamma}")
    ax.set_xticks(range(len(gammas)))
    ax.set_xticklabels([f"{g}" for g in gammas])
    ax.set_xlabel("gamma (data similarity)")
    ax.set_ylabel("honest gradient norm ||g_i||")
    ax.set_title(f"Per-client gradient norm spread -- {config_tag}")
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{config_tag}_norm_spread.png"), dpi=150)
    plt.close(fig)


def plot_pca_effective_rank(df, config_tag, out_dir):
    sub = df[df["config_tag"] == config_tag]
    if sub.empty or "pca_n_components_90pct" not in sub.columns:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for gamma in _gammas_sorted(sub):
        rows = sub[sub["gamma"] == gamma].sort_values("step")
        if rows.empty:
            continue
        color = _gamma_color(gamma)
        ax.plot(rows["step"], rows["pca_n_components_90pct"], color=color,
                linewidth=2, marker="o", markersize=6, label=f"gamma={gamma}")
    ax.set_xlabel("training step")
    ax.set_ylabel("# components for 90% explained variance")
    ax.set_title(f"PCA effective rank over training -- {config_tag}")
    ax.set_ylim(bottom=0)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{config_tag}_pca_effective_rank.png"), dpi=150)
    plt.close(fig)


def plot_pca_scatter(df, config_tag, out_dir):
    sub = df[df["config_tag"] == config_tag]
    proj_cols = _matching_columns(sub, PCA_PROJ_RE)
    if sub.empty or not proj_cols:
        return
    clients = sorted({int(PCA_PROJ_RE.match(c).group(1)) for c in proj_cols})
    gammas = _gammas_sorted(sub)
    last_step = sub["step"].max()
    fig, axes = plt.subplots(1, len(gammas), figsize=(4 * len(gammas), 4), sharex=True, sharey=True)
    if len(gammas) == 1:
        axes = [axes]
    for ax, gamma in zip(axes, gammas):
        rows = sub[(sub["gamma"] == gamma) & (sub["step"] == last_step)]
        if rows.empty:
            continue
        row = rows.iloc[0]
        xs = [row.get(f"pca_proj_c{c}_pc0", np.nan) for c in clients]
        ys = [row.get(f"pca_proj_c{c}_pc1", np.nan) for c in clients]
        color = _gamma_color(gamma)
        ax.scatter(xs, ys, s=64, color=color, edgecolor="white", linewidth=0.6, zorder=3)
        ax.axhline(0, color=GRIDLINE, linewidth=0.8, zorder=0)
        ax.axvline(0, color=GRIDLINE, linewidth=0.8, zorder=0)
        ax.set_title(f"gamma={gamma}", fontsize=10)
        _style_axes(ax)
    fig.suptitle(f"Honest clients projected on top-2 PCs (step {last_step}) -- {config_tag}")
    fig.text(0.5, 0.02, "PC1", ha="center", color=MUTED)
    axes[0].set_ylabel("PC2")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(os.path.join(out_dir, f"{config_tag}_pca_scatter.png"), dpi=150)
    plt.close(fig)


def plot_subset_repartition(df, config_tag, out_dir):
    sub = df[df["config_tag"] == config_tag]
    hist_cols = _matching_columns(sub, SUBSET_HIST_RE)
    if sub.empty or not hist_cols or "subset_hist_lo" not in sub.columns:
        return
    positions = sorted({int(SUBSET_HIST_RE.match(c).group(1)) for c in hist_cols})
    gammas = _gammas_sorted(sub)
    last_step = sub["step"].max()
    fig, axes = plt.subplots(1, len(gammas), figsize=(4 * len(gammas), 4), sharey=True)
    if len(gammas) == 1:
        axes = [axes]
    for ax, gamma in zip(axes, gammas):
        rows = sub[(sub["gamma"] == gamma) & (sub["step"] == last_step)]
        if rows.empty:
            continue
        row = rows.iloc[0]
        lo, hi, bins = row["subset_hist_lo"], row["subset_hist_hi"], int(row["subset_hist_bins"])
        edges = np.linspace(lo, hi, bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        for pos in positions:
            client_cols = [c for c in hist_cols if c.startswith(f"subset{pos}_client")]
            if not client_cols:
                continue
            client_id = SUBSET_HIST_RE.match(client_cols[0]).group(2)
            counts = [row[c] for c in sorted(client_cols, key=lambda c: int(SUBSET_HIST_RE.match(c).group(3)))]
            ax.step(centers, counts, where="mid", linewidth=1.6,
                    color=SUBSET_COLORS[pos % len(SUBSET_COLORS)], label=f"client {client_id}")
        ax.set_title(f"gamma={gamma}", fontsize=10)
        ax.set_yscale("log")
        _style_axes(ax)
    axes[0].set_ylabel("coordinate count (log)")
    fig.text(0.5, 0.02, "coordinate value", ha="center", color=MUTED)
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Subset coordinate-value repartition (step {last_step}) -- {config_tag}")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(os.path.join(out_dir, f"{config_tag}_subset_repartition.png"), dpi=150)
    plt.close(fig)


def plot_active_coord_mean(df, config_tag, out_dir):
    sub = df[df["config_tag"] == config_tag]
    mean_cols = _matching_columns(sub, ACTIVE_MEAN_RE)
    if sub.empty or not mean_cols:
        return
    thresholds = sorted({float(ACTIVE_MEAN_RE.match(c).group(1)) for c in mean_cols})
    last_step = sub["step"].max()
    sub = sub[sub["step"] == last_step]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for gamma in sorted(sub["gamma"].unique(), reverse=True):
        rows = sub[sub["gamma"] == gamma]
        if rows.empty:
            continue
        row = rows.iloc[0]
        ys = [row.get(f"active_mean_thr_{th:g}", np.nan) for th in thresholds]
        color = _gamma_color(gamma)
        ax.plot(thresholds, ys, color=color, linewidth=2, marker="o", markersize=6,
                label=f"gamma={gamma}")
    ax.set_xscale("symlog", linthresh=1e-6)
    ax.set_xlabel("active-coordinate threshold on |g|")
    ax.set_ylabel("mean |g| over active coordinates")
    ax.set_title(f"Active-coordinate mean magnitude (step {last_step}) -- {config_tag}")
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{config_tag}_active_coord_mean.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats-dir", default="analysis/gradient_structure/output",
                         help="Directory containing gradient_stats_combined.csv "
                              "(the --out-dir passed to collect_gradient_stats.py)")
    parser.add_argument("--out-dir", default=None,
                         help="Where to write PNGs (default: <stats-dir>/plots)")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.stats_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(args.stats_dir, "gradient_stats_combined.csv"))

    for config_tag in sorted(df["config_tag"].unique()):
        print(f"Plotting {config_tag}...")
        plot_norm_spread(df, config_tag, out_dir)
        plot_pca_effective_rank(df, config_tag, out_dir)
        plot_pca_scatter(df, config_tag, out_dir)
        plot_subset_repartition(df, config_tag, out_dir)
        plot_active_coord_mean(df, config_tag, out_dir)

    print(f"Plots written to {out_dir}")


if __name__ == "__main__":
    main()
