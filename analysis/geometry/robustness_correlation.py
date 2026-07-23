"""Cross gradient-geometry baselines (f=0, results/geometry_baseline/*) against
existing robustness sweeps (f=0..5, under attack) already on disk.

IMPORTANT data-availability caveat (see report): these big robustness sweeps
were originally run on a separate Docker host, and only a subset of the full
grid was imported into this local repo/results tree -- not the complete sweep
output. As a result, no single locally-available sweep matches our f=0
geometry baseline's exact hyperparameters (TrMean aggregator, SNN alpha=1.2,
CNN lr=0.15) with full f=0..5 coverage across all 3 attacks and all 3 models
simultaneously; only whatever combos were imported are usable here. The best
available *common* aggregator with complete (f, gamma, attack) coverage for
all 3 models among the imported data is GeometricMedian+NNM+ARC.
Per-model substitutions actually used (see ROBUSTNESS_SOURCES below):
  - SNN atan:   results/snn/robust_new_atan_sweep. Only alpha values
    {0.5,0.75,1.0,1.25,1.5,2.0,3.0} were imported for this aggregator (not
    1.2) -- alpha=1.25 used as the closest available, threshold=1.0, lr=0.1
    (matches). Only Optimal_ALittleIsEnough_neg1 was imported for this
    aggregator at this alpha -- single-attack score, not worst-of-3.
  - CNN ReLU:   results/cnn/weekend, lr=0.05 (not 0.15 -- the lr=0.15 import,
    robust_comparison_sweep, only covers f=0..2 for every aggregator). All 3
    attacks available -> worst-of-3 score.
  - CNN Tanh:   results/cnn/tanh_heatmap_sweep, lr=0.15 (exact match to the
    geometry baseline). All 3 attacks available -> worst-of-3 score.

Read-only on all existing result files; writes only into analysis/geometry/.
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import genfromtxt
from scipy import stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

GAMMAS = [1.0, 0.66, 0.33, 0.0]
F_VALUES = [0, 1, 2, 3, 4, 5]
EVAL_DELTA = 50
NB_STEPS = 500
NB_CHECKPOINTS = 1 + NB_STEPS // EVAL_DELTA  # 11
LATE_WINDOW_IDX = [8, 9, 10]  # checkpoints at steps 400, 450, 500
NB_SEEDS = 5
TR_SEED0, DD_SEED = 42, 42

ALIE = "Optimal_ALittleIsEnough_neg1"
SIGNFLIP = "SignFlipping"
IPM = "Optimal_InnerProductManipulation"

ROBUSTNESS_SOURCES = {
    "snn_atan12": dict(
        results_dir=os.path.join(REPO_ROOT, "results", "snn", "robust_new_atan_sweep"),
        attacks=[ALIE],  # only attack imported locally for this aggregator/alpha
        dirname=lambda f, gamma, attack: (
            f"mnist_cnn_mnist_snn_n_{10+f}_f_{f}_d_{f}_gamma_similarity_niid_{gamma}_"
            f"GeometricMedian_NNM_ARC_{attack}_lr_0.1_mom_0.9_wd_0.0001_ts_10_enc_constant_"
            f"beta_0.95_learn_threshold_False_surrogate_gradient_atan_alpha_1.25_threshold_1.0"
        ),
        note="alpha=1.25 (closest imported to our 1.2), single-attack (ALIE only)",
    ),
    "cnn_relu": dict(
        results_dir=os.path.join(REPO_ROOT, "results", "cnn", "weekend"),
        attacks=[ALIE, SIGNFLIP, IPM],
        dirname=lambda f, gamma, attack: (
            f"mnist_cnn_mnist_n_{10+f}_f_{f}_d_{f}_gamma_similarity_niid_{gamma}_"
            f"GeometricMedian_NNM_ARC_{attack}_lr_0.05_mom_0.9_wd_0.0001"
        ),
        note="lr=0.05 (not 0.15 -- the lr=0.15 import only covers f=0..2), worst-of-3 attacks",
    ),
    "cnn_tanh": dict(
        results_dir=os.path.join(REPO_ROOT, "results", "cnn", "tanh_heatmap_sweep"),
        attacks=[ALIE, SIGNFLIP, IPM],
        dirname=lambda f, gamma, attack: (
            f"mnist_cnn_mnist_tanh_n_{10+f}_f_{f}_d_{f}_gamma_similarity_niid_{gamma}_"
            f"GeometricMedian_NNM_ARC_{attack}_lr_0.15_mom_0.9_wd_0.0001"
        ),
        note="lr=0.15, exact match to the geometry baseline",
    ),
}


def load_late_window_accuracy(run_dir):
    """Mean test accuracy over steps {400,450,500}, averaged over the 5 training seeds."""
    vals = []
    for run in range(NB_SEEDS):
        path = os.path.join(run_dir, f"test_accuracy_tr_seed_{TR_SEED0+run}_dd_seed_{DD_SEED}.txt")
        arr = genfromtxt(path, delimiter=",")
        vals.append(arr[LATE_WINDOW_IDX].mean())
    return float(np.mean(vals))


def build_robustness_table():
    records = []
    missing = []
    for model_key, src in ROBUSTNESS_SOURCES.items():
        for gamma in GAMMAS:
            for f in F_VALUES:
                attack_accs = []
                for attack in src["attacks"]:
                    run_dir = os.path.join(src["results_dir"], src["dirname"](f, gamma, attack))
                    if not os.path.isdir(run_dir):
                        missing.append(run_dir)
                        continue
                    try:
                        attack_accs.append(load_late_window_accuracy(run_dir))
                    except Exception as e:
                        missing.append(f"{run_dir} ({e})")
                if not attack_accs:
                    continue
                worst_case_acc = min(attack_accs)  # worst-of-attacks, same convention as test_heatmap()
                records.append({"model": model_key, "gamma": gamma, "f": f, "accuracy": worst_case_acc})
    if missing:
        print(f"WARNING: {len(missing)} run dirs missing/failed to load, e.g.:")
        for m in missing[:5]:
            print(f"  {m}")
    return pd.DataFrame(records), missing


def plot_robustness_heatmaps(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (model_key, src) in zip(axes, ROBUSTNESS_SOURCES.items()):
        sub = df[df["model"] == model_key]
        pivot = sub.pivot(index="gamma", columns="f", values="accuracy").reindex(index=sorted(GAMMAS, reverse=True))
        im = ax.imshow(pivot.values, vmin=0, vmax=1, cmap="rocket_r" if False else "viridis_r", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v < 0.5 else "black", fontsize=9)
        ax.set_xlabel("f (Byzantine clients)")
        ax.set_ylabel("gamma")
        ax.set_title(f"{model_key}\n({src['note']})", fontsize=9)
    fig.suptitle("Robustness heatmaps: worst-case test accuracy (late-window, GeometricMedian+NNM+ARC)")
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "robustness_heatmaps.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_merged_table(robustness_df, geometry_summary_path):
    geo = pd.read_csv(geometry_summary_path)
    clean = robustness_df[robustness_df["f"] == 0][["model", "gamma", "accuracy"]].rename(columns={"accuracy": "acc_clean_f0"})
    attacked = robustness_df[robustness_df["f"] == 5][["model", "gamma", "accuracy"]].rename(columns={"accuracy": "acc_f5"})
    mean_attacked = (
        robustness_df[robustness_df["f"] > 0]
        .groupby(["model", "gamma"])["accuracy"].mean()
        .reset_index().rename(columns={"accuracy": "acc_f1to5_mean"})
    )
    merged = geo.merge(clean, on=["model", "gamma"], how="left")
    merged = merged.merge(attacked, on=["model", "gamma"], how="left")
    merged = merged.merge(mean_attacked, on=["model", "gamma"], how="left")
    merged["robustness_drop"] = merged["acc_clean_f0"] - merged["acc_f5"]
    out_path = os.path.join(OUT_DIR, "geometry_robustness_merged.csv")
    merged.to_csv(out_path, index=False)
    return merged, out_path


GEOMETRY_METRICS = ["N_median", "Q_median", "S_median", "S_layer0_median", "cos_mean_median"]
ROBUSTNESS_METRICS = ["acc_clean_f0", "acc_f5", "acc_f1to5_mean", "robustness_drop"]


def correlation_scatter(merged):
    fig, axes = plt.subplots(len(GEOMETRY_METRICS), len(ROBUSTNESS_METRICS), figsize=(4 * len(ROBUSTNESS_METRICS), 3.2 * len(GEOMETRY_METRICS)))
    colors = {"snn_atan12": "tab:blue", "cnn_relu": "tab:orange", "cnn_tanh": "tab:green"}
    corr_records = []

    for i, gcol in enumerate(GEOMETRY_METRICS):
        for j, rcol in enumerate(ROBUSTNESS_METRICS):
            ax = axes[i, j]
            sub = merged.dropna(subset=[gcol, rcol])
            for model_key in ROBUSTNESS_SOURCES:
                m = sub[sub["model"] == model_key]
                ax.scatter(m[gcol], m[rcol], color=colors[model_key], label=model_key, s=35)
            if len(sub) >= 3:
                r, p = stats.pearsonr(sub[gcol], sub[rcol])
            else:
                r, p = float("nan"), float("nan")
            corr_records.append({"geometry_metric": gcol, "robustness_metric": rcol, "pearson_r": r, "p_value": p, "n": len(sub)})
            ax.set_title(f"r={r:.2f}" + (f", p={p:.3f}" if not np.isnan(p) else ""), fontsize=9)
            if i == len(GEOMETRY_METRICS) - 1:
                ax.set_xlabel(gcol, fontsize=8)
            ax.set_ylabel(rcol, fontsize=8)
            ax.tick_params(labelsize=7)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Geometry (f=0 baseline) vs. robustness (attacked, f>0), n=12 (model x gamma) points", y=1.05)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "correlation_scatter.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    corr_df = pd.DataFrame(corr_records).sort_values("pearson_r", key=lambda s: s.abs(), ascending=False)
    corr_csv = os.path.join(OUT_DIR, "correlation_table.csv")
    corr_df.to_csv(corr_csv, index=False)
    return out_path, corr_df, corr_csv


if __name__ == "__main__":
    robustness_df, missing = build_robustness_table()
    robustness_df.to_csv(os.path.join(OUT_DIR, "robustness_table.csv"), index=False)
    heatmap_path = plot_robustness_heatmaps(robustness_df)

    geometry_summary_path = os.path.join(OUT_DIR, "summary_table.csv")
    merged, merged_path = build_merged_table(robustness_df, geometry_summary_path)

    scatter_path, corr_df, corr_csv = correlation_scatter(merged)

    print(f"\nWrote:\n  {heatmap_path}\n  {merged_path}\n  {scatter_path}\n  {corr_csv}")
    print(f"\n{len(missing)} missing run dirs (see warnings above)")
    print("\nTop correlations by |r|:")
    print(corr_df.head(10).to_string(index=False))
