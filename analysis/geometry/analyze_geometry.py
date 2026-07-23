"""Gradient-geometry baseline analysis (f=0, unattacked) for the SNN/ReLU/Tanh
correlation study.

Reads the online metrics_geometry_*.csv files already produced by the training
hook under results/geometry_baseline/{snn_atan12,cnn_relu,cnn_tanh}/ (read-only,
nothing is recomputed from raw vectors -- N/Q/S/cos_mean/per-layer columns are
used exactly as logged). Produces:
  - analysis/geometry/trajectories_N_Q_S.png : N(step), Q(step), S(step) per (model, gamma)
  - analysis/geometry/summary_table.csv       : median/q25/q75 over steps in [100,400]
  - analysis/geometry/summary_table.png       : rendered table
  - analysis/geometry/predictions_check.txt   : the 4 registered predictions, checked
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_ROOT = os.path.join(REPO_ROOT, "results", "geometry_baseline")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "snn_atan12": {"label": "SNN (atan, alpha=1.2)", "layer0": "conv1", "layers": ["conv1", "conv2", "fc1", "fc2"]},
    "cnn_relu": {"label": "CNN ReLU", "layer0": "_c1", "layers": ["_c1", "_c2", "_f1", "_f2"]},
    "cnn_tanh": {"label": "CNN Tanh", "layer0": "_c1", "layers": ["_c1", "_c2", "_f1", "_f2"]},
}
GAMMAS = [1.0, 0.66, 0.33, 0.0]
WINDOW = (100, 400)  # steps considered "established regime" for the summary stats


def find_run_dir(model_key, gamma):
    pattern = os.path.join(RESULTS_ROOT, model_key, f"*gamma_similarity_niid_{gamma}_*")
    matches = glob.glob(pattern)
    assert len(matches) == 1, f"expected exactly 1 dir for {model_key} gamma={gamma}, got {matches}"
    return matches[0]


def load_all():
    rows = []
    for model_key, meta in MODELS.items():
        for gamma in GAMMAS:
            run_dir = find_run_dir(model_key, gamma)
            csv_path = glob.glob(os.path.join(run_dir, "metrics_geometry_*.csv"))[0]
            df = pd.read_csv(csv_path)
            df["model"] = model_key
            df["gamma"] = gamma
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


def plot_trajectories(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [("dispersion_N", "N (dispersion)"), ("consensus_Q", "Q (consensus, raw)"), ("consensus_S", "S (consensus, signal-weighted)")]

    colors = {"snn_atan12": "tab:blue", "cnn_relu": "tab:orange", "cnn_tanh": "tab:green"}
    alphas = {1.0: 1.0, 0.66: 0.75, 0.33: 0.5, 0.0: 0.3}
    styles = {1.0: "-", 0.66: "--", 0.33: "-.", 0.0: ":"}

    for ax, (col, title) in zip(axes, metrics):
        for model_key, meta in MODELS.items():
            for gamma in GAMMAS:
                sub = df[(df["model"] == model_key) & (df["gamma"] == gamma)].sort_values("step")
                ax.plot(
                    sub["step"], sub[col],
                    color=colors[model_key], linestyle=styles[gamma], alpha=alphas[gamma],
                    linewidth=1.6,
                    label=f"{meta['label']}, gamma={gamma}",
                )
        ax.set_xlabel("step")
        ax.set_title(title)
        ax.axvspan(WINDOW[0], WINDOW[1], color="gray", alpha=0.08, zorder=0)
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Gradient-geometry trajectories, f=0, TrMean+NNM+ARC (gray band = summary window [100,400])")
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "trajectories_N_Q_S.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_trajectories_by_gamma(df):
    """One figure per gamma, so the 3 models can be compared directly at fixed gamma."""
    metrics = [("dispersion_N", "N (dispersion)"), ("consensus_Q", "Q (consensus, raw)"), ("consensus_S", "S (consensus, signal-weighted)")]
    colors = {"snn_atan12": "tab:blue", "cnn_relu": "tab:orange", "cnn_tanh": "tab:green"}

    out_paths = []
    for gamma in GAMMAS:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, (col, title) in zip(axes, metrics):
            for model_key, meta in MODELS.items():
                sub = df[(df["model"] == model_key) & (df["gamma"] == gamma)].sort_values("step")
                ax.plot(sub["step"], sub[col], color=colors[model_key], linewidth=1.8, label=meta["label"])
            ax.set_xlabel("step")
            ax.set_title(title)
            ax.axvspan(WINDOW[0], WINDOW[1], color="gray", alpha=0.08, zorder=0)
            ax.grid(alpha=0.25)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.05))
        fig.suptitle(f"Gradient-geometry trajectories at gamma={gamma}, f=0, TrMean+NNM+ARC (gray band = summary window [100,400])")
        fig.tight_layout()
        gamma_tag = str(gamma).replace(".", "")
        out_path = os.path.join(OUT_DIR, f"trajectories_gamma_{gamma_tag}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(out_path)
    return out_paths


def build_summary_table(df):
    mask = (df["step"] >= WINDOW[0]) & (df["step"] <= WINDOW[1])
    windowed = df[mask]
    records = []
    for model_key, meta in MODELS.items():
        layer0 = meta["layer0"]
        for gamma in GAMMAS:
            sub = windowed[(windowed["model"] == model_key) & (windowed["gamma"] == gamma)]
            rec = {"model": model_key, "gamma": gamma}
            for col, name in [
                ("dispersion_N", "N"), ("consensus_Q", "Q"), ("consensus_S", "S"),
                (f"S_{layer0}", "S_layer0"), ("cos_mean", "cos_mean"),
            ]:
                rec[f"{name}_median"] = sub[col].median()
                rec[f"{name}_q25"] = sub[col].quantile(0.25)
                rec[f"{name}_q75"] = sub[col].quantile(0.75)
            # mean S over layers other than layer0 ("deep" layers), for the layer-1 prediction
            deep_layers = [l for l in meta["layers"] if l != layer0]
            deep_s_cols = [f"S_{l}" for l in deep_layers]
            rec["S_deep_mean_median"] = sub[deep_s_cols].mean(axis=1).median()
            records.append(rec)
    table = pd.DataFrame(records)
    csv_path = os.path.join(OUT_DIR, "summary_table.csv")
    table.to_csv(csv_path, index=False)

    # rendered version
    display_cols = ["model", "gamma", "N_median", "Q_median", "S_median", "S_layer0_median", "S_deep_mean_median", "cos_mean_median"]
    disp = table[display_cols].copy()
    for c in display_cols[2:]:
        disp[c] = disp[c].map(lambda v: f"{v:.4f}")
    fig, ax = plt.subplots(figsize=(12, 0.4 * len(disp) + 1.2))
    ax.axis("off")
    tbl = ax.table(cellText=disp.values, colLabels=display_cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    fig.suptitle(f"Summary (median over steps in {WINDOW})")
    png_path = os.path.join(OUT_DIR, "summary_table.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return table, csv_path, png_path


def check_predictions(table):
    lines = []

    def add(text):
        lines.append(text)
        print(text)

    add("=== Prediction checks (median over steps [100,400]) ===\n")

    piv_S = table.pivot(index="gamma", columns="model", values="S_median")
    add("-- Prediction: S_SNN < S_ReLU and S_SNN < S_Tanh at every gamma --")
    for gamma in GAMMAS:
        s_snn, s_relu, s_tanh = piv_S.loc[gamma, "snn_atan12"], piv_S.loc[gamma, "cnn_relu"], piv_S.loc[gamma, "cnn_tanh"]
        ok = (s_snn < s_relu) and (s_snn < s_tanh)
        add(f"  gamma={gamma}: S_SNN={s_snn:.4f}, S_ReLU={s_relu:.4f}, S_Tanh={s_tanh:.4f} -> {'HOLDS' if ok else 'VIOLATED'}")

    piv_N = table.pivot(index="gamma", columns="model", values="N_median")
    add("\n-- Prediction: N_ReLU >> N_SNN ~ N_Tanh (ReLU alone is high-N) --")
    for gamma in GAMMAS:
        n_snn, n_relu, n_tanh = piv_N.loc[gamma, "snn_atan12"], piv_N.loc[gamma, "cnn_relu"], piv_N.loc[gamma, "cnn_tanh"]
        ok = n_relu > n_snn and n_relu > n_tanh
        add(f"  gamma={gamma}: N_SNN={n_snn:.4f}, N_ReLU={n_relu:.4f}, N_Tanh={n_tanh:.4f} -> {'HOLDS' if ok else 'VIOLATED'} (ReLU highest)")

    add("\n-- Prediction: S decreases as gamma decreases (1.0 -> 0.0), for all models --")
    for model_key in MODELS:
        s_by_gamma = table[table["model"] == model_key].set_index("gamma")["S_median"].loc[GAMMAS]
        monotone = all(s_by_gamma.iloc[i] >= s_by_gamma.iloc[i + 1] for i in range(len(GAMMAS) - 1))
        add(f"  {model_key}: S(gamma) = {[round(v,4) for v in s_by_gamma.tolist()]} -> {'MONOTONE NON-INCREASING' if monotone else 'NOT MONOTONE'}")

    add("\n-- Prediction: S_layer0(SNN) >> S_deep_mean(SNN) --")
    snn_rows = table[table["model"] == "snn_atan12"]
    for _, row in snn_rows.iterrows():
        ok = row["S_layer0_median"] > row["S_deep_mean_median"]
        add(f"  gamma={row['gamma']}: S_layer0={row['S_layer0_median']:.4f}, S_deep_mean={row['S_deep_mean_median']:.4f} -> {'HOLDS' if ok else 'VIOLATED'}")

    txt_path = os.path.join(OUT_DIR, "predictions_check.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return txt_path


if __name__ == "__main__":
    df = load_all()
    traj_path = plot_trajectories(df)
    per_gamma_paths = plot_trajectories_by_gamma(df)
    table, table_csv, table_png = build_summary_table(df)
    pred_path = check_predictions(table)
    per_gamma_list = "\n  ".join(per_gamma_paths)
    print(f"\nWrote:\n  {traj_path}\n  {per_gamma_list}\n  {table_csv}\n  {table_png}\n  {pred_path}")
