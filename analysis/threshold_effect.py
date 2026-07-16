"""Phase 4 analysis for the threshold-only sweep (theta in {1.0, 0.8, 0.6, 0.4},
alpha=1.2 fixed, restricted grid f in {0,3,5} x gamma in {1.0,0.66}).

Ingests every run directory under --sweep_root (default:
./results/snn/threshold_sweep, i.e. thr_10/thr_08/thr_06/thr_04 -- theta=1.0
is a real run in this sweep, not a reused stale baseline; see the threshold
sweep write-up for why the previously-existing "baseline" wasn't usable).

Each run directory's own config.json is read to identify (threshold,
aggregator, attack, f, gamma); accuracy is the final (last evaluation_delta
checkpoint) test accuracy, averaged over the 5 training seeds. If the Phase 2
instrumentation files are present (layer_firing_rate_*.csv,
client_vectors_*/step_*.npy) they are joined in and used for the fingerprint
plots; otherwise those plots are skipped with a message instead of failing.

Usage:
    python -m analysis.threshold_effect [--sweep_root ...] [--output_dir ...]
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SEED_RE = re.compile(r"tr_seed_(\d+)_dd_seed_(\d+)")


def _parse_seed(path):
    m = SEED_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else None


def discover_runs(sweep_root):
    """One row per (threshold, aggregator, attack, f, gamma, seed) with its
    final test accuracy, by reading each run directory's own config.json."""
    rows = []
    thr_dirs = sorted(glob.glob(os.path.join(sweep_root, "thr_*")))
    for thr_dir in thr_dirs:
        run_dirs = [d for d in glob.glob(os.path.join(thr_dir, "*")) if os.path.isdir(d)]
        for run_dir in run_dirs:
            config_path = os.path.join(run_dir, "config.json")
            if not os.path.exists(config_path):
                continue
            with open(config_path) as f:
                cfg = json.load(f)
            try:
                threshold = cfg["model"]["model_params"]["threshold"]
                aggregator = cfg["aggregator"]["name"]
                attack = cfg["attack"]["name"]
                f_val = cfg["benchmark_config"]["f"]
                gamma = cfg["benchmark_config"]["data_distribution"]["distribution_parameter"]
            except (KeyError, TypeError) as e:
                print(f"Warning: could not parse config at {config_path}: {e}")
                continue

            acc_files = sorted(glob.glob(os.path.join(run_dir, "test_accuracy_tr_seed_*_dd_seed_*.txt")))
            for acc_path in acc_files:
                seed = _parse_seed(acc_path)
                try:
                    arr = np.atleast_1d(np.genfromtxt(acc_path, delimiter=","))
                except Exception as e:
                    print(f"Warning: could not read {acc_path}: {e}")
                    continue
                if arr.size == 0 or np.all(np.isnan(arr)):
                    continue
                rows.append({
                    "threshold": threshold,
                    "aggregator": aggregator,
                    "attack": attack,
                    "f": f_val,
                    "gamma": gamma,
                    "seed": seed,
                    "final_test_accuracy": float(arr[-1]),
                    "run_dir": run_dir,
                })
    return pd.DataFrame(rows)


def load_firing_rate(run_dir):
    """Mean firing rate (across all logged layers, at the last logged step)
    per seed, from the Phase 2a layer_firing_rate CSV."""
    rows = []
    for csv_path in sorted(glob.glob(os.path.join(run_dir, "layer_firing_rate_tr_seed_*_dd_seed_*.csv"))):
        seed = _parse_seed(csv_path)
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if df.empty:
            continue
        layer_cols = [c for c in df.columns if c != "step"]
        if not layer_cols:
            continue
        mean_rate = float(df.iloc[-1][layer_cols].mean())
        rows.append({"seed": seed, "firing_rate": mean_rate})
    return rows


def load_sign_agreement(run_dir, agreement_threshold=0.8):
    """Mean fraction of coordinates with sign agreement A_k >= agreement_threshold
    among honest clients, averaged over available step snapshots, per seed.
    A_k = fraction of honest clients whose sign at coordinate k matches the
    majority sign at that coordinate. From the Phase 2b per-client vector dumps."""
    rows = []
    for vec_dir in sorted(glob.glob(os.path.join(run_dir, "client_vectors_tr_seed_*_dd_seed_*"))):
        seed = _parse_seed(vec_dir)
        step_files = sorted(glob.glob(os.path.join(vec_dir, "step_*.npy")))
        if not step_files:
            continue
        fracs = []
        for step_path in step_files:
            vectors = np.load(step_path)  # (nb_honest_clients, d)
            if vectors.shape[0] < 2:
                continue
            signs = np.sign(vectors)
            majority_sign = np.sign(signs.sum(axis=0))
            agreement = (signs == majority_sign).mean(axis=0)  # A_k per coordinate
            fracs.append(float((agreement >= agreement_threshold).mean()))
        if fracs:
            rows.append({"seed": seed, "sign_agreement_frac": float(np.mean(fracs))})
    return rows


def attach_instrumentation(df):
    fr_rows, sa_rows = [], []
    for run_dir in df["run_dir"].unique():
        for r in load_firing_rate(run_dir):
            r["run_dir"] = run_dir
            fr_rows.append(r)
        for r in load_sign_agreement(run_dir):
            r["run_dir"] = run_dir
            sa_rows.append(r)

    if fr_rows:
        df = df.merge(pd.DataFrame(fr_rows), on=["run_dir", "seed"], how="left")
    else:
        df["firing_rate"] = np.nan

    if sa_rows:
        df = df.merge(pd.DataFrame(sa_rows), on=["run_dir", "seed"], how="left")
    else:
        df["sign_agreement_frac"] = np.nan

    return df


def plot_accuracy_vs_theta(df, output_dir):
    thresholds = sorted(df["threshold"].unique())
    aggregators = sorted(df["aggregator"].unique())
    attacks = sorted(df["attack"].unique())

    for f_val in [3, 5]:
        sub = df[(df["gamma"] == 0.66) & (df["f"] == f_val)]
        if sub.empty:
            print(f"No data for gamma=0.66, f={f_val}; skipping accuracy-vs-theta plot.")
            continue

        fig, axes = plt.subplots(
            len(aggregators), len(attacks),
            figsize=(4 * len(attacks), 3 * len(aggregators)),
            squeeze=False, sharey=True,
        )
        for i, agg in enumerate(aggregators):
            for j, atk in enumerate(attacks):
                ax = axes[i][j]
                cell = sub[(sub["aggregator"] == agg) & (sub["attack"] == atk)]
                xs, means, stds = [], [], []
                for thr in thresholds:
                    vals = cell[cell["threshold"] == thr]["final_test_accuracy"].values
                    if len(vals) == 0:
                        continue
                    xs.append(thr)
                    means.append(vals.mean())
                    stds.append(vals.std())
                ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3)
                ax.set_title(f"{agg} / {atk}", fontsize=9)
                ax.set_xlabel("theta")
                ax.set_ylim(0, 1)
                if j == 0:
                    ax.set_ylabel("test accuracy")

        fig.suptitle(f"Accuracy vs theta (gamma=0.66, f={f_val}), error bars = std over seeds")
        fig.tight_layout()
        out_path = os.path.join(output_dir, f"accuracy_vs_theta_gamma0.66_f{f_val}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_path}")


def plot_heatmap_strips(df, output_dir, gamma=0.66):
    thresholds = sorted(df["threshold"].unique())
    f_vals = sorted(df["f"].unique())

    for thr in thresholds:
        sub = df[(df["threshold"] == thr) & (df["gamma"] == gamma)]
        row = np.full(len(f_vals), np.nan)
        for idx, f_val in enumerate(f_vals):
            cell = sub[sub["f"] == f_val]
            if cell.empty:
                continue
            per_agg_worst = []
            for agg in cell["aggregator"].unique():
                agg_cell = cell[cell["aggregator"] == agg]
                worst = agg_cell.groupby("attack")["final_test_accuracy"].mean().min()
                per_agg_worst.append(worst)
            if per_agg_worst:
                row[idx] = max(per_agg_worst)

        fig, ax = plt.subplots(figsize=(1.5 * len(f_vals) + 1, 2))
        sns.heatmap(
            row.reshape(1, -1), annot=True, fmt=".2f", vmin=0.0, vmax=1.0,
            xticklabels=[str(f) for f in f_vals], yticklabels=[f"gamma={gamma}"],
            ax=ax, cmap="viridis",
        )
        ax.set_title(f"theta={thr}: best-aggregator worst-case-attack accuracy")
        ax.set_xlabel("f")
        fig.tight_layout()
        out_path = os.path.join(output_dir, f"heatmap_strip_theta_{thr}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_path}")


def plot_fingerprints(df, output_dir):
    has_fr = "firing_rate" in df.columns and df["firing_rate"].notna().any()
    has_sa = "sign_agreement_frac" in df.columns and df["sign_agreement_frac"].notna().any()

    if not has_fr and not has_sa:
        print("No Phase 2 instrumentation data found (layer_firing_rate_*.csv / "
              "client_vectors_*/step_*.npy) under any run directory -- skipping "
              "firing-rate / sign-agreement fingerprint plots. These require runs "
              "executed with the updated client.py/train.py instrumentation.")
        return

    thresholds = sorted(df["threshold"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {thr: cmap(i / max(1, len(thresholds) - 1)) for i, thr in enumerate(thresholds)}

    if has_fr:
        fig, ax = plt.subplots(figsize=(6, 5))
        for thr in thresholds:
            sub = df[(df["threshold"] == thr) & df["firing_rate"].notna()]
            if sub.empty:
                continue
            ax.scatter(sub["firing_rate"], sub["final_test_accuracy"],
                       label=f"theta={thr}", color=colors[thr], alpha=0.7)
        ax.set_xlabel("mean firing rate")
        ax.set_ylabel("test accuracy")
        ax.legend()
        fig.tight_layout()
        out_path = os.path.join(output_dir, "accuracy_vs_firing_rate.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_path}")
    else:
        print("No firing-rate data found; skipping accuracy-vs-firing-rate plot.")

    if has_sa:
        fig, ax = plt.subplots(figsize=(6, 5))
        for thr in thresholds:
            sub = df[(df["threshold"] == thr) & df["sign_agreement_frac"].notna()]
            if sub.empty:
                continue
            ax.scatter(sub["sign_agreement_frac"], sub["final_test_accuracy"],
                       label=f"theta={thr}", color=colors[thr], alpha=0.7)
        ax.set_xlabel("fraction of coordinates with sign agreement >= 0.8")
        ax.set_ylabel("test accuracy")
        ax.legend()
        fig.tight_layout()
        out_path = os.path.join(output_dir, "accuracy_vs_sign_agreement.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_path}")
    else:
        print("No per-client vector dumps found; skipping accuracy-vs-sign-agreement plot.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep_root", type=str, default="./results/snn/threshold_sweep")
    parser.add_argument("--output_dir", type=str, default="./results/snn/threshold_sweep/analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = discover_runs(args.sweep_root)
    if df.empty:
        print(f"No runs found under {args.sweep_root}. Nothing to analyze.")
        return

    df = attach_instrumentation(df)

    csv_path = os.path.join(args.output_dir, "threshold_sweep_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote full results table: {csv_path} ({len(df)} rows)")

    plot_accuracy_vs_theta(df, args.output_dir)
    plot_heatmap_strips(df, args.output_dir)
    plot_fingerprints(df, args.output_dir)


if __name__ == "__main__":
    main()
