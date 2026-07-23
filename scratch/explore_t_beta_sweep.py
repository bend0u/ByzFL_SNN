import glob
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ROOT = "results/snn_t_beta_sweep_server"
SEED_RE = re.compile(r"tr_seed_(\d+)_dd_seed_(\d+)")


def parse_seed(path):
    m = SEED_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else None


rows = []
bad_configs = []
run_dirs = [d for d in glob.glob(os.path.join(ROOT, "*")) if os.path.isdir(d)]
print(f"Total run directories: {len(run_dirs)}")

for run_dir in run_dirs:
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg_path):
        bad_configs.append((run_dir, "no config.json"))
        continue
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except Exception as e:
        bad_configs.append((run_dir, f"json error: {e}"))
        continue

    try:
        mp = cfg["model"]["model_params"]
        threshold = mp["threshold"]
        beta = mp["beta"]
        alpha = mp["surrogate_params"]["alpha"]
        surrogate = mp["surrogate_gradient"]
        learn_threshold = mp.get("learn_threshold")
        T = cfg["model"]["encoding"]["time_steps"]
        agg = cfg["aggregator"]["name"]
        attack = cfg["attack"]["name"]
        f_val = cfg["benchmark_config"]["f"]
        gamma = cfg["benchmark_config"]["data_distribution"]["distribution_parameter"]
        nb_steps = cfg["benchmark_config"]["nb_steps"]
        eval_delta = cfg["evaluation_and_results"]["evaluation_delta"]
        lr = cfg["model"]["learning_rate"]
    except (KeyError, TypeError) as e:
        bad_configs.append((run_dir, f"missing key: {e}"))
        continue

    acc_files = sorted(glob.glob(os.path.join(run_dir, "test_accuracy_tr_seed_*_dd_seed_*.txt")))
    seeds_present = []
    for acc_path in acc_files:
        seed = parse_seed(acc_path)
        try:
            arr = np.atleast_1d(np.genfromtxt(acc_path, delimiter=","))
        except Exception:
            continue
        if arr.size == 0 or np.all(np.isnan(arr)):
            continue
        seeds_present.append(seed)
        rows.append({
            "T": T, "beta": beta, "alpha": alpha, "threshold": threshold,
            "surrogate": surrogate, "learn_threshold": learn_threshold,
            "aggregator": agg, "attack": attack, "f": f_val, "gamma": gamma,
            "seed": seed, "nb_steps": nb_steps, "eval_delta": eval_delta, "lr": lr,
            "n_eval_points": arr.size,
            "final_test_accuracy": float(arr[-1]),
            "max_test_accuracy": float(np.nanmax(arr)),
            "run_dir": run_dir,
        })

df = pd.DataFrame(rows)
print(f"Total (run, seed) rows loaded: {len(df)}")
print(f"Bad/unparseable dirs: {len(bad_configs)}")
for d, reason in bad_configs[:10]:
    print("  ", d, "->", reason)

print("\n=== distinct values per axis ===")
for col in ["T", "beta", "alpha", "threshold", "surrogate", "learn_threshold",
            "aggregator", "attack", "f", "gamma", "nb_steps", "eval_delta", "lr"]:
    print(f"{col}: {sorted(df[col].unique(), key=str)}")

print("\n=== seeds per (T,beta,agg,attack,f,gamma) combo: distribution ===")
combo_cols = ["T", "beta", "aggregator", "attack", "f", "gamma"]
seed_counts = df.groupby(combo_cols)["seed"].nunique()
print(seed_counts.value_counts().sort_index())
print(f"Total distinct combos: {seed_counts.shape[0]}")

full_grid = 3 * 2 * 4 * 3 * 5 * 2  # T x beta x agg x attack x f x gamma
print(f"Full theoretical grid (T x beta x agg x attack x f x gamma): {full_grid}")

print("\n=== n_eval_points distribution (expect 11 = 1+500/50 if complete) ===")
print(df["n_eval_points"].value_counts().sort_index())

print("\n=== incomplete runs (n_eval_points < 11) ===")
incomplete = df[df["n_eval_points"] < 11]
print(f"count: {len(incomplete)}")
if len(incomplete) > 0:
    print(incomplete[combo_cols + ["seed", "n_eval_points"]].head(20))

df.to_csv("scratch/t_beta_sweep_table.csv", index=False)
print("\nWrote scratch/t_beta_sweep_table.csv")
