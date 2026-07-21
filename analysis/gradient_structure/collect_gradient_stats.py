#!/usr/bin/env python3
"""Collect gradient-structure statistics already computed during training.

The metrics themselves (PCA effective rank, active-coordinate magnitude,
support overlap, subset coordinate histograms, per-client norms, honest-honest
cosine similarity) are computed online, in-process, by
byzfl/utils/gradient_structure_metrics.py and byzfl/utils/gradient_geometry.py
(gated behind the store_gradient_structure_metrics opt-in flag) -- no raw
per-client vectors are ever written to disk. This script is purely a
read-only joiner: for each run folder under a results directory, it reads
  - config.json (gamma, f, model name)
  - metrics_geometry_tr_seed_*_dd_seed_*.csv (per-client norms, cosine)
  - metrics_gradient_structure_tr_seed_*_dd_seed_*.csv (PCA, active-coord,
    jaccard, subset histograms)
joins them on step, tags each row with config_tag/gamma/model, and writes one
combined tidy CSV for plot_gradient_stats.py.
"""
import argparse
import glob
import json
import os
import re

import pandas as pd

SEED_RE = re.compile(r"_tr_seed_(\d+)_dd_seed_(\d+)\.csv$")


def _folder_config(folder):
    config_path = os.path.join(folder, "config.json")
    if not os.path.isfile(config_path):
        return None
    with open(config_path) as f:
        return json.load(f)


def _gamma_of(config):
    dist = config["benchmark_config"]["data_distribution"]
    if isinstance(dist, list):
        dist = dist[0]
    return dist.get("distribution_parameter")


def process_results_dir(results_dir, config_tag):
    rows = []
    combo_folders = sorted(
        p for p in glob.glob(os.path.join(results_dir, "*")) if os.path.isdir(p)
    )
    for folder in combo_folders:
        config = _folder_config(folder)
        if config is None:
            continue
        gamma = _gamma_of(config)
        f_val = config["benchmark_config"]["f"]
        model_name = config["model"]["name"]
        is_snn = config["model"].get("is_snn", False)

        gs_files = glob.glob(os.path.join(folder, "metrics_gradient_structure_tr_seed_*_dd_seed_*.csv"))
        for gs_file in gs_files:
            m = SEED_RE.search(gs_file)
            if not m:
                continue
            training_seed, dd_seed = m.group(1), m.group(2)
            geometry_file = os.path.join(
                folder, f"metrics_geometry_tr_seed_{training_seed}_dd_seed_{dd_seed}.csv"
            )
            if not os.path.isfile(geometry_file):
                print(f"  [WARN] no matching {geometry_file}, skipping norms/cosine for {gs_file}")
                gs_df = pd.read_csv(gs_file)
            else:
                gs_df = pd.read_csv(gs_file)
                geom_df = pd.read_csv(geometry_file)
                gs_df = gs_df.merge(geom_df, on="step", how="left", suffixes=("", "_geom"))

            tag_columns = pd.DataFrame({
                "config_tag": [config_tag] * len(gs_df),
                "model_name": model_name,
                "is_snn": is_snn,
                "gamma": gamma,
                "f": f_val,
                "training_seed": training_seed,
                "dd_seed": dd_seed,
            })
            rows.append(pd.concat([gs_df, tag_columns], axis=1))

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", action="append", required=True,
                         help="Results directory for one config (repeatable, paired with --config-tag)")
    parser.add_argument("--config-tag", action="append", required=True,
                         help="Short label for this results dir, e.g. snn_atan12 (repeatable)")
    parser.add_argument("--out-dir", default="analysis/gradient_structure/output",
                         help="Where to write the combined CSV")
    args = parser.parse_args()

    if len(args.results_dir) != len(args.config_tag):
        parser.error("--results-dir and --config-tag must be repeated the same number of times")

    os.makedirs(args.out_dir, exist_ok=True)

    all_dfs = []
    for results_dir, config_tag in zip(args.results_dir, args.config_tag):
        print(f"Processing {config_tag}: {results_dir}")
        df = process_results_dir(results_dir, config_tag)
        print(f"  {len(df)} snapshot row(s) found")
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    if combined.empty:
        print("No metrics_gradient_structure_*.csv found -- did you run with "
              "store_gradient_structure_metrics: true?")
        return

    out_path = os.path.join(args.out_dir, "gradient_stats_combined.csv")
    combined.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
