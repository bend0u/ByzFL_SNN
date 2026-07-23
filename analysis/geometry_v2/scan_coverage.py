"""STEP 0 (read-only): scan candidate robustness-sweep directories and report
exactly which (aggregator, attack, f, gamma, lr/alpha) combinations exist on
disk, by parsing directory names only (no file reads, no training, no GPU).
"""
import os
import re
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CANDIDATES = {
    "snn_atan12 (target alpha=1.2)": os.path.join(REPO_ROOT, "results", "snn", "robust_new_atan_sweep"),
    "cnn_relu (target lr=0.15)": os.path.join(REPO_ROOT, "results", "cnn", "robust_comparison_sweep"),
    "cnn_tanh (target lr=0.15)": os.path.join(REPO_ROOT, "results", "cnn", "tanh_heatmap_sweep"),
}

AGGREGATORS = ["CenteredClipping", "GeometricMedian", "MultiKrum", "TrMean"]
ATTACKS = ["Optimal_ALittleIsEnough_neg1", "SignFlipping", "Optimal_InnerProductManipulation"]
GAMMAS = ["1.0", "0.66", "0.33", "0.0"]
F_VALUES = list(range(6))

DIRNAME_RE = re.compile(
    r"_n_(\d+)_f_(\d+)_d_(\d+)_gamma_similarity_niid_([0-9.]+)_"
    r"(CenteredClipping|GeometricMedian|MultiKrum|TrMean)_NNM_ARC_"
    r"(Optimal_ALittleIsEnough_neg1|SignFlipping|Optimal_InnerProductManipulation)_"
    r"lr_([0-9.]+)_mom_([0-9.]+)_wd_([0-9.]+)(.*)"
)
ALPHA_RE = re.compile(r"alpha_([0-9.]+)_threshold")


def scan(results_dir):
    if not os.path.isdir(results_dir):
        return None, "directory does not exist"
    entries = os.listdir(results_dir)
    records = []
    for name in entries:
        full = os.path.join(results_dir, name)
        if not os.path.isdir(full):
            continue
        m = DIRNAME_RE.search(name)
        if not m:
            continue
        n, f, d, gamma, agg, attack, lr, mom, wd, rest = m.groups()
        alpha_m = ALPHA_RE.search(rest)
        alpha = alpha_m.group(1) if alpha_m else None
        records.append(dict(f=int(f), gamma=gamma, aggregator=agg, attack=attack, lr=lr, alpha=alpha))
    return records, None


def summarize(model_label, records):
    print(f"\n{'='*100}\n{model_label}\n{'='*100}")
    if records is None:
        print("  MISSING DIRECTORY")
        return

    lr_values = sorted(set(r["lr"] for r in records))
    alpha_values = sorted(set(r["alpha"] for r in records if r["alpha"] is not None))
    agg_values = sorted(set(r["aggregator"] for r in records))
    print(f"  total matched dirs: {len(records)}")
    print(f"  lr values present: {lr_values}")
    if alpha_values:
        print(f"  alpha values present: {alpha_values}")
    print(f"  aggregators present: {agg_values}")

    # For each lr (and alpha, if applicable), build coverage table: aggregator x attack -> set of (f,gamma) present
    key_field = "alpha" if alpha_values else "lr"
    key_values = alpha_values if alpha_values else lr_values

    for key_val in key_values:
        sub = [r for r in records if r[key_field] == key_val]
        print(f"\n  --- {key_field}={key_val} ---")
        for agg in AGGREGATORS:
            for attack in ATTACKS:
                cells = set((r["f"], r["gamma"]) for r in sub if r["aggregator"] == agg and r["attack"] == attack)
                expected = set((f, g) for f in F_VALUES for g in GAMMAS)
                missing = expected - cells
                status = "COMPLETE" if not missing else f"MISSING {len(missing)}/{len(expected)}"
                if cells:
                    fs_present = sorted(set(f for f, g in cells))
                    print(f"    {agg:18s} x {attack:32s}: {len(cells):3d}/24 cells -> {status:14s} f_present={fs_present}")


if __name__ == "__main__":
    for label, path in CANDIDATES.items():
        records, err = scan(path)
        if err:
            print(f"\n{'='*100}\n{label}\n{'='*100}\n  {err}: {path}")
            continue
        summarize(f"{label}\n  path: {path}", records)
