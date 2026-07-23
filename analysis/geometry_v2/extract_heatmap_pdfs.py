"""Extract exact accuracy values from the already-rendered heatmap PDFs under
plots/, instead of asking for manual transcription. These PDFs are vector
graphics (matplotlib/seaborn `annot=True`), so every number on the grid is
real selectable text with a position -- `pdftotext -bbox` gives us that text
plus its bounding box, and the grid geometry (4 gamma rows x 6 f columns) is
fully determined by evaluate_results.test_heatmap()'s layout convention:
  - distribution_parameter_list = [1.0, 0.66, 0.33, 0.0] (declared order),
    reversed -> yticklabels = [0.0, 0.33, 0.66, 1.0], top row = 0.0.
  - f column order left-to-right = the declared f list, [0,1,2,3,4,5].
So: sort matched cell words by y ascending -> 4 row-groups of 6 -> gamma in
[0.0, 0.33, 0.66, 1.0] order; within each row sort by x ascending -> f in
[0,1,2,3,4,5] order.

No training, no GPU, read-only on plots/. pdftotext.exe (MiKTeX/poppler) must
be on disk; path is hardcoded below (same one used for compiling the LaTeX
reports in this repo).
"""
import os
import re
import subprocess

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PDFTOTEXT = r"C:\Users\7430\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdftotext.exe"

GAMMAS_TOP_TO_BOTTOM = [0.0, 0.33, 0.66, 1.0]
F_LEFT_TO_RIGHT = [0, 1, 2, 3, 4, 5]

WORD_RE = re.compile(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">([^<]+)</word>')
NUMERIC_RE = re.compile(r"^\d+\.\d+$")

AGGREGATORS = ["TrMean", "GeometricMedian", "MultiKrum", "CenteredClipping"]
ATTACKS = ["Optimal_ALittleIsEnough_neg1", "SignFlipping", "Optimal_InnerProductManipulation"]


def parse_heatmap_pdf(pdf_path):
    """Returns dict[(f, gamma)] -> accuracy, or raises if the grid doesn't
    parse to exactly 24 cells (fail loudly rather than silently mis-map)."""
    result = subprocess.run([PDFTOTEXT, "-bbox", pdf_path, "-"], capture_output=True, text=True, check=True)
    html = result.stdout

    page_m = re.search(r'<page width="([\d.]+)" height="([\d.]+)">', html)
    page_width = float(page_m.group(1))

    words = []
    for m in WORD_RE.finditer(html):
        xmin, ymin, xmax, ymax, text = m.groups()
        if NUMERIC_RE.match(text):
            xc = (float(xmin) + float(xmax)) / 2
            yc = (float(ymin) + float(ymax)) / 2
            words.append((xc, yc, float(text)))

    # Data grid sits away from the left row-labels and the right colorbar ticks.
    grid_words = [(x, y, v) for x, y, v in words if 0.08 * page_width < x < 0.80 * page_width]

    if len(grid_words) != 24:
        raise ValueError(f"expected 24 grid cells, got {len(grid_words)} in {pdf_path}")

    grid_words.sort(key=lambda t: t[1])  # by y ascending -> row groups
    result_map = {}
    for row_idx in range(4):
        row = sorted(grid_words[row_idx * 6:(row_idx + 1) * 6], key=lambda t: t[0])  # by x ascending -> f groups
        gamma = GAMMAS_TOP_TO_BOTTOM[row_idx]
        for col_idx, (x, y, v) in enumerate(row):
            f = F_LEFT_TO_RIGHT[col_idx]
            result_map[(f, gamma)] = v
    return result_map


def build_filename(model_key, agg, attack):
    if model_key == "snn_atan12":
        return (
            f"test_{attack}_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_{agg}_"
            f"nb_honest_clients_10_tolerated_f_equal_real.pdf"
        )
    elif model_key == "cnn_relu":
        return (
            f"test_{attack}_mnist_cnn_mnist_gamma_similarity_niid_NNM_ARC_{agg}_"
            f"nb_honest_clients_10_tolerated_f_equal_real.pdf"
        )
    elif model_key == "cnn_tanh":
        return (
            f"test_{attack}_mnist_cnn_mnist_tanh_gamma_similarity_niid_NNM_ARC_{agg}_"
            f"nb_honest_clients_10_tolerated_f_equal_real.pdf"
        )
    raise ValueError(model_key)


SOURCES = {
    "snn_atan12": os.path.join(REPO_ROOT, "plots", "robust_new_atan_sweep", "alpha_1.2"),
    "cnn_relu": os.path.join(REPO_ROOT, "plots", "robust_comparison_sweep", "learning_rate_0.15"),
    "cnn_tanh": os.path.join(REPO_ROOT, "plots", "cnn_tanh_heatmaps"),
}


if __name__ == "__main__":
    all_rows = []
    problems = []
    for model_key, plot_dir in SOURCES.items():
        for agg in AGGREGATORS:
            for attack in ATTACKS:
                fname = build_filename(model_key, agg, attack)
                fpath = os.path.join(plot_dir, fname)
                if not os.path.exists(fpath):
                    problems.append(f"MISSING FILE: {fpath}")
                    continue
                try:
                    grid = parse_heatmap_pdf(fpath)
                except Exception as e:
                    problems.append(f"PARSE ERROR ({model_key}, {agg}, {attack}): {e}")
                    continue
                for (f, gamma), acc in grid.items():
                    all_rows.append(dict(model=model_key, gamma=gamma, f=f, aggregator=agg, attack=attack, accuracy=acc))

    df = pd.DataFrame(all_rows)
    out_path = os.path.join(OUT_DIR, "robustness_table_extracted.csv")
    df.to_csv(out_path, index=False)

    print(f"Extracted {len(df)} rows (expected {3*4*3*24} = 864)")
    print(f"Wrote {out_path}")
    if problems:
        print(f"\n{len(problems)} PROBLEMS:")
        for p in problems:
            print(f"  {p}")
    else:
        print("No problems -- full grid extracted for both models.")

    # Sanity check: values must be in [0,1] and each (model,agg,attack) must have exactly 24 rows
    print("\nSanity check -- rows per (model, aggregator, attack):")
    counts = df.groupby(["model", "aggregator", "attack"]).size()
    bad = counts[counts != 24]
    if len(bad):
        print("  MISMATCHES:")
        print(bad)
    else:
        print("  all 24/24 OK")
    print(f"\naccuracy range: [{df['accuracy'].min():.3f}, {df['accuracy'].max():.3f}]")
