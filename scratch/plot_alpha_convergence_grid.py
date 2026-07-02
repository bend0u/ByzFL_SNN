"""
Generate individual convergence plots per (f, gamma) panel, then compose
them in a LaTeX document.

Two types of plots per panel:
  - Test Accuracy: SNN alphas (1.0, 1.25, 1.5, 2.0, 3.0) + CNN baseline
  - Train Loss: SNN alphas only (CNN loss is not comparable to SNN rate-coded loss)

Each panel is saved as an individual PDF file. A LaTeX .tex file assembles
them in a 3x3 grid layout.
"""

import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ─── Configuration ────────────────────────────────────────────────────────────

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SNN_RESULTS_DIR = os.path.join(WORKSPACE, "results", "snn", "robust_new_atan_sweep")
CNN_RESULTS_DIR = os.path.join(WORKSPACE, "results", "cnn", "weekend")
OUTPUT_DIR = os.path.join(WORKSPACE, "plots", "snn", "alpha_convergence_grid")
LATEX_DIR = os.path.join(WORKSPACE, "latex_plots")

ALPHAS = [1.0, 1.25, 1.5, 2.0, 3.0]
NB_HONEST = 10
EVALUATION_DELTA = 50
NB_STEPS = 500
TRAINING_SEEDS = [42, 43, 44, 45, 46]
DD_SEED = 42
AGGREGATOR = "GeometricMedian"

# Grid layout: rows = f values, cols = gamma values
F_VALUES = [0, 3, 5]
GAMMA_VALUES = [1.0, 0.33, 0.0]

# ─── Styling ──────────────────────────────────────────────────────────────────

ALPHA_COLORS = {
    1.0:  "#2196F3",   # blue
    1.25: "#4CAF50",   # green
    1.5:  "#FF9800",   # orange
    2.0:  "#E91E63",   # pink/red
    3.0:  "#9C27B0",   # purple
}
CNN_COLOR = "#212121"
CNN_LINESTYLE = "--"

# Theoretical minimum loss values
# SNN: log(1 + (C-1)*e^{-1}) where C=10 classes, because spikes are binary {0,1}
SNN_LOSS_MIN = np.log(1 + 9 * np.exp(-1))  # ≈ 1.4602
# CNN: NLLLoss with unbounded logits → minimum is 0
CNN_LOSS_MIN = 0.0

# ─── Path helpers ─────────────────────────────────────────────────────────────

def snn_dir_name(f, gamma, alpha):
    n = NB_HONEST + f
    d = f
    return (
        f"mnist_cnn_mnist_snn_n_{n}_f_{f}_d_{d}_gamma_similarity_niid_{gamma}_"
        f"{AGGREGATOR}_NNM_ARC_Optimal_ALittleIsEnough_neg1_"
        f"lr_0.1_mom_0.9_wd_0.0001_ts_10_enc_constant_beta_0.95_"
        f"learn_threshold_False_surrogate_gradient_atan_alpha_{alpha}_threshold_1.0"
    )


def cnn_dir_name(f, gamma):
    n = NB_HONEST + f
    d = f
    return (
        f"mnist_cnn_mnist_n_{n}_f_{f}_d_{d}_gamma_similarity_niid_{gamma}_"
        f"{AGGREGATOR}_NNM_ARC_Optimal_ALittleIsEnough_neg1_"
        f"lr_0.05_mom_0.9_wd_0.0001"
    )


# ─── Data loading ────────────────────────────────────────────────────────────

def load_metric_curves(results_base, dir_name, metric_prefix):
    """
    Load time-series metric files for all seeds.
    Returns (mean_array, std_array, steps_array) or (None, None, None).
    """
    folder = os.path.join(results_base, dir_name)
    if not os.path.isdir(folder):
        print(f"  WARNING: missing directory {dir_name}")
        return None, None, None

    curves = []
    for seed in TRAINING_SEEDS:
        fname = f"{metric_prefix}_tr_seed_{seed}_dd_seed_{DD_SEED}.txt"
        fpath = os.path.join(folder, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            data = np.loadtxt(fpath, delimiter=',')
            if data.ndim == 0:
                data = np.array([data.item()])
            curves.append(data)
        except Exception as e:
            print(f"  WARNING: could not load {fpath}: {e}")

    if not curves:
        return None, None, None

    min_len = min(len(c) for c in curves)
    curves = np.array([c[:min_len] for c in curves])

    mean = np.mean(curves, axis=0)
    std = np.std(curves, axis=0)
    n_points = len(mean)

    # test_accuracy: sparse (11 points), train_loss: dense (500 points)
    if n_points <= 1 + (NB_STEPS // EVALUATION_DELTA) + 2:
        steps = np.arange(0, NB_STEPS + EVALUATION_DELTA, EVALUATION_DELTA)[:n_points]
    else:
        steps = np.arange(1, n_points + 1)

    return mean, std, steps


# ─── Individual panel plotting ────────────────────────────────────────────────

def setup_matplotlib():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'legend.fontsize': 8,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 200,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })


def plot_single_panel(f, gamma, metric_prefix, ylabel, include_cnn, ylim, outpath):
    """
    Plot a single (f, gamma) panel and save as PDF + PNG.
    """
    fig, ax = plt.subplots(figsize=(5, 3.5))

    # SNN curves for each alpha
    for alpha in ALPHAS:
        dname = snn_dir_name(f, gamma, alpha)
        mean, std, steps = load_metric_curves(SNN_RESULTS_DIR, dname, metric_prefix)
        if mean is None:
            continue
        color = ALPHA_COLORS[alpha]
        label = f"SNN α={alpha}"
        ax.plot(steps, mean, color=color, linewidth=1.8, label=label, zorder=3)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.12, zorder=2)

    # CNN baseline (only if requested)
    if include_cnn:
        cnn_dname = cnn_dir_name(f, gamma)
        mean_cnn, std_cnn, steps_cnn = load_metric_curves(CNN_RESULTS_DIR, cnn_dname, metric_prefix)
        if mean_cnn is not None:
            ax.plot(steps_cnn, mean_cnn, color=CNN_COLOR, linewidth=2.0,
                    linestyle=CNN_LINESTYLE, label="CNN", zorder=4)
            ax.fill_between(steps_cnn, mean_cnn - std_cnn, mean_cnn + std_cnn,
                            color=CNN_COLOR, alpha=0.08, zorder=1)

    gamma_str = f"{gamma}" if gamma != 0.0 else "0.0"
    ax.set_title(f"f = {f},  γ = {gamma_str}", fontweight='bold', pad=8)
    ax.set_xlabel("Training Step")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, NB_STEPS)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(100))

    # Theoretical minimum line (for loss plots only)
    if metric_prefix == "train_loss" and not include_cnn:
        ax.axhline(y=SNN_LOSS_MIN, color='#D32F2F', linestyle=':', linewidth=1.5,
                   label=f'Theoretical min ≈ {SNN_LOSS_MIN:.2f}', zorder=5)

    # Legend: compact, inside the plot
    ax.legend(loc='best', fontsize=7, ncol=2, framealpha=0.8)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches='tight', facecolor='white')
    # Also save PNG version
    png_path = outpath.replace('.pdf', '.png')
    fig.savefig(png_path, bbox_inches='tight', facecolor='white', dpi=200)
    plt.close(fig)
    print(f"  Saved: {os.path.basename(outpath)}")


def plot_cnn_loss_panel(f, gamma, outpath):
    """
    Plot a single CNN-only train loss panel.
    """
    fig, ax = plt.subplots(figsize=(5, 3.5))

    cnn_dname = cnn_dir_name(f, gamma)
    mean_cnn, std_cnn, steps_cnn = load_metric_curves(CNN_RESULTS_DIR, cnn_dname, "train_loss")
    if mean_cnn is not None:
        ax.plot(steps_cnn, mean_cnn, color=CNN_COLOR, linewidth=2.0,
                label="CNN", zorder=4)
        ax.fill_between(steps_cnn, mean_cnn - std_cnn, mean_cnn + std_cnn,
                        color=CNN_COLOR, alpha=0.15, zorder=1)

    gamma_str = f"{gamma}" if gamma != 0.0 else "0.0"
    ax.set_title(f"f = {f},  γ = {gamma_str}", fontweight='bold', pad=8)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Train Loss")
    ax.set_xlim(0, NB_STEPS)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(100))
    ax.axhline(y=CNN_LOSS_MIN, color='#D32F2F', linestyle=':', linewidth=1.5,
               label=f'Theoretical min = {CNN_LOSS_MIN:.1f}', zorder=5)
    ax.legend(loc='best', fontsize=8, framealpha=0.8)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches='tight', facecolor='white')
    png_path = outpath.replace('.pdf', '.png')
    fig.savefig(png_path, bbox_inches='tight', facecolor='white', dpi=200)
    plt.close(fig)
    print(f"  Saved: {os.path.basename(outpath)}")


def generate_all_panels():
    """Generate all individual panel PDFs for both metrics."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Generating Test Accuracy panels (SNN + CNN)...")
    print("=" * 60)
    for f in F_VALUES:
        for gamma in GAMMA_VALUES:
            gamma_str = str(gamma).replace('.', '_')
            fname = f"test_accuracy_f{f}_gamma{gamma_str}.pdf"
            outpath = os.path.join(OUTPUT_DIR, fname)
            plot_single_panel(f, gamma, "test_accuracy", "Test Accuracy",
                              include_cnn=True, ylim=(0.0, 1.0), outpath=outpath)

    print()
    print("=" * 60)
    print("Generating Train Loss panels (SNN only, no CNN)...")
    print("=" * 60)
    for f in F_VALUES:
        for gamma in GAMMA_VALUES:
            gamma_str = str(gamma).replace('.', '_')
            fname = f"train_loss_f{f}_gamma{gamma_str}.pdf"
            outpath = os.path.join(OUTPUT_DIR, fname)
            plot_single_panel(f, gamma, "train_loss", "Train Loss",
                              include_cnn=False, ylim=(0, 3.5), outpath=outpath)

    print()
    print("=" * 60)
    print("Generating Train Loss panels (CNN only)...")
    print("=" * 60)
    for f in F_VALUES:
        for gamma in GAMMA_VALUES:
            gamma_str = str(gamma).replace('.', '_')
            fname = f"train_loss_cnn_f{f}_gamma{gamma_str}.pdf"
            outpath = os.path.join(OUTPUT_DIR, fname)
            plot_cnn_loss_panel(f, gamma, outpath=outpath)


# ─── LaTeX generation ─────────────────────────────────────────────────────────

def generate_latex():
    """Generate a LaTeX document composing all panels in a grid."""
    os.makedirs(LATEX_DIR, exist_ok=True)

    # Compute relative path from latex_dir to output_dir
    rel_plots = os.path.relpath(OUTPUT_DIR, LATEX_DIR).replace("\\", "/")

    tex_path = os.path.join(LATEX_DIR, "convergence_alpha_vs_cnn.tex")

    with open(tex_path, "w") as f:
        f.write(r"""\documentclass{article}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{subcaption}
\usepackage{amsmath}

\geometry{
    a4paper,
    margin=0.6in
}

\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,
    urlcolor=cyan,
}

\title{SNN Atan $\alpha$ Convergence Analysis\\vs CNN Baseline (MNIST)}
\author{ByzFL SNN Benchmark}
\date{\today}

\begin{document}

\maketitle

\section{Overview}
This document presents training convergence plots comparing Spiking Neural Network (SNN) models with different ArcTangent surrogate gradient scaling factors $\alpha \in \{1.0, 1.25, 1.5, 2.0, 3.0\}$ and the CNN baseline.

\textbf{Experimental settings:}
\begin{itemize}
    \item \textbf{Dataset:} MNIST
    \item \textbf{Aggregator:} GeometricMedian (with NNM + ARC pre-aggregation)
    \item \textbf{Attack:} Optimal A Little Is Enough
    \item \textbf{Honest clients:} $n_h = 10$, Byzantine: $f \in \{0, 3, 5\}$
    \item \textbf{Data heterogeneity:} $\gamma \in \{1.0, 0.33, 0.0\}$ (IID $\to$ extreme non-IID)
    \item \textbf{Seeds:} 5 seeds (mean $\pm$ std shading)
\end{itemize}

""")

        # ── Test Accuracy Grid ──
        f.write(r"""
\newpage
\section{Test Accuracy over Training Steps}
Each panel shows test accuracy as a function of training step for SNN with different $\alpha$ values (solid colored lines) and the CNN baseline (dashed black).

\begin{figure}[htbp]
    \centering
""")
        for row_idx, fval in enumerate(F_VALUES):
            for col_idx, gamma in enumerate(GAMMA_VALUES):
                gamma_str = str(gamma).replace('.', '_')
                fname = f"test_accuracy_f{fval}_gamma{gamma_str}.pdf"
                label = f"fig:acc_f{fval}_g{gamma_str}"
                caption = f"$f={fval}$, $\\gamma={gamma}$"

                f.write(f"    \\begin{{subfigure}}[b]{{0.32\\textwidth}}\n")
                f.write(f"        \\centering\n")
                f.write(f"        \\includegraphics[width=\\textwidth]{{{rel_plots}/{fname}}}\n")
                f.write(f"        \\caption{{{caption}}}\n")
                f.write(f"        \\label{{{label}}}\n")
                f.write(f"    \\end{{subfigure}}\n")
                if col_idx < len(GAMMA_VALUES) - 1:
                    f.write("    \\hfill\n")

            if row_idx < len(F_VALUES) - 1:
                f.write("\n    \\vspace{0.3cm}\n\n")

        f.write(r"""
    \caption{Test Accuracy convergence: SNN ($\alpha \in \{1.0, 1.25, 1.5, 2.0, 3.0\}$) vs CNN baseline. Rows: increasing Byzantine fraction ($f$). Columns: decreasing data IID-ness ($\gamma$).}
    \label{fig:test_accuracy_grid}
\end{figure}

""")

        # ── Train Loss Grid ──
        f.write(r"""
\newpage
\section{Train Loss over Training Steps (SNN Only)}
Train loss for SNN models only. CNN loss is omitted because the loss functions are not comparable (SNN uses rate-coded cross-entropy, CNN uses NLL loss).

\begin{figure}[htbp]
    \centering
""")
        for row_idx, fval in enumerate(F_VALUES):
            for col_idx, gamma in enumerate(GAMMA_VALUES):
                gamma_str = str(gamma).replace('.', '_')
                fname = f"train_loss_f{fval}_gamma{gamma_str}.pdf"
                label = f"fig:loss_f{fval}_g{gamma_str}"
                caption = f"$f={fval}$, $\\gamma={gamma}$"

                f.write(f"    \\begin{{subfigure}}[b]{{0.32\\textwidth}}\n")
                f.write(f"        \\centering\n")
                f.write(f"        \\includegraphics[width=\\textwidth]{{{rel_plots}/{fname}}}\n")
                f.write(f"        \\caption{{{caption}}}\n")
                f.write(f"        \\label{{{label}}}\n")
                f.write(f"    \\end{{subfigure}}\n")
                if col_idx < len(GAMMA_VALUES) - 1:
                    f.write("    \\hfill\n")

            if row_idx < len(F_VALUES) - 1:
                f.write("\n    \\vspace{0.3cm}\n\n")

        f.write(r"""
    \caption{Train Loss convergence for SNN only ($\alpha \in \{1.0, 1.25, 1.5, 2.0, 3.0\}$). CNN is excluded because loss functions are not directly comparable.}
    \label{fig:train_loss_grid}
\end{figure}

""")

        # ── CNN Train Loss Grid ──
        f.write(r"""
\newpage
\section{Train Loss over Training Steps (CNN Only)}
Train loss for the CNN baseline. Shown separately from SNN because the loss functions differ (CNN uses NLL loss, SNN uses rate-coded cross-entropy).

\begin{figure}[htbp]
    \centering
""")
        for row_idx, fval in enumerate(F_VALUES):
            for col_idx, gamma in enumerate(GAMMA_VALUES):
                gamma_str = str(gamma).replace('.', '_')
                fname = f"train_loss_cnn_f{fval}_gamma{gamma_str}.pdf"
                label = f"fig:cnn_loss_f{fval}_g{gamma_str}"
                caption = f"$f={fval}$, $\\gamma={gamma}$"

                f.write(f"    \\begin{{subfigure}}[b]{{0.32\\textwidth}}\n")
                f.write(f"        \\centering\n")
                f.write(f"        \\includegraphics[width=\\textwidth]{{{rel_plots}/{fname}}}\n")
                f.write(f"        \\caption{{{caption}}}\n")
                f.write(f"        \\label{{{label}}}\n")
                f.write(f"    \\end{{subfigure}}\n")
                if col_idx < len(GAMMA_VALUES) - 1:
                    f.write("    \\hfill\n")

            if row_idx < len(F_VALUES) - 1:
                f.write("\n    \\vspace{0.3cm}\n\n")

        f.write(r"""
    \caption{Train Loss convergence for CNN baseline only (NLL Loss).}
    \label{fig:cnn_train_loss_grid}
\end{figure}

\section{Key Observations}
\begin{itemize}
    \item \textbf{Benign settings} ($f=0$, any $\gamma$): All $\alpha$ values converge similarly; higher $\alpha$ is slightly faster initially.
    \item \textbf{Moderate adversarial} ($f=3$, $\gamma=0.33$): $\alpha=1.0$ shows the most robustness, climbing to $\sim$65\% while $\alpha \geq 2.0$ collapse to random. CNN achieves higher but unstable accuracy.
    \item \textbf{IID under attack} ($f=3$ or $f=5$, $\gamma=1.0$): Surprisingly, higher $\alpha$ converges faster even under attack when data is IID.
    \item \textbf{Extreme adversarial} ($f=5$, $\gamma=0.0$): All models collapse to random accuracy.
    \item \textbf{Train loss stability}: SNN train loss is much smoother than CNN under Byzantine attack, suggesting intrinsic gradient stability of spiking networks.
\end{itemize}

\end{document}
""")

    print(f"\nLaTeX file written: {tex_path}")
    return tex_path


def compile_latex(tex_path):
    """Compile the LaTeX file to PDF."""
    tex_dir = os.path.dirname(tex_path)
    tex_name = os.path.basename(tex_path)
    pdf_name = tex_name.replace(".tex", ".pdf")

    print("Compiling LaTeX to PDF...")
    try:
        for _ in range(2):
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-disable-installer", tex_name],
                cwd=tex_dir,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        # Cleanup LaTeX auxiliary files
        for ext in [".aux", ".log", ".out", ".toc", ".fls", ".fdb_latexmk", ".synctex.gz"]:
            aux_file = os.path.join(tex_dir, tex_name.replace(".tex", ext))
            if os.path.exists(aux_file):
                os.remove(aux_file)
        print(f"PDF generated: latex_plots/{pdf_name}")
    except FileNotFoundError:
        print("[INFO] pdflatex not found. LaTeX file ready for manual compilation.")
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] pdflatex returned non-zero. Check the .log file for details.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    setup_matplotlib()
    generate_all_panels()
    tex_path = generate_latex()
    compile_latex(tex_path)
    print(f"\nDone! Individual plots in: {OUTPUT_DIR}")
    print(f"LaTeX document: {tex_path}")


if __name__ == "__main__":
    main()
