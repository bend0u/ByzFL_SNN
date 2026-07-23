import os
import subprocess

# Six configurations compared, mirroring the structure of robust_mixed_report.tex
CONFIGS = [
    dict(key="noclip", caption="CNN ReLU (No Clip)",
         dir="../plots/mnist_clipping_heatmaps/cnn_mnist",
         model="mnist_cnn_mnist", has_full_agg=True),
    dict(key="clip1", caption="CNN ReLU (Clip $=1$)",
         dir="../plots/mnist_clipping_heatmaps/cnn_mnist_clipping_1",
         model="mnist_cnn_mnist_clipping_1", has_full_agg=True),
    dict(key="clip2", caption="CNN ReLU (Clip $=2$)",
         dir="../plots/mnist_clipping_heatmaps/cnn_mnist_clipping_2",
         model="mnist_cnn_mnist_clipping_2", has_full_agg=True),
    dict(key="clip4", caption="CNN ReLU (Clip $=4$)",
         dir="../plots/mnist_clipping_heatmaps/cnn_mnist_clipping_4",
         model="mnist_cnn_mnist_clipping_4", has_full_agg=True),
    dict(key="tanh", caption="CNN Tanh",
         dir="../plots/cnn_tanh_heatmaps",
         model="mnist_cnn_mnist_tanh", has_full_agg=True),
    dict(key="gradclip21", caption="CNN ReLU (Grad-Norm Clip $=21$)",
         dir="../plots/cnn_clipped_heatmaps",
         model="mnist_cnn_mnist", has_full_agg=False),
]

SUFFIX = "gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real"

ATTACKS = [
    dict(key="alie", tag="Optimal_ALittleIsEnough_neg1", title="Optimal ALIE Attack", short="ALIE"),
    dict(key="sf", tag="SignFlipping", title="Sign Flipping Attack", short="SF"),
    dict(key="ipm", tag="Optimal_InnerProductManipulation", title="Optimal IPM Attack", short="IPM"),
]

AGGREGATORS = [
    dict(key="cc", tag="CenteredClipping", title="Centered Clipping (CC)", short="CC"),
    dict(key="gm", tag="GeometricMedian", title="Geometric Median (GM)", short="GM"),
    dict(key="mk", tag="MultiKrum", title="Multi-Krum (MK)", short="MK"),
    dict(key="tm", tag="TrMean", title="Trimmed Mean (TM)", short="TM"),
]


def best_test_path(cfg, attack_tag=None):
    prefix = "best_test"
    parts = [prefix]
    if attack_tag:
        parts.append(attack_tag)
    parts.append(cfg["model"])
    parts.append(SUFFIX)
    return f"{cfg['dir']}/{'_'.join(parts)}.pdf"


def test_path(cfg, agg_tag, attack_tag=None):
    parts = ["test"]
    if attack_tag:
        parts.append(attack_tag)
    parts.append(cfg["model"])
    parts.append(SUFFIX.replace("gamma_similarity_niid_NNM_ARC",
                                 f"gamma_similarity_niid_NNM_ARC_{agg_tag}"))
    return f"{cfg['dir']}/{'_'.join(parts)}.pdf"


def subfig(width, path, caption, label):
    return "\n".join([
        r"    \begin{subfigure}[b]{%s\textwidth}" % width,
        r"        \centering",
        f"        \\includegraphics[width=\\textwidth]{{{path}}}",
        f"        \\caption{{{caption}}}",
        f"        \\label{{fig:{label}}}",
        r"    \end{subfigure}",
    ])


def make_grid(paths_captions_labels, width, caption, label, per_row=3):
    lines = [r"\begin{figure}[htbp]", r"    \centering"]
    for i, (path, cap, lab) in enumerate(paths_captions_labels):
        lines.append(subfig(width, path, cap, lab))
        if i != len(paths_captions_labels) - 1:
            if (i + 1) % per_row == 0:
                lines.append(r"    \vspace{0.3cm}")
            else:
                lines.append(r"    \hfill")
    lines.append(f"    \\caption{{{caption}}}")
    lines.append(f"    \\label{{fig:{label}}}")
    lines.append(r"\end{figure}")
    return "\n".join(lines)


def main():
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    latex_dir = os.path.join(workspace_dir, "latex_plots")
    os.makedirs(latex_dir, exist_ok=True)
    tex_path = os.path.join(latex_dir, "mnist_clipping_report.tex")

    with open(tex_path, "w") as f:
        f.write(r"""\documentclass{article}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{subcaption}
\usepackage{amsmath}

\geometry{a4paper, margin=0.65in}

\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,
    urlcolor=cyan,
}

\title{MNIST CNN ReLU Clipping Sweep Report:\\No Clip vs.\ Clip 1/2/4 vs.\ Tanh vs.\ Grad-Norm Clip 21}
\author{ByzFL SNN Benchmark}
\date{\today}

\begin{document}

\maketitle

\clearpage
\tableofcontents
\clearpage

\section{Introduction}
This report compares six MNIST CNN configurations:
\begin{enumerate}
    \item \textbf{CNN ReLU (No Clip)}: Standard CNN with ReLU, no activation clipping.
    \item \textbf{CNN ReLU (Clip $=1$)}: ReLU activations clipped elementwise to $[0, 1]$.
    \item \textbf{CNN ReLU (Clip $=2$)}: ReLU activations clipped elementwise to $[0, 2]$.
    \item \textbf{CNN ReLU (Clip $=4$)}: ReLU activations clipped elementwise to $[0, 4]$.
    \item \textbf{CNN Tanh}: CNN with Tanh activation (implicit saturation).
    \item \textbf{CNN ReLU (Grad-Norm Clip $=21$)}: ReLU network with the gradient's overall norm clipped to 21 (not a per-neuron/elementwise clip).
\end{enumerate}

\textbf{Note:} The Grad-Norm Clip $=21$ configuration only has heatmaps available for the best-aggregator (worst-case) sections and the Geometric Median sections.

\clearpage

""")

        # Best overall (worst-case) section
        f.write(r"\section{Best Overall Performance (Worst-Case Across Attacks)}" + "\n")
        items = []
        for cfg in CONFIGS:
            items.append((best_test_path(cfg), cfg["caption"], f"best_overall_{cfg['key']}"))
        f.write(make_grid(items, "0.32", "Best overall test accuracy (worst-case across attacks).",
                           "best_overall_grid"))
        f.write("\n\\clearpage\n")

        # Best aggregator under specific attacks
        f.write(r"\section{Best Aggregator under Specific Attacks}" + "\n\n")
        for attack in ATTACKS:
            f.write(f"\\subsection{{{attack['title']}}}\n")
            items = []
            for cfg in CONFIGS:
                items.append((best_test_path(cfg, attack["tag"]), cfg["caption"],
                              f"best_{attack['key']}_{cfg['key']}"))
            f.write(make_grid(items, "0.32", f"Best test accuracy under {attack['short']}.",
                               f"best_{attack['key']}_grid"))
            f.write("\n\\clearpage\n")

        # Per-aggregator sections
        for agg in AGGREGATORS:
            f.write(f"\\subsection{{{agg['title']}}}\n")

            # Worst-case across all attacks
            f.write(f"\\subsubsection{{{agg['short']} under Worst-Case across all Attacks}}\n")
            items = []
            for cfg in CONFIGS:
                if not cfg["has_full_agg"] and agg["key"] != "gm":
                    continue
                items.append((test_path(cfg, agg["tag"]), cfg["caption"],
                              f"{agg['key']}_worst_case_{cfg['key']}"))
            f.write(make_grid(items, "0.24" if len(items) > 3 else "0.32",
                               f"{agg['title']} under Worst-Case across all Attacks.",
                               f"{agg['key']}_worst_case_grid"))
            f.write("\n\\clearpage\n")

            for attack in ATTACKS:
                f.write(f"\\subsubsection{{{agg['short']} under {attack['short']}}}\n")
                items = []
                for cfg in CONFIGS:
                    if not cfg["has_full_agg"] and agg["key"] != "gm":
                        continue
                    items.append((test_path(cfg, agg["tag"], attack["tag"]), cfg["caption"],
                                  f"{agg['key']}_{attack['key']}_{cfg['key']}"))
                f.write(make_grid(items, "0.24" if len(items) > 3 else "0.32",
                                   f"{agg['title']} under {attack['title']}.",
                                   f"{agg['key']}_{attack['key']}_grid"))
                f.write("\n\\clearpage\n")

        f.write("\n\\end{document}\n")

    print(f"LaTeX file successfully written to {tex_path}")

    print("Compiling LaTeX report to PDF...")
    try:
        tex_filename = os.path.basename(tex_path)
        for _ in range(2):
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", tex_filename],
                cwd=latex_dir,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        print("LaTeX report compiled successfully: mnist_clipping_report.pdf")
    except Exception as e:
        print(f"[ERROR] Failed to compile LaTeX: {e}")


if __name__ == "__main__":
    main()
