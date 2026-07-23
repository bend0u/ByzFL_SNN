import os
import subprocess

CONFIGS = [
    dict(key="relu", caption="CNN ReLU", dir="../plots/cifar-relu-heatmap-22-07",
         model="cifar10_cnn_cifar"),
    dict(key="tanh", caption="CNN Tanh", dir="../plots/cifar-tanh-heatmap-22-07",
         model="cifar10_cnn_cifar_tanh"),
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
    parts = ["best_test"]
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


def make_grid(paths_captions_labels, width, caption, label, per_row=2):
    lines = [r"\begin{figure}[htbp]", r"    \centering"]
    for i, (path, cap, lab) in enumerate(paths_captions_labels):
        lines.append(subfig(width, path, cap, lab))
        if i != len(paths_captions_labels) - 1:
            lines.append(r"    \hfill")
    lines.append(f"    \\caption{{{caption}}}")
    lines.append(f"    \\label{{fig:{label}}}")
    lines.append(r"\end{figure}")
    return "\n".join(lines)


def main():
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    latex_dir = os.path.join(workspace_dir, "latex_plots")
    os.makedirs(latex_dir, exist_ok=True)
    tex_path = os.path.join(latex_dir, "cifar_relu_vs_tanh_report.tex")

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

\title{CIFAR-10 CNN Report: ReLU vs.\ Tanh}
\author{ByzFL SNN Benchmark}
\date{\today}

\begin{document}

\maketitle

\clearpage
\tableofcontents
\clearpage

\section{Introduction}
This report compares two CIFAR-10 CNN configurations side by side:
\begin{enumerate}
    \item \textbf{CNN ReLU}: Standard CNN with ReLU activations, no clipping.
    \item \textbf{CNN Tanh}: CNN with Tanh activation (implicit saturation).
\end{enumerate}
Both share the same DSGD training setup, NNM+ARC pre-aggregation, and $f=0..5$ / $\gamma=0..1$ sweep grid.

\clearpage

""")

        f.write(r"\section{Best Overall Performance (Worst-Case Across Attacks)}" + "\n")
        items = [(best_test_path(cfg), cfg["caption"], f"best_overall_{cfg['key']}") for cfg in CONFIGS]
        f.write(make_grid(items, "0.48", "Best overall test accuracy (worst-case across attacks).",
                           "best_overall_grid"))
        f.write("\n\\clearpage\n")

        f.write(r"\section{Best Aggregator under Specific Attacks}" + "\n\n")
        for attack in ATTACKS:
            f.write(f"\\subsection{{{attack['title']}}}\n")
            items = [(best_test_path(cfg, attack["tag"]), cfg["caption"], f"best_{attack['key']}_{cfg['key']}")
                     for cfg in CONFIGS]
            f.write(make_grid(items, "0.48", f"Best test accuracy under {attack['short']}.",
                               f"best_{attack['key']}_grid"))
            f.write("\n\\clearpage\n")

        for agg in AGGREGATORS:
            f.write(f"\\subsection{{{agg['title']}}}\n")

            f.write(f"\\subsubsection{{{agg['short']} under Worst-Case across all Attacks}}\n")
            items = [(test_path(cfg, agg["tag"]), cfg["caption"], f"{agg['key']}_worst_case_{cfg['key']}")
                     for cfg in CONFIGS]
            f.write(make_grid(items, "0.48", f"{agg['title']} under Worst-Case across all Attacks.",
                               f"{agg['key']}_worst_case_grid"))
            f.write("\n\\clearpage\n")

            for attack in ATTACKS:
                f.write(f"\\subsubsection{{{agg['short']} under {attack['short']}}}\n")
                items = [(test_path(cfg, agg["tag"], attack["tag"]), cfg["caption"],
                          f"{agg['key']}_{attack['key']}_{cfg['key']}") for cfg in CONFIGS]
                f.write(make_grid(items, "0.48", f"{agg['title']} under {attack['title']}.",
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
        print("LaTeX report compiled successfully: cifar_relu_vs_tanh_report.pdf")
    except Exception as e:
        print(f"[ERROR] Failed to compile LaTeX: {e}")


if __name__ == "__main__":
    main()
