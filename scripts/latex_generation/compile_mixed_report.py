import os
import subprocess

def make_figure_6(title_caption, label_prefix, snn_filename, has_clipped=True):
    """Generate a 2x3 or 2x2+1 grid figure."""
    latex = []
    latex.append(r"\begin{figure}[htbp]")
    latex.append(r"    \centering")
    
    cnn_filename = snn_filename.replace("cnn_mnist_snn", "cnn_mnist")
    tanh_filename = snn_filename.replace("cnn_mnist_snn", "cnn_mnist_tanh")
    
    if has_clipped:
        sub_width = "0.32\\textwidth"
        configs = [
            (f"../plots/robust_comparison_sweep/learning_rate_0.15/{cnn_filename}", "CNN ReLU ($LR=0.15$)", f"{label_prefix}_cnn"),
            (f"../plots/robust_new_atan_sweep/alpha_1.2/{snn_filename}", "SNN Atan ($\\alpha = 1.2$)", f"{label_prefix}_atan"),
            (f"../plots/robust_new_tri_sweep/beta_2.0/{snn_filename}", "SNN Tri ($\\beta = 2.0$)", f"{label_prefix}_tri"),
            (f"../plots/robust_new_box_sweep/beta_2.0/{snn_filename}", "SNN Box ($\\beta = 2.0$)", f"{label_prefix}_box"),
            (f"../plots/cnn_clipped_heatmaps/{cnn_filename}", "CNN ReLU (Grad Clipped)", f"{label_prefix}_clip"),
            (f"../plots/cnn_tanh_heatmaps/{tanh_filename}", "CNN Tanh", f"{label_prefix}_tanh"),
        ]
    else:
        sub_width = "0.24\\textwidth"
        configs = [
            (f"../plots/robust_comparison_sweep/learning_rate_0.15/{cnn_filename}", "CNN ReLU ($LR=0.15$)", f"{label_prefix}_cnn"),
            (f"../plots/robust_new_atan_sweep/alpha_1.2/{snn_filename}", "SNN Atan ($\\alpha = 1.2$)", f"{label_prefix}_atan"),
            (f"../plots/robust_new_tri_sweep/beta_2.0/{snn_filename}", "SNN Tri ($\\beta = 2.0$)", f"{label_prefix}_tri"),
            (f"../plots/robust_new_box_sweep/beta_2.0/{snn_filename}", "SNN Box ($\\beta = 2.0$)", f"{label_prefix}_box"),
            (f"../plots/cnn_tanh_heatmaps/{tanh_filename}", "CNN Tanh", f"{label_prefix}_tanh"),
        ]
    
    for i, (path, caption, label) in enumerate(configs):
        latex.append(
            f"    \\begin{{subfigure}}[b]{{{sub_width}}}\n"
            f"        \\centering\n"
            f"        \\includegraphics[width=\\textwidth]{{{path}}}\n"
            f"        \\caption{{{caption}}}\n"
            f"        \\label{{fig:{label}}}\n"
            f"    \\end{{subfigure}}"
        )
        # Add spacing
        if has_clipped:
            if i == 2:  # End of row 1
                latex.append(r"    \vspace{0.3cm}")
            elif i < len(configs) - 1:
                latex.append(r"    \hfill")
        else:
            if i == 2:
                latex.append(r"    \hfill")
            elif i == 3:
                latex.append(r"    \vspace{0.3cm}")
            elif i < len(configs) - 1:
                latex.append(r"    \hfill")

    latex.append(f"    \\caption{{{title_caption}}}")
    latex.append(f"    \\label{{fig:{label_prefix}_grid}}")
    latex.append(r"\end{figure}")
    return "\n".join(latex)


def main():
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    latex_dir = os.path.join(workspace_dir, "latex_plots")
    os.makedirs(latex_dir, exist_ok=True)

    tex_path = os.path.join(latex_dir, "robust_mixed_report.tex")

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

\title{Mixed Comparison Report:\\CNN ReLU vs. SNN (Atan, Tri, Box) vs. CNN Clipped vs. CNN Tanh}
\author{ByzFL SNN Benchmark}
\date{\today}

\begin{document}

\maketitle

\clearpage
\tableofcontents
\clearpage

\section{Introduction}
This report compares six configurations:
\begin{enumerate}
    \item \textbf{CNN ReLU ($LR=0.15$)}: Standard CNN with ReLU, no clipping.
    \item \textbf{SNN Atan ($\alpha=1.2$)}: Spiking CNN with ArcTangent surrogate.
    \item \textbf{SNN Tri ($\beta=2.0$)}: Spiking CNN with Triangular surrogate.
    \item \textbf{SNN Box ($\beta=2.0$)}: Spiking CNN with Box surrogate.
    \item \textbf{CNN ReLU (Grad Clipped)}: CNN with explicit gradient norm clipping.
    \item \textbf{CNN Tanh}: CNN with Tanh activation (implicit saturation).
\end{enumerate}

\textbf{Note:} CNN Clipped heatmaps are available for best-aggregator and Geometric Median sections only. Other per-aggregator sections show 5 configs (without Clipped).

\clearpage

\section{Best Overall Performance (Worst-Case Across Attacks)}

""")
        cap = "All 6 configurations"
        f.write(make_figure_6(
            f"Best overall test accuracy (worst-case across attacks).",
            "best_overall",
            "best_test_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real.pdf",
            has_clipped=True
        ))

        f.write(r"""

\clearpage
\section{Best Aggregator under Specific Attacks}

\subsection{Optimal ALIE Attack}
""")
        f.write(make_figure_6(
            "Best test accuracy under OptALIE.", "best_alie",
            "best_test_Optimal_ALittleIsEnough_neg1_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real.pdf",
            has_clipped=True))

        f.write(r"""
\clearpage
\subsection{Sign Flipping Attack}
""")
        f.write(make_figure_6(
            "Best test accuracy under Sign Flipping.", "best_sf",
            "best_test_SignFlipping_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real.pdf",
            has_clipped=True))

        f.write(r"""
\clearpage
\subsection{Optimal IPM Attack}
""")
        f.write(make_figure_6(
            "Best test accuracy under OptIPM.", "best_ipm",
            "best_test_Optimal_InnerProductManipulation_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real.pdf",
            has_clipped=True))

        # Per-aggregator
        for agg_name, agg_short, agg_key in [
            ("Centered Clipping", "CC", "CenteredClipping"),
            ("Geometric Median", "GM", "GeometricMedian"),
            ("Multi-Krum", "MK", "MultiKrum"),
            ("Trimmed Mean", "TM", "TrMean"),
        ]:
            has_clip = (agg_key == "GeometricMedian")
            f.write(f"\n\\clearpage\n\\subsection{{{agg_name} ({agg_short})}}\n")
            
            for attack_name, attack_short, attack_key in [
                ("Worst-Case across all Attacks", "worst_case", ""),
                ("Optimal ALIE", "alie", "Optimal_ALittleIsEnough_neg1_"),
                ("Sign Flipping", "sf", "SignFlipping_"),
                ("Optimal IPM", "ipm", "Optimal_InnerProductManipulation_"),
            ]:
                label = f"{agg_short.lower()}_{attack_short}"
                filename = f"test_{attack_key}mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_{agg_key}_nb_honest_clients_10_tolerated_f_equal_real.pdf"
                
                f.write(f"\\subsubsection{{{agg_short} under {attack_name}}}\n")
                f.write(make_figure_6(
                    f"{agg_name} under {attack_name}.",
                    label, filename, has_clipped=has_clip))
                f.write("\n\\clearpage\n")

        f.write(r"""
\end{document}
""")

    print(f"LaTeX file successfully written to {tex_path}")

    print("Compiling LaTeX report to PDF...")
    try:
        tex_filename = os.path.basename(tex_path)
        for _ in range(2):
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-disable-installer", tex_filename],
                cwd=latex_dir,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        for ext in [".aux", ".log", ".out", ".toc", ".fls", ".fdb_latexmk", ".synctex.gz"]:
            aux_file = os.path.join(latex_dir, tex_filename.replace(".tex", ext))
            if os.path.exists(aux_file):
                os.remove(aux_file)
        print(f"LaTeX report compiled successfully: robust_mixed_report.pdf")
    except Exception as e:
        print(f"[ERROR] Failed to compile LaTeX: {e}")

if __name__ == "__main__":
    main()
