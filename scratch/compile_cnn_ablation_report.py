import os
import subprocess

def make_figure_3col(title_caption, label_prefix, base_filename, noclip_dir, clip_dir, tanh_dir, tanh_prefix="cnn_mnist_tanh"):
    """3-column figure: CNN NoClip | CNN Clipped | CNN Tanh."""
    latex = []
    latex.append(r"\begin{figure}[htbp]")
    latex.append(r"    \centering")
    sub_width = "0.32\\textwidth"
    
    cnn_filename = base_filename
    tanh_filename = base_filename.replace("cnn_mnist", tanh_prefix)
    
    for path, caption, label in [
        (f"{noclip_dir}/{cnn_filename}", "CNN ReLU (No Clip)", f"{label_prefix}_noclip"),
        (f"{clip_dir}/{cnn_filename}", "CNN ReLU (Grad Clipped)", f"{label_prefix}_clipped"),
        (f"{tanh_dir}/{tanh_filename}", "CNN Tanh", f"{label_prefix}_tanh"),
    ]:
        latex.append(
            f"    \\begin{{subfigure}}[b]{{{sub_width}}}\n"
            f"        \\centering\n"
            f"        \\includegraphics[width=\\textwidth]{{{path}}}\n"
            f"        \\caption{{{caption}}}\n"
            f"        \\label{{fig:{label}}}\n"
            f"    \\end{{subfigure}}"
        )
        latex.append(r"    \hfill")
    
    latex.pop()  # Remove trailing \hfill
    latex.append(f"    \\caption{{{title_caption}}}")
    latex.append(f"    \\label{{fig:{label_prefix}_grid}}")
    latex.append(r"\end{figure}")
    return "\n".join(latex)


def make_figure_2col(title_caption, label_prefix, base_filename, noclip_dir, tanh_dir, tanh_prefix="cnn_mnist_tanh"):
    """2-column figure (for aggregators where Clipped is unavailable): CNN NoClip | CNN Tanh."""
    latex = []
    latex.append(r"\begin{figure}[htbp]")
    latex.append(r"    \centering")
    sub_width = "0.48\\textwidth"
    
    cnn_filename = base_filename
    tanh_filename = base_filename.replace("cnn_mnist", tanh_prefix)
    
    for path, caption, label in [
        (f"{noclip_dir}/{cnn_filename}", "CNN ReLU (No Clip)", f"{label_prefix}_noclip"),
        (f"{tanh_dir}/{tanh_filename}", "CNN Tanh", f"{label_prefix}_tanh"),
    ]:
        latex.append(
            f"    \\begin{{subfigure}}[b]{{{sub_width}}}\n"
            f"        \\centering\n"
            f"        \\includegraphics[width=\\textwidth]{{{path}}}\n"
            f"        \\caption{{{caption}}}\n"
            f"        \\label{{fig:{label}}}\n"
            f"    \\end{{subfigure}}"
        )
        latex.append(r"    \hfill")
    
    latex.pop()
    latex.append(f"    \\caption{{{title_caption}}}")
    latex.append(f"    \\label{{fig:{label_prefix}_grid}}")
    latex.append(r"\end{figure}")
    return "\n".join(latex)


def main():
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    latex_dir = os.path.join(workspace_dir, "latex_plots")
    os.makedirs(latex_dir, exist_ok=True)

    noclip_dir = "../plots/robust_comparison_sweep/learning_rate_0.1"
    clip_dir = "../plots/cnn_clipped_heatmaps"
    tanh_dir = "../plots/cnn_tanh_heatmaps"

    tex_path = os.path.join(latex_dir, "cnn_ablation_report.tex")

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

\title{CNN Activation \& Gradient Clipping Ablation Report\\CNN ReLU (No Clip) vs. CNN ReLU (Clipped) vs. CNN Tanh}
\author{ByzFL SNN Benchmark}
\date{\today}

\begin{document}

\maketitle

\clearpage
\tableofcontents
\clearpage

\section{Introduction}
This report compares three CNN configurations to isolate the effect of gradient saturation/clipping on Byzantine robustness:
\begin{enumerate}
    \item \textbf{CNN ReLU (No Clip)}: Standard CNN with ReLU activation, no gradient clipping ($LR=0.1$).
    \item \textbf{CNN ReLU (Grad Clipped)}: Same CNN with ReLU + explicit gradient norm clipping.
    \item \textbf{CNN Tanh}: Same CNN architecture with Tanh activation (natural gradient saturation).
\end{enumerate}

\textbf{Note:} CNN Clipped results are currently available only for the ``best aggregator'' heatmaps and the Geometric Median aggregator. Other aggregator sections show CNN ReLU vs CNN Tanh only.

\clearpage

\section{Best Overall Performance (Worst-Case Across Attacks)}

""")
        # Best heatmaps (all 3 available)
        cap = "CNN ReLU (No Clip) vs. CNN ReLU (Clipped) vs. CNN Tanh"
        for title, label, filename in [
            ("Best overall (worst-case across attacks)", "best_overall",
             "best_test_mnist_cnn_mnist_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real.pdf"),
        ]:
            f.write(make_figure_3col(f"{title}: {cap}.", label, filename, noclip_dir, clip_dir, tanh_dir))

        f.write(r"""

\clearpage
\section{Best Aggregator under Specific Attacks}

\subsection{Optimal ALIE Attack}
""")
        f.write(make_figure_3col(
            f"Best under OptALIE: {cap}.", "best_alie",
            "best_test_Optimal_ALittleIsEnough_neg1_mnist_cnn_mnist_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real.pdf",
            noclip_dir, clip_dir, tanh_dir))

        f.write(r"""
\clearpage
\subsection{Sign Flipping Attack}
""")
        f.write(make_figure_3col(
            f"Best under Sign Flipping: {cap}.", "best_sf",
            "best_test_SignFlipping_mnist_cnn_mnist_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real.pdf",
            noclip_dir, clip_dir, tanh_dir))

        f.write(r"""
\clearpage
\subsection{Optimal IPM Attack}
""")
        f.write(make_figure_3col(
            f"Best under OptIPM: {cap}.", "best_ipm",
            "best_test_Optimal_InnerProductManipulation_mnist_cnn_mnist_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real.pdf",
            noclip_dir, clip_dir, tanh_dir))

        # Geometric Median (all 3 available)
        f.write(r"""
\clearpage
\section{Per-Aggregator Results}

\subsection{Geometric Median (GM) --- All 3 Configs}
""")
        for attack_name, attack_short, attack_key in [
            ("Worst-Case", "worst_case", ""),
            ("Optimal ALIE", "alie", "Optimal_ALittleIsEnough_neg1_"),
            ("Sign Flipping", "sf", "SignFlipping_"),
            ("Optimal IPM", "ipm", "Optimal_InnerProductManipulation_"),
        ]:
            filename = f"test_{attack_key}mnist_cnn_mnist_gamma_similarity_niid_NNM_ARC_GeometricMedian_nb_honest_clients_10_tolerated_f_equal_real.pdf"
            f.write(f"\\subsubsection{{GM under {attack_name}}}\n")
            f.write(make_figure_3col(
                f"Geometric Median under {attack_name}: {cap}.",
                f"gm_{attack_short}", filename, noclip_dir, clip_dir, tanh_dir))
            f.write("\n\\clearpage\n")

        # Other aggregators (only NoClip vs Tanh)
        cap2 = "CNN ReLU (No Clip) vs. CNN Tanh"
        for agg_name, agg_short, agg_key in [
            ("Centered Clipping", "CC", "CenteredClipping"),
            ("Multi-Krum", "MK", "MultiKrum"),
            ("Trimmed Mean", "TM", "TrMean"),
        ]:
            f.write(f"\n\\subsection{{{agg_name} ({agg_short}) --- NoClip vs Tanh}}\n")
            for attack_name, attack_short, attack_key in [
                ("Worst-Case", "worst_case", ""),
                ("Optimal ALIE", "alie", "Optimal_ALittleIsEnough_neg1_"),
                ("Sign Flipping", "sf", "SignFlipping_"),
                ("Optimal IPM", "ipm", "Optimal_InnerProductManipulation_"),
            ]:
                filename = f"test_{attack_key}mnist_cnn_mnist_gamma_similarity_niid_NNM_ARC_{agg_key}_nb_honest_clients_10_tolerated_f_equal_real.pdf"
                f.write(f"\\subsubsection{{{agg_short} under {attack_name}}}\n")
                f.write(make_figure_2col(
                    f"{agg_name} under {attack_name}: {cap2}.",
                    f"{agg_short.lower()}_{attack_short}", filename, noclip_dir, tanh_dir))
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
        print(f"LaTeX report compiled successfully: cnn_ablation_report.pdf")
    except Exception as e:
        print(f"[ERROR] Failed to compile LaTeX: {e}")

if __name__ == "__main__":
    main()
