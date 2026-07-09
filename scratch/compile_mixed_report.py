import os
import sys
import subprocess

def make_figure_latex(title_caption, label_prefix, filename):
    latex = []
    latex.append(r"\begin{figure}[htbp]")
    latex.append(r"    \centering")
    
    sub_width = "0.48\\textwidth"
    
    # 1. Add CNN Baseline
    cnn_filename = filename.replace("cnn_mnist_snn", "cnn_mnist")
    cnn_path = f"../plots/robust_comparison_sweep/{cnn_filename}"
    latex.append(
        f"    \\begin{{subfigure}}[b]{{{sub_width}}}\n"
        f"        \\centering\n"
        f"        \\includegraphics[width=\\textwidth]{{{cnn_path}}}\n"
        f"        \\caption{{CNN Baseline}}\n"
        f"        \\label{{fig:{label_prefix}_cnn}}\n"
        f"    \\end{{subfigure}}"
    )
    latex.append(r"    \hfill")
    
    # 2. Add SNN Atan (alpha=1.2)
    atan_path_1_2 = f"../plots/robust_new_atan_sweep/alpha_1.2/{filename}"
    latex.append(
        f"    \\begin{{subfigure}}[b]{{{sub_width}}}\n"
        f"        \\centering\n"
        f"        \\includegraphics[width=\\textwidth]{{{atan_path_1_2}}}\n"
        f"        \\caption{{SNN Atan ($\\alpha = 1.2$)}}\n"
        f"        \\label{{fig:{label_prefix}_atan_1_2}}\n"
        f"    \\end{{subfigure}}"
    )
    latex.append(r"    \vspace{0.2cm}")

    # 3. Add SNN Atan (alpha=2.0)
    atan_path_2_0 = f"../plots/robust_new_atan_sweep/alpha_2.0/{filename}"
    latex.append(
        f"    \\begin{{subfigure}}[b]{{{sub_width}}}\n"
        f"        \\centering\n"
        f"        \\includegraphics[width=\\textwidth]{{{atan_path_2_0}}}\n"
        f"        \\caption{{SNN Atan ($\\alpha = 2.0$)}}\n"
        f"        \\label{{fig:{label_prefix}_atan_2_0}}\n"
        f"    \\end{{subfigure}}"
    )
    latex.append(r"    \hfill")

    # 4. Add SNN Tri (beta=2.0)
    tri_path = f"../plots/robust_new_tri_sweep/beta_2.0/{filename}"
    latex.append(
        f"    \\begin{{subfigure}}[b]{{{sub_width}}}\n"
        f"        \\centering\n"
        f"        \\includegraphics[width=\\textwidth]{{{tri_path}}}\n"
        f"        \\caption{{SNN Tri ($\\beta = 2.0$)}}\n"
        f"        \\label{{fig:{label_prefix}_tri_2_0}}\n"
        f"    \\end{{subfigure}}"
    )
            
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
        # Preamble
        f.write(r"""\documentclass{article}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{subcaption}
\usepackage{amsmath}

\geometry{
    a4paper,
    margin=0.65in
}

\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,      
    urlcolor=cyan,
}

\title{Spiking Neural Network (SNN) Robustness Benchmark\\Mixed Comparison Report: CNN Baseline vs. SNN (Atan $\alpha \in \{1.2, 2.0\}$ and Tri $\beta=2.0$)}
\author{ByzFL SNN Benchmark}
\date{\today}

\begin{document}

\maketitle

\clearpage
\tableofcontents
\clearpage

\section{Introduction and Experimental Settings}
This report provides a side-by-side mixed comparison of four distinct configurations evaluated under identical federated learning setups featuring varying data heterogeneity (non-IID levels) and Byzantine attacks:
\begin{enumerate}
    \item \textbf{CNN Baseline}: A standard (non-spiking) Convolutional Neural Network (\texttt{cnn\_mnist}).
    \item \textbf{SNN Atan ($\alpha=1.2$)}: A Spiking CNN (\texttt{cnn\_mnist\_snn}) training with the ArcTangent surrogate gradient at stiffness $\alpha = 1.2$.
    \item \textbf{SNN Atan ($\alpha=2.0$)}: A Spiking CNN (\texttt{cnn\_mnist\_snn}) training with the ArcTangent surrogate gradient at stiffness $\alpha = 2.0$.
    \item \textbf{SNN Tri ($\beta=2.0$)}: A Spiking CNN (\texttt{cnn\_mnist\_snn}) training with the Triangular surrogate gradient at parameter $\beta = 2.0$.
\end{enumerate}

These configurations allow direct comparative analysis of the non-spiking model performance against optimized Spiking Neural Networks using different surrogate gradient functions and parameters.

\subsection{Parameter Configurations}
Table~\ref{tab:params} details the parameters, hyperparameters, and algorithms used in this comparison.

\begin{table}[htbp]
    \centering
    \caption{Experiment Parameters and Hyperparameters}
    \label{tab:params}
    \begin{tabular}{ll}
        \toprule
        \textbf{Component / Parameter} & \textbf{Configuration Value} \\
        \midrule
        \textbf{Dataset} & MNIST (60,000 train, 10,000 test) \\
        \textbf{Model Structure (CNN)} & Convolutional Neural Network (\texttt{cnn\_mnist}) \\
        \textbf{Model Structure (SNN)} & Spiking Convolutional Neural Network (\texttt{cnn\_mnist\_snn}) \\
        \textbf{Loss Function} & Rate-coded Cross-Entropy Loss (\texttt{ce\_rate\_loss}) \\
        \textbf{Optimizer} & SGD \\
        \textbf{Learning Rate} & $0.10$ \\
        \textbf{Momentum} & $0.9$ \\
        \textbf{Weight Decay} & $0.0001$ \\
        \textbf{Batch Size} & $128$ \\
        \textbf{SNN Encoding} & Constant encoding, $10$ time steps \\
        \textbf{SNN Decay ($\beta$)} & $0.95$ \\
        \textbf{SNN Threshold} & $1.0$ (fixed) \\
        \textbf{Training Steps} & $500$ \\
        \textbf{Training Algorithm} & DSGD \\
        \textbf{Honest Nodes ($N$)} & $10$ \\
        \textbf{Byzantine Nodes ($f$)} & $f \in \{0, 1, 2, 3, 4, 5\}$ \\
        \textbf{Data Heterogeneity ($\gamma$)} & $\gamma \in \{1.0, 0.66, 0.33, 0.0\}$ (from IID to extreme Non-IID) \\
        \textbf{Pre-aggregators} & NNM, ARC \\
        \textbf{Aggregators Swept} & Centered Clipping, Geometric Median, Multi-Krum, Trimmed Mean \\
        \textbf{Attacks Evaluated} & Optimal ALIE, Sign Flipping, Optimal Inner Product Manipulation \\
        \textbf{Training Seeds} & 5 seeds (42 to 46) \\
        \textbf{Data Dist. Seed} & 42 \\
        \bottomrule
    \end{tabular}
\end{table}

\clearpage

\section{Best Overall Performance (Worst-Case Across Attacks)}
The heatmaps in this section display the worst-case test accuracy at convergence across all attacks achieved by selecting the best robust aggregator for each cell.

""")
        f.write(make_figure_latex(
            "Best overall test accuracy heatmaps (worst-case across attacks) for CNN Baseline vs. SNN Atan ($\\alpha=1.2$, $2.0$) vs. SNN Tri ($\\beta=2.0$).",
            "best_overall",
            "best_test_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real.pdf"
        ))

        # Best under specific attacks
        f.write(r"""

\clearpage
\section{Best Aggregator under Specific Attacks}
This section presents heatmaps showing the best aggregator performance under each specific attack.

\subsection{Optimal A Little Is Enough (ALIE) Attack}
""")
        f.write(make_figure_latex(
            "Best test accuracy heatmaps under the Optimal ALIE attack for CNN Baseline vs. SNN Atan ($\\alpha=1.2$, $2.0$) vs. SNN Tri ($\\beta=2.0$).",
            "best_alie",
            "best_test_Optimal_ALittleIsEnough_neg1_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real.pdf"
        ))

        f.write(r"""
\clearpage
\subsection{Sign Flipping Attack}
""")
        f.write(make_figure_latex(
            "Best test accuracy heatmaps under the Sign Flipping attack for CNN Baseline vs. SNN Atan ($\\alpha=1.2$, $2.0$) vs. SNN Tri ($\\beta=2.0$).",
            "best_sf",
            "best_test_SignFlipping_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real.pdf"
        ))

        f.write(r"""
\clearpage
\subsection{Optimal Inner Product Manipulation Attack}
""")
        f.write(make_figure_latex(
            "Best test accuracy heatmaps under the Optimal Inner Product Manipulation attack for CNN Baseline vs. SNN Atan ($\\alpha=1.2$, $2.0$) vs. SNN Tri ($\\beta=2.0$).",
            "best_ipm",
            "best_test_Optimal_InnerProductManipulation_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real.pdf"
        ))

        # Fixed aggregators under specific attacks
        f.write(r"""

\clearpage
\section{Robustness of Fixed Aggregators across all Attacks}
In this section, we analyze the performance of each specific aggregator (Centered Clipping, Geometric Median, Multi-Krum, Trimmed Mean) across all Byzantine attack settings.

\clearpage
\subsection{Centered Clipping (CC)}
""")
        f.write("\\subsubsection{CC under Worst-Case across all Attacks}\n")
        f.write(make_figure_latex(
            "Centered Clipping performance under the worst-case across all attacks for CNN Baseline vs. SNN Atan ($\\alpha=1.2$, $2.0$) vs. SNN Tri ($\\beta=2.0$).",
            "cc_worst_case",
            "test_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_CenteredClipping_nb_honest_clients_10_tolerated_f_equal_real.pdf"
        ))
        
        f.write("\\clearpage\n\\subsubsection{CC under Optimal ALIE Attack}\n")
        f.write(make_figure_latex(
            "Centered Clipping performance under Optimal ALIE for CNN Baseline vs. SNN Atan ($\\alpha=1.2$, $2.0$) vs. SNN Tri ($\\beta=2.0$).",
            "cc_alie",
            "test_Optimal_ALittleIsEnough_neg1_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_CenteredClipping_nb_honest_clients_10_tolerated_f_equal_real.pdf"
        ))

        f.write("\\clearpage\n\\subsubsection{CC under Sign Flipping Attack}\n")
        f.write(make_figure_latex(
            "Centered Clipping performance under Sign Flipping for CNN Baseline vs. SNN Atan ($\\alpha=1.2$, $2.0$) vs. SNN Tri ($\\beta=2.0$).",
            "cc_sf",
            "test_SignFlipping_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_CenteredClipping_nb_honest_clients_10_tolerated_f_equal_real.pdf"
        ))

        f.write("\\clearpage\n\\subsubsection{CC under Optimal Inner Product Manipulation Attack}\n")
        f.write(make_figure_latex(
            "Centered Clipping performance under Optimal Inner Product Manipulation for CNN Baseline vs. SNN Atan ($\\alpha=1.2$, $2.0$) vs. SNN Tri ($\\beta=2.0$).",
            "cc_ipm",
            "test_Optimal_InnerProductManipulation_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_CenteredClipping_nb_honest_clients_10_tolerated_f_equal_real.pdf"
        ))

        f.write(r"""
\clearpage
\subsection{Geometric Median (GM)}
""")
        f.write("\\subsubsection{GM under Worst-Case across all Attacks}\n")
        f.write(make_figure_latex(
            "Geometric Median performance under the worst-case across all attacks for CNN Baseline vs. SNN Atan ($\\alpha=1.2$, $2.0$) vs. SNN Tri ($\\beta=2.0$).",
            "gm_worst_case",
            "test_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_GeometricMedian_nb_honest_clients_10_tolerated_f_equal_real.pdf"
        ))
        
        f.write("\\clearpage\n\\subsubsection{GM under Optimal ALIE Attack}\n")
        f.write(make_figure_latex(
            "Geometric Median performance under Optimal ALIE for CNN Baseline vs. SNN Atan ($\\alpha=1.2$, $2.0$) vs. SNN Tri ($\\beta=2.0$).",
            "gm_alie",
            "test_Optimal_ALittleIsEnough_neg1_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_GeometricMedian_nb_honest_clients_10_tolerated_f_equal_real.pdf"
        ))

        f.write("\\clearpage\n\\subsubsection{GM under Sign Flipping Attack}\n")
        f.write(make_figure_latex(
            "Geometric Median performance under Sign Flipping for CNN Baseline vs. SNN Atan ($\\alpha=1.2$, $2.0$) vs. SNN Tri ($\\beta=2.0$).",
            "gm_sf",
            "test_SignFlipping_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_GeometricMedian_nb_honest_clients_10_tolerated_f_equal_real.pdf"
        ))

        f.write("\\clearpage\n\\subsubsection{GM under Optimal Inner Product Manipulation Attack}\n")
        f.write(make_figure_latex(
            "Geometric Median performance under Optimal Inner Product Manipulation for CNN Baseline vs. SNN Atan ($\\alpha=1.2$, $2.0$) vs. SNN Tri ($\\beta=2.0$).",
            "gm_ipm",
            "test_Optimal_InnerProductManipulation_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_GeometricMedian_nb_honest_clients_10_tolerated_f_equal_real.pdf"
        ))

        f.write(r"""
\clearpage
\subsection{Multi-Krum (MK)}
""")
        f.write("\\subsubsection{MK under Worst-Case across all Attacks}\n")
        f.write(make_figure_latex(
            "Multi-Krum performance under the worst-case across all attacks for CNN Baseline vs. SNN Atan ($\\alpha=1.2$, $2.0$) vs. SNN Tri ($\\beta=2.0$).",
            "mk_worst_case",
            "test_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_MultiKrum_nb_honest_clients_10_tolerated_f_equal_real.pdf"
        ))
        
        f.write("\\clearpage\n\\subsubsection{MK under Optimal ALIE Attack}\n")
        f.write(make_figure_latex(
            "Multi-Krum performance under Optimal ALIE for CNN Baseline vs. SNN Atan ($\\alpha=1.2$, $2.0$) vs. SNN Tri ($\\beta=2.0$).",
            "mk_alie",
            "test_Optimal_ALittleIsEnough_neg1_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_MultiKrum_nb_honest_clients_10_tolerated_f_equal_real.pdf"
        ))

        f.write("\\clearpage\n\\subsubsection{MK under Sign Flipping Attack}\n")
        f.write(make_figure_latex(
            "Multi-Krum performance under Sign Flipping for CNN Baseline vs. SNN Atan ($\\alpha=1.2$, $2.0$) vs. SNN Tri ($\\beta=2.0$).",
            "mk_sf",
            "test_SignFlipping_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_MultiKrum_nb_honest_clients_10_tolerated_f_equal_real.pdf"
        ))

        f.write("\\clearpage\n\\subsubsection{MK under Optimal Inner Product Manipulation Attack}\n")
        f.write(make_figure_latex(
            "Multi-Krum performance under Optimal Inner Product Manipulation for CNN Baseline vs. SNN Atan ($\\alpha=1.2$, $2.0$) vs. SNN Tri ($\\beta=2.0$).",
            "mk_ipm",
            "test_Optimal_InnerProductManipulation_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_MultiKrum_nb_honest_clients_10_tolerated_f_equal_real.pdf"
        ))

        f.write(r"""
\clearpage
\subsection{Trimmed Mean (TM)}
""")
        f.write("\\subsubsection{TM under Worst-Case across all Attacks}\n")
        f.write(make_figure_latex(
            "Trimmed Mean performance under the worst-case across all attacks for CNN Baseline vs. SNN Atan ($\\alpha=1.2$, $2.0$) vs. SNN Tri ($\\beta=2.0$).",
            "tm_worst_case",
            "test_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_TrMean_nb_honest_clients_10_tolerated_f_equal_real.pdf"
        ))
        
        f.write("\\clearpage\n\\subsubsection{TM under Optimal ALIE Attack}\n")
        f.write(make_figure_latex(
            "Trimmed Mean performance under Optimal ALIE for CNN Baseline vs. SNN Atan ($\\alpha=1.2$, $2.0$) vs. SNN Tri ($\\beta=2.0$).",
            "tm_alie",
            "test_Optimal_ALittleIsEnough_neg1_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_TrMean_nb_honest_clients_10_tolerated_f_equal_real.pdf"
        ))

        f.write("\\clearpage\n\\subsubsection{TM under Sign Flipping Attack}\n")
        f.write(make_figure_latex(
            "Trimmed Mean performance under Sign Flipping for CNN Baseline vs. SNN Atan ($\\alpha=1.2$, $2.0$) vs. SNN Tri ($\\beta=2.0$).",
            "tm_sf",
            "test_SignFlipping_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_TrMean_nb_honest_clients_10_tolerated_f_equal_real.pdf"
        ))

        f.write("\\clearpage\n\\subsubsection{TM under Optimal Inner Product Manipulation Attack}\n")
        f.write(make_figure_latex(
            "Trimmed Mean performance under Optimal Inner Product Manipulation for CNN Baseline vs. SNN Atan ($\\alpha=1.2$, $2.0$) vs. SNN Tri ($\\beta=2.0$).",
            "tm_ipm",
            "test_Optimal_InnerProductManipulation_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_TrMean_nb_honest_clients_10_tolerated_f_equal_real.pdf"
        ))

        f.write(r"""
\end{document}
""")

    print(f"LaTeX file successfully written to {tex_path}")

    # Compile the LaTeX document
    print("Compiling LaTeX report to PDF...")
    try:
        tex_filename = os.path.basename(tex_path)
        pdf_filename = tex_filename.replace(".tex", ".pdf")
        for _ in range(2):
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-disable-installer", tex_filename],
                cwd=latex_dir,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        # Cleanup LaTeX auxiliary files
        for ext in [".aux", ".log", ".out", ".toc", ".fls", ".fdb_latexmk", ".synctex.gz"]:
            aux_file = os.path.join(latex_dir, tex_filename.replace(".tex", ext))
            if os.path.exists(aux_file):
                os.remove(aux_file)
        print(f"LaTeX report compiled successfully: robust_mixed_report.pdf")
    except Exception as e:
        print(f"[ERROR] Failed to compile LaTeX: {e}")

if __name__ == "__main__":
    main()
