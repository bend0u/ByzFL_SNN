import os
import subprocess

def make_2x2_threshold_grid(title_caption, label_prefix, filename_template):
    """Generate a 2x2 grid figure: theta=1.0, 0.8, 0.6, 0.4."""
    latex = []
    latex.append(r"\begin{figure}[htbp]")
    latex.append(r"    \centering")
    
    sub_width = "0.48\\textwidth"
    
    thresholds = [
        ("1.0", "10"),
        ("0.8", "08"),
        ("0.6", "06"),
        ("0.4", "04")
    ]
    
    for i, (thr_name, folder) in enumerate(thresholds):
        path = f"../plots/threshold_sweep/{folder}/plots/{filename_template}"
        
        latex.append(
            f"    \\begin{{subfigure}}[b]{{{sub_width}}}\n"
            f"        \\centering\n"
            f"        \\includegraphics[width=\\textwidth]{{{path}}}\n"
            f"        \\caption{{$\\theta = {thr_name}$}}\n"
            f"    \\end{{subfigure}}"
        )
        if i % 2 == 0:
            latex.append(r"    \hfill")
        elif i == 1:
            latex.append(r"    \vspace{0.3cm}")

    latex.append(f"    \\caption{{{title_caption}}}")
    latex.append(f"    \\label{{fig:{label_prefix}_grid}}")
    latex.append(r"\end{figure}")
    return "\n".join(latex)

def main():
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    latex_dir = os.path.join(workspace_dir, "latex_plots")
    os.makedirs(latex_dir, exist_ok=True)

    tex_path = os.path.join(latex_dir, "threshold_comparison_report.tex")

    with open(tex_path, "w") as f:
        f.write(r"""\documentclass{article}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{hyperref}
\usepackage{subcaption}
\usepackage{amsmath}

\geometry{a4paper, margin=0.65in}

\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,      
    urlcolor=cyan,
}

\title{Spiking Threshold Sweep Comparison Report:\\$\theta \in \{1.0, 0.8, 0.6, 0.4\}$ on MNIST SNN}
\author{ByzFL SNN Benchmark}
\date{\today}

\begin{document}

\maketitle

\clearpage
\tableofcontents
\clearpage

\section{Introduction}
This report compiles and compares the robust aggregation sweep results for different values of the spiking threshold $\theta \in \{1.0, 0.8, 0.6, 0.4\}$. All models are SNNs trained with constant encoding ($T=10$) and the Atan surrogate gradient ($\alpha = 1.2$).
We compare the worst-case test accuracy across four robust aggregators (Centered Clipping, Geometric Median, Multi-Krum, and Trimmed Mean) under three different attacks:
\begin{enumerate}
    \item Optimal A Little Is Enough (OptALIE)
    \item Optimal Inner Product Manipulation (OptIPM)
    \item Sign Flipping
\end{enumerate}
Our objective is to verify whether varying the spiking threshold alters the gradient sparsity and consequently shifts the vulnerability window of the SNN.

\clearpage
\section{Worst-Case Robustness (Across All Attacks)}
The following figures show the worst-case test accuracy across all three attacks for the different threshold values.

""")
        
        # 1. Worst Case
        f.write(make_2x2_threshold_grid(
            "Worst-case test accuracy across all attacks (Worst-Case Heatmaps) for different spiking thresholds.",
            "worst_case",
            "best_test_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real.pdf"
        ))
        f.write("\\clearpage\n")

        # 2. OptALIE
        f.write("\\section{Robustness under Optimal ALIE}")
        f.write(make_2x2_threshold_grid(
            "Test accuracy under Optimal A Little Is Enough (OptALIE) attack for different spiking thresholds.",
            "opt_alie",
            "best_test_Optimal_ALittleIsEnough_neg1_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real.pdf"
        ))
        f.write("\\clearpage\n")

        # 3. OptIPM
        f.write("\\section{Robustness under Optimal IPM}")
        f.write(make_2x2_threshold_grid(
            "Test accuracy under Optimal Inner Product Manipulation (OptIPM) attack for different spiking thresholds.",
            "opt_ipm",
            "best_test_Optimal_InnerProductManipulation_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real.pdf"
        ))
        f.write("\\clearpage\n")

        # 4. Sign Flipping
        f.write("\\section{Robustness under Sign Flipping}")
        f.write(make_2x2_threshold_grid(
            "Test accuracy under Sign Flipping attack for different spiking thresholds.",
            "sign_flipping",
            "best_test_SignFlipping_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real.pdf"
        ))
        f.write("\\clearpage\n")

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
        print(f"LaTeX report compiled successfully: threshold_comparison_report.pdf")
    except Exception as e:
        print(f"[ERROR] Failed to compile LaTeX: {e}")

if __name__ == "__main__":
    main()
