import os
import subprocess

def make_1x2_grid(title_caption, label_prefix, filename_template):
    """Generate a 1x2 grid figure comparing normal Atan vs Threshold 0.4."""
    latex = []
    latex.append(r"\begin{figure}[htbp]")
    latex.append(r"    \centering")
    
    sub_width = "0.48\\textwidth"
    
    configs = [
        (f"../plots/robust_new_atan_sweep/alpha_1.2/{filename_template}", "SNN Atan ($\\alpha=1.2, \\theta=1.0$)"),
        (f"../plots/full-sweep-threshold04/{filename_template}", "SNN Atan ($\\alpha=1.2, \\theta=0.4$)")
    ]
    
    for i, (path, caption) in enumerate(configs):
        latex.append(
            f"    \\begin{{subfigure}}[b]{{{sub_width}}}\n"
            f"        \\centering\n"
            f"        \\includegraphics[width=\\textwidth]{{{path}}}\n"
            f"        \\caption{{{caption}}}\n"
            f"    \\end{{subfigure}}"
        )
        if i == 0:
            latex.append(r"    \hfill")

    latex.append(f"    \\caption{{{title_caption}}}")
    latex.append(f"    \\label{{fig:{label_prefix}_grid}}")
    latex.append(r"\end{figure}")
    return "\n".join(latex)

def main():
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    latex_dir = os.path.join(workspace_dir, "latex_plots")
    os.makedirs(latex_dir, exist_ok=True)

    tex_path = os.path.join(latex_dir, "threshold04_vs_atan_report.tex")

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

\title{Comparison Report:\\Standard SNN Atan ($\theta=1.0$) vs SNN Atan ($\theta=0.4$)}
\author{ByzFL SNN Benchmark}
\date{\today}

\begin{document}

\maketitle

\clearpage
\tableofcontents
\clearpage

\section{Introduction}
This report compares the robustness of the standard SNN Atan configuration (spiking threshold $\theta=1.0$, surrogate gradient sharpness $\alpha=1.2$) against the same architecture with a lowered spiking threshold ($\theta=0.4$).
We compare the worst-case test accuracy across four robust aggregators (Centered Clipping, Geometric Median, Multi-Krum, and Trimmed Mean) under three different attacks:
\begin{enumerate}
    \item Optimal A Little Is Enough (OptALIE)
    \item Optimal Inner Product Manipulation (OptIPM)
    \item Sign Flipping
\end{enumerate}

\clearpage
\section{Worst-Case Robustness (Across All Attacks)}
The following figures show the worst-case test accuracy across all three attacks.

""")
        
        # 1. Worst Case
        f.write(make_1x2_grid(
            "Worst-case test accuracy across all attacks.",
            "worst_case",
            "best_test_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real.pdf"
        ))
        f.write("\\clearpage\n")

        # 2. OptALIE
        f.write("\\section{Robustness under Optimal ALIE}\n")
        f.write(make_1x2_grid(
            "Test accuracy under Optimal A Little Is Enough (OptALIE) attack.",
            "opt_alie",
            "best_test_Optimal_ALittleIsEnough_neg1_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real.pdf"
        ))
        f.write("\\clearpage\n")

        # 3. OptIPM
        f.write("\\section{Robustness under Optimal IPM}\n")
        f.write(make_1x2_grid(
            "Test accuracy under Optimal Inner Product Manipulation (OptIPM) attack.",
            "opt_ipm",
            "best_test_Optimal_InnerProductManipulation_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real.pdf"
        ))
        f.write("\\clearpage\n")

        # 4. Sign Flipping
        f.write("\\section{Robustness under Sign Flipping}\n")
        f.write(make_1x2_grid(
            "Test accuracy under Sign Flipping attack.",
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
        
        pdf_path = os.path.join(latex_dir, "threshold04_vs_atan_report.pdf")
        artifact_dir = os.environ.get("ARTIFACT_DIR", ".")
        if os.path.exists(artifact_dir):
            subprocess.run(["copy", pdf_path, artifact_dir], shell=True)
            
        print(f"LaTeX report compiled successfully: threshold04_vs_atan_report.pdf")
    except Exception as e:
        print(f"[ERROR] Failed to compile LaTeX: {e}")

if __name__ == "__main__":
    main()
