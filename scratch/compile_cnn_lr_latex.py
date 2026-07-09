import os
import sys
import subprocess

def make_figure_latex(title_caption, label_prefix, path_prefix, filename, learning_rates):
    latex = []
    latex.append(r"\begin{figure}[htbp]")
    latex.append(r"    \centering")
    
    sub_width = "0.24\\textwidth"
    
    for idx, lr in enumerate(learning_rates):
        # Determine subfigure placement
        sub_fig = (
            f"    \\begin{{subfigure}}[b]{{{sub_width}}}\n"
            f"        \\centering\n"
            f"        \\includegraphics[width=\\textwidth]{{{path_prefix}/learning_rate_{lr}/{filename}}}\n"
            f"        \\caption{{$\\text{{LR}} = {lr}$}}\n"
            f"        \\label{{fig:{label_prefix}_lr_{str(lr).replace('.', '_')}}}\n"
            f"    \\end{{subfigure}}"
        )
        latex.append(sub_fig)
        
        # Format layout
        total_idx = idx + 1
        total_count = len(learning_rates)
        if total_idx % 4 == 0 and total_idx < total_count:
            latex.append(r"    \vspace{0.2cm}")
        elif total_idx < total_count:
            latex.append(r"    \hfill")
            
    latex.append(f"    \\caption{{{title_caption}}}")
    latex.append(f"    \\label{{fig:{label_prefix}_grid}}")
    latex.append(r"\end{figure}")
    return "\n".join(latex)


def main():
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    latex_dir = os.path.join(workspace_dir, "latex_plots")
    os.makedirs(latex_dir, exist_ok=True)

    # ADJUST THESE based on the actual folders created for the learning rate sweep
    learning_rates = ["0.01", "0.05", "0.08", "0.1", "0.12", "0.15"]
    path_prefix = "../plots/robust_comparison_sweep"

    tex_path = os.path.join(latex_dir, "cnn_lr_sweep_report.tex")

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

\title{Convolutional Neural Network (CNN) Baseline\\Learning Rate Sweeps under Data Heterogeneity and Attacks}
\author{ByzFL SNN Benchmark}
\date{\today}

\begin{document}

\maketitle

\clearpage
\tableofcontents
\clearpage

\section{Introduction}
This report documents the robustness sweeps of standard Convolutional Neural Networks (CNN) using various Learning Rates.

\clearpage

\section{Best Overall Performance (Worst-Case Across Attacks)}
""")
        f.write(make_figure_latex(
            "Best overall test accuracy heatmaps (worst-case across attacks) for different learning rates.",
            "best_overall", path_prefix,
            "best_test_mnist_cnn_mnist_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real.pdf",
            learning_rates
        ))

        # Best under specific attacks
        f.write(r"""

\clearpage
\section{Best Aggregator under Specific Attacks}

\subsection{Optimal A Little Is Enough (ALIE) Attack}
""")
        f.write(make_figure_latex(
            "Best test accuracy heatmaps under the Optimal ALIE attack for different learning rates.",
            "best_alie", path_prefix,
            "best_test_Optimal_ALittleIsEnough_neg1_mnist_cnn_mnist_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real.pdf",
            learning_rates
        ))

        f.write(r"""
\clearpage
\subsection{Sign Flipping Attack}
""")
        f.write(make_figure_latex(
            "Best test accuracy heatmaps under the Sign Flipping attack for different learning rates.",
            "best_sf", path_prefix,
            "best_test_SignFlipping_mnist_cnn_mnist_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real.pdf",
            learning_rates
        ))

        f.write(r"""
\clearpage
\subsection{Optimal Inner Product Manipulation Attack}
""")
        f.write(make_figure_latex(
            "Best test accuracy heatmaps under the Optimal Inner Product Manipulation attack for different learning rates.",
            "best_ipm", path_prefix,
            "best_test_Optimal_InnerProductManipulation_mnist_cnn_mnist_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real.pdf",
            learning_rates
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
        print(f"LaTeX report compiled successfully: cnn_lr_sweep_report.pdf")
    except Exception as e:
        print(f"[ERROR] Failed to compile LaTeX: {e}")

if __name__ == "__main__":
    main()
