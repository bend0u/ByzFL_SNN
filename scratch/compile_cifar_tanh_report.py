import os
import subprocess

def make_single_image(title_caption, label_prefix, path):
    """Generate a figure with a single image."""
    latex = []
    latex.append(r"\begin{figure}[htbp]")
    latex.append(r"    \centering")
    latex.append(f"    \\includegraphics[width=0.8\\textwidth]{{{path}}}")
    latex.append(f"    \\caption{{{title_caption}}}")
    latex.append(f"    \\label{{fig:{label_prefix}}}")
    latex.append(r"\end{figure}")
    return "\n".join(latex)

def main():
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    latex_dir = os.path.join(workspace_dir, "latex_plots")
    os.makedirs(latex_dir, exist_ok=True)

    tex_path = os.path.join(latex_dir, "cifar_tanh_report.tex")
    
    base_dir = "../plots/cifar-tanh-sweep20"

    with open(tex_path, "w") as f:
        f.write(r"""\documentclass{article}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{hyperref}

\geometry{a4paper, margin=0.65in}

\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,      
    urlcolor=cyan,
}

\title{CIFAR-10 CNN Tanh Sweep 20 Report}
\author{ByzFL Benchmark}
\date{\today}

\begin{document}

\maketitle

\clearpage
\tableofcontents
\clearpage

\section{Best Overall Performance (Worst-Case Across Attacks)}
""")
        f.write(make_single_image(
            "Best overall test accuracy (worst-case across attacks).",
            "best_overall",
            f"{base_dir}/best_test_cifar10_cnn_cifar_tanh_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real.pdf"
        ))

        f.write(r"""
\clearpage
\section{Best Aggregator under Specific Attacks}

\subsection{Optimal ALIE Attack}
""")
        f.write(make_single_image(
            "Best test accuracy under OptALIE.", "best_alie",
            f"{base_dir}/best_test_Optimal_ALittleIsEnough_neg1_cifar10_cnn_cifar_tanh_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real.pdf"
        ))

        f.write(r"""
\clearpage
\subsection{Sign Flipping Attack}
""")
        f.write(make_single_image(
            "Best test accuracy under Sign Flipping.", "best_sf",
            f"{base_dir}/best_test_SignFlipping_cifar10_cnn_cifar_tanh_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real.pdf"
        ))

        f.write(r"""
\clearpage
\subsection{Optimal IPM Attack}
""")
        f.write(make_single_image(
            "Best test accuracy under OptIPM.", "best_ipm",
            f"{base_dir}/best_test_Optimal_InnerProductManipulation_cifar10_cnn_cifar_tanh_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real.pdf"
        ))

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
        
        pdf_path = os.path.join(latex_dir, "cifar_tanh_report.pdf")
        artifact_dir = os.environ.get("ARTIFACT_DIR", ".")
        if os.path.exists(artifact_dir):
            subprocess.run(["copy", pdf_path, artifact_dir], shell=True)
            
        print(f"LaTeX report compiled successfully: cifar_tanh_report.pdf")
    except Exception as e:
        print(f"[ERROR] Failed to compile LaTeX: {e}")

if __name__ == "__main__":
    main()
