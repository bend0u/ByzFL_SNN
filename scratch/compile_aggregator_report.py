import os
import subprocess

def make_2x2_grid(title_caption, label_prefix, base_dir, filename_template):
    """Generate a 2x2 grid figure: CC, GM, MK, TM."""
    latex = []
    latex.append(r"\begin{figure}[htbp]")
    latex.append(r"    \centering")
    
    sub_width = "0.48\\textwidth"
    
    aggregators = [
        ("Centered Clipping", "CenteredClipping"),
        ("Geometric Median", "GeometricMedian"),
        ("Multi-Krum", "MultiKrum"),
        ("Trimmed Mean", "TrMean")
    ]
    
    for i, (agg_name, agg_key) in enumerate(aggregators):
        filename = filename_template.replace("{agg}", agg_key)
        path = f"../{base_dir}/{filename}"
        
        latex.append(
            f"    \\begin{{subfigure}}[b]{{{sub_width}}}\n"
            f"        \\centering\n"
            f"        \\includegraphics[width=\\textwidth]{{{path}}}\n"
            f"        \\caption{{{agg_name}}}\n"
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


def make_1x1_grid(title_caption, label_prefix, base_dir, filename_template):
    """Generate a 1x1 figure for when only GM is available."""
    latex = []
    latex.append(r"\begin{figure}[htbp]")
    latex.append(r"    \centering")
    
    filename = filename_template.replace("{agg}", "GeometricMedian")
    path = f"../{base_dir}/{filename}"
    
    latex.append(
        f"    \\includegraphics[width=0.6\\textwidth]{{{path}}}\n"
        f"    \\caption{{Geometric Median (Only available aggregator)\\newline {title_caption}}}\n"
        f"    \\label{{fig:{label_prefix}_grid}}"
    )
    latex.append(r"\end{figure}")
    return "\n".join(latex)


def main():
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    latex_dir = os.path.join(workspace_dir, "latex_plots")
    os.makedirs(latex_dir, exist_ok=True)

    tex_path = os.path.join(latex_dir, "aggregator_comparison_report.tex")

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

\title{Per-Model Aggregator Comparison:\\Centered Clipping, Geometric Median, Multi-Krum, Trimmed Mean}
\author{ByzFL SNN Benchmark}
\date{\today}

\begin{document}

\maketitle

\clearpage
\tableofcontents
\clearpage

\section{Introduction}
This report provides a grid of heatmaps for each model architecture, showing the worst-case test accuracy across all attacks for four different robust aggregators:
\begin{enumerate}
    \item Centered Clipping
    \item Geometric Median
    \item Multi-Krum
    \item Trimmed Mean
\end{enumerate}
This allows for a direct visual comparison of which aggregators perform best for Sparse vs. Dense gradients.

\clearpage
""")

        # 1. SNN Atan
        f.write("\\section{SNN Atan ($\\alpha=1.2$)}\n")
        f.write(make_2x2_grid(
            "SNN Atan ($\\alpha=1.2$) - Worst-case accuracy across all attacks for different aggregators.",
            "snn_atan",
            "plots/robust_new_atan_sweep/alpha_1.2",
            "test_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_{agg}_nb_honest_clients_10_tolerated_f_equal_real.pdf"
        ))
        f.write("\\clearpage\n")

        # 2. SNN Tri
        f.write("\\section{SNN Tri ($\\beta=2.0$)}\n")
        f.write(make_2x2_grid(
            "SNN Tri ($\\beta=2.0$) - Worst-case accuracy across all attacks for different aggregators.",
            "snn_tri",
            "plots/robust_new_tri_sweep/beta_2.0",
            "test_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_{agg}_nb_honest_clients_10_tolerated_f_equal_real.pdf"
        ))
        f.write("\\clearpage\n")

        # 3. SNN Box
        f.write("\\section{SNN Box ($\\beta=2.0$)}\n")
        f.write(make_2x2_grid(
            "SNN Box ($\\beta=2.0$) - Worst-case accuracy across all attacks for different aggregators.",
            "snn_box",
            "plots/robust_new_box_sweep/beta_2.0",
            "test_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_{agg}_nb_honest_clients_10_tolerated_f_equal_real.pdf"
        ))
        f.write("\\clearpage\n")

        # 4. CNN ReLU (Baseline LR=0.15)
        f.write("\\section{CNN ReLU (Baseline, $LR=0.15$)}\n")
        f.write(make_2x2_grid(
            "CNN ReLU (No Clip, $LR=0.15$) - Worst-case accuracy across all attacks for different aggregators.",
            "cnn_relu",
            "plots/robust_comparison_sweep/learning_rate_0.15",
            "test_mnist_cnn_mnist_gamma_similarity_niid_NNM_ARC_{agg}_nb_honest_clients_10_tolerated_f_equal_real.pdf"
        ))
        f.write("\\clearpage\n")

        # 5. CNN Tanh
        f.write("\\section{CNN Tanh}\n")
        f.write(make_2x2_grid(
            "CNN Tanh - Worst-case accuracy across all attacks for different aggregators.",
            "cnn_tanh",
            "plots/cnn_tanh_heatmaps",
            "test_mnist_cnn_mnist_tanh_gamma_similarity_niid_NNM_ARC_{agg}_nb_honest_clients_10_tolerated_f_equal_real.pdf"
        ))
        f.write("\\clearpage\n")

        # 6. CNN Clipped
        f.write("\\section{CNN ReLU (Gradient Clipped)}\n")
        f.write(make_1x1_grid(
            "CNN ReLU (Grad Clipped) - Worst-case accuracy.",
            "cnn_clipped",
            "plots/cnn_clipped_heatmaps",
            "test_mnist_cnn_mnist_gamma_similarity_niid_NNM_ARC_{agg}_nb_honest_clients_10_tolerated_f_equal_real.pdf"
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
        print(f"LaTeX report compiled successfully: aggregator_comparison_report.pdf")
    except Exception as e:
        print(f"[ERROR] Failed to compile LaTeX: {e}")

if __name__ == "__main__":
    main()
