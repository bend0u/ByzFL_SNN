import os
import sys
import json
import shutil
import subprocess

# Ensure workspace root is in Python path
workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, workspace_dir)

from byzfl.benchmark.evaluate_results import aggregated_test_heatmap

def main():
    latex_dir = os.path.join(workspace_dir, "latex_plots")
    os.makedirs(latex_dir, exist_ok=True)

    # 1. Atan alphas
    atan_results_dir = os.path.join(workspace_dir, "results", "snn", "robust_new_atan_sweep")
    atan_plots_dir = os.path.join(workspace_dir, "plots", "snn", "robust_new_atan_sweep")
    alphas = [0.5, 0.75, 1.0, 1.5, 2.0]
    
    atan_config_path = os.path.join(atan_results_dir, "config.json")
    if not os.path.exists(atan_config_path):
        print(f"Error: {atan_config_path} not found.")
        sys.exit(1)
        
    atan_config_backup = atan_config_path + ".bak"
    shutil.copyfile(atan_config_path, atan_config_backup)

    print("Generating and copying Atan alpha plots...")
    try:
        with open(atan_config_path, "r") as f:
            config_data = json.load(f)

        for alpha in alphas:
            print(f"\n---> Processing Atan alpha = {alpha}")
            temp_config = json.loads(json.dumps(config_data))
            temp_config["model"]["model_params"]["surrogate_params"]["alpha"] = alpha
            
            with open(atan_config_path, "w") as f:
                json.dump(temp_config, f, indent=4)
                
            val_plots_dir = os.path.join(atan_plots_dir, f"alpha_{alpha}")
            os.makedirs(val_plots_dir, exist_ok=True)
            
            # Generate for:
            # A. Worst-case over all attacks (best aggregator)
            print("Generating heatmap for Worst-case (Best Overall Aggregator)...")
            aggregated_test_heatmap(atan_results_dir, val_plots_dir, target_attack=None, metric="best_step")
            
            # B. Optimal ALIE attack
            print("Generating heatmap for attack Optimal_ALittleIsEnough_neg1...")
            aggregated_test_heatmap(atan_results_dir, val_plots_dir, target_attack="Optimal_ALittleIsEnough_neg1", metric="best_step")
            
            # C. Sign Flipping attack
            print("Generating heatmap for attack SignFlipping...")
            aggregated_test_heatmap(atan_results_dir, val_plots_dir, target_attack="SignFlipping", metric="best_step")
            
            # Copy to latex_plots with nice names
            alpha_str = str(alpha).replace(".", "_")
            
            # Names in plots folder
            src_best_name = "best_test_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real"
            src_alie_name = "best_test_Optimal_ALittleIsEnough_neg1_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real"
            src_sf_name = "best_test_SignFlipping_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real"
            
            # Destination names
            dest_best_name = f"heatmap_alpha_{alpha_str}_best"
            dest_alie_name = f"heatmap_alpha_{alpha_str}_optalie"
            dest_sf_name = f"heatmap_alpha_{alpha_str}_sf"
            
            for ext in [".pdf", ".png"]:
                # Copy Best
                src_best = os.path.join(val_plots_dir, src_best_name + ext)
                dest_best = os.path.join(latex_dir, dest_best_name + ext)
                if os.path.exists(src_best):
                    shutil.copyfile(src_best, dest_best)
                    print(f"Copied {src_best} to {dest_best}")
                else:
                    print(f"[WARNING] Could not find {src_best}")

                # Copy OptAlie
                src_alie = os.path.join(val_plots_dir, src_alie_name + ext)
                dest_alie = os.path.join(latex_dir, dest_alie_name + ext)
                if os.path.exists(src_alie):
                    shutil.copyfile(src_alie, dest_alie)
                    print(f"Copied {src_alie} to {dest_alie}")
                else:
                    print(f"[WARNING] Could not find {src_alie}")
                    
                # Copy SignFlipping
                src_sf = os.path.join(val_plots_dir, src_sf_name + ext)
                dest_sf = os.path.join(latex_dir, dest_sf_name + ext)
                if os.path.exists(src_sf):
                    shutil.copyfile(src_sf, dest_sf)
                    print(f"Copied {src_sf} to {dest_sf}")
                else:
                    print(f"[WARNING] Could not find {src_sf}")
    finally:
        if os.path.exists(atan_config_backup):
            shutil.move(atan_config_backup, atan_config_path)
            print("Restored original Atan config.json successfully.")

    # 2. Tri betas
    tri_results_dir = os.path.join(workspace_dir, "results", "snn", "robust_new_tri_sweep")
    tri_plots_dir = os.path.join(workspace_dir, "plots", "snn", "robust_new_tri_sweep")
    betas = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    
    tri_config_path = os.path.join(tri_results_dir, "config.json")
    if not os.path.exists(tri_config_path):
        print(f"Error: {tri_config_path} not found.")
        sys.exit(1)
        
    tri_config_backup = tri_config_path + ".bak"
    shutil.copyfile(tri_config_path, tri_config_backup)

    print("\nGenerating and copying Tri beta plots...")
    try:
        with open(tri_config_path, "r") as f:
            config_data = json.load(f)

        for beta in betas:
            print(f"\n---> Processing Tri beta = {beta}")
            temp_config = json.loads(json.dumps(config_data))
            temp_config["model"]["model_params"]["surrogate_params"]["beta"] = beta
            
            with open(tri_config_path, "w") as f:
                json.dump(temp_config, f, indent=4)
                
            val_plots_dir = os.path.join(tri_plots_dir, f"beta_{beta}")
            os.makedirs(val_plots_dir, exist_ok=True)
            
            # Generate for:
            # A. Worst-case over all attacks (best aggregator)
            print("Generating heatmap for Worst-case (Best Overall Aggregator)...")
            aggregated_test_heatmap(tri_results_dir, val_plots_dir, target_attack=None, metric="best_step")
            
            # B. Optimal ALIE attack
            print("Generating heatmap for attack Optimal_ALittleIsEnough_neg1...")
            aggregated_test_heatmap(tri_results_dir, val_plots_dir, target_attack="Optimal_ALittleIsEnough_neg1", metric="best_step")
            
            # C. Sign Flipping attack
            print("Generating heatmap for attack SignFlipping...")
            aggregated_test_heatmap(tri_results_dir, val_plots_dir, target_attack="SignFlipping", metric="best_step")
            
            # Copy to latex_plots with nice names
            beta_str = str(beta).replace(".", "_")
            
            # Names in plots folder
            src_best_name = "best_test_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real"
            src_alie_name = "best_test_Optimal_ALittleIsEnough_neg1_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real"
            src_sf_name = "best_test_SignFlipping_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real"
            
            # Destination names
            dest_best_name = f"heatmap_tri_beta_{beta_str}_best"
            dest_alie_name = f"heatmap_tri_beta_{beta_str}_optalie"
            dest_sf_name = f"heatmap_tri_beta_{beta_str}_sf"
            
            for ext in [".pdf", ".png"]:
                # Copy Best
                src_best = os.path.join(val_plots_dir, src_best_name + ext)
                dest_best = os.path.join(latex_dir, dest_best_name + ext)
                if os.path.exists(src_best):
                    shutil.copyfile(src_best, dest_best)
                    print(f"Copied {src_best} to {dest_best}")
                else:
                    print(f"[WARNING] Could not find {src_best}")

                # Copy OptAlie
                src_alie = os.path.join(val_plots_dir, src_alie_name + ext)
                dest_alie = os.path.join(latex_dir, dest_alie_name + ext)
                if os.path.exists(src_alie):
                    shutil.copyfile(src_alie, dest_alie)
                    print(f"Copied {src_alie} to {dest_alie}")
                else:
                    print(f"[WARNING] Could not find {src_alie}")
                    
                # Copy SignFlipping
                src_sf = os.path.join(val_plots_dir, src_sf_name + ext)
                dest_sf = os.path.join(latex_dir, dest_sf_name + ext)
                if os.path.exists(src_sf):
                    shutil.copyfile(src_sf, dest_sf)
                    print(f"Copied {src_sf} to {dest_sf}")
                else:
                    print(f"[WARNING] Could not find {src_sf}")
    finally:
        if os.path.exists(tri_config_backup):
            shutil.move(tri_config_backup, tri_config_path)
            print("Restored original Tri config.json successfully.")

    # 3. Box betas
    box_results_dir = os.path.join(workspace_dir, "results", "snn", "robust_new_box_sweep")
    box_plots_dir = os.path.join(workspace_dir, "plots", "snn", "robust_new_box_sweep")
    betas_box = [0.25, 0.5, 0.75, 1.0, 1.25, 2.0]
    
    box_config_path = os.path.join(box_results_dir, "config.json")
    if not os.path.exists(box_config_path):
        print(f"Error: {box_config_path} not found.")
        sys.exit(1)
        
    box_config_backup = box_config_path + ".bak"
    shutil.copyfile(box_config_path, box_config_backup)

    print("\nGenerating and copying Box beta plots...")
    try:
        with open(box_config_path, "r") as f:
            config_data = json.load(f)

        for beta in betas_box:
            print(f"\n---> Processing Box beta = {beta}")
            temp_config = json.loads(json.dumps(config_data))
            temp_config["model"]["model_params"]["surrogate_params"]["beta"] = beta
            
            with open(box_config_path, "w") as f:
                json.dump(temp_config, f, indent=4)
                
            val_plots_dir = os.path.join(box_plots_dir, f"beta_{beta}")
            os.makedirs(val_plots_dir, exist_ok=True)
            
            # Generate for:
            # A. Worst-case over all attacks (best aggregator)
            print("Generating heatmap for Worst-case (Best Overall Aggregator)...")
            aggregated_test_heatmap(box_results_dir, val_plots_dir, target_attack=None, metric="best_step")
            
            # B. Optimal ALIE attack
            print("Generating heatmap for attack Optimal_ALittleIsEnough_neg1...")
            aggregated_test_heatmap(box_results_dir, val_plots_dir, target_attack="Optimal_ALittleIsEnough_neg1", metric="best_step")
            
            # C. Sign Flipping attack
            print("Generating heatmap for attack SignFlipping...")
            aggregated_test_heatmap(box_results_dir, val_plots_dir, target_attack="SignFlipping", metric="best_step")
            
            # Copy to latex_plots with nice names
            beta_str = str(beta).replace(".", "_")
            
            # Names in plots folder
            src_best_name = "best_test_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real"
            src_alie_name = "best_test_Optimal_ALittleIsEnough_neg1_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real"
            src_sf_name = "best_test_SignFlipping_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real"
            
            # Destination names
            dest_best_name = f"heatmap_box_beta_{beta_str}_best"
            dest_alie_name = f"heatmap_box_beta_{beta_str}_optalie"
            dest_sf_name = f"heatmap_box_beta_{beta_str}_sf"
            
            for ext in [".pdf", ".png"]:
                # Copy Best
                src_best = os.path.join(val_plots_dir, src_best_name + ext)
                dest_best = os.path.join(latex_dir, dest_best_name + ext)
                if os.path.exists(src_best):
                    shutil.copyfile(src_best, dest_best)
                    print(f"Copied {src_best} to {dest_best}")
                else:
                    print(f"[WARNING] Could not find {src_best}")

                # Copy OptAlie
                src_alie = os.path.join(val_plots_dir, src_alie_name + ext)
                dest_alie = os.path.join(latex_dir, dest_alie_name + ext)
                if os.path.exists(src_alie):
                    shutil.copyfile(src_alie, dest_alie)
                    print(f"Copied {src_alie} to {dest_alie}")
                else:
                    print(f"[WARNING] Could not find {src_alie}")
                    
                # Copy SignFlipping
                src_sf = os.path.join(val_plots_dir, src_sf_name + ext)
                dest_sf = os.path.join(latex_dir, dest_sf_name + ext)
                if os.path.exists(src_sf):
                    shutil.copyfile(src_sf, dest_sf)
                    print(f"Copied {src_sf} to {dest_sf}")
                else:
                    print(f"[WARNING] Could not find {src_sf}")
    finally:
        if os.path.exists(box_config_backup):
            shutil.move(box_config_backup, box_config_path)
            print("Restored original Box config.json successfully.")

    # 4. Generate LaTeX content
    tex_path = os.path.join(latex_dir, "comparison_plots_atan_vs_tri.tex")
    
    with open(tex_path, "w") as f:
        f.write(r"""\documentclass{article}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{subcaption}
\usepackage{amsmath}

\geometry{
    a4paper,
    margin=0.6in
}

\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,      
    urlcolor=cyan,
}

\title{SNN Surrogate Gradient Parameter Comparison:\\ArcTangent ($\alpha$) vs. Triangular ($\beta$) vs. Box ($\beta$) under Robust Sweeps}
\author{ByzFL SNN Benchmark}
\date{\today}

\begin{document}

\maketitle

\section{Introduction}
This document presents a comprehensive comparison of Spiking Neural Network (SNN) robustness on the MNIST dataset under data heterogeneity ($\gamma$) and Byzantine attacks ($f$). We evaluate and compare three surrogate gradient functions under fixed attacks and overall:
\begin{itemize}
    \item \textbf{ArcTangent (\texttt{atan})}: parameterized by stiffness factor $\alpha \in \{0.5, 0.75, 1.0, 1.5, 2.0\}$.
    \item \textbf{Triangular (\texttt{tri})}: parameterized by stiffness factor $\beta \in \{0.5, 0.75, 1.0, 1.25, 1.5, 2.0\}$.
    \item \textbf{Box (\texttt{box})}: parameterized by stiffness factor $\beta \in \{0.25, 0.5, 0.75, 1.0, 1.25, 2.0\}$.
\end{itemize}

The evaluations are carried out on $N = 10$ honest nodes + $f$ Byzantine nodes (up to $f=5$ for \texttt{atan}, \texttt{tri}, and \texttt{box}), across non-IID levels $\gamma \in \{1.0, 0.66, 0.33, 0.0\}$. 

The heatmaps display the best test accuracy at convergence (validation-based best step) achieved by the optimal aggregator choice (Centered Clipping or Geometric Median, with NNM and ARC pre-aggregation). We evaluate performance under three scenarios:
\begin{enumerate}
    \item \textbf{Best Overall Aggregator (Worst-Case Across Attacks)}
    \item \textbf{Optimal A Little Is Enough (ALIE) Attack} (\texttt{Optimal\_ALittleIsEnough\_neg1})
    \item \textbf{Sign Flipping (SF) Attack} (\texttt{SignFlipping})
\end{enumerate}

\clearpage

\section{ArcTangent (\texttt{atan}) Surrogate Gradient stiffness sweeps}

\subsection{Best Overall Aggregator (Worst-Case Across Attacks)}
The figure below displays the test accuracy heatmaps under the worst-case attack scenario for different $\alpha$ values of the ArcTangent surrogate gradient.

\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_alpha_0_5_best.pdf}
        \caption{$\alpha = 0.5$}
        \label{fig:alpha_0_5_best}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_alpha_0_75_best.pdf}
        \caption{$\alpha = 0.75$}
        \label{fig:alpha_0_75_best}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_alpha_1_0_best.pdf}
        \caption{$\alpha = 1.0$}
        \label{fig:alpha_1_0_best}
    \end{subfigure}
    
    \vspace{0.4cm}
    
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_alpha_1_5_best.pdf}
        \caption{$\alpha = 1.5$}
        \label{fig:alpha_1_5_best}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_alpha_2_0_best.pdf}
        \caption{$\alpha = 2.0$}
        \label{fig:alpha_2_0_best}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \vspace{2cm}
        \label{fig:alpha_spacer_best}
    \end{subfigure}
    
    \caption{ArcTangent (\texttt{atan}) comparison for best overall aggregator (worst-case) for $\alpha \in \{0.5, 0.75, 1.0, 1.5, 2.0\}$.}
    \label{fig:atan_comparison_best}
\end{figure}

\clearpage

\subsection{Optimal ALIE Attack}
The figure below displays the best test accuracy heatmaps under the Optimal ALIE attack for different $\alpha$ values of the ArcTangent surrogate gradient.

\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_alpha_0_5_optalie.pdf}
        \caption{$\alpha = 0.5$}
        \label{fig:alpha_0_5_optalie}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_alpha_0_75_optalie.pdf}
        \caption{$\alpha = 0.75$}
        \label{fig:alpha_0_75_optalie}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_alpha_1_0_optalie.pdf}
        \caption{$\alpha = 1.0$}
        \label{fig:alpha_1_0_optalie}
    \end{subfigure}
    
    \vspace{0.4cm}
    
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_alpha_1_5_optalie.pdf}
        \caption{$\alpha = 1.5$}
        \label{fig:alpha_1_5_optalie}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_alpha_2_0_optalie.pdf}
        \caption{$\alpha = 2.0$}
        \label{fig:alpha_2_0_optalie}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \vspace{2cm}
        \label{fig:alpha_spacer_optalie}
    \end{subfigure}
    
    \caption{ArcTangent (\texttt{atan}) comparison under Optimal ALIE attack for $\alpha \in \{0.5, 0.75, 1.0, 1.5, 2.0\}$.}
    \label{fig:atan_comparison_optalie}
\end{figure}

\clearpage

\subsection{Sign Flipping Attack}
The figure below displays the best test accuracy heatmaps under the Sign Flipping attack for different $\alpha$ values of the ArcTangent surrogate gradient.

\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_alpha_0_5_sf.pdf}
        \caption{$\alpha = 0.5$}
        \label{fig:alpha_0_5_sf}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_alpha_0_75_sf.pdf}
        \caption{$\alpha = 0.75$}
        \label{fig:alpha_0_75_sf}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_alpha_1_0_sf.pdf}
        \caption{$\alpha = 1.0$}
        \label{fig:alpha_1_0_sf}
    \end{subfigure}
    
    \vspace{0.4cm}
    
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_alpha_1_5_sf.pdf}
        \caption{$\alpha = 1.5$}
        \label{fig:alpha_1_5_sf}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_alpha_2_0_sf.pdf}
        \caption{$\alpha = 2.0$}
        \label{fig:alpha_2_0_sf}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \vspace{2cm}
        \label{fig:alpha_spacer_sf}
    \end{subfigure}
    
    \caption{ArcTangent (\texttt{atan}) comparison under Sign Flipping attack for $\alpha \in \{0.5, 0.75, 1.0, 1.5, 2.0\}$.}
    \label{fig:atan_comparison_sf}
\end{figure}

\clearpage

\section{Triangular (\texttt{tri}) Surrogate Gradient stiffness sweeps}

\subsection{Best Overall Aggregator (Worst-Case Across Attacks)}
The figure below displays the test accuracy heatmaps under the worst-case attack scenario for different $\beta$ values of the Triangular surrogate gradient.

\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_tri_beta_0_5_best.pdf}
        \caption{$\beta = 0.5$}
        \label{fig:beta_0_5_best}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_tri_beta_0_75_best.pdf}
        \caption{$\beta = 0.75$}
        \label{fig:beta_0_75_best}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_tri_beta_1_0_best.pdf}
        \caption{$\beta = 1.0$}
        \label{fig:beta_1_0_best}
    \end{subfigure}
    
    \vspace{0.4cm}
    
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_tri_beta_1_25_best.pdf}
        \caption{$\beta = 1.25$}
        \label{fig:beta_1_25_best}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_tri_beta_1_5_best.pdf}
        \caption{$\beta = 1.5$}
        \label{fig:beta_1_5_best}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_tri_beta_2_0_best.pdf}
        \caption{$\beta = 2.0$}
        \label{fig:beta_2_0_best}
    \end{subfigure}
    
    \caption{Triangular (\texttt{tri}) comparison for best overall aggregator (worst-case) for $\beta \in \{0.5, 0.75, 1.0, 1.25, 1.5, 2.0\}$.}
    \label{fig:tri_comparison_best}
\end{figure}

\clearpage

\subsection{Optimal ALIE Attack}
The figure below displays the best test accuracy heatmaps under the Optimal ALIE attack for different $\beta$ values of the Triangular surrogate gradient.

\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_tri_beta_0_5_optalie.pdf}
        \caption{$\beta = 0.5$}
        \label{fig:beta_0_5_optalie}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_tri_beta_0_75_optalie.pdf}
        \caption{$\beta = 0.75$}
        \label{fig:beta_0_75_optalie}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_tri_beta_1_0_optalie.pdf}
        \caption{$\beta = 1.0$}
        \label{fig:beta_1_0_optalie}
    \end{subfigure}
    
    \vspace{0.4cm}
    
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_tri_beta_1_25_optalie.pdf}
        \caption{$\beta = 1.25$}
        \label{fig:beta_1_25_optalie}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_tri_beta_1_5_optalie.pdf}
        \caption{$\beta = 1.5$}
        \label{fig:beta_1_5_optalie}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_tri_beta_2_0_optalie.pdf}
        \caption{$\beta = 2.0$}
        \label{fig:beta_2_0_optalie}
    \end{subfigure}
    
    \caption{Triangular (\texttt{tri}) comparison under Optimal ALIE attack for $\beta \in \{0.5, 0.75, 1.0, 1.25, 1.5, 2.0\}$.}
    \label{fig:tri_comparison_optalie}
\end{figure}

\clearpage

\subsection{Sign Flipping Attack}
The figure below displays the best test accuracy heatmaps under the Sign Flipping attack for different $\beta$ values of the Triangular surrogate gradient.

\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_tri_beta_0_5_sf.pdf}
        \caption{$\beta = 0.5$}
        \label{fig:beta_0_5_sf}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_tri_beta_0_75_sf.pdf}
        \caption{$\beta = 0.75$}
        \label{fig:beta_0_75_sf}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_tri_beta_1_0_sf.pdf}
        \caption{$\beta = 1.0$}
        \label{fig:beta_1_0_sf}
    \end{subfigure}
    
    \vspace{0.4cm}
    
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_tri_beta_1_25_sf.pdf}
        \caption{$\beta = 1.25$}
        \label{fig:beta_1_25_sf}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_tri_beta_1_5_sf.pdf}
        \caption{$\beta = 1.5$}
        \label{fig:beta_1_5_sf}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_tri_beta_2_0_sf.pdf}
        \caption{$\beta = 2.0$}
        \label{fig:beta_2_0_sf}
    \end{subfigure}
    
    \caption{Triangular (\texttt{tri}) comparison under Sign Flipping attack for $\beta \in \{0.5, 0.75, 1.0, 1.25, 1.5, 2.0\}$.}
    \label{fig:tri_comparison_sf}
\end{figure}

\clearpage

\section{Box (\texttt{box}) Surrogate Gradient stiffness sweeps}

\subsection{Best Overall Aggregator (Worst-Case Across Attacks)}
The figure below displays the test accuracy heatmaps under the worst-case attack scenario for different $\beta$ values of the Box surrogate gradient.

\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_box_beta_0_25_best.pdf}
        \caption{$\beta = 0.25$}
        \label{fig:box_beta_0_25_best}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_box_beta_0_5_best.pdf}
        \caption{$\beta = 0.5$}
        \label{fig:box_beta_0_5_best}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_box_beta_0_75_best.pdf}
        \caption{$\beta = 0.75$}
        \label{fig:box_beta_0_75_best}
    \end{subfigure}
    
    \vspace{0.4cm}
    
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_box_beta_1_0_best.pdf}
        \caption{$\beta = 1.0$}
        \label{fig:box_beta_1_0_best}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_box_beta_1_25_best.pdf}
        \caption{$\beta = 1.25$}
        \label{fig:box_beta_1_25_best}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_box_beta_2_0_best.pdf}
        \caption{$\beta = 2.0$}
        \label{fig:box_beta_2_0_best}
    \end{subfigure}
    
    \caption{Box (\texttt{box}) comparison for best overall aggregator (worst-case) for $\beta \in \{0.25, 0.5, 0.75, 1.0, 1.25, 2.0\}$.}
    \label{fig:box_comparison_best}
\end{figure}

\clearpage

\subsection{Optimal ALIE Attack}
The figure below displays the best test accuracy heatmaps under the Optimal ALIE attack for different $\beta$ values of the Box surrogate gradient.

\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_box_beta_0_25_optalie.pdf}
        \caption{$\beta = 0.25$}
        \label{fig:box_beta_0_25_optalie}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_box_beta_0_5_optalie.pdf}
        \caption{$\beta = 0.5$}
        \label{fig:box_beta_0_5_optalie}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_box_beta_0_75_optalie.pdf}
        \caption{$\beta = 0.75$}
        \label{fig:box_beta_0_75_optalie}
    \end{subfigure}
    
    \vspace{0.4cm}
    
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_box_beta_1_0_optalie.pdf}
        \caption{$\beta = 1.0$}
        \label{fig:box_beta_1_0_optalie}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_box_beta_1_25_optalie.pdf}
        \caption{$\beta = 1.25$}
        \label{fig:box_beta_1_25_optalie}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_box_beta_2_0_optalie.pdf}
        \caption{$\beta = 2.0$}
        \label{fig:box_beta_2_0_optalie}
    \end{subfigure}
    
    \caption{Box (\texttt{box}) comparison under Optimal ALIE attack for $\beta \in \{0.25, 0.5, 0.75, 1.0, 1.25, 2.0\}$.}
    \label{fig:box_comparison_optalie}
\end{figure}

\clearpage

\subsection{Sign Flipping Attack}
The figure below displays the best test accuracy heatmaps under the Sign Flipping attack for different $\beta$ values of the Box surrogate gradient.

\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_box_beta_0_25_sf.pdf}
        \caption{$\beta = 0.25$}
        \label{fig:box_beta_0_25_sf}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_box_beta_0_5_sf.pdf}
        \caption{$\beta = 0.5$}
        \label{fig:box_beta_0_5_sf}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_box_beta_0_75_sf.pdf}
        \caption{$\beta = 0.75$}
        \label{fig:box_beta_0_75_sf}
    \end{subfigure}
    
    \vspace{0.4cm}
    
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_box_beta_1_0_sf.pdf}
        \caption{$\beta = 1.0$}
        \label{fig:box_beta_1_0_sf}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_box_beta_1_25_sf.pdf}
        \caption{$\beta = 1.25$}
        \label{fig:box_beta_1_25_sf}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_box_beta_2_0_sf.pdf}
        \caption{$\beta = 2.0$}
        \label{fig:box_beta_2_0_sf}
    \end{subfigure}
    
    \caption{Box (\texttt{box}) comparison under Sign Flipping attack for $\beta \in \{0.25, 0.5, 0.75, 1.0, 1.25, 2.0\}$.}
    \label{fig:box_comparison_sf}
\end{figure}

\end{document}
""")

    print(f"\nLaTeX file written to: {tex_path}")
    
    # 5. Compile the LaTeX document
    print("Compiling LaTeX to PDF...")
    try:
        for _ in range(2):
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-disable-installer", "comparison_plots_atan_vs_tri.tex"],
                cwd=latex_dir,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        # Cleanup LaTeX auxiliary files
        for ext in [".aux", ".log", ".out", ".toc", ".fls", ".fdb_latexmk", ".synctex.gz"]:
            aux_file = os.path.join(latex_dir, "comparison_plots_atan_vs_tri" + ext)
            if os.path.exists(aux_file):
                os.remove(aux_file)
        print(f"--> Successfully generated PDF: {os.path.join(latex_dir, 'comparison_plots_atan_vs_tri.pdf')}")
    except FileNotFoundError:
        print("[INFO] pdflatex not found. LaTeX file ready for manual compilation.")
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] pdflatex returned non-zero. Check log for details.")

if __name__ == "__main__":
    main()
