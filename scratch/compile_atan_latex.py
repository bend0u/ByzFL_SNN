import os
import sys
import json
import shutil
import numpy as np
import subprocess

# Ensure workspace root is in Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from byzfl.benchmark.managers import ParamsManager, get_snn_suffix
from byzfl.benchmark.evaluate_results import (
    get_accuracy_at_best_step,
    get_max_test_accuracy,
    get_dist_param_val,
    custom_dict_to_str,
    ensure_list
)

def get_heatmap_data(results_dir, alpha_val):
    config_path = os.path.join(results_dir, "config.json")
    with open(config_path, "r") as file:
        data = json.load(file)

    # Set up parameters
    data["model"]["model_params"]["surrogate_params"]["alpha"] = alpha_val
    data["benchmark_config"]["f"] = [0, 1, 2, 3, 4, 5]

    pm = ParamsManager(data)
    snn_suffix = get_snn_suffix(pm)
    clean = False

    path_to_hyperparameters = os.path.join(results_dir, "best_hyperparameters")

    training_seed = data["benchmark_config"]["training_seed"]
    nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
    nb_honest_clients = data["benchmark_config"]["nb_honest_clients"]
    nb_byz = data["benchmark_config"]["f"]
    nb_declared = [5]  # tolerated_f = 5
    data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
    nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
    data_distributions = data["benchmark_config"]["data_distribution"]
    set_honest_clients_as_clients = data["benchmark_config"]["set_honest_clients_as_clients"]
    evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]

    model_name = data["model"]["name"]
    dataset_name = data["model"]["dataset_name"]
    lr_list = ensure_list(data["model"]["learning_rate"])

    momentum_list = ensure_list(data["honest_clients"]["momentum"])
    wd_list = ensure_list(data["honest_clients"]["weight_decay"])

    aggregators = ensure_list(data["aggregator"])
    pre_aggregators = [data["pre_aggregators"]]  # Unify pre_aggregators

    attacks = ensure_list(data["attack"])

    nb_honest_clients = ensure_list(nb_honest_clients)
    nb_byz = ensure_list(nb_byz)
    nb_declared = ensure_list(nb_declared)
    data_distributions = ensure_list(data_distributions)

    pre_agg = pre_aggregators[0]
    pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
    pre_agg_names = "_".join(pre_agg_list_names)

    nb_honest = nb_honest_clients[0]
    nb_decl = nb_declared[0]
    actual_nb_byz = [item for item in nb_byz if item <= nb_decl]
    data_dist = data_distributions[0]
    distribution_parameter_list = ensure_list(data_dist["distribution_parameter"])

    heat_map_cube = np.zeros((len(aggregators), len(distribution_parameter_list), len(actual_nb_byz)))

    for z, agg in enumerate(aggregators):
        heat_map_table = np.zeros((len(distribution_parameter_list), len(actual_nb_byz)))

        for y, nb_byzantine in enumerate(actual_nb_byz):
            nb_decl = nb_byzantine  # declared_equal_real is True
            nb_nodes = nb_honest + nb_byzantine

            for x, dist_param in enumerate(distribution_parameter_list):
                dist_param_val = get_dist_param_val(data_dist['name'], dist_param)

                hyper_file_name = (
                    f"{dataset_name}_"
                    f"{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                    f"{custom_dict_to_str(data_dist['name'])}_{dist_param_val}_"
                    f"{pre_agg_names}_{agg['name']}.txt"
                )

                full_path = os.path.join(path_to_hyperparameters, "hyperparameters", hyper_file_name)

                if os.path.exists(full_path):
                    hyperparameters = np.loadtxt(full_path)
                    lr = hyperparameters[0]
                    momentum = hyperparameters[1]
                    wd = hyperparameters[2]
                else:
                    lr = lr_list[0]
                    momentum = momentum_list[0]
                    wd = wd_list[0]

                worst_accuracy = np.inf
                for attack in attacks:
                    accuracy = get_accuracy_at_best_step(
                        results_dir, clean, dataset_name, model_name, nb_nodes, nb_byzantine, nb_decl,
                        data_dist['name'], dist_param_val, agg['name'], pre_agg_names, attack['name'],
                        lr, momentum, wd, snn_suffix, pm,
                        nb_data_distribution_seeds, nb_training_seeds,
                        training_seed, data_distribution_seed, evaluation_delta
                    )
                    if accuracy < worst_accuracy:
                        worst_accuracy = accuracy

                heat_map_table[len(heat_map_table)-1-x][y] = worst_accuracy

        heat_map_cube[z] = heat_map_table

    # Best aggregator is the max across the aggregator axis
    final_heat_map = np.max(heat_map_cube, axis=0)
    return final_heat_map, distribution_parameter_list, actual_nb_byz

def main():
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(workspace_dir, "results", "snn", "robust_new_atan_sweep")
    plots_dir = os.path.join(workspace_dir, "plots", "snn", "robust_new_atan_sweep_f5")
    latex_dir = os.path.join(workspace_dir, "latex_plots")
    os.makedirs(latex_dir, exist_ok=True)

    alphas = [1.0, 1.25, 1.5, 2.0, 3.0]
    data_summary = {}

    print("Extracting accuracy data for each alpha value...")
    for alpha in alphas:
        grid, gammas, f_vals = get_heatmap_data(results_dir, alpha)
        data_summary[alpha] = {
            "grid": grid,
            "gammas": gammas,
            "f_vals": f_vals
        }

        # Copy the PDF plot file to latex_plots directory with a clean filename
        # E.g. alpha = 1.25 -> alpha_str = "1_25"
        alpha_str = str(alpha).replace(".", "_")
        src_pdf_name = "best_test_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real.pdf"
        src_png_name = "best_test_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real.png"
        
        src_pdf_path = os.path.join(plots_dir, f"alpha_{alpha}", src_pdf_name)
        src_png_path = os.path.join(plots_dir, f"alpha_{alpha}", src_png_name)
        
        dest_pdf_path = os.path.join(latex_dir, f"heatmap_alpha_{alpha_str}.pdf")
        dest_png_path = os.path.join(latex_dir, f"heatmap_alpha_{alpha_str}.png")
        
        if os.path.exists(src_pdf_path):
            shutil.copyfile(src_pdf_path, dest_pdf_path)
            print(f"--> Copied PDF plot for alpha={alpha} to {dest_pdf_path}")
        else:
            print(f"--> [WARNING] PDF plot not found at {src_pdf_path}")
            
        if os.path.exists(src_png_path):
            shutil.copyfile(src_png_path, dest_png_path)
            print(f"--> Copied PNG plot for alpha={alpha} to {dest_png_path}")

    # Generate LaTeX content
    tex_path = os.path.join(latex_dir, "comparison_plots_atan_alpha_f5.tex")
    
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
    margin=0.7in
}

\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,      
    urlcolor=cyan,
}

\title{SNN Surrogate Gradient $\alpha$ Parameter Comparison\\Robustness Sweep under Data Heterogeneity ($f \le 5$)}
\author{ByzFL SNN Benchmark}
\date{\today}

\begin{document}

\maketitle

\section{Introduction}
This document presents a detailed comparison of the Spiking Neural Network (SNN) robustness on the MNIST dataset using different scaling factors $\alpha$ for the ArcTangent (\texttt{atan}) surrogate gradient. The evaluation spans:
\begin{itemize}
    \item \textbf{Surrogate Gradient Scaling}: $\alpha \in \{1.0, 1.25, 1.5, 2.0, 3.0\}$
    \item \textbf{Byzantine Nodes}: $f \in \{0, 1, 2, 3, 4, 5\}$ out of $N=10$ honest nodes
    \item \textbf{Data Heterogeneity}: Non-IID levels $\gamma \in \{1.0, 0.66, 0.33, 0.0\}$ (where $\gamma = 0.0$ represents extreme Non-IID client partitions)
    \item \textbf{Attack Model}: Optimal A Little Is Enough (\texttt{Optimal\_ALittleIsEnough\_neg1})
    \item \textbf{Aggregators}: Best aggregated result across Centered Clipping and Geometric Median (pre-aggregated with NNM and ARC).
\end{itemize}

\section{Aggregated Test Accuracy Heatmaps}
The figure below displays the grid layout of the best aggregated test accuracy heatmaps for each tested $\alpha$ parameter value.

\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_alpha_1_0.pdf}
        \caption{$\alpha = 1.0$}
        \label{fig:alpha_1_0}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_alpha_1_25.pdf}
        \caption{$\alpha = 1.25$}
        \label{fig:alpha_1_25}
    \end{subfigure}
    
    \vspace{0.3cm}
    
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_alpha_1_5.pdf}
        \caption{$\alpha = 1.5$}
        \label{fig:alpha_1_5}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_alpha_2_0.pdf}
        \caption{$\alpha = 2.0$}
        \label{fig:alpha_2_0}
    \end{subfigure}
    
    \vspace{0.3cm}
    
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_alpha_3_0.pdf}
        \caption{$\alpha = 3.0$}
        \label{fig:alpha_3_0}
    \end{subfigure}
    
    \caption{Aggregated Test Accuracy Heatmaps across different surrogate gradient $\alpha$ values for SNN ($f \le 5$).}
    \label{fig:atan_alpha_heatmaps}
\end{figure}

\newpage

\section{Quantitative Performance Summary}
Table~\ref{tab:summary} compares the overall performance metrics for each $\alpha$ parameter value. The metrics include the average accuracy across all 24 cells in the grid, the worst-case accuracy, and the accuracy under the most challenging experimental setup ($\gamma=0.0$, $f=5$).

\begin{table}[htbp]
    \centering
    \caption{Performance Summary for Different Surrogate Gradient scaling values ($\alpha$).}
    \label{tab:summary}
    \begin{tabular}{lccc}
        \toprule
        \textbf{Surrogate Scaling ($\alpha$)} & \textbf{Average Accuracy} & \textbf{Worst-case Accuracy} & \textbf{Accuracy ($\gamma=0.0, f=5$)} \\
        \midrule
""")

        # Add data to the summary table
        for alpha in alphas:
            grid = data_summary[alpha]["grid"]
            mean_acc = np.mean(grid) * 100
            min_acc = np.min(grid) * 100
            extreme_acc = grid[0, 5] * 100
            f.write(f"        $\\alpha = {alpha}$ & {mean_acc:.2f}\\% & {min_acc:.2f}\\% & {extreme_acc:.2f}\\% \\\\\n")

        f.write(r"""        \bottomrule
    \end{tabular}
\end{table}

\section{Detailed Grid Tables for each $\alpha$}
This section lists the exact aggregated test accuracies for each data heterogeneity level ($\gamma$) and number of Byzantine clients ($f$).

""")

        # Generate 4x6 tables for each alpha
        for alpha in alphas:
            grid = data_summary[alpha]["grid"]
            f.write(f"\\subsection{{Surrogate scaling $\\alpha = {alpha}$}}\n")
            f.write(r"""\begin{table}[htbp]
    \centering
    \begin{tabular}{ccccccc}
        \toprule
        & \textbf{f = 0} & \textbf{f = 1} & \textbf{f = 2} & \textbf{f = 3} & \textbf{f = 4} & \textbf{f = 5} \\
        \midrule
""")
            gammas = [1.0, 0.66, 0.33, 0.0]
            for idx, gamma in enumerate(gammas):
                row_idx = 3 - idx
                vals = grid[row_idx] * 100
                f.write(f"        $\\gamma = {gamma}$ & {vals[0]:.2f}\\% & {vals[1]:.2f}\\% & {vals[2]:.2f}\\% & {vals[3]:.2f}\\% & {vals[4]:.2f}\\% & {vals[5]:.2f}\\% \\\\\n")

            f.write(r"""        \bottomrule
    \end{tabular}
\end{table}
""")

        f.write(r"""
\section{Key Interpretations}
Based on the sweep results:
\begin{itemize}
    \item \textbf{Optimal $\alpha$ Value}: Tuning the surrogate gradient scaling factor $\alpha$ has a significant impact on learning under non-IID conditions and Byzantine attacks. Larger $\alpha$ values tend to sharpen the gradient approximation, while smaller values smooth it.
    \item \textbf{Non-IID Stability}: The worst-case accuracy drops are mostly concentrated in the $\gamma = 0.0$ and $f = 3$ regime. Comparing this cell across different $\alpha$ values allows us to identify the most robust configurations.
\end{itemize}

\end{document}
""")

    print(f"LaTeX file written to {tex_path}")

    # Compile the LaTeX document
    print("Compiling LaTeX to PDF...")
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
        print(f"LaTeX compiled successfully! PDF generated: latex_plots/{pdf_filename}")
    except Exception as e:
        print(f"[ERROR] Failed to compile LaTeX: {e}")

if __name__ == "__main__":
    main()
