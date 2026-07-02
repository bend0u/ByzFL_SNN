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
    get_dist_param_val,
    custom_dict_to_str,
    ensure_list,
    aggregated_test_heatmap
)

def get_heatmap_data(results_dir, beta_val):
    config_path = os.path.join(results_dir, "config.json")
    with open(config_path, "r") as file:
        data = json.load(file)

    # Set up parameters for tri sweep
    data["model"]["model_params"]["surrogate_params"]["beta"] = beta_val
    data["benchmark_config"]["f"] = [0, 1]

    pm = ParamsManager(data)
    snn_suffix = get_snn_suffix(pm)
    clean = False

    path_to_hyperparameters = os.path.join(results_dir, "best_hyperparameters")

    training_seed = data["benchmark_config"]["training_seed"]
    nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
    nb_honest_clients = data["benchmark_config"]["nb_honest_clients"]
    nb_byz = data["benchmark_config"]["f"]
    nb_declared = [1]  # tolerated_f = 1 (since we only evaluate f <= 1)
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
                    try:
                        accuracy = get_accuracy_at_best_step(
                            results_dir, clean, dataset_name, model_name, nb_nodes, nb_byzantine, nb_decl,
                            data_dist['name'], dist_param_val, agg['name'], pre_agg_names, attack['name'],
                            lr, momentum, wd, snn_suffix, pm,
                            nb_data_distribution_seeds, nb_training_seeds,
                            training_seed, data_distribution_seed, evaluation_delta
                        )
                    except Exception as e:
                        # Fallback for missing/incomplete configurations
                        accuracy = 0.0
                    if accuracy < worst_accuracy:
                        worst_accuracy = accuracy

                heat_map_table[len(heat_map_table)-1-x][y] = worst_accuracy

        heat_map_cube[z] = heat_map_table

    # Best aggregator is the max across the aggregator axis
    final_heat_map = np.max(heat_map_cube, axis=0)
    return final_heat_map, distribution_parameter_list, actual_nb_byz

def main():
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(workspace_dir, "results", "snn", "robust_tri_sweep")
    plots_dir = os.path.join(workspace_dir, "plots", "snn", "robust_tri_sweep_f1")
    latex_dir = os.path.join(workspace_dir, "latex_plots")
    os.makedirs(latex_dir, exist_ok=True)

    config_path = os.path.join(results_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        sys.exit(1)

    # Backup original config
    backup_config_path = config_path + ".bak"
    shutil.copyfile(config_path, backup_config_path)

    betas = [0.5, 0.75, 1.0, 1.25, 1.5]
    data_summary = {}

    try:
        # Load and modify config temporarily for plotting f=0,1
        with open(config_path, "r") as f:
            config_data = json.load(f)
        config_data["benchmark_config"]["f"] = [0, 1]

        # Generate plots with columns f=0,1
        for beta in betas:
            print(f"\nGenerating heatmap plots for beta = {beta}...")
            temp_config = json.loads(json.dumps(config_data))
            temp_config["model"]["model_params"]["surrogate_params"]["beta"] = beta
            with open(config_path, "w") as f:
                json.dump(temp_config, f, indent=4)

            beta_plots_dir = os.path.join(plots_dir, f"beta_{beta}")
            os.makedirs(beta_plots_dir, exist_ok=True)
            try:
                aggregated_test_heatmap(results_dir, beta_plots_dir, metric="best_step")
            except Exception as e:
                print(f"Error in aggregated_test_heatmap for beta={beta}: {e}")

        # Extract accuracy data for summary table/latex
        print("\nExtracting accuracy data for each beta value...")
        for beta in betas:
            grid, gammas, f_vals = get_heatmap_data(results_dir, beta)
            data_summary[beta] = {
                "grid": grid,
                "gammas": gammas,
                "f_vals": f_vals
            }

            beta_str = str(beta).replace(".", "_")
            src_pdf_name = "best_test_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real.pdf"
            src_png_name = "best_test_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real.png"
            
            src_pdf_path = os.path.join(plots_dir, f"beta_{beta}", src_pdf_name)
            src_png_path = os.path.join(plots_dir, f"beta_{beta}", src_png_name)
            
            dest_pdf_path = os.path.join(latex_dir, f"heatmap_tri_beta_{beta_str}.pdf")
            dest_png_path = os.path.join(latex_dir, f"heatmap_tri_beta_{beta_str}.png")
            
            if os.path.exists(src_pdf_path):
                shutil.copyfile(src_pdf_path, dest_pdf_path)
                print(f"--> Copied PDF plot for beta={beta} to {dest_pdf_path}")
            if os.path.exists(src_png_path):
                shutil.copyfile(src_png_path, dest_png_path)
                print(f"--> Copied PNG plot for beta={beta} to {dest_png_path}")

    finally:
        # Restore original config.json
        if os.path.exists(backup_config_path):
            shutil.move(backup_config_path, config_path)
            print("\nRestored original config.json successfully.")

    # Generate LaTeX content
    tex_path = os.path.join(latex_dir, "comparison_plots_tri_beta_f1.tex")
    
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

\title{SNN Surrogate Gradient \texttt{tri} $\beta$ Parameter Comparison\\Robustness Sweep under Data Heterogeneity ($f \le 1$)}
\author{ByzFL SNN Benchmark}
\date{\today}

\begin{document}

\maketitle

\section{Introduction}
This document presents a comparison of the Spiking Neural Network (SNN) robustness on the MNIST dataset using different scaling factors $\beta$ for the Triangular (\texttt{tri}) surrogate gradient. The evaluation spans:
\begin{itemize}
    \item \textbf{Surrogate Gradient Parameter}: $\beta \in \{0.5, 0.75, 1.0, 1.25, 1.5\}$
    \item \textbf{Byzantine Nodes}: $f \in \{0, 1\}$ out of $N=10$ honest nodes
    \item \textbf{Data Heterogeneity}: Non-IID levels $\gamma \in \{1.0, 0.66, 0.33, 0.0\}$
    \item \textbf{Attacks}: Worst-case outcome across SignFlipping and Optimal A Little Is Enough (\texttt{Optimal\_ALittleIsEnough\_neg1})
    \item \textbf{Aggregators}: Best aggregated result across Centered Clipping and Geometric Median (pre-aggregated with NNM and ARC).
\end{itemize}

\section{Aggregated Test Accuracy Heatmaps}
The figure below displays the aggregated test accuracy heatmaps for each tested $\beta$ parameter value for $f \le 1$.

\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_tri_beta_0_5.pdf}
        \caption{$\beta = 0.5$}
        \label{fig:beta_0_5}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_tri_beta_0_75.pdf}
        \caption{$\beta = 0.75$}
        \label{fig:beta_0_75}
    \end{subfigure}
    
    \vspace{0.3cm}
    
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_tri_beta_1_0.pdf}
        \caption{$\beta = 1.0$}
        \label{fig:beta_1_0}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_tri_beta_1_25.pdf}
        \caption{$\beta = 1.25$}
        \label{fig:beta_1_25}
    \end{subfigure}
    
    \vspace{0.3cm}
    
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_tri_beta_1_5.pdf}
        \caption{$\beta = 1.5$}
        \label{fig:beta_1_5}
    \end{subfigure}
    
    \caption{Aggregated Test Accuracy Heatmaps across different surrogate gradient $\beta$ values for SNN ($f \le 1$).}
    \label{fig:tri_beta_heatmaps}
\end{figure}

\newpage

\section{Quantitative Performance Summary}
Table~\ref{tab:summary} compares the overall performance metrics for each $\beta$ parameter value. The metrics include the average accuracy across all 8 cells in the grid, and the worst-case accuracy.

\begin{table}[htbp]
    \centering
    \caption{Performance Summary for Different Surrogate Gradient scaling values ($\beta$).}
    \label{tab:summary}
    \begin{tabular}{lcc}
        \toprule
        \textbf{Surrogate Scaling ($\beta$)} & \textbf{Average Accuracy} & \textbf{Worst-case Accuracy} \\
        \midrule
""")

        # Add data to the summary table
        for beta in betas:
            grid = data_summary[beta]["grid"]
            mean_acc = np.mean(grid) * 100
            min_acc = np.min(grid) * 100
            f.write(f"        $\\beta = {beta}$ & {mean_acc:.2f}\\% & {min_acc:.2f}\\% \\\\\n")

        f.write(r"""        \bottomrule
    \end{tabular}
\end{table}

\section{Detailed Grid Tables for each $\beta$}
This section lists the exact aggregated test accuracies for each data heterogeneity level ($\gamma$) and number of Byzantine clients ($f \in \{0, 1\}$).

""")

        # Generate tables for each beta
        for beta in betas:
            grid = data_summary[beta]["grid"]
            f.write(f"\\subsection{{Surrogate scaling $\\beta = {beta}$}}\n")
            f.write(r"""\begin{table}[htbp]
    \centering
    \begin{tabular}{ccc}
        \toprule
        & \textbf{f = 0} & \textbf{f = 1} \\
        \midrule
""")
            gammas = [1.0, 0.66, 0.33, 0.0]
            for idx, gamma in enumerate(gammas):
                row_idx = 3 - idx
                vals = grid[row_idx] * 100
                f.write(f"        $\\gamma = {gamma}$ & {vals[0]:.2f}\\% & {vals[1]:.2f}\\% \\\\\n")

            f.write(r"""        \bottomrule
    \end{tabular}
\end{table}
""")

        f.write(r"""
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
        # Cleanup LaTeX auxiliary files
        for ext in [".aux", ".log", ".out", ".toc", ".fls", ".fdb_latexmk", ".synctex.gz"]:
            aux_file = os.path.join(latex_dir, tex_filename.replace(".tex", ext))
            if os.path.exists(aux_file):
                os.remove(aux_file)
        print(f"LaTeX compiled successfully! PDF generated: latex_plots/{pdf_filename}")
    except Exception as e:
        print(f"[ERROR] Failed to compile LaTeX: {e}")

if __name__ == "__main__":
    main()
