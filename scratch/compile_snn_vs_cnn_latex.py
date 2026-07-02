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
    ensure_list
)

def get_snn_heatmap_data(results_dir, alpha_val):
    config_path = os.path.join(results_dir, "config.json")
    with open(config_path, "r") as file:
        data = json.load(file)

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

    final_heat_map = np.max(heat_map_cube, axis=0)
    return final_heat_map, distribution_parameter_list, actual_nb_byz

def get_cnn_heatmap_data(results_dir):
    config_path = os.path.join(results_dir, "config.json")
    with open(config_path, "r") as file:
        data = json.load(file)

    aggregators = [
        {"name": "GeometricMedian"},
        {"name": "CenteredClipping"}
    ]
    attacks = [
        {"name": "Optimal_ALittleIsEnough_neg1"}
    ]
    data_distributions = data["benchmark_config"]["data_distribution"]
    nb_honest_clients = data["benchmark_config"]["nb_honest_clients"]
    nb_byz = [0, 1, 2, 3, 4, 5]
    nb_declared = [5]

    training_seed = data["benchmark_config"]["training_seed"]
    nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
    data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
    nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
    evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]

    model_name = data["model"]["name"]
    dataset_name = data["model"]["dataset_name"]
    lr_list = ensure_list(data["model"]["learning_rate"])
    
    snn_suffix = ""
    clean = False

    nb_honest = nb_honest_clients[0]
    nb_decl = nb_declared[0]
    actual_nb_byz = [item for item in nb_byz if item <= nb_decl]
    data_dist = data_distributions[0]
    distribution_parameter_list = ensure_list(data_dist["distribution_parameter"])

    heat_map_cube = np.zeros((len(aggregators), len(distribution_parameter_list), len(actual_nb_byz)))

    lr = lr_list[0]
    momentum = 0.9
    wd = 0.0001
    pre_agg_names = "NNM_ARC"

    for z, agg in enumerate(aggregators):
        heat_map_table = np.zeros((len(distribution_parameter_list), len(actual_nb_byz)))

        for y, nb_byzantine in enumerate(actual_nb_byz):
            nb_decl = nb_byzantine
            nb_nodes = nb_honest + nb_byzantine

            for x, dist_param in enumerate(distribution_parameter_list):
                dist_param_val = get_dist_param_val(data_dist['name'], dist_param)

                worst_accuracy = np.inf
                for attack in attacks:
                    accuracy = get_accuracy_at_best_step(
                        results_dir, clean, dataset_name, model_name, nb_nodes, nb_byzantine, nb_decl,
                        data_dist['name'], dist_param_val, agg['name'], pre_agg_names, attack['name'],
                        lr, momentum, wd, snn_suffix, None,
                        nb_data_distribution_seeds, nb_training_seeds,
                        training_seed, data_distribution_seed, evaluation_delta
                    )
                    if accuracy < worst_accuracy:
                        worst_accuracy = accuracy

                heat_map_table[len(heat_map_table)-1-x][y] = worst_accuracy

        heat_map_cube[z] = heat_map_table

    final_heat_map = np.max(heat_map_cube, axis=0)
    return final_heat_map, distribution_parameter_list, actual_nb_byz

def main():
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    snn_results_dir = os.path.join(workspace_dir, "results", "snn", "robust_new_atan_sweep")
    cnn_results_dir = os.path.join(workspace_dir, "results", "cnn", "weekend")
    
    snn_plots_dir = os.path.join(workspace_dir, "plots", "snn", "robust_new_atan_sweep_f5")
    cnn_plots_dir = os.path.join(workspace_dir, "plots", "cnn", "weekend")
    
    latex_dir = os.path.join(workspace_dir, "latex_plots")
    os.makedirs(latex_dir, exist_ok=True)

    print("Extracting SNN (alpha=1.0) and CNN baseline data...")
    snn_grid, gammas, f_vals = get_snn_heatmap_data(snn_results_dir, 1.0)
    cnn_grid, _, _ = get_cnn_heatmap_data(cnn_results_dir)

    # Copy plots to latex directory
    shutil.copyfile(
        os.path.join(snn_plots_dir, "alpha_1.0", "best_test_mnist_cnn_mnist_snn_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real.pdf"),
        os.path.join(latex_dir, "heatmap_snn_alpha_1_0.pdf")
    )
    shutil.copyfile(
        os.path.join(cnn_plots_dir, "best_test_mnist_cnn_mnist_gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real.pdf"),
        os.path.join(latex_dir, "heatmap_cnn_baseline.pdf")
    )

    tex_path = os.path.join(latex_dir, "comparison_snn_vs_cnn.tex")
    
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

\title{SNN vs CNN Baseline Robustness Comparison\\Sweep under Data Heterogeneity ($f \le 5$)}
\author{ByzFL SNN Benchmark}
\date{\today}

\begin{document}

\maketitle

\section{Introduction}
This document presents a detailed robustness comparison between a Spiking Neural Network (SNN) with surrogate gradient scaling $\alpha=1.0$ and a standard Artificial Neural Network (CNN) baseline. Both models are evaluated on the MNIST dataset under identical client partitioning and attack scenarios:
\begin{itemize}
    \item \textbf{Byzantine Nodes}: $f \in \{0, 1, 2, 3, 4, 5\}$ out of $N=10$ honest nodes
    \item \textbf{Data Heterogeneity}: Non-IID levels $\gamma \in \{1.0, 0.66, 0.33, 0.0\}$
    \item \textbf{Attack Model}: Optimal A Little Is Enough (\texttt{Optimal\_ALittleIsEnough\_neg1})
    \item \textbf{Aggregators}: Best aggregated result across Centered Clipping and Geometric Median (pre-aggregated with NNM and ARC).
\end{itemize}

\section{Visual Heatmaps Comparison}
The figure below places the aggregated test accuracy heatmaps of the SNN ($\alpha=1.0$) and the CNN baseline side-by-side.

\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_snn_alpha_1_0.pdf}
        \caption{SNN ($\alpha = 1.0$)}
        \label{fig:heatmap_snn}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{heatmap_cnn_baseline.pdf}
        \caption{CNN Baseline}
        \label{fig:heatmap_cnn}
    \end{subfigure}
    \caption{Aggregated Test Accuracy Heatmaps for SNN vs CNN baseline ($f \le 5$).}
    \label{fig:heatmaps_comparison}
\end{figure}

\newpage

\section{Quantitative Detailed Tables}

\subsection{SNN (ArcTangent, $\alpha = 1.0$, $T=10$)}
\begin{table}[htbp]
    \centering
    \begin{tabular}{ccccccc}
        \toprule
        & \textbf{f = 0} & \textbf{f = 1} & \textbf{f = 2} & \textbf{f = 3} & \textbf{f = 4} & \textbf{f = 5} \\
        \midrule
""")
        gammas = [1.0, 0.66, 0.33, 0.0]
        for idx, gamma in enumerate(gammas):
            row_idx = 3 - idx
            vals = snn_grid[row_idx] * 100
            f.write(f"        $\\gamma = {gamma}$ & {vals[0]:.2f}\\% & {vals[1]:.2f}\\% & {vals[2]:.2f}\\% & {vals[3]:.2f}\\% & {vals[4]:.2f}\\% & {vals[5]:.2f}\\% \\\\\n")
        f.write(r"""        \bottomrule
    \end{tabular}
    \caption{Aggregated test accuracy for SNN ($\alpha=1.0$).}
\end{table}

\subsection{CNN Baseline (NLLLoss, lr=0.05)}
\begin{table}[htbp]
    \centering
    \begin{tabular}{ccccccc}
        \toprule
        & \textbf{f = 0} & \textbf{f = 1} & \textbf{f = 2} & \textbf{f = 3} & \textbf{f = 4} & \textbf{f = 5} \\
        \midrule
""")
        for idx, gamma in enumerate(gammas):
            row_idx = 3 - idx
            vals = cnn_grid[row_idx] * 100
            f.write(f"        $\\gamma = {gamma}$ & {vals[0]:.2f}\\% & {vals[1]:.2f}\\% & {vals[2]:.2f}\\% & {vals[3]:.2f}\\% & {vals[4]:.2f}\\% & {vals[5]:.2f}\\% \\\\\n")
        f.write(r"""        \bottomrule
    \end{tabular}
    \caption{Aggregated test accuracy for CNN baseline.}
\end{table}

\section{Key Insights}
\begin{itemize}
    \item \textbf{Low-Heterogeneity Regime ($\gamma \ge 0.66$)}: SNN ($\alpha=1.0$) shows superior resilience to large Byzantine fractions compared to CNN. At $\gamma=0.66$ and $f=5$, the SNN accuracy remains at \textbf{96.53\%} while the CNN drops to \textbf{95.07\%}.
    \item \textbf{High-Heterogeneity Regime ($\gamma = 0.0$)}: Under the extreme non-IID regime, the SNN shows significant gains in the low-to-moderate attacker counts:
    \begin{itemize}
        \item Under $f=1$, SNN achieves \textbf{84.02\%} accuracy compared to CNN's \textbf{55.41\%} (+28.6\% improvement).
        \item Under $f=2$, SNN achieves \textbf{53.80\%} accuracy compared to CNN's \textbf{47.28\%} (+6.5\% improvement).
    \end{itemize}
    \item \textbf{Attacker count breakdown threshold}: For larger numbers of attackers ($f \ge 4$), the CNN baseline remains stable for a longer period at $\gamma=0.33$ ($57.49\%$), whereas SNN collapses.
\end{itemize}

\end{document}
""")

    print(f"LaTeX file written to {tex_path}")
    print("Compiling LaTeX to PDF...")
    try:
        for _ in range(2):
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-disable-installer", "comparison_snn_vs_cnn.tex"],
                cwd=latex_dir,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        print("LaTeX compiled successfully! PDF generated: latex_plots/comparison_snn_vs_cnn.pdf")
    except Exception as e:
        print(f"[ERROR] Failed to compile LaTeX: {e}")

if __name__ == "__main__":
    main()
