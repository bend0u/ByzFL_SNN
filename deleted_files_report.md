# Deleted Files Report

This document records all files and directories removed during the repository cleanup process, along with their functions, historical context, and rationale for deletion.

---

## 1. Throwaway & Temporary Files (Root Level)

| Deleted Path | Function / Description | Reason for Removal |
|---|---|---|
| `test_clip.py`, `test_clip2.py` | One-off manual test scripts for gradient and activation clipping. | Temporary dev test scripts; functionality integrated into `byzfl/`. |
| `test_path.py`, `test_path2.py` | Debugging scripts for local module import paths. | One-off path resolution checks. |
| `prove_clipping.py` | Isolated script attempting to prove clipping properties mathematically/empirically. | One-off proof script; work completed. |
| `analyze_attacks.py` | Early script analyzing Byzantine attack behavior on model weights. | Superseded by online metrics instrumentation. |
| `cleanup_stale_results.py` | Utility script to purge incomplete or mismatched sweep runs. | Single-use maintenance utility. |
| `acc_clusters.txt` | Raw text dump of client accuracy clustering metrics. | Temporary unformatted data output. |
| `nohup.out` | Standard output log from background execution. | Stale process output. |
| `extracted_results.txt` | Unstructured text summary extracted from sweep runs. | Superseded by structured JSON and LaTeX reports. |
| `test_accuracy_grid.png` | Orphan plot image generated at repository root. | Intermediate plot asset not linked to reports. |
| `extract_results.py` | Script used to parse raw experiment folders into `extracted_results.txt`. | One-off parsing script. |
| `theoretical_analysis_report.*` | LaTeX source (`.tex`), compiled PDF (`.pdf`), auxiliary file (`.aux`), and metadata (`.metadata.json`) for early theoretical analysis document. | Obsolete theoretical report draft; superseded by active empirical LaTeX reports in `reports/`. |

---

## 2. Log Files (Root Level & `logs/` Directory)

| Deleted Path | Function / Description | Reason for Removal |
|---|---|---|
| `full_sweep_docker.log` | Execution log from full Docker sweep run. | Stale run output (13 KB). |
| `interarch_experiment.log` | Raw execution log from inter-architecture experiments. | Stale run output (150 KB); results saved in `results/`. |
| `sfma_sweep.log` | Execution log from early SFMA attack/defense sweep. | Stale run output (170 KB). |
| `sfma_test.log` | Debug execution log for SFMA test runs. | Stale run output (17 KB). |
| `run_targeted.log` | Execution log for targeted attack experiment. | Small stale log file (624 B). |
| `logs/` (directory) | 11 MB directory containing raw terminal output logs (`atan_sweep.log`, `box_sweep.log`, `cnn_cifar.log`, `robust_atan.log`, `robust_box.log`, `robust_tri.log`, `tri_sweep.log`, `weekend_cnn.log`, `weekend_snn.log`). | Redundant execution logs; metrics preserved in structured `results/` folder. |

---

## 3. Exploratory & Obsolete Scripts (Root & `scratch/` & `script_archives/`)

| Deleted Path | Function / Description | Reason for Removal |
|---|---|---|
| `run_targeted_comparison.py` | Runner for targeted attack comparison. | Superseded by `run_cnn_robust_sweeps.py` & `run_snn_robust_sweeps.py`. |
| `run_full_sweep_docker.py`, `.sh` | Docker container sweep launchers. | Obsolete container sweep flow. |
| `run_docker.sh`, `run_docker_v100.sh` | Shell wrappers to start Docker containers locally and on V100 GPU nodes. | Legacy execution environment setup scripts. |
| `run_new_sweeps.sh` | Generic shell script for running sweeps. | Unclear scope; superseded by explicit launchers in `scripts/launchers/`. |
| `run_cifar_tuning.py`, `_rcp.sh` | CIFAR-10 hyperparameter tuning runner and RCP launcher. | One-off hyperparameter search; best params saved in configs. |
| `run_dropout_experiment.py`, `run_dropout_calibration.py`, `run_dropout_heatmap_rcp.sh`, `run_dense_dropout_baseline.py` | Experiments evaluating dropout as a Byzantine defense/regularizer in SNNs and CNNs. | Exploratory investigation line; completed and not featured in main reports. |
| `run_sparsity_experiment.py`, `run_sparsity_measure.py` | Experiment runners measuring firing and gradient sparsity. | Replaced by online sparsity hooks inside core framework. |
| `run_and_plot_mnist.py` | Combined script running MNIST FL rounds and immediately plotting results. | Monolithic combined runner; split into modular `scripts/experiments/` and `scripts/plotting/`. |
| `run_mnist_heatmap_clipping_rcp.sh` | Shell launcher for MNIST clipping heatmaps. | Superseded by current RCP launchers in `scripts/launchers/rcp/`. |
| `plot_cifar_tuning.py` | Plotting script for CIFAR hyperparameter search. | Associated with completed tuning phase. |
| `plot_dropout_experiment.py`, `plot_dropout_40_experiment.py`, `plot_dropout_calibration.py` | Plotting scripts for dropout experiments. | Associated with completed dropout exploration. |
| `generate_dropout_calibration_config.py` | Config generator for dropout calibration. | One-off config generator. |
| `run_mnist_cliff_smoketest_local.sh`, `run_mnist_exp2_smoketest_local.sh`, `run_mnist_irreversibility_smoketest_local.sh`, `run_activation_clip_smoketest_local.sh` | Local bash smoketest scripts for short trial runs. | Redundant local test scripts; production SSH/RCP launchers preserved in `scripts/launchers/`. |
| `script_archives/` (directory) | 31 legacy python files (e.g. `run_weekend_experiments.py`, `run_overnight1.py`, `run_mnist_byzantine_sweep.py`). | Superseded historical experiment code from earlier project iterations. |
| `scratch/` (directory) | Temporary script scratchpad containing obsolete report compilers (`compile_ablation_report.py`, `compile_comparison_latex.py`, etc.) and heatmap generators. | Active compile scripts moved to `scripts/latex_generation/` and `scripts/utils/`; remainder purged. |

---

## 4. Old Report LaTeX Files & Orphan Plot Images (`reports/`)

| Deleted Path | Function / Description | Reason for Removal |
|---|---|---|
| `reports/heatmap_*.png`, `reports/heatmap_*.pdf` (~120 files) | Standalone heatmap raster & vector images generated in earlier experiment iterations. | Unreferenced by any active LaTeX reports (active reports import directly from `../plots/`). |
| `reports/ablation_clipping_report.*` | Report evaluating early clipping ablation runs. | Obsolete report draft. |
| `reports/aggregator_comparison_report.*` | Report comparing basic Byzantine aggregators. | Superseded by `robust_mixed_report.tex`. |
| `reports/cnn_ablation_report.*` | Report detailing CNN activation ablation experiments. | Obsolete report draft. |
| `reports/comparison_plots.*` (and variants `_alie_delta`, `_atan_alpha_f3/f4/f5`, `_atan_vs_tri`, `_cnn_vs_snn_direct_f10`, `_extra_seeds`, `_f10`, `_f8`, `_simplest`, `_snn_direct_vs_rate_f10`, `_tri_beta_f1`, `_snn_vs_cnn`, `convergence_alpha_vs_cnn`) | Intermediate comparison PDF/TeX report compilations. | Legacy compilation targets superseded by the 5 primary reports. |

---

## 5. Intermediate Plot Directories (`plots/` & Root)

| Deleted Path | Function / Description | Reason for Removal |
|---|---|---|
| `extra_seeds_plots/` | 4 heatmap image files for extra seed evaluations. | Standalone plot outputs no longer referenced. |
| `snn_simplest_plots/` | 20 PDF plots from early SNN simplest experiments. | Intermediate plot files from early exploration. |
| `plots/robust_comparison_sweep_old/` | Plots from early robustness sweep runs. | Superceded by `plots/robust_comparison_sweep/`. |
| `plots/robustness_metrics/` | Figures for early robustness metrics. | Obsolete plot outputs. |
| `plots/variance_comparison/`, `plots/variance_vs_step/` | Variance analysis plot subdirectories. | Superseded by inter-architecture metrics plots. |
| `plots/sparsity_experiment/`, `plots/sparsity_measure/` | Sparsity plot output subdirectories. | Replaced by `scripts/plotting/plot_sparsity_comparison.py`. |

---

## Note on Preserved & Isolated Assets

- **SMEA & SFMA Experiments:** Per user instructions, all SMEA and SFMA scripts, tests, and configuration files (`run_smea*.py`, `run_sfma*.py`, `plot_smea*.py`, `configs/smea/`) were preserved and isolated into `smea_sfma_archive/`.
- **Results Directory (`results/`):** 5.4 GB of experiment outputs completely untouched.
- **Core Package (`byzfl/`):** Core framework code completely untouched.
