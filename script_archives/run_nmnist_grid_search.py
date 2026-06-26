import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from byzfl import run_benchmark

RESULTS_DIR = "./snn_benchmark_results"
PLOTS_DIR = "./snn_benchmark_plots"

def main():
    print("==========================================================")
    print("STARTING OPTIMIZED SINGLE N-MNIST SNN RUN ON GPU 1 (TS=15)")
    print("==========================================================")
    
    # 1. Run the optimized benchmark (strictly 1 job, no competition)
    run_benchmark("snn_nmnist_decay.json", nb_jobs=1)
    
    # 2. Parse and display results
    print("\nParsing results...")
    
    # The folder name created by the framework for this configuration
    folder_name = "nmnist_nmnist_snn_n_16_f_0_d_0_iid_None_Average__NoAttack_lr_0.02_mom_0.9_wd_0.0001_ts_15_enc_constant_beta_0.95_surrogate_gradient_atan_threshold_1.0"
    folder_path = os.path.join(RESULTS_DIR, folder_name)
    
    if not os.path.exists(folder_path):
        print(f"Results folder {folder_path} not found.")
        return
        
    val_acc_file = os.path.join(folder_path, "val_accuracy_tr_seed_42_dd_seed_42.txt")
    test_acc_file = os.path.join(folder_path, "test_accuracy_tr_seed_42_dd_seed_42.txt")
    train_time_file = os.path.join(folder_path, "train_time_tr_seed_42_dd_seed_42.txt")
    
    if not os.path.exists(val_acc_file):
        print("Validation accuracy file not found. The run might not have completed.")
        return
        
    try:
        val_accs = np.loadtxt(val_acc_file, delimiter=",")
        if val_accs.ndim == 0: val_accs = np.array([val_accs])
        
        test_accs = None
        if os.path.exists(test_acc_file):
            test_accs = np.loadtxt(test_acc_file, delimiter=",")
            if test_accs.ndim == 0: test_accs = np.array([test_accs])
        
        train_time = 0.0
        if os.path.exists(train_time_file):
            train_time = float(np.loadtxt(train_time_file))
            
        print("\n==========================================================")
        print("RUN COMPLETED SUCCESSFULLY")
        print("==========================================================")
        print(f"Time Steps: 15")
        print(f"Learning Rate: 0.02")
        print(f"Total Steps: 250")
        print(f"Final Validation Accuracy: {val_accs[-1]:.4f}")
        if test_accs is not None:
            print(f"Final Test Accuracy: {test_accs[-1]:.4f}")
        else:
            print("Final Test Accuracy: Not Evaluated (evaluate_on_test=false)")
        print(f"Total Training Time: {train_time:.1f} seconds ({train_time/3600:.2f} hours)")
        print("==========================================================")
        
        # Plot convergence curve
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plt.figure(figsize=(8, 5))
        # Evaluated at steps 0, 50, 100, 150, 200, 250
        steps = [i * 50 for i in range(len(val_accs))]
        steps = [min(s, 250) for s in steps]
        
        plt.plot(steps, val_accs, label="Validation Accuracy", color="#3b82f6", linewidth=2.5, marker="o")
        if test_accs is not None and len(test_accs) == len(val_accs):
            plt.plot(steps, test_accs, label="Test Accuracy", color="#10b981", linewidth=2.5, marker="s")
            
        plt.title("N-MNIST Spiking Model Convergence (TS=15, LR=0.02)")
        plt.xlabel("Communication Step")
        plt.ylabel("Accuracy")
        plt.ylim(-0.02, 1.02)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="lower right")
        
        plot_path = os.path.join(PLOTS_DIR, "nmnist_fast_run_convergence.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved convergence plot to: {plot_path}")
        
        # Generate Comparison Plot with previous runs
        print("\nGenerating comparison plot with previous grid search runs...")
        plt.figure(figsize=(11, 6))
        
        all_runs = []
        for folder in sorted(os.listdir(RESULTS_DIR)):
            if not folder.startswith("nmnist_nmnist_snn_n_16_f_0_"):
                continue
            folder_path = os.path.join(RESULTS_DIR, folder)
            if not os.path.isdir(folder_path):
                continue
                
            val_file = os.path.join(folder_path, "val_accuracy_tr_seed_42_dd_seed_42.txt")
            if not os.path.exists(val_file):
                continue
                
            import re
            ts_match = re.search(r'_ts_(\d+)_', folder)
            lr_match = re.search(r'_lr_([\d\.]+)_', folder)
            
            if not (ts_match and lr_match):
                continue
                
            run_ts = int(ts_match.group(1))
            run_lr = float(lr_match.group(1))
            
            try:
                accs = np.loadtxt(val_file, delimiter=",")
                if accs.ndim == 0:
                    accs = np.array([accs])
                
                config_path = os.path.join(folder_path, "config.json")
                delta = 10
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        cfg = json.load(f)
                        delta = cfg.get("evaluation_and_results", {}).get("evaluation_delta", 10)
                        
                all_runs.append({
                    "folder": folder,
                    "ts": run_ts,
                    "lr": run_lr,
                    "val_accs": accs,
                    "delta": delta
                })
            except Exception as ex:
                print(f"Error loading {folder}: {ex}")
                
        for r in all_runs:
            steps_r = [i * r["delta"] for i in range(len(r["val_accs"]))]
            steps_r = [min(s, 250) for s in steps_r]
            
            if r["ts"] == 15 and r["lr"] == 0.02:
                plt.plot(steps_r, r["val_accs"], label=f"NEW: LR={r['lr']}, TS={r['ts']} (Fast Run)", 
                         color="#ef4444", linewidth=3.5, marker="o", zorder=10)
            elif r["ts"] == 20 and r["lr"] == 0.02:
                plt.plot(steps_r, r["val_accs"], label=f"LR={r['lr']}, TS={r['ts']} (Previous Best)", 
                         color="#10b981", linewidth=2.5, marker="s", zorder=5)
            else:
                plt.plot(steps_r, r["val_accs"], label=f"LR={r['lr']}, TS={r['ts']}", 
                         alpha=0.5, linewidth=1.5)
                         
        plt.title("N-MNIST Spiking Model: Fast Run vs Previous Grid Search Runs")
        plt.xlabel("Communication Step")
        plt.ylabel("Validation Accuracy")
        plt.ylim(-0.02, 1.02)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize="small")
        
        comp_plot_path = os.path.join(PLOTS_DIR, "nmnist_comparison_with_grid.png")
        plt.tight_layout()
        plt.savefig(comp_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison plot to: {comp_plot_path}")
        
    except Exception as e:
        print(f"Error parsing results: {e}")

if __name__ == "__main__":
    main()
