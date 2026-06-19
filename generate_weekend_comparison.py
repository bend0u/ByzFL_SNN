import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Resolve paths relative to script directory to ensure robustness
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

import run_weekend_experiments
# Dynamically override the RESULTS_DIR in the imported module to use absolute paths
run_weekend_experiments.RESULTS_DIR = os.path.join(script_dir, "weekendexperiments")

from run_weekend_experiments import CONFIGS, get_folder_path, load_final_accuracy

RESULTS_DIR = run_weekend_experiments.RESULTS_DIR
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots_comparison")
SUMMARY_COMP_FILE = os.path.join(RESULTS_DIR, "accuracy_comparison_summary.txt")

# Set up matplotlib aesthetics
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 15,
    'legend.fontsize': 10,
    'grid.alpha': 0.5,
    'grid.linestyle': '--'
})

def load_accuracy_history(folder_path):
    path = os.path.join(folder_path, "test_accuracy.txt")
    if not os.path.exists(path):
        return None
    try:
        data = np.loadtxt(path, delimiter=",")
        if data.ndim == 0:
            data = np.array([data])
        return data
    except Exception:
        return None

def generate_bar_chart(attack, dist, dist_param, f_val, grouped_configs):
    # Filter groups matching this attack, dist, f_val
    matching_keys = [
        k for k in grouped_configs.keys()
        if k[0] == attack and k[1] == dist and k[2] == dist_param and k[3] == f_val
    ]
    if not matching_keys:
        return
        
    # Sort keys by defense name for consistent ordering
    matching_keys = sorted(matching_keys, key=lambda k: f"{k[4] or 'None'}/{k[5]}")
    
    defenses = []
    accs_direct = []
    accs_rate = []
    
    for k in matching_keys:
        pre_agg = k[4] if k[4] else "None"
        agg = k[5]
        defense_label = f"{pre_agg}/{agg}"
        
        acc_rate = load_final_accuracy(get_folder_path(grouped_configs[k]["rate"]))
        acc_direct = load_final_accuracy(get_folder_path(grouped_configs[k]["constant"]))
        
        defenses.append(defense_label)
        accs_direct.append(acc_direct if acc_direct is not None else 0.0)
        accs_rate.append(acc_rate if acc_rate is not None else 0.0)
        
    x = np.arange(len(defenses))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, accs_direct, width, label='Direct Encoding', color='#1d4ed8')  # Royal Blue
    rects2 = ax.bar(x + width/2, accs_rate, width, label='Rate Encoding', color='#fb923c')   # Orange
    
    dist_title = "IID" if dist == "iid" else f"Dirichlet NIID ({dist_param})"
    ax.set_title(f"Encoding Comparison: Direct vs. Rate\nAttack: {attack} (f={f_val}) | {dist_title}", fontweight='bold', pad=15)
    ax.set_ylabel('Final Test Accuracy')
    ax.set_xlabel('Defense Setup (Pre-Aggregator / Aggregator)')
    ax.set_xticks(x)
    ax.set_xticklabels(defenses, rotation=15, ha='right')
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax.legend(loc='lower left', frameon=True)
    
    # Add accuracy values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0.01:
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
                            
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, f"weekend_bar_{attack}_{dist}_f{f_val}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved bar chart: {out_path}")

def plot_learning_curve_pair(attack, dist, dist_param, f_val, pre_agg, agg, grouped_configs):
    key = (attack, dist, dist_param, f_val, pre_agg, agg)
    matching_key = None
    for k in grouped_configs.keys():
        if k[:6] == key:
            matching_key = k
            break
            
    if not matching_key:
        print(f"Warning: Configuration not found for learning curve plot: {key}")
        return
        
    cfg_rate = grouped_configs[matching_key]["rate"]
    cfg_direct = grouped_configs[matching_key]["constant"]
    
    path_rate = get_folder_path(cfg_rate)
    path_direct = get_folder_path(cfg_direct)
    
    acc_rate = load_accuracy_history(path_rate)
    acc_direct = load_accuracy_history(path_direct)
    
    if acc_rate is None and acc_direct is None:
        return
        
    fig, ax = plt.subplots(figsize=(9, 6))
    
    if acc_direct is not None:
        steps_direct = np.arange(len(acc_direct)) * 10
        ax.plot(steps_direct, acc_direct, label="Direct Encoding (Constant)", color='#1d4ed8', linewidth=2.5, linestyle='-')
        
    if acc_rate is not None:
        steps_rate = np.arange(len(acc_rate)) * 10
        ax.plot(steps_rate, acc_rate, label="Rate Encoding", color='#fb923c', linewidth=2.5, linestyle='--')
        
    dist_title = "IID" if dist == "iid" else f"Dirichlet NIID ({dist_param})"
    pre_str = f"{pre_agg}/" if pre_agg else ""
    ax.set_title(f"Learning Curve: Direct vs. Rate\n{attack} (f={f_val}) | Defense: {pre_str}{agg} | {dist_title}", fontweight='bold', pad=15)
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Test Accuracy')
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', frameon=True)
    
    plt.tight_layout()
    pre_filename = f"{pre_agg}_" if pre_agg else ""
    out_path = os.path.join(PLOTS_DIR, f"weekend_curve_{attack}_{pre_filename}{agg}_{dist}_f{f_val}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved learning curve plot: {out_path}")

def main():
    print("==========================================================")
    print("GENERATING ACCURACY COMPARISON SUMMARY & PLOTS")
    print("==========================================================")
    
    if not os.path.isdir(RESULTS_DIR):
        print(f"Error: results directory '{RESULTS_DIR}' does not exist.")
        return
        
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # 1. Group CONFIGS by unique scenario parameters
    grouped_configs = {}
    for cfg in CONFIGS:
        key = (cfg["attack"], cfg["dist"], cfg["dist_param"], cfg["f"], cfg["pre_agg"], cfg["agg"], cfg["desc"])
        if key not in grouped_configs:
            grouped_configs[key] = {}
        grouped_configs[key][cfg["encoding"]] = cfg

    # Find control baseline accuracies dynamically (NoAttack, Average, IID, f=0)
    ctrl_key_direct = None
    ctrl_key_rate = None
    for k, enc_dict in grouped_configs.items():
        if k[0] == "NoAttack" and k[1] == "iid" and k[3] == 0 and k[4] is None and k[5] == "Average":
            ctrl_key_direct = get_folder_path(enc_dict["constant"])
            ctrl_key_rate = get_folder_path(enc_dict["rate"])
            break

    base_direct = load_final_accuracy(ctrl_key_direct) if ctrl_key_direct else 0.9751
    base_rate = load_final_accuracy(ctrl_key_rate) if ctrl_key_rate else 0.9847
    if base_direct is None: base_direct = 0.9751
    if base_rate is None: base_rate = 0.9847
        
    # 2. Build comparison text lines
    lines = [
        "===========================================================================================================================================",
        "SNN WEEKEND EXPERIMENTS COMPARISON REPORT (DIRECT VS RATE)",
        "===========================================================================================================================================",
        "",
        f"{'Attack':<32} | {'Dist':<14} | {'f':<2} | {'Pre-Agg':<8} | {'Aggregator':<16} | {'Direct Acc':<10} | {'Rate Acc':<10} | {'D-Drop %':<10} | {'R-Drop %':<10} | {'R-Gain %':<10}",
        "-" * 139
    ]
    
    # Let's group keys by (attack, dist, dist_param, f) for output grouping
    scenarios = {}
    for k in grouped_configs.keys():
        scen_key = (k[0], k[1], k[2], k[3]) # attack, dist, dist_param, f
        if scen_key not in scenarios:
            scenarios[scen_key] = []
        scenarios[scen_key].append(k)
        
    # Sort scenarios logically (NoAttack first, then by attack name, dist)
    sorted_scenarios = sorted(scenarios.keys(), key=lambda s: (0 if s[0] == "NoAttack" else 1, s[0], s[1], s[3]))
    
    for scen in sorted_scenarios:
        keys_in_scen = scenarios[scen]
        # Sort defense configs in scenario alphabetically
        keys_in_scen = sorted(keys_in_scen, key=lambda k: f"{k[4] or 'None'}/{k[5]}")
        
        for k in keys_in_scen:
            attack, dist, dist_param, f_val, pre_agg, agg, desc = k
            
            acc_rate = load_final_accuracy(get_folder_path(grouped_configs[k]["rate"]))
            acc_direct = load_final_accuracy(get_folder_path(grouped_configs[k]["constant"]))
            
            str_direct = f"{acc_direct:.4f}" if acc_direct is not None else "N/A"
            str_rate = f"{acc_rate:.4f}" if acc_rate is not None else "N/A"
            
            # Compute relative drops based on baseline control
            if acc_direct is not None:
                d_drop = ((acc_direct - base_direct) / base_direct) * 100
                str_d_drop = f"{d_drop:+.2f}%"
            else:
                d_drop = None
                str_d_drop = "N/A"
                
            if acc_rate is not None:
                r_drop = ((acc_rate - base_rate) / base_rate) * 100
                str_r_drop = f"{r_drop:+.2f}%"
            else:
                r_drop = None
                str_r_drop = "N/A"
                
            if d_drop is not None and r_drop is not None:
                r_gain = r_drop - d_drop
                str_r_gain = f"{r_gain:+.2f}%"
            else:
                str_r_gain = "N/A"
                
            dist_title = "IID" if dist == "iid" else f"{dist}_{dist_param}"
            pre_str = pre_agg if pre_agg else "None"
            
            lines.append(
                f"{attack:<32} | "
                f"{dist_title:<14} | "
                f"{f_val:<2} | "
                f"{pre_str:<8} | "
                f"{agg:<16} | "
                f"{str_direct:<10} | "
                f"{str_rate:<10} | "
                f"{str_d_drop:<10} | "
                f"{str_r_drop:<10} | "
                f"{str_r_gain:<10}"
            )
        lines.append("-" * 139)
        
    # Append explanatory Appendix
    lines.append("")
    lines.append("===========================================================================================================================================")
    lines.append("APPENDIX: METRIC DEFINITIONS & CALCULATIONS")
    lines.append("===========================================================================================================================================")
    lines.append("To mathematically compare the resilience of Direct vs. Rate SNN encodings under attacks and label skew, we measure how")
    lines.append("much each encoding degrades relative to its baseline ceiling (control experiment).")
    lines.append("")
    lines.append("1. Control Experiment (Baseline ceiling):")
    lines.append("   - Setup: NoAttack, Average Aggregator, IID distribution, f = 0")
    lines.append(f"   - Direct Baseline Accuracy (Direct_Baseline): {base_direct:.4f}")
    lines.append(f"   - Rate Baseline Accuracy (Rate_Baseline):     {base_rate:.4f}")
    lines.append("")
    lines.append("2. Direct Drop % (D-Drop %):")
    lines.append("   The relative percentage change in Direct accuracy compared to the Direct Baseline.")
    lines.append("   Formula: D-Drop % = ((Direct_Accuracy - Direct_Baseline) / Direct_Baseline) * 100")
    lines.append("")
    lines.append("3. Rate Drop % (R-Drop %):")
    lines.append("   The relative percentage change in Rate accuracy compared to the Rate Baseline.")
    lines.append("   Formula: R-Drop % = ((Rate_Accuracy - Rate_Baseline) / Rate_Baseline) * 100")
    lines.append("")
    lines.append("4. Resilience Gain % (R-Gain %):")
    lines.append("   The difference in degradation between Rate and Direct encodings. Measures the relative resilience advantage.")
    lines.append("   Formula: R-Gain % = R-Drop % - D-Drop %")
    lines.append("   - A positive value (e.g., +1.50%) indicates Rate encoding is relatively more resilient (it dropped less).")
    lines.append("   - A negative value (e.g., -2.00%) indicates Direct encoding is relatively more resilient.")
    lines.append("   - A value of +0.00% indicates both encodings degraded by the same relative percentage.")
    lines.append("===========================================================================================================================================")

    summary_comp_text = "\n".join(lines)
    with open(SUMMARY_COMP_FILE, "w") as f:
        f.write(summary_comp_text)
        
    print(f"Comparison summary written to {SUMMARY_COMP_FILE}")
    
    # 3. Generate relevant bar charts for f=5
    print("\nGenerating bar charts...")
    for attack in ["SignFlipping", "Optimal_ALittleIsEnough", "Optimal_InnerProductManipulation"]:
        for dist in ["iid", "dirichlet_niid"]:
            dist_param = 1.0 if dist == "iid" else 0.5
            generate_bar_chart(attack, dist, dist_param, 5, grouped_configs)
            
    # 4. Generate learning curves comparison plots for key configurations
    print("\nGenerating key learning curve comparison plots...")
    # Key Scenario 1: SignFlipping + CenteredClipping (Dirichlet NIID, f=5)
    plot_learning_curve_pair("SignFlipping", "dirichlet_niid", 0.5, 5, None, "CenteredClipping", grouped_configs)
    # Key Scenario 2: Optimal_InnerProductManipulation + NNM/TrMean (Dirichlet NIID, f=5)
    plot_learning_curve_pair("Optimal_InnerProductManipulation", "dirichlet_niid", 0.5, 5, "NNM", "TrMean", grouped_configs)
    # Key Scenario 3: Optimal_ALittleIsEnough + MultiKrum (Dirichlet NIID, f=5)
    plot_learning_curve_pair("Optimal_ALittleIsEnough", "dirichlet_niid", 0.5, 5, None, "MultiKrum", grouped_configs)
    # Key Scenario 4: Optimal_ALittleIsEnough + NNM/TrMean (Dirichlet NIID, f=5)
    plot_learning_curve_pair("Optimal_ALittleIsEnough", "dirichlet_niid", 0.5, 5, "NNM", "TrMean", grouped_configs)
    # Key Scenario 5: Optimal_InnerProductManipulation + Median (Dirichlet NIID, f=5)
    plot_learning_curve_pair("Optimal_InnerProductManipulation", "dirichlet_niid", 0.5, 5, None, "Median", grouped_configs)
    
    print(f"\nAll plots generated successfully in: {PLOTS_DIR}")

if __name__ == "__main__":
    main()
