import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR_DIRECT = os.path.join(script_dir, "snn_benchmark_results_11/mnist_direct")
RESULTS_DIR_RATE = os.path.join(script_dir, "snn_benchmark_results_11/mnist_rate")
OUTPUT_FILE = os.path.join(script_dir, "snn_benchmark_results_11/accuracy_summary.txt")

ATTACKS = ["SignFlipping", "Optimal_InnerProductManipulation"]
DISTRIBUTIONS = ["iid", "dirichlet_niid_0.5"]
AGGREGATORS = ["Average", "TrMean"]
F_VALUES = [0, 1, 2, 3, 4]

def load_final_accuracy(results_dir, folder_name):
    path = os.path.join(results_dir, folder_name, "test_accuracy.txt")
    if not os.path.exists(path):
        return None
    try:
        data = np.loadtxt(path, delimiter=",")
        if data.ndim == 0:
            return float(data)
        return float(data[-1])
    except Exception as e:
        return None

def main():
    lines = []
    lines.append("==========================================================")
    lines.append("SNN BENCHMARK FINAL ACCURACY SUMMARY (DIRECT VS RATE)")
    lines.append("==========================================================")
    lines.append("")
    
    for attack in ATTACKS:
        for dist in DISTRIBUTIONS:
            dist_title = "IID" if dist == "iid" else "Dirichlet NIID (0.5)"
            lines.append(f"Attack: {attack}")
            lines.append(f"Distribution: {dist_title}")
            lines.append("-" * 40)
            
            for agg in AGGREGATORS:
                lines.append(f"  Aggregator: {agg}")
                for f in F_VALUES:
                    if f == 0:
                        folder_name = f"NoAttack_Average_f_0_{dist}"
                    else:
                        folder_name = f"{attack}_{agg}_f_{f}_{dist}"
                        
                    acc_direct = load_final_accuracy(RESULTS_DIR_DIRECT, folder_name)
                    acc_rate = load_final_accuracy(RESULTS_DIR_RATE, folder_name)
                    
                    str_direct = f"{acc_direct:.4f}" if acc_direct is not None else "N/A"
                    str_rate = f"{acc_rate:.4f}" if acc_rate is not None else "N/A"
                    
                    lines.append(f"    f = {f}: (encoding = direct : acc = {str_direct}, encoding = rate : acc = {str_rate})")
                lines.append("")
            lines.append("=" * 60)
            lines.append("")
            
    summary_text = "\n".join(lines)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        f.write(summary_text)
        
    print(f"Summary written to {OUTPUT_FILE}")
    print("\nSummary Content:")
    print(summary_text)

if __name__ == "__main__":
    main()
