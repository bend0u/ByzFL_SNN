import os
import numpy as np

def get_folder_path(result_dir, attack_name, pre_aggs, agg, f):
    pre_aggs_str = "_".join(pre_aggs)
    preaggs_aggregator = f"{pre_aggs_str}_{agg}"
    folder_name = f"{attack_name}_{preaggs_aggregator}_f_{f}_extreme_niid"
    return os.path.join(result_dir, "mnist_direct", folder_name)

def generate_report():
    print("\n" + "=" * 80)
    print("MNIST ARC SWEEP REPORT")
    print("=" * 80)
    
    modes = [
        ("Default ALIE (delta = 10.0)", "./arc_mnist_results_default_alie"),
        ("Corrected ALIE (delta = -10.0)", "./arc_mnist_results_corrected_alie")
    ]
    
    f_values = [1, 2, 3, 4, 5]
    
    for mode_title, result_dir in modes:
        print(f"\n>>> {mode_title} <<<")
        print(f"{'Aggregator':<16} | {'Pre-Agg':<12} | " + " | ".join([f"f={f:<7}" for f in f_values]))
        print("-" * 90)
        
        for agg in ["TrMean", "GeometricMedian"]:
            for pre_agg_label, pre_aggs in [("NNM", ["NNM"]), ("NNM+ARC", ["NNM", "ARC"])]:
                acc_list = []
                peak_list = []
                for f in f_values:
                    path = get_folder_path(result_dir, "Optimal_ALittleIsEnough", pre_aggs, agg, f)
                    acc_file = os.path.join(path, "test_accuracy.txt")
                    if os.path.exists(acc_file):
                        try:
                            data = np.loadtxt(acc_file, delimiter=",")
                            if data.ndim == 0:
                                acc_list.append(f"{data:.4f}")
                                peak_list.append(f"{data:.4f}")
                            else:
                                acc_list.append(f"{data[-1]:.4f}")
                                peak_list.append(f"{np.max(data):.4f}")
                        except Exception:
                            acc_list.append("Error ")
                            peak_list.append("Error ")
                    else:
                        acc_list.append("N/A   ")
                        peak_list.append("N/A   ")
                
                print(f"{agg:<16} | {pre_agg_label:<12} | " + " | ".join([f"{a} (Pk:{p})" for a, p in zip(acc_list, peak_list)]))

if __name__ == "__main__":
    generate_report()
