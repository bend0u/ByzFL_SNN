import os
import argparse
import numpy as np
from byzfl import run_benchmark

def get_folder_path(result_dir, pre_aggs, agg, f):
    pre_aggs_str = "_".join(pre_aggs)
    preaggs_aggregator = f"{pre_aggs_str}_{agg}"
    folder_name = f"Optimal_ALittleIsEnough_{preaggs_aggregator}_f_{f}_extreme_niid"
    return os.path.join(result_dir, "mnist_direct", folder_name)

def generate_report(result_dir):
    print("\n" + "=" * 80)
    print("MNIST ARC SWEEP REPRODUCTION REPORT (Default ALIE, delta = 10.0)")
    print("=" * 80)
    
    f_values = [1, 2, 3, 4, 5]
    print(f"{'Aggregator':<16} | {'Pre-Agg':<12} | " + " | ".join([f"f={f:<7}" for f in f_values]))
    print("-" * 90)
    
    for agg in ["TrMean", "GeometricMedian"]:
        acc_list = []
        peak_list = []
        for f in f_values:
            path = get_folder_path(result_dir, ["NNM", "ARC"], agg, f)
            acc_file = os.path.join(path, "test_accuracy_tr_seed_42_dd_seed_42.txt")
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
        
        print(f"{agg:<16} | {'NNM+ARC':<12} | " + " | ".join([f"{a} (Pk:{p})" for a, p in zip(acc_list, peak_list)]))

def main():
    parser = argparse.ArgumentParser(description="Reproduce Default ALIE results using official run_benchmark pipeline.")
    parser.add_argument("--nb_jobs", type=int, default=4, help="Number of parallel trainings.")
    parser.add_argument("--report-only", action="store_true", help="Only parse and print the report from existing files.")
    args = parser.parse_args()
    
    result_dir = "./reproduce_default_alie"
    
    if not args.report_only:
        print("Starting reproduction runs for NNM+ARC configurations...")
        run_benchmark("reproduce_arc_default.json", nb_jobs=args.nb_jobs, distribute_gpus=True)
        
    generate_report(result_dir)

if __name__ == "__main__":
    main()
