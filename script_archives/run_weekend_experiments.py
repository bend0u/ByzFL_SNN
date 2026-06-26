import os
import argparse
import json
import traceback
import numpy as np

from byzfl import run_benchmark

# Define results directory
RESULTS_DIR = "./weekendexperiments"
SUMMARY_FILE = os.path.join(RESULTS_DIR, "accuracy_summary.txt")

# Define all 68 configurations
CONFIGS = [
    # Block A: Clean Baselines & Raw Skew Taxes (12 Runs)
    {"id": 1, "encoding": "rate", "dist": "iid", "dist_param": 1.0, "f": 0, "attack": "NoAttack", "pre_agg": None, "agg": "Average", "desc": "Maximum baseline ceiling for both encodings."},
    {"id": 2, "encoding": "constant", "dist": "iid", "dist_param": 1.0, "f": 0, "attack": "NoAttack", "pre_agg": None, "agg": "Average", "desc": "Maximum baseline ceiling for both encodings."},
    {"id": 3, "encoding": "rate", "dist": "dirichlet_niid", "dist_param": 0.5, "f": 0, "attack": "NoAttack", "pre_agg": None, "agg": "Average", "desc": "Clean impact of organic label skew on SNN updates."},
    {"id": 4, "encoding": "constant", "dist": "dirichlet_niid", "dist_param": 0.5, "f": 0, "attack": "NoAttack", "pre_agg": None, "agg": "Average", "desc": "Clean impact of organic label skew on SNN updates."},
    {"id": 5, "encoding": "rate", "dist": "iid", "dist_param": 1.0, "f": 0, "attack": "NoAttack", "pre_agg": None, "agg": "CenteredClipping", "desc": "Measures if Centered Clipping alters clean SNN convergence."},
    {"id": 6, "encoding": "constant", "dist": "iid", "dist_param": 1.0, "f": 0, "attack": "NoAttack", "pre_agg": None, "agg": "CenteredClipping", "desc": "Measures if Centered Clipping alters clean SNN convergence."},
    {"id": 7, "encoding": "rate", "dist": "dirichlet_niid", "dist_param": 0.5, "f": 0, "attack": "NoAttack", "pre_agg": None, "agg": "CenteredClipping", "desc": "Tax of Centered Clipping under Non-IID conditions."},
    {"id": 8, "encoding": "constant", "dist": "dirichlet_niid", "dist_param": 0.5, "f": 0, "attack": "NoAttack", "pre_agg": None, "agg": "CenteredClipping", "desc": "Tax of Centered Clipping under Non-IID conditions."},
    {"id": 9, "encoding": "rate", "dist": "dirichlet_niid", "dist_param": 0.5, "f": 0, "attack": "NoAttack", "pre_agg": None, "agg": "Median", "desc": "Standard coordinate trimming tax on raw skew."},
    {"id": 10, "encoding": "constant", "dist": "dirichlet_niid", "dist_param": 0.5, "f": 0, "attack": "NoAttack", "pre_agg": None, "agg": "Median", "desc": "Standard coordinate trimming tax on raw skew."},
    {"id": 11, "encoding": "rate", "dist": "dirichlet_niid", "dist_param": 0.5, "f": 0, "attack": "NoAttack", "pre_agg": None, "agg": "MultiKrum", "desc": "Standard distance-based tax on raw skew."},
    {"id": 12, "encoding": "constant", "dist": "dirichlet_niid", "dist_param": 0.5, "f": 0, "attack": "NoAttack", "pre_agg": None, "agg": "MultiKrum", "desc": "Standard distance-based tax on raw skew."},
    
    # Block B: The SignFlipping Crucible (24 Runs)
    {"id": 13, "encoding": "rate", "dist": "iid", "dist_param": 1.0, "f": 2, "attack": "SignFlipping", "pre_agg": None, "agg": "Average", "desc": "Unprotected system vulnerability under mild attack."},
    {"id": 14, "encoding": "constant", "dist": "iid", "dist_param": 1.0, "f": 2, "attack": "SignFlipping", "pre_agg": None, "agg": "Average", "desc": "Unprotected system vulnerability under mild attack."},
    {"id": 15, "encoding": "rate", "dist": "dirichlet_niid", "dist_param": 0.5, "f": 2, "attack": "SignFlipping", "pre_agg": None, "agg": "Average", "desc": "Compounding destruction of attack + label skew."},
    {"id": 16, "encoding": "constant", "dist": "dirichlet_niid", "dist_param": 0.5, "f": 2, "attack": "SignFlipping", "pre_agg": None, "agg": "Average", "desc": "Compounding destruction of attack + label skew."},
    {"id": 17, "encoding": "rate", "dist": "iid", "dist_param": 1.0, "f": 5, "attack": "SignFlipping", "pre_agg": None, "agg": "Average", "desc": "Absolute baseline system breakdown point (~31% malicious)."},
    {"id": 18, "encoding": "constant", "dist": "iid", "dist_param": 1.0, "f": 5, "attack": "SignFlipping", "pre_agg": None, "agg": "Average", "desc": "Absolute baseline system breakdown point (~31% malicious)."},
    {"id": 19, "encoding": "rate", "dist": "dirichlet_niid", "dist_param": 0.5, "f": 5, "attack": "SignFlipping", "pre_agg": None, "agg": "Average", "desc": "Total network collapse baseline."},
    {"id": 20, "encoding": "constant", "dist": "dirichlet_niid", "dist_param": 0.5, "f": 5, "attack": "SignFlipping", "pre_agg": None, "agg": "Average", "desc": "Total network collapse baseline."},
    {"id": 21, "encoding": "rate", "dist": "iid", "dist_param": 1.0, "f": 5, "attack": "SignFlipping", "pre_agg": None, "agg": "Median", "desc": "Can coordinate-wise Median cleanly filter inverted signs?"},
    {"id": 22, "encoding": "constant", "dist": "iid", "dist_param": 1.0, "f": 5, "attack": "SignFlipping", "pre_agg": None, "agg": "Median", "desc": "Can coordinate-wise Median cleanly filter inverted signs?"},
    {"id": 23, "encoding": "rate", "dist": "dirichlet_niid", "dist_param": 0.5, "f": 5, "attack": "SignFlipping", "pre_agg": None, "agg": "Median", "desc": "Does Non-IID layout shield sign flippers from Median?"},
    {"id": 24, "encoding": "constant", "dist": "dirichlet_niid", "dist_param": 0.5, "f": 5, "attack": "SignFlipping", "pre_agg": None, "agg": "Median", "desc": "Does Non-IID layout shield sign flippers from Median?"},
    {"id": 25, "encoding": "rate", "dist": "iid", "dist_param": 1.0, "f": 5, "attack": "SignFlipping", "pre_agg": None, "agg": "CenteredClipping", "desc": "Can Centered Clipping track and drop inverted magnitudes?"},
    {"id": 26, "encoding": "constant", "dist": "iid", "dist_param": 1.0, "f": 5, "attack": "SignFlipping", "pre_agg": None, "agg": "CenteredClipping", "desc": "Can Centered Clipping track and drop inverted magnitudes?"},
    {"id": 27, "encoding": "rate", "dist": "dirichlet_niid", "dist_param": 0.5, "f": 5, "attack": "SignFlipping", "pre_agg": None, "agg": "CenteredClipping", "desc": "Critical SOTA test: Centered Clipping vs SignFlipping on skewed data."},
    {"id": 28, "encoding": "constant", "dist": "dirichlet_niid", "dist_param": 0.5, "f": 5, "attack": "SignFlipping", "pre_agg": None, "agg": "CenteredClipping", "desc": "Critical SOTA test: Centered Clipping vs SignFlipping on skewed data."},
    {"id": 29, "encoding": "rate", "dist": "iid", "dist_param": 1.0, "f": 5, "attack": "SignFlipping", "pre_agg": None, "agg": "MultiKrum", "desc": "Can distance geometry instantly decouple inverted vectors?"},
    {"id": 30, "encoding": "constant", "dist": "iid", "dist_param": 1.0, "f": 5, "attack": "SignFlipping", "pre_agg": None, "agg": "MultiKrum", "desc": "Can distance geometry instantly decouple inverted vectors?"},
    {"id": 31, "encoding": "rate", "dist": "dirichlet_niid", "dist_param": 0.5, "f": 5, "attack": "SignFlipping", "pre_agg": None, "agg": "MultiKrum", "desc": "Does Krum drop unique honest nodes over explicit sign-flippers?"},
    {"id": 32, "encoding": "constant", "dist": "dirichlet_niid", "dist_param": 0.5, "f": 5, "attack": "SignFlipping", "pre_agg": None, "agg": "MultiKrum", "desc": "Does Krum drop unique honest nodes over explicit sign-flippers?"},
    {"id": 33, "encoding": "rate", "dist": "iid", "dist_param": 1.0, "f": 5, "attack": "SignFlipping", "pre_agg": "Clipping", "agg": "TrMean", "desc": "Frontline pipeline protection vs aggressive scale shifts."},
    {"id": 34, "encoding": "constant", "dist": "iid", "dist_param": 1.0, "f": 5, "attack": "SignFlipping", "pre_agg": "Clipping", "agg": "TrMean", "desc": "Frontline pipeline protection vs aggressive scale shifts."},
    {"id": 35, "encoding": "rate", "dist": "dirichlet_niid", "dist_param": 0.5, "f": 5, "attack": "SignFlipping", "pre_agg": "Clipping", "agg": "TrMean", "desc": "Evaluates if static clipping helps Trimmed Mean under skew."},
    {"id": 36, "encoding": "constant", "dist": "dirichlet_niid", "dist_param": 0.5, "f": 5, "attack": "SignFlipping", "pre_agg": "Clipping", "agg": "TrMean", "desc": "Evaluates if static clipping helps Trimmed Mean under skew."},

    # Block C: The Stealthy ALIE Challenge (24 Runs)
    {"id": 37, "encoding": "rate", "dist": "iid", "dist_param": 1.0, "f": 2, "attack": "Optimal_ALittleIsEnough", "pre_agg": None, "agg": "Average", "desc": "Does continuous optimization drag effortlessly pull down SNNs?"},
    {"id": 38, "encoding": "constant", "dist": "iid", "dist_param": 1.0, "f": 2, "attack": "Optimal_ALittleIsEnough", "pre_agg": None, "agg": "Average", "desc": "Does continuous optimization drag effortlessly pull down SNNs?"},
    {"id": 39, "encoding": "rate", "dist": "dirichlet_niid", "dist_param": 0.5, "f": 2, "attack": "Optimal_ALittleIsEnough", "pre_agg": None, "agg": "Average", "desc": "Does ALIE successfully mask itself inside natural client variance?"},
    {"id": 40, "encoding": "constant", "dist": "dirichlet_niid", "dist_param": 0.5, "f": 2, "attack": "Optimal_ALittleIsEnough", "pre_agg": None, "agg": "Average", "desc": "Does ALIE successfully mask itself inside natural client variance?"},
    {"id": 41, "encoding": "rate", "dist": "iid", "dist_param": 1.0, "f": 5, "attack": "Optimal_ALittleIsEnough", "pre_agg": None, "agg": "Median", "desc": "The Median Trap: Does ALIE safely cluster inside the SNN median?"},
    {"id": 42, "encoding": "constant", "dist": "iid", "dist_param": 1.0, "f": 5, "attack": "Optimal_ALittleIsEnough", "pre_agg": None, "agg": "Median", "desc": "The Median Trap: Does ALIE safely cluster inside the SNN median?"},
    {"id": 43, "encoding": "rate", "dist": "dirichlet_niid", "dist_param": 0.5, "f": 5, "attack": "Optimal_ALittleIsEnough", "pre_agg": None, "agg": "Median", "desc": "Does data skew worsen the Median Trap for SNNs?"},
    {"id": 44, "encoding": "constant", "dist": "dirichlet_niid", "dist_param": 0.5, "f": 5, "attack": "Optimal_ALittleIsEnough", "pre_agg": None, "agg": "Median", "desc": "Does data skew worsen the Median Trap for SNNs?"},
    {"id": 45, "encoding": "rate", "dist": "iid", "dist_param": 1.0, "f": 5, "attack": "Optimal_ALittleIsEnough", "pre_agg": None, "agg": "CenteredClipping", "desc": "Can Centered Clipping's iterative momentum catch the subtle ALIE shift?"},
    {"id": 46, "encoding": "constant", "dist": "iid", "dist_param": 1.0, "f": 5, "attack": "Optimal_ALittleIsEnough", "pre_agg": None, "agg": "CenteredClipping", "desc": "Can Centered Clipping's iterative momentum catch the subtle ALIE shift?"},
    {"id": 47, "encoding": "rate", "dist": "dirichlet_niid", "dist_param": 0.5, "f": 5, "attack": "Optimal_ALittleIsEnough", "pre_agg": None, "agg": "CenteredClipping", "desc": "High-value test: Does Centered Clipping drop or absorb ALIE drift?"},
    {"id": 48, "encoding": "constant", "dist": "dirichlet_niid", "dist_param": 0.5, "f": 5, "attack": "Optimal_ALittleIsEnough", "pre_agg": None, "agg": "CenteredClipping", "desc": "High-value test: Does Centered Clipping drop or absorb ALIE drift?"},
    {"id": 49, "encoding": "rate", "dist": "iid", "dist_param": 1.0, "f": 5, "attack": "Optimal_ALittleIsEnough", "pre_agg": None, "agg": "MultiKrum", "desc": "The Stagnation Cliff: Does ALIE freeze Krum selection loops?"},
    {"id": 50, "encoding": "constant", "dist": "iid", "dist_param": 1.0, "f": 5, "attack": "Optimal_ALittleIsEnough", "pre_agg": None, "agg": "MultiKrum", "desc": "The Stagnation Cliff: Does ALIE freeze Krum selection loops?"},
    {"id": 51, "encoding": "rate", "dist": "dirichlet_niid", "dist_param": 0.5, "f": 5, "attack": "Optimal_ALittleIsEnough", "pre_agg": None, "agg": "MultiKrum", "desc": "Does Krum freeze faster when data distributions are heavily skewed?"},
    {"id": 52, "encoding": "constant", "dist": "dirichlet_niid", "dist_param": 0.5, "f": 5, "attack": "Optimal_ALittleIsEnough", "pre_agg": None, "agg": "MultiKrum", "desc": "Does Krum freeze faster when data distributions are heavily skewed?"},
    {"id": 53, "encoding": "rate", "dist": "iid", "dist_param": 1.0, "f": 5, "attack": "Optimal_ALittleIsEnough", "pre_agg": "NNM", "agg": "TrMean", "desc": "Can Nearest Neighbor Mixing wash away the stealthy ALIE group?"},
    {"id": 54, "encoding": "constant", "dist": "iid", "dist_param": 1.0, "f": 5, "attack": "Optimal_ALittleIsEnough", "pre_agg": "NNM", "agg": "TrMean", "desc": "Can Nearest Neighbor Mixing wash away the stealthy ALIE group?"},
    {"id": 55, "encoding": "rate", "dist": "dirichlet_niid", "dist_param": 0.5, "f": 5, "attack": "Optimal_ALittleIsEnough", "pre_agg": "NNM", "agg": "TrMean", "desc": "Hybrid pipeline protection performance vs stealth."},
    {"id": 56, "encoding": "constant", "dist": "dirichlet_niid", "dist_param": 0.5, "f": 5, "attack": "Optimal_ALittleIsEnough", "pre_agg": "NNM", "agg": "TrMean", "desc": "Hybrid pipeline protection performance vs stealth."},

    # Block D: Inner Product Manipulation (IPM) Optimization Drag (12 Runs)
    {"id": 57, "encoding": "rate", "dist": "iid", "dist_param": 1.0, "f": 5, "attack": "Optimal_InnerProductManipulation", "pre_agg": None, "agg": "Average", "desc": "Baseline directional optimization drag on SNN parameters."},
    {"id": 58, "encoding": "constant", "dist": "iid", "dist_param": 1.0, "f": 5, "attack": "Optimal_InnerProductManipulation", "pre_agg": None, "agg": "Average", "desc": "Baseline directional optimization drag on SNN parameters."},
    {"id": 59, "encoding": "rate", "dist": "dirichlet_niid", "dist_param": 0.5, "f": 5, "attack": "Optimal_InnerProductManipulation", "pre_agg": None, "agg": "Average", "desc": "Does directional drag break the network faster under heavy skew?"},
    {"id": 60, "encoding": "constant", "dist": "dirichlet_niid", "dist_param": 0.5, "f": 5, "attack": "Optimal_InnerProductManipulation", "pre_agg": None, "agg": "Average", "desc": "Does directional drag break the network faster under heavy skew?"},
    {"id": 61, "encoding": "rate", "dist": "iid", "dist_param": 1.0, "f": 5, "attack": "Optimal_InnerProductManipulation", "pre_agg": None, "agg": "Median", "desc": "Can coordinate tracking stop unified directional pushbacks?"},
    {"id": 62, "encoding": "constant", "dist": "iid", "dist_param": 1.0, "f": 5, "attack": "Optimal_InnerProductManipulation", "pre_agg": None, "agg": "Median", "desc": "Can coordinate tracking stop unified directional pushbacks?"},
    {"id": 63, "encoding": "rate", "dist": "dirichlet_niid", "dist_param": 0.5, "f": 5, "attack": "Optimal_InnerProductManipulation", "pre_agg": None, "agg": "Median", "desc": "Accuracy drop measurement: tracking the exact percentage bled."},
    {"id": 64, "encoding": "constant", "dist": "dirichlet_niid", "dist_param": 0.5, "f": 5, "attack": "Optimal_InnerProductManipulation", "pre_agg": None, "agg": "Median", "desc": "Accuracy drop measurement: tracking the exact percentage bled."},
    {"id": 65, "encoding": "rate", "dist": "iid", "dist_param": 1.0, "f": 5, "attack": "Optimal_InnerProductManipulation", "pre_agg": None, "agg": "CenteredClipping", "desc": "Does Centered Clipping correctly identify a structurally reversed gradient?"},
    {"id": 66, "encoding": "constant", "dist": "iid", "dist_param": 1.0, "f": 5, "attack": "Optimal_InnerProductManipulation", "pre_agg": None, "agg": "CenteredClipping", "desc": "Does Centered Clipping correctly identify a structurally reversed gradient?"},
    {"id": 67, "encoding": "rate", "dist": "dirichlet_niid", "dist_param": 0.5, "f": 5, "attack": "Optimal_InnerProductManipulation", "pre_agg": "NNM", "agg": "TrMean", "desc": "SOTA defensive pipeline vs deliberate gradient vector reversal."},
    {"id": 68, "encoding": "constant", "dist": "dirichlet_niid", "dist_param": 0.5, "f": 5, "attack": "Optimal_InnerProductManipulation", "pre_agg": "NNM", "agg": "TrMean", "desc": "SOTA defensive pipeline vs deliberate gradient vector reversal."},
]

# Helper function to get the subdirectory name matching FileManager.clean directory structure
def get_folder_path(cfg_info):
    pre_aggs = [cfg_info["pre_agg"]] if cfg_info["pre_agg"] else []
    preaggs_aggregator = '_'.join(pre_aggs + [cfg_info["agg"]])
    
    dist_part = cfg_info["dist"]
    if cfg_info["dist"] not in ["iid", "extreme_niid"]:
        dist_part = f"{cfg_info['dist']}_{cfg_info['dist_param']}"
        
    folder_name = (
        f"{cfg_info['attack']}_"
        f"{preaggs_aggregator}_"
        f"f_{cfg_info['f']}_"
        f"{dist_part}"
    )
    enc_name = "direct" if cfg_info["encoding"] == "constant" else cfg_info["encoding"]
    parent_dir = f"mnist_{enc_name}"
    
    return os.path.join(RESULTS_DIR, parent_dir, folder_name)

def load_final_accuracy(folder_path):
    path = os.path.join(folder_path, "test_accuracy.txt")
    if not os.path.exists(path):
        return None
    try:
        data = np.loadtxt(path, delimiter=",")
        if data.ndim == 0:
            return float(data)
        return float(data[-1])
    except Exception:
        return None

def make_config(cfg_info, template_rate, template_constant, nb_steps, evaluation_delta):
    if cfg_info["encoding"] == "rate":
        cfg = json.loads(json.dumps(template_rate))
    else:
        cfg = json.loads(json.dumps(template_constant))
        
    cfg["benchmark_config"]["device"] = "cuda"
    cfg["benchmark_config"]["nb_steps"] = nb_steps
    cfg["benchmark_config"]["nb_honest_clients"] = 16
    cfg["benchmark_config"]["training_seed"] = 42
    cfg["benchmark_config"]["data_distribution_seed"] = 42
    cfg["benchmark_config"]["nb_training_seeds"] = 1
    cfg["benchmark_config"]["nb_data_distribution_seeds"] = 1
    
    cfg["benchmark_config"]["data_distribution"] = [
        {"name": cfg_info["dist"], "distribution_parameter": cfg_info["dist_param"]}
    ]
    
    cfg["benchmark_config"]["f"] = [cfg_info["f"]]
    cfg["benchmark_config"]["tolerated_f"] = [cfg_info["f"]]
    
    cfg["attack"] = [
        {"name": cfg_info["attack"], "parameters": {}}
    ]
    
    if cfg_info["pre_agg"] is None:
        cfg["pre_aggregators"] = []
    else:
        cfg["pre_aggregators"] = [
            {"name": cfg_info["pre_agg"], "parameters": {}}
        ]
        
    cfg["aggregator"] = [
        {"name": cfg_info["agg"], "parameters": {}}
    ]
    
    cfg["evaluation_and_results"]["results_directory"] = RESULTS_DIR
    cfg["evaluation_and_results"]["evaluation_delta"] = evaluation_delta
    cfg["evaluation_and_results"]["clean_directory_structure"] = True
    
    return cfg

def main():
    parser = argparse.ArgumentParser(description="Run SNN Byzantine sweeps for weekend experiments.")
    parser.add_argument("--test", action="store_true", help="Run a quick test of the pipeline with 2 steps.")
    args = parser.parse_args()
    
    nb_steps = 2 if args.test else 500
    evaluation_delta = 1 if args.test else 10
    
    # Load templates
    with open("snn_mnist_rate.json", "r") as f:
        template_rate = json.load(f)
    with open("snn_mnist_direct.json", "r") as f:
        template_constant = json.load(f)
        
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Save a master config for these weekend experiments
    master_config_path = os.path.join(RESULTS_DIR, "config.json")
    with open(master_config_path, "w") as f:
        json.dump({
            "benchmark_config": {
                "training_seed": 42,
                "nb_training_seeds": 1,
                "nb_honest_clients": 16,
                "f": [0, 2, 5],
                "data_distribution_seed": 42,
                "nb_data_distribution_seeds": 1,
                "data_distribution": [
                    {"name": "iid", "distribution_parameter": 1.0},
                    {"name": "dirichlet_niid", "distribution_parameter": 0.5}
                ],
                "nb_steps": nb_steps
            },
            "model": {
                "name": "convnet_snn",
                "dataset_name": "mnist",
                "learning_rate": [0.05],
                "learning_rate_decay": 1.0,
                "milestones": []
            },
            "honest_clients": {
                "momentum": [0.9],
                "weight_decay": [0.0001],
                "batch_size": 128
            },
            "aggregator": [
                {"name": "Average"}, {"name": "CenteredClipping"}, {"name": "Median"}, {"name": "MultiKrum"}, {"name": "TrMean"}
            ],
            "pre_aggregators": [
                [], [{"name": "Clipping"}], [{"name": "NNM"}]
            ],
            "attack": [
                {"name": "NoAttack"}, {"name": "SignFlipping"}, {"name": "Optimal_ALittleIsEnough"}, {"name": "Optimal_InnerProductManipulation"}
            ],
            "evaluation_and_results": {
                "evaluation_delta": evaluation_delta,
                "batch_size_evaluation": 128,
                "results_directory": RESULTS_DIR,
                "clean_directory_structure": True
            }
        }, f, indent=4)
        
    print(f"Starting SNN weekend experiments sweep execution (Total configs: {len(CONFIGS)})...")
    
    # Temp configuration filename
    temp_fn = "temp_weekend_sweep_run.json"
    
    for idx, cfg_info in enumerate(CONFIGS):
        print(f"\n--- Running Config ID {cfg_info['id']}/{len(CONFIGS)}: {cfg_info['encoding'].upper()} encoding, {cfg_info['agg']} vs {cfg_info['attack']} ({cfg_info['dist']}) ---")
        
        cfg = make_config(cfg_info, template_rate, template_constant, nb_steps, evaluation_delta)
        
        with open(temp_fn, "w") as f:
            json.dump(cfg, f, indent=4)
            
        try:
            run_benchmark(temp_fn, nb_jobs=1)
        except Exception as e:
            print(f"Error running Config ID {cfg_info['id']}: {e}")
            traceback.print_exc()
        finally:
            if os.path.exists(temp_fn):
                os.remove(temp_fn)
                
    # Generate accuracy summary
    print("\n==========================================================")
    print("GENERATING SUMMARY REPORT FOR WEEKEND EXPERIMENTS")
    print("==========================================================")
    
    lines = [
        "==========================================================",
        "SNN WEEKEND EXPERIMENTS SUMMARY",
        "==========================================================",
        f"Test mode: {args.test}",
        f"Steps: {nb_steps}",
        "",
        f"{'Config ID':<10} | {'Encoding':<9} | {'Dist':<14} | {'f':<2} | {'Attack':<30} | {'Pre-Agg':<8} | {'Aggregator':<16} | {'Accuracy':<8} | {'Objective/Description'}",
        "-" * 140
    ]
    
    for cfg_info in CONFIGS:
        folder_path = get_folder_path(cfg_info)
        acc = load_final_accuracy(folder_path)
        acc_str = f"{acc:.4f}" if acc is not None else "N/A"
        pre_str = cfg_info["pre_agg"] if cfg_info["pre_agg"] else "None"
        
        lines.append(
            f"{cfg_info['id']:<10} | "
            f"{cfg_info['encoding']:<9} | "
            f"{cfg_info['dist']:<14} | "
            f"{cfg_info['f']:<2} | "
            f"{cfg_info['attack']:<30} | "
            f"{pre_str:<8} | "
            f"{cfg_info['agg']:<16} | "
            f"{acc_str:<8} | "
            f"{cfg_info['desc']}"
        )
        
    summary_text = "\n".join(lines)
    with open(SUMMARY_FILE, "w") as f:
        f.write(summary_text)
        
    print(f"Summary written to {SUMMARY_FILE}")
    print("\nSummary Results:")
    print("\n".join(lines[:20])) # Print first 20 lines as preview
    print("...")

if __name__ == "__main__":
    main()
