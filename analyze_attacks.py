import os
import json
import numpy as np
from collections import defaultdict

def main():
    results_dir = "/localhome/bendouro/ByzFL_snn/byzfl/results"
    
    # Structure: data[f][gamma][attack][model][aggregator] = mean_acc
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    
    for root, dirs, files in os.walk(results_dir):
        if "config.json" in files:
            config_path = os.path.join(root, "config.json")
            try:
                with open(config_path, "r") as f_obj:
                    config = json.load(f_obj)
            except Exception:
                continue
            
            # Extract key parameters
            try:
                # Need to handle both "number_of_faulty_nodes" (old?) and "f"
                f = config["benchmark_config"].get("f", config["benchmark_config"].get("number_of_faulty_nodes"))
                if f is None:
                    continue
                if isinstance(f, list): f = f[0]
                    
                dist = config["benchmark_config"]["data_distribution"]
                if isinstance(dist, list): dist = dist[0]
                gamma = dist["distribution_parameter"]
                if isinstance(gamma, list): gamma = gamma[0]
                if gamma is None: continue
                if f is None: continue
                
                atk = config.get("attack", config["benchmark_config"].get("attack"))
                if isinstance(atk, list): atk = atk[0]
                attack = atk["name"]
                
                agg = config.get("aggregator", config["benchmark_config"].get("aggregator"))
                if isinstance(agg, list): agg = agg[0]
                aggregator = agg["name"]
                
                pre_aggs = config.get("pre_aggregators", config["benchmark_config"].get("pre_aggregators", []))
                if pre_aggs:
                    # e.g., NNM_ARC
                    aggregator = "_".join([pa["name"] for pa in pre_aggs]) + "_" + aggregator
                
                # Model type
                mod = config.get("model", config["benchmark_config"].get("model"))
                if isinstance(mod, list): mod = mod[0]
                model_name = mod["name"].lower()
                
                if "snn" in model_name:
                    model_type = "SNN"
                elif "cnn" in model_name:
                    if mod.get("clip_norm", None) is not None or "clip_norm" in mod:
                        model_type = "CNN Clipped"
                    else:
                        activation = mod.get("activation", "relu").lower()
                        if "tanh" in model_name:
                            activation = "tanh"
                        if "dropout" in model_name:
                            import re
                            m = re.search(r"dropout_(\d+)", model_name)
                            drop = f" Drop{m.group(1)}" if m else ""
                            model_type = f"CNN {activation.capitalize()}{drop}"
                        else:
                            model_type = f"CNN {activation.capitalize()}"
                else:
                    model_type = "Unknown"
                
                # Get test accuracy
                test_accs = []
                for file in os.listdir(root):
                    if file.startswith("test_accuracy_") and file.endswith(".txt"):
                        with open(os.path.join(root, file), "r") as f_acc:
                            content = f_acc.read().strip()
                            if not content:
                                continue
                            vals = content.split(",")
                            try:
                                acc = float(vals[-1])
                                test_accs.append(acc)
                            except ValueError:
                                pass
                
                if test_accs:
                    mean_acc = np.mean(test_accs)
                    data[f][gamma][attack][model_type][aggregator] = mean_acc
                    
            except KeyError as e:
                pass

    pivots = [(1, 0.0), (1, 1.0), (5, 0.66), (5, 0.33), (1, 0.33), (5, 0.0)] # standard pivots
    
    print("\n--- DETAILED PIVOTS ---\n")

    for f in sorted(data.keys()):
        if f is None: continue
        for gamma in sorted(data[f].keys()):
            if gamma is None: continue
            if (f, gamma) in pivots or (f == 5 and gamma > 0.6 and gamma < 0.7) or (f == 1 and gamma < 0.1) or (f == 10):
                print(f"\n{'='*60}")
                print(f"PIVOT: f = {f}, gamma = {gamma:.2f}")
                print(f"{'='*60}")
                attacks = sorted(data[f][gamma].keys())
                for attack in attacks:
                    print(f"\nAttack: {attack}")
                    models = sorted(data[f][gamma][attack].keys())
                    # Sort models to compare easily: SNN first, then CNNs
                    models = sorted(models, key=lambda x: ("A" if "SNN" in x else "B") + x)
                    for model in models:
                        aggs = data[f][gamma][attack][model]
                        # Let's sort aggregators
                        for agg in sorted(aggs.keys()):
                            acc = aggs[agg]
                            print(f"  {model:20s} | Agg: {agg:15s} | Acc: {acc:.4f}")

if __name__ == "__main__":
    main()
