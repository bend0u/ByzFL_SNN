import os
import numpy as np

GAMMAS = [1.0, 0.0]
SPARSITY_DIR = "results/sparsity_measure"
VARIANCE_DIR = "results/snn/variance_sweep"

MODELS = {
    'CNN ReLU': {
        'sparsity_pattern': 'mnist_cnn_mnist_n_10_f_0_d_0_gamma_similarity_niid_',
        'variance_dir': 'results/cnn/variance_sweep',
        'variance_pattern': 'mnist_cnn_mnist_n_10_f_0_d_0_gamma_similarity_niid_'
    },
    'CNN Tanh': {
        'sparsity_pattern': 'mnist_cnn_mnist_tanh_n_10_f_0_d_0_gamma_similarity_niid_',
        'variance_dir': 'results/cnn/variance_sweep',
        'variance_pattern': 'mnist_cnn_mnist_tanh_n_10_f_0_d_0_gamma_similarity_niid_'
    },
    'SNN': {
        'sparsity_pattern': 'mnist_cnn_mnist_snn_n_10_f_0_d_0_gamma_similarity_niid_',
        'variance_dir': 'results/snn/variance_sweep',
        'variance_pattern': 'mnist_cnn_mnist_snn_n_10_f_0_d_0_gamma_similarity_niid_' # wait, let's just match any snn with gamma
    }
}

def load_sparsity(pattern, gamma, metric):
    dirs = []
    if not os.path.exists(SPARSITY_DIR): return "N/A"
    full_pat = f"{pattern}{gamma}_"
    for e in os.listdir(SPARSITY_DIR):
        if full_pat in e or (e.startswith(pattern) and e.endswith(f"{gamma}")):
            dirs.append(os.path.join(SPARSITY_DIR, e))
    
    vals = []
    for d in dirs:
        for f in os.listdir(d):
            if f.startswith(f"sparsity_{metric}_mean") and f.endswith(".txt"):
                try:
                    data = np.loadtxt(os.path.join(d, f), delimiter=',')
                    if data.ndim == 1: vals.append(data[-1])
                    elif data.ndim == 2: vals.append(data[0][-1])
                except Exception: pass
    if vals: return f"{np.mean(vals):.4f}"
    return "N/A"

def load_variance(vdir, gamma, metric):
    dirs = []
    import json
    if not os.path.exists(vdir): return "N/A"
    for e in os.listdir(vdir):
        path = os.path.join(vdir, e)
        if os.path.isdir(path):
            cp = os.path.join(path, "config.json")
            if os.path.exists(cp):
                try:
                    with open(cp) as f: config = json.load(f)
                    g = config["benchmark_config"]["data_distribution"]["distribution_parameter"]
                    if type(g) == list: g = g[0]
                    if abs(g - gamma) < 1e-6:
                        dirs.append(path)
                except Exception: pass
    vals = []
    for d in dirs:
        for f in os.listdir(d):
            if f.startswith(metric) and f.endswith(".txt"):
                try:
                    data = np.loadtxt(os.path.join(d, f), delimiter=',')
                    vals.append(data[-1])
                except Exception: pass
    if vals: return f"{np.mean(vals):.4f}"
    return "N/A"

def main():
    print("--- EMPIRICAL COMPARISON: SPARSITY vs MAGNITUDE ---")
    for gamma in GAMMAS:
        print(f"\nGamma = {gamma}:")
        print(f"| {'Model':15} | {'Hoyer Sparsity':15} | {'Max Abs Coord':15} | {'Max Deviation':15} |")
        print(f"|{'-'*17}|{'-'*17}|{'-'*17}|{'-'*17}|")
        
        for name, info in MODELS.items():
            hoyer = load_sparsity(info['sparsity_pattern'], gamma, 'hoyer')
            max_abs = load_variance(info['variance_dir'], gamma, 'honest_max_abs_grad')
            max_dev = load_variance(info['variance_dir'], gamma, 'honest_max_deviation')
            
            print(f"| {name:15} | {hoyer:15} | {max_abs:15} | {max_dev:15} |")

if __name__ == "__main__":
    main()
