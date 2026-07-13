import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_avg_metric(model_name, metric_prefix):
    # Depending on how the directories were generated, they might be in clean structure or direct
    # Since clean_directory_structure=True was in the config, it should be:
    # results/sparsity_experiment/model_name/model_name_n_10.../
    base_dir = f"results/sparsity_experiment/{model_name}"
    if not os.path.isdir(base_dir):
        return None
    
    metric_arrays = []
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.startswith(metric_prefix) and file.endswith(".txt"):
                filepath = os.path.join(root, file)
                try:
                    data = np.loadtxt(filepath, delimiter=',')
                    metric_arrays.append(data)
                except Exception as e:
                    print(f"Failed to load {filepath}: {e}")
                    
    if len(metric_arrays) == 0:
        return None
        
    return np.mean(metric_arrays, axis=0)


def plot_variance():
    os.makedirs('plots/sparsity_experiment', exist_ok=True)
    
    # We plot two things: std of grad norm, and max deviation
    metrics = {
        "honest_grad_norm_std": "Standard Deviation of Honest Gradients Norms",
        "honest_max_deviation": "Max Euclidean Deviation Among Honest Clients"
    }
    
    for metric_prefix, metric_title in metrics.items():
        cnn_val = get_avg_metric('cnn_mnist', metric_prefix)
        tanh_val = get_avg_metric('cnn_mnist_tanh', metric_prefix)
        snn_val = get_avg_metric('cnn_mnist_snn', metric_prefix)
        
        plt.figure(figsize=(10, 6))
        
        if cnn_val is not None:
            plt.plot(np.arange(len(cnn_val)) * 10, cnn_val, label="CNN ReLU (Dense)", linewidth=2)
        if tanh_val is not None:
            plt.plot(np.arange(len(tanh_val)) * 10, tanh_val, label="CNN Tanh (Dense)", linewidth=2)
        if snn_val is not None:
            plt.plot(np.arange(len(snn_val)) * 10, snn_val, label="SNN Tri (Sparse)", linewidth=2)
            
        plt.xlabel('Communication Rounds', fontsize=14)
        plt.ylabel(metric_title, fontsize=14)
        plt.title(f'{metric_title} ($f=0$, $\gamma=0.33$)', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.yscale('log') # Use log scale because variance can be orders of magnitude different
        plt.tight_layout()
        plt.savefig(f'plots/sparsity_experiment/{metric_prefix}_comparison.pdf')
        plt.savefig(f'plots/sparsity_experiment/{metric_prefix}_comparison.png')
        print(f"Plot saved to plots/sparsity_experiment/{metric_prefix}_comparison.png")

if __name__ == "__main__":
    plot_variance()
