import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from byzfl.benchmark.benchmark import run_benchmark

import subprocess

def run_experiments():
    os.makedirs('results/sparsity_experiment', exist_ok=True)
    
    print("Running all 15 jobs in parallel (3 models * 5 seeds)...")
    
    # Use subprocesses to avoid CUDA multiprocessing context issues
    cmd1 = ["python", "-c", "from byzfl.benchmark.benchmark import run_benchmark; run_benchmark('configs/sparsity_cnn.json', nb_jobs=5, distribute_gpus=True)"]
    cmd2 = ["python", "-c", "from byzfl.benchmark.benchmark import run_benchmark; run_benchmark('configs/sparsity_cnn_tanh.json', nb_jobs=5, distribute_gpus=True)"]
    cmd3 = ["python", "-c", "from byzfl.benchmark.benchmark import run_benchmark; run_benchmark('configs/sparsity_snn_atan.json', nb_jobs=5, distribute_gpus=True)"]
    
    p1 = subprocess.Popen(cmd1)
    p2 = subprocess.Popen(cmd2)
    p3 = subprocess.Popen(cmd3)
    
    # Wait for all 3 sweeps to complete
    p1.wait()
    p2.wait()
    p3.wait()
    print("All 15 jobs completed!")


def get_avg_cos_sim(model_name):
    base_dir = f"results/sparsity_experiment/{model_name}"
    if not os.path.isdir(base_dir):
        return None
        
    exp_dir = None
    for d in os.listdir(base_dir):
        if d.startswith(f"mnist_{model_name}"):
            exp_dir = os.path.join(base_dir, d)
            break
            
    if exp_dir is None:
        return None
        
    cos_sims = []
    # Find all honest_mean_cos_sim files
    for f in os.listdir(exp_dir):
        if f.startswith("honest_mean_cos_sim"):
            cos_file = os.path.join(exp_dir, f)
            try:
                data = np.loadtxt(cos_file, delimiter=',')
                cos_sims.append(data)
            except Exception as e:
                pass
                    
    if len(cos_sims) == 0:
        return None
        
    return np.mean(cos_sims, axis=0)


def plot_results():
    os.makedirs('plots/sparsity_experiment', exist_ok=True)
    
    cnn_cos = get_avg_cos_sim('cnn_mnist')
    tanh_cos = get_avg_cos_sim('cnn_mnist_tanh')
    snn_cos = get_avg_cos_sim('cnn_mnist_snn')
    
    plt.figure(figsize=(10, 6))
    
    if cnn_cos is not None:
        plt.plot(np.arange(len(cnn_cos)), cnn_cos, label="CNN ReLU (Dense)", linewidth=2)
    if tanh_cos is not None:
        plt.plot(np.arange(len(tanh_cos)), tanh_cos, label="CNN Tanh (Dense)", linewidth=2)
    if snn_cos is not None:
        plt.plot(np.arange(len(snn_cos)), snn_cos, label="SNN Atan (Sparse)", linewidth=2)
        
    plt.xlabel('Communication Rounds', fontsize=14)
    plt.ylabel('Mean Cosine Similarity', fontsize=14)
    plt.title(r'Honest Clients Gradient Alignment ($f=0$, $\gamma=0.33$)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('plots/sparsity_experiment/honest_gradient_alignment.pdf')
    plt.savefig('plots/sparsity_experiment/honest_gradient_alignment.png')
    print("Plot saved to plots/sparsity_experiment/honest_gradient_alignment.pdf")


if __name__ == "__main__":
    run_experiments()
    plot_results()
