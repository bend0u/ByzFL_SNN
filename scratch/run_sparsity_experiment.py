import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from byzfl.benchmark.benchmark import run_benchmark

def create_config(config_path, model_name, lr, is_snn=False, surrogate=None, surrogate_param_name=None, surrogate_param_val=None):
    config = {
        "benchmark_config": {
            "nb_training_seeds": 3,
            "nb_data_distribution_seeds": 1,
            "f": 0,
            "nb_honest_clients": 10,
            "data_distribution": {
                "name": "gamma_similarity_niid",
                "distribution_parameter": [0.33]
            },
            "size_train_set": 1.0,
            "training_algorithm": {
                "name": "DSGD",
                "parameters": {}
            },
            "nb_steps": 500
        },
        "model": {
            "name": model_name,
            "is_snn": is_snn,
            "dataset_name": "mnist",
            "nb_labels": 10,
            "loss": "ce_rate_loss" if is_snn else "ce_loss",
            "accuracy_name": "accuracy_rate" if is_snn else "accuracy",
            "optimizer_name": "SGD",
            "learning_rate": lr,
            "learning_rate_decay": 1.0,
            "milestones": []
        },
        "honest_clients": {
            "momentum": 0.9,
            "weight_decay": 0.0001,
            "batch_size": 128
        },
        "attack": [
            {
                "name": "SignFlipping",
                "parameters": {}
            }
        ],
        "aggregator": [
            {
                "name": "Average",
                "parameters": {}
            }
        ],
        "pre_aggregators": [],
        "evaluation_and_results": {
            "evaluation_delta": 10,
            "evaluate_on_test": False,
            "clean_directory_structure": True,
            "store_models": False,
            "store_per_client_metrics": False,
            "results_directory": f"results/sparsity_experiment/{model_name}"
        }
    }
    
    if is_snn:
        config["model"]["encoding"] = {
            "type": "constant",
            "time_steps": 10
        }
        config["model"]["model_params"] = {
            "beta": 0.95,
            "surrogate_gradient": surrogate,
            "surrogate_params": {
                surrogate_param_name: surrogate_param_val
            }
        }
        
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)


def run_experiments():
    os.makedirs('configs', exist_ok=True)
    os.makedirs('results/sparsity_experiment', exist_ok=True)
    
    # 1. CNN ReLU (Dense, unclipped)
    create_config('configs/sparsity_cnn.json', 'cnn_mnist', lr=0.15, is_snn=False)
    # 2. CNN Tanh (Dense, saturated)
    create_config('configs/sparsity_cnn_tanh.json', 'cnn_mnist_tanh', lr=0.15, is_snn=False)
    # 3. SNN Tri (Sparse, saturated)
    create_config('configs/sparsity_snn_tri.json', 'cnn_mnist_snn', lr=0.005, is_snn=True, surrogate='tri', surrogate_param_name='beta', surrogate_param_val=2.0)
    
    print("Running CNN ReLU...")
    run_benchmark('configs/sparsity_cnn.json', nb_jobs=5, distribute_gpus=True)
    print("Running CNN Tanh...")
    run_benchmark('configs/sparsity_cnn_tanh.json', nb_jobs=5, distribute_gpus=True)
    print("Running SNN Tri...")
    run_benchmark('configs/sparsity_snn_tri.json', nb_jobs=5, distribute_gpus=True)
    
    
def get_avg_cos_sim(model_name):
    base_dir = f"results/sparsity_experiment/{model_name}/mnist_direct/SignFlipping_Average_f_0_gamma_similarity_niid_0.33"
    if not os.path.isdir(base_dir):
        return None
    
    cos_sims = []
    # Find seed folders
    for f in os.listdir(base_dir):
        if "seed" in f:
            seed_dir = os.path.join(base_dir, f)
            cos_file = os.path.join(seed_dir, "honest_mean_cos_sim_tr_seed_42_dd_seed_42.txt")
            if not os.path.exists(cos_file):
                # Check other tr_seed
                for seed_file in os.listdir(seed_dir):
                    if seed_file.startswith("honest_mean_cos_sim"):
                        cos_file = os.path.join(seed_dir, seed_file)
                        break
                        
            if os.path.exists(cos_file):
                try:
                    data = np.loadtxt(cos_file)
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
        plt.plot(np.arange(len(cnn_cos)) * 10, cnn_cos, label="CNN ReLU (Dense)", linewidth=2)
    if tanh_cos is not None:
        plt.plot(np.arange(len(tanh_cos)) * 10, tanh_cos, label="CNN Tanh (Dense)", linewidth=2)
    if snn_cos is not None:
        plt.plot(np.arange(len(snn_cos)) * 10, snn_cos, label="SNN Tri (Sparse)", linewidth=2)
        
    plt.xlabel('Communication Rounds', fontsize=14)
    plt.ylabel('Mean Cosine Similarity', fontsize=14)
    plt.title('Honest Clients Gradient Alignment ($f=0$, $\gamma=0.33$)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('plots/sparsity_experiment/honest_gradient_alignment.pdf')
    plt.savefig('plots/sparsity_experiment/honest_gradient_alignment.png')
    print("Plot saved to plots/sparsity_experiment/honest_gradient_alignment.pdf")


if __name__ == "__main__":
    run_experiments()
    plot_results()
