"""Sanity check: does the threshold ladder theta in {1.0, 0.8, 0.6, 0.4}
actually space the firing probability p before GPU is spent on the full
threshold sweep?

For each theta: if a checkpoint exists under ./results matching that
threshold, load it and measure firing rate on one real MNIST batch. Since
every sweep config in this repo has store_models=false, this will normally
find nothing and fall back to a fresh, tiny f=0 warm-start (default 50 local
SGD steps, 4 clients) instead. Either way, prints the per-layer mean firing
rate.

Usage (run as a module from the repo root, so `byzfl` resolves):
    python -m scripts.theta_p_check [--device cuda] [--warmup_steps 50] [--nb_clients 4]
"""
import argparse
import glob
import os
from collections import defaultdict

import numpy as np
import torch

from byzfl import Client, DataDistributor
from byzfl.benchmark.data import load_and_split_data
from byzfl.benchmark.managers import ParamsManager
from byzfl.utils.misc import set_random_seed

THRESHOLDS = [1.0, 0.8, 0.6, 0.4]


def find_checkpoint(results_root, threshold):
    """Look for an existing store_models=true checkpoint matching this threshold."""
    pattern = os.path.join(
        results_root, "**", f"*threshold_{threshold}*", "models_tr_seed_*", "model_step_*.pth"
    )
    matches = sorted(glob.glob(pattern, recursive=True))
    return matches[-1] if matches else None


def build_mini_config(threshold, device, nb_steps, nb_clients):
    return {
        "benchmark_config": {
            "device": device,
            "training_seed": 42,
            "nb_honest_clients": nb_clients,
            "f": 0,
            "size_train_set": 0.8,
            "data_distribution_seed": 42,
            "data_distribution": {"name": "gamma_similarity_niid", "distribution_parameter": 1.0},
            "nb_steps": nb_steps,
        },
        "model": {
            "name": "cnn_mnist_snn",
            "is_snn": True,
            "dataset_name": "mnist",
            "nb_labels": 10,
            "loss": "ce_rate_loss",
            "accuracy_name": "accuracy_rate",
            "optimizer_name": "SGD",
            "encoding": {"type": "constant", "time_steps": 10},
            "learning_rate": 0.10,
            "learning_rate_decay": 1.0,
            "milestones": [],
            "model_params": {
                "beta": 0.95,
                "surrogate_gradient": "atan",
                "surrogate_params": {"alpha": 1.2},
                "threshold": threshold,
                "learn_threshold": False,
            },
        },
        "honest_clients": {"momentum": 0.9, "weight_decay": 0.0001, "batch_size": 128},
        "evaluation_and_results": {
            "batch_size_evaluation": 128,
            "data_folder": "./data",
        },
    }


def build_pm_and_data(threshold, device, nb_steps, nb_clients):
    params = build_mini_config(threshold, device, nb_steps, nb_clients)
    pm = ParamsManager(params)
    set_random_seed(pm.get_data_distribution_seed())
    train_dataset, _, _ = load_and_split_data(pm)
    return pm, train_dataset


def make_clients(pm, train_dataset, nb_clients):
    data_distributor = DataDistributor({
        "data_distribution_name": pm.get_name_data_distribution(),
        "distribution_parameter": pm.get_parameter_data_distribution(),
        "nb_honest": nb_clients,
        "data_loader": train_dataset,
        "batch_size": pm.get_honest_clients_batch_size(),
    })
    client_dataloaders = data_distributor.split_data()

    model_params = pm.get_model_params().copy()
    model_params["time_steps"] = pm.get_time_steps()

    return [
        Client({
            "model_name": pm.get_model_name(),
            "model_params": model_params,
            "device": pm.get_device(),
            "optimizer_name": pm.get_optimizer_name(),
            "learning_rate": pm.get_learning_rate(),
            "loss_name": pm.get_loss_name(),
            "loss_params": pm.get_loss_params(),
            "accuracy_name": pm.get_accuracy_name(),
            "weight_decay": pm.get_honest_clients_weight_decay(),
            "milestones": pm.get_milestones(),
            "learning_rate_decay": pm.get_learning_rate_decay(),
            "LabelFlipping": False,
            "training_dataloader": client_dataloaders[i],
            "momentum": pm.get_honest_clients_momentum(),
            "nb_labels": pm.get_nb_labels(),
            "store_per_client_metrics": True,
            "gradient_clip_val": 0.0,
        }) for i in range(nb_clients)
    ]


def measure_from_checkpoint(threshold, device, ckpt_path):
    """Load a checkpoint's weights into a fresh model of matching threshold
    and measure firing rate on one real MNIST batch (no training)."""
    pm, train_dataset = build_pm_and_data(threshold, device, nb_steps=1, nb_clients=1)
    client = make_clients(pm, train_dataset, nb_clients=1)[0]
    state_dict = torch.load(ckpt_path, map_location=device)
    client.set_model_state(state_dict)
    client.compute_gradients()  # one forward+backward on a real batch; weights untouched (no optimizer.step())
    return client.get_last_layer_firing_rates()


def measure_from_warmup(threshold, device, warmup_steps, nb_clients):
    """Fresh f=0 warm-start: each client trains locally for warmup_steps with
    plain SGD, then per-layer firing rate is averaged across clients."""
    pm, train_dataset = build_pm_and_data(threshold, device, warmup_steps, nb_clients)
    clients = make_clients(pm, train_dataset, nb_clients)

    set_random_seed(pm.get_training_seed())
    for c in clients:
        c.compute_model_update(warmup_steps)

    layer_rates = defaultdict(list)
    for c in clients:
        for layer_name, rate in c.get_last_layer_firing_rates().items():
            layer_rates[layer_name].append(rate)
    return {name: float(np.mean(vals)) for name, vals in layer_rates.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--nb_clients", type=int, default=4, help="Clients for the fresh mini warm-start (kept small for speed)")
    parser.add_argument("--results_root", type=str, default="./results")
    args = parser.parse_args()

    print(f"Device: {args.device} | warmup_steps: {args.warmup_steps} | nb_clients: {args.nb_clients}\n")

    for threshold in THRESHOLDS:
        ckpt = find_checkpoint(args.results_root, threshold)
        if ckpt is not None:
            print(f"theta={threshold}: found checkpoint {ckpt}, measuring on a real batch (no training)...")
            layer_rates = measure_from_checkpoint(threshold, args.device, ckpt)
        else:
            print(f"theta={threshold}: no checkpoint found, running {args.warmup_steps}-step f=0 warm-start "
                  f"({args.nb_clients} clients)...")
            layer_rates = measure_from_warmup(threshold, args.device, args.warmup_steps, args.nb_clients)

        overall = float(np.mean(list(layer_rates.values()))) if layer_rates else float("nan")
        per_layer_str = ", ".join(f"{name}={rate:.4f}" for name, rate in sorted(layer_rates.items()))
        print(f"  theta={threshold}: overall p={overall:.4f} | {per_layer_str}\n")


if __name__ == "__main__":
    main()
