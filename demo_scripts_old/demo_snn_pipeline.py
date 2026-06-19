import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import copy
import re
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from snntorch import spikegen
import snntorch.functional as sf
from byzfl import Client, Server, ByzantineClient, DataDistributor
from byzfl.utils.misc import set_random_seed
import matplotlib.pyplot as plt
import tonic
import tonic.transforms as tonic_transforms

# 1. Set seed for reproducibility
SEED = 42
set_random_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Helper to recursively merge dictionaries for overrides
def merge_configs(base, overrides):
    merged = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
            merged[k] = merge_configs(merged[k], v)
        else:
            merged[k] = copy.deepcopy(v)
    return merged

# Helper to sanitize string into a safe filename
def sanitize_filename(name):
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', name)

# 2. Load Configuration from JSON
config_path = "snn_config.json"
if os.path.exists(config_path):
    print(f"Loading SNN config from '{config_path}'...")
    with open(config_path, "r") as f:
        config = json.load(f)
else:
    print(f"Config file '{config_path}' not found. Using default configurations...")
    config = {
        "data": {
            "dataset_name": "mnist",
            "time_steps": 25,
            "nb_honest_clients": 17,
            "nb_byz_clients": 0,
            "data_distribution_name": "iid",
            "distribution_parameter": 0.5,
            "batch_size": 128,
            "dataloader_batch_size": 256
        },
        "model": {
            "hidden_dim": 256,
            "beta": 0.95,
            "surrogate_gradient": "atan"
        },
        "training": {
            "loss_name": "ce_rate_loss",
            "loss_params": {},
            "accuracy_name": "accuracy_rate",
            "num_epochs": 5,
            "learning_rate": 0.05,
            "weight_decay": 0.0001,
            "momentum": 0.9,
            "learning_rate_decay": 0.5,
            "byzantine_attack_name": "InnerProductManipulation",
            "byzantine_attack_tau": 3.0,
            "aggregator_name": "TrMean"
        },
        "plot": {
            "output_directory": "experiments",
            "output_filename": "snn_hyperparameters_exploration.png"
        },
        "experiments": []
    }

# 3. Define a temporal wrapper for static MNIST
class TemporalMNIST(torch.utils.data.Dataset):
    def __init__(self, mnist_dataset, time_steps=25, encoding_type="constant", encoding_params=None):
        self.mnist_dataset = mnist_dataset
        self.time_steps = time_steps
        self.targets = mnist_dataset.targets
        self.encoding_type = encoding_type.lower()
        self.encoding_params = encoding_params if encoding_params is not None else {}

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        img, target = self.mnist_dataset[idx]
        
        if self.encoding_type == "rate":
            # Poisson rate coding (spikes are proportional to pixel intensity)
            temporal_img = spikegen.rate(img, num_steps=self.time_steps, **self.encoding_params)
        elif self.encoding_type == "latency":
            # Latency coding (brighter pixels fire spikes earlier)
            temporal_img = spikegen.latency(img, num_steps=self.time_steps, **self.encoding_params)
        elif self.encoding_type == "delta":
            # Delta modulation
            temporal_img = spikegen.delta(img, **self.encoding_params)
        else:
            # Constant current coding (original pixel intensities repeated at each step)
            temporal_img = img.unsqueeze(0).repeat(self.time_steps, 1, 1, 1)
            
        return temporal_img, target

# Define a wrapper for Tonic Neuromorphic datasets (like NMNIST)
class TonicDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, tonic_dataset):
        self.tonic_dataset = tonic_dataset
        self.targets = torch.tensor(tonic_dataset.targets)

    def __len__(self):
        return len(self.tonic_dataset)

    def __getitem__(self, idx):
        x, y = self.tonic_dataset[idx]
        # x is a numpy array of shape (time_steps, 2, 34, 34). We convert to float32 tensor
        return torch.tensor(x, dtype=torch.float32), y

# Preload raw MNIST datasets once if needed by any config
print("Preloading raw MNIST datasets...")
os.makedirs('./data', exist_ok=True)
raw_mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
raw_mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

# Extract experiments sweep
experiments = config.get("experiments", [])
if not experiments:
    experiments = [
        {
            "name": "Default Baseline: Constant Coding + CE Rate Loss",
            "group": "General",
            "encoding_type": "constant"
        }
    ]

# Setup output directories & results store
results = {}
plot_dir = config.get("plot", {}).get("output_directory", "experiments")
os.makedirs(plot_dir, exist_ok=True)

summary_file_path = os.path.join(plot_dir, "snn_experiments_summary.txt")

# Initialize summary file with header
with open(summary_file_path, "w") as f:
    f.write("==================================================\n")
    f.write("SNN Federated Learning Experiment Sweep Summary\n")
    f.write("==================================================\n\n")

# Color palette and markers for plotting
colors = ["#3b82f6", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6", "#ef4444", "#06b6d4", "#14b8a6", "#f43f5e", "#6366f1"]
markers = ["o", "s", "^", "D", "v", "<", ">", "p", "P", "*"]

for idx, exp in enumerate(experiments):
    exp_name = exp.get("name", f"Experiment {idx+1}")
    exp_group = exp.get("group", "General")
    
    print(f"\n==================================================")
    print(f"Running Experiment {idx+1}/{len(experiments)}: {exp_name}")
    print(f"Group: {exp_group}")
    print(f"==================================================")
    
    # 1. Merge configurations (Experiment Overrides base config)
    exp_config = merge_configs(config, exp)
    
    # Resolve all dynamic hyper-parameters for this run
    dataset_name = exp_config["data"].get("dataset_name", "mnist").lower()
    TIME_STEPS = exp_config["data"].get("time_steps", 25)
    nb_honest_clients = exp_config["data"].get("nb_honest_clients", 17)
    nb_byz_clients = exp_config["data"].get("nb_byz_clients", 0)
    batch_size = exp_config["data"].get("batch_size", 128)
    dataloader_batch_size = exp_config["data"].get("dataloader_batch_size", 256)
    data_distribution_name = exp_config["data"].get("data_distribution_name", "iid")
    distribution_parameter = exp_config["data"].get("distribution_parameter", 0.5)
    
    encoding_type = exp.get("encoding_type", "constant").lower()
    encoding_params = exp.get("encoding_params", {})
    
    hidden_dim = exp_config["model"].get("hidden_dim", 256)
    beta = exp_config["model"].get("beta", 0.95)
    surrogate_gradient = exp_config["model"].get("surrogate_gradient", "atan")
    
    NUM_EPOCHS = exp_config["training"].get("num_epochs", 5)
    learning_rate = exp_config["training"].get("learning_rate", 0.05)
    weight_decay = exp_config["training"].get("weight_decay", 0.0001)
    momentum = exp_config["training"].get("momentum", 0.9)
    learning_rate_decay = exp_config["training"].get("learning_rate_decay", 0.5)
    byzantine_attack_name = exp_config["training"].get("byzantine_attack_name", "InnerProductManipulation")
    byzantine_attack_tau = exp_config["training"].get("byzantine_attack_tau", 3.0)
    aggregator_name = exp_config["training"].get("aggregator_name", "TrMean")
    loss_name = exp_config["training"].get("loss_name", "ce_rate_loss")
    loss_params = exp_config["training"].get("loss_params", {})
    accuracy_name = exp_config["training"].get("accuracy_name", "accuracy_rate")

    # 2. Instantiate datasets based on chosen dataset name & temporal encoding overrides
    if dataset_name == "nmnist":
        print("Instantiating neuromorphic N-MNIST dataset...")
        time_window = int(300000 / TIME_STEPS)
        sensor_size = tonic.datasets.NMNIST.sensor_size
        frame_transform = tonic_transforms.Compose([
            tonic_transforms.ToFrame(sensor_size=sensor_size, time_window=time_window),
            tonic_transforms.Denoise(filter_time=time_window),
        ])
        raw_nmnist_train = tonic.datasets.NMNIST(save_to='./data', train=True, transform=frame_transform)
        raw_nmnist_test = tonic.datasets.NMNIST(save_to='./data', train=False, transform=frame_transform)
        input_dim = 2312  # 34 x 34 x 2 polarities
        train_dataset = TonicDatasetWrapper(raw_nmnist_train)
        test_dataset = TonicDatasetWrapper(raw_nmnist_test)
    else:
        input_dim = 784
        train_dataset = TemporalMNIST(raw_mnist_train, time_steps=TIME_STEPS, encoding_type=encoding_type, encoding_params=encoding_params)
        test_dataset = TemporalMNIST(raw_mnist_test, time_steps=TIME_STEPS, encoding_type=encoding_type, encoding_params=encoding_params)

    train_loader = DataLoader(train_dataset, batch_size=dataloader_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=dataloader_batch_size, shuffle=False)

    # 3. Partition data using DataDistributor
    print(f"Partitioning data among honest clients using distribution: '{data_distribution_name}'...")
    data_distributor = DataDistributor({
        "data_distribution_name": data_distribution_name,
        "distribution_parameter": distribution_parameter,
        "nb_honest": nb_honest_clients,
        "data_loader": train_loader,
        "batch_size": batch_size,
    })
    client_dataloaders = data_distributor.split_data()

    # Calculate steps dynamically
    dataset_length = len(train_dataset)
    samples_per_client = dataset_length / nb_honest_clients
    steps_per_epoch = int(samples_per_client / batch_size)
    nb_training_steps = NUM_EPOCHS * steps_per_epoch

    print(f"Steps per epoch: {steps_per_epoch} | Total communication steps: {nb_training_steps}")

    # 4. Construct SNN model params
    model_params = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "output_dim": 10,
        "beta": beta,
        "surrogate_gradient": surrogate_gradient,
    }

    # Initialize Honest Clients
    honest_clients = [
        Client({
            "model_name": "fc_snn",
            "model_params": model_params,
            "device": device,
            "loss_name": loss_name,
            "loss_params": loss_params,
            "accuracy_name": accuracy_name,
            "LabelFlipping": False,
            "training_dataloader": client_dataloaders[i],
            "momentum": momentum,
            "nb_labels": 10,
            "store_per_client_metrics": False,
        }) for i in range(nb_honest_clients)
    ]

    # Initialize Server
    server = Server({
        "device": device,
        "model_name": "fc_snn",
        "model_params": model_params,
        "accuracy_name": accuracy_name,
        "test_loader": test_loader,
        "optimizer_name": "SGD",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "milestones": [int(NUM_EPOCHS * 0.4) * steps_per_epoch, int(NUM_EPOCHS * 0.8) * steps_per_epoch],
        "learning_rate_decay": learning_rate_decay,
        "aggregator_info": {"name": aggregator_name, "parameters": {"f": nb_byz_clients}},
        "pre_agg_list": []
    })

    # Initialize Byzantine Attacker
    attack = {
        "name": byzantine_attack_name,
        "f": nb_byz_clients,
        "parameters": {"tau": byzantine_attack_tau},
    }
    byz_client = ByzantineClient(attack)

    step_list = []
    loss_history = []
    acc_history = []

    # 5. Training loop
    for step in range(1, nb_training_steps + 1):
        # Synchronize parameters from server to clients
        server_model_state = server.get_dict_parameters()
        for client in honest_clients:
            client.set_model_state(server_model_state)

        # Clients compute local updates
        losses = []
        for client in honest_clients:
            loss = client.compute_gradients()
            losses.append(loss)
        mean_loss = sum(losses) / len(losses)

        # Aggregate and Update server weights
        honest_gradients = [client.get_flat_gradients_with_momentum() for client in honest_clients]
        byz_vector = byz_client.apply_attack(honest_gradients)
        server.update_model_with_gradients(honest_gradients + byz_vector)

        # Evaluate model accuracy on server
        test_acc = server.compute_test_accuracy()
        if step % 5 == 0 or step == 1 or step == nb_training_steps:
            print(f"Step {step:02d}/{nb_training_steps} | Avg Loss: {mean_loss:.4f} | Test Acc: {test_acc:.4f}")

        step_list.append(step)
        loss_history.append(mean_loss)
        acc_history.append(test_acc)

    # 6. Store and save results
    results[exp_name] = {
        "steps": step_list,
        "losses": loss_history,
        "accuracies": acc_history,
        "group": exp_group,
        "final_loss": loss_history[-1],
        "final_accuracy": acc_history[-1]
    }

    # Write summary entry to file immediately (progressive logging)
    with open(summary_file_path, "a") as f:
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Group: {exp_group}\n")
        f.write(f"Configuration:\n")
        f.write(f"  - Dataset: {dataset_name}\n")
        f.write(f"  - Time Steps: {TIME_STEPS}\n")
        f.write(f"  - Hidden Layer Dim: {hidden_dim}\n")
        f.write(f"  - Beta (Decay): {beta}\n")
        f.write(f"  - Data Encoding: {encoding_type}\n")
        if encoding_params:
            f.write(f"  - Encoding Params: {encoding_params}\n")
        f.write(f"  - Loss Name: {loss_name}\n")
        f.write(f"  - Accuracy Name: {accuracy_name}\n")
        f.write(f"  - Learning Rate: {learning_rate}\n")
        f.write(f"  - Epochs: {NUM_EPOCHS}\n")
        f.write(f"Results:\n")
        f.write(f"  - Final Avg Loss: {loss_history[-1]:.4f}\n")
        f.write(f"  - Final Test Accuracy: {acc_history[-1]:.4f}\n")
        f.write("-" * 50 + "\n\n")

    # PROGRESSIVE PLOT 1: Single plot for this experiment
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    ax1.plot(step_list, loss_history, color="#3b82f6", linewidth=2.5)
    ax1.set_title(f"Loss - {exp_name}", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2.plot(step_list, acc_history, color="#10b981", linewidth=2.5)
    ax2.set_title(f"Test Accuracy - {exp_name}", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    single_plot_dir = os.path.join(plot_dir, "single")
    os.makedirs(single_plot_dir, exist_ok=True)
    single_plot_name = f"single_{sanitize_filename(exp_name)}.png"
    plt.savefig(os.path.join(single_plot_dir, single_plot_name), dpi=120)
    plt.close()
    print(f"Saved progressive single plot to: {os.path.join(single_plot_dir, single_plot_name)}")

    # PROGRESSIVE PLOT 2: Group Plot Comparison
    group_exps = {name: data for name, data in results.items() if data["group"] == exp_group}
    if len(group_exps) >= 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        for i, (g_name, g_data) in enumerate(group_exps.items()):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            markevery = max(1, len(g_data["steps"]) // 10)
            ax1.plot(g_data["steps"], g_data["losses"], label=g_name, color=color, marker=marker, markevery=markevery, linewidth=2)
            ax2.plot(g_data["steps"], g_data["accuracies"], label=g_name, color=color, marker=marker, markevery=markevery, linewidth=2)
            
        ax1.set_title(f"Loss comparison - {exp_group}", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Training Step")
        ax1.set_ylabel("Loss")
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(fontsize=8, loc="upper right")
        
        ax2.set_title(f"Accuracy comparison - {exp_group}", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Training Step")
        ax2.set_ylabel("Accuracy")
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(fontsize=8, loc="lower right")
        
        plt.tight_layout()
        group_plot_dir = os.path.join(plot_dir, "groups")
        os.makedirs(group_plot_dir, exist_ok=True)
        group_plot_name = f"group_{sanitize_filename(exp_group)}.png"
        plt.savefig(os.path.join(group_plot_dir, group_plot_name), dpi=120)
        plt.close()
        print(f"Updated group comparison plot to: {os.path.join(group_plot_dir, group_plot_name)}")

    # PROGRESSIVE PLOT 3: Master Plot Comparison (all experiments completed so far)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    for i, (m_name, m_data) in enumerate(results.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        markevery = max(1, len(m_data["steps"]) // 10)
        ax1.plot(m_data["steps"], m_data["losses"], label=m_name, color=color, marker=marker, markevery=markevery, linewidth=2)
        ax2.plot(m_data["steps"], m_data["accuracies"], label=m_name, color=color, marker=marker, markevery=markevery, linewidth=2)
        
    ax1.set_title("Master Average Honest Client Loss Comparison (All Exps)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=8, loc="upper right")
    
    ax2.set_title("Master Global Server Test Accuracy Comparison (All Exps)", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=8, loc="lower right")
    
    plt.tight_layout()
    master_plot_path = os.path.join(plot_dir, config.get("plot", {}).get("output_filename", "snn_hyperparameters_exploration.png"))
    plt.savefig(master_plot_path, dpi=150)
    plt.close()
    print(f"Updated master comparison plot to: {master_plot_path}")

print(f"\nAll experiments finished successfully!")
print(f"Summary log file located at: {os.path.abspath(summary_file_path)}")
print(f"Plot comparisons located under: {os.path.abspath(plot_dir)}")
