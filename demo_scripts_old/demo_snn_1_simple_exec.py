import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import snntorch.functional as sf
from byzfl import Client, Server, ByzantineClient, DataDistributor
from byzfl.utils.misc import set_random_seed
import matplotlib.pyplot as plt

# 1. Set seed for reproducibility
SEED = 42
set_random_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2. Define a temporal wrapper for static MNIST
class TemporalMNIST(torch.utils.data.Dataset):
    def __init__(self, mnist_dataset, time_steps=25):
        self.mnist_dataset = mnist_dataset
        self.time_steps = time_steps
        self.targets = mnist_dataset.targets

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        img, target = self.mnist_dataset[idx]
        temporal_img = img.unsqueeze(0).repeat(self.time_steps, 1, 1, 1)
        return temporal_img, target

TIME_STEPS = 25

print("Downloading and preparing real MNIST dataset...")
os.makedirs('./data', exist_ok=True)
raw_mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
raw_mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

train_dataset = TemporalMNIST(raw_mnist_train, time_steps=TIME_STEPS)
test_dataset = TemporalMNIST(raw_mnist_test, time_steps=TIME_STEPS)

# Configurations
nb_honest_clients = 17
nb_byz_clients = 0
batch_size = 128  # local batch size for clients
dataloader_batch_size = 256  # dataloader batch size

# Calculate training steps dynamically based on Epochs
NUM_EPOCHS = 5
samples_per_client = len(train_dataset) / nb_honest_clients
steps_per_epoch = int(samples_per_client / batch_size)  # split based on client batch size
nb_training_steps = NUM_EPOCHS * steps_per_epoch

print(f"Dataset: Full MNIST ({len(train_dataset)} train samples, {len(test_dataset)} test samples)")
print(f"Training for {NUM_EPOCHS} epochs ({nb_training_steps} total communication steps)...")

# 3. Create DataLoaders on full dataset
train_loader = DataLoader(train_dataset, batch_size=dataloader_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=dataloader_batch_size, shuffle=False)

# 4. Partition data
print("Partitioning temporal data among honest clients using IID distribution...")
data_distributor = DataDistributor({
    "data_distribution_name": "iid",
    "distribution_parameter": 0.5,
    "nb_honest": nb_honest_clients,
    "data_loader": train_loader,
    "batch_size": batch_size,
})
client_dataloaders = data_distributor.split_data()

# SNN Model base parameters
model_params = {
    "input_dim": 784,
    "hidden_dim": 512,
    "output_dim": 10,
    "beta": 0.95,
    "surrogate_gradient": "atan",
}

# Define the experiments to run
experiments = [
    {
        "name": "MSE Count Loss + Rate Acc",
        "loss_name": "mse_count_loss",
        "loss_params": {"correct_rate": 1.0, "incorrect_rate": 0.0},
        "accuracy_name": "accuracy_rate",
    }
]

# Dictionary to store results of all experiments
results = {}

for exp in experiments:
    exp_name = exp["name"]
    print(f"\n==================================================")
    print(f"Running Experiment: {exp_name}")
    print(f"==================================================")

    # Initialize Honest Clients
    honest_clients = [
        Client({
            "model_name": "fc_snn",
            "model_params": model_params,
            "device": device,
            "loss_name": exp["loss_name"],
            "loss_params": exp["loss_params"],
            "accuracy_name": exp["accuracy_name"],
            "LabelFlipping": False,
            "training_dataloader": client_dataloaders[i],
            "momentum": 0.9,
            "nb_labels": 10,
            "store_per_client_metrics": False,
        }) for i in range(nb_honest_clients)
    ]

    # Initialize Server
    server = Server({
        "device": device,
        "model_name": "fc_snn",
        "model_params": model_params,
        "accuracy_name": exp["accuracy_name"],
        "test_loader": test_loader,
        "optimizer_name": "SGD",
        "learning_rate": 0.05,  # Robust learning rate for stable convergence on full dataset
        "weight_decay": 0.0001,
        "milestones": [4 * steps_per_epoch, 8 * steps_per_epoch],
        "learning_rate_decay": 0.5,
        "aggregator_info": {"name": "Average", "parameters": {}},
        "pre_agg_list": []
    })

    # Initialize Byzantine Client
    attack = {
        "name": "InnerProductManipulation",
        "f": nb_byz_clients,
        "parameters": {"tau": 3.0},
    }
    byz_client = ByzantineClient(attack)

    step_list = []
    loss_history = []
    acc_history = []

    # Training loop
    for step in range(1, nb_training_steps + 1):
        # Sync states
        server_model_state = server.get_dict_parameters()
        for client in honest_clients:
            client.set_model_state(server_model_state)

        # Local computations
        losses = []
        for client in honest_clients:
            loss = client.compute_gradients()
            losses.append(loss)
        mean_loss = sum(losses) / len(losses)

        # Aggregate and Update
        honest_gradients = [client.get_flat_gradients_with_momentum() for client in honest_clients]
        byz_vector = byz_client.apply_attack(honest_gradients)
        server.update_model_with_gradients(honest_gradients + byz_vector)

        # Evaluate
        test_acc = server.compute_test_accuracy()
        if step % 5 == 0 or step == 1 or step == nb_training_steps:
            print(f"Step {step:02d}/{nb_training_steps} | Avg Loss: {mean_loss:.4f} | Test Acc: {test_acc:.4f}")

        step_list.append(step)
        loss_history.append(mean_loss)
        acc_history.append(test_acc)

    results[exp_name] = {
        "steps": step_list,
        "losses": loss_history,
        "accuracies": acc_history,
    }

print("\nAll experiments completed successfully!")

# 9. Plot Comparative Results
print("Generating comparative evaluation plots...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

colors = ["#3b82f6", "#10b981", "#f59e0b", "#ec4899"]
markers = ["o", "s", "^", "D"]

for i, (name, data) in enumerate(results.items()):
    ax1.plot(data["steps"], data["losses"], label=name, color=colors[i], marker=markers[i], markevery=3, linewidth=2)
    ax2.plot(data["steps"], data["accuracies"], label=name, color=colors[i], marker=markers[i], markevery=3, linewidth=2)

# Format loss plot
ax1.set_title("Average Honest Client Loss comparison", fontsize=13, fontweight='bold')
ax1.set_xlabel("Training Step", fontsize=11)
ax1.set_ylabel("Loss", fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(fontsize=9, loc="upper right")

# Format accuracy plot
ax2.set_title("Global Server Test Accuracy comparison", fontsize=13, fontweight='bold')
ax2.set_xlabel("Training Step", fontsize=11)
ax2.set_ylabel("Accuracy", fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(fontsize=9, loc="lower right")

plt.tight_layout()
plot_dir = "experiments"
os.makedirs(plot_dir, exist_ok=True)
plot_path = os.path.join(plot_dir, "snn_experiments_comparison_full_mnist.png")
plt.savefig(plot_path, dpi=150)
print(f"\nComparative plots saved successfully as '{os.path.abspath(plot_path)}'")