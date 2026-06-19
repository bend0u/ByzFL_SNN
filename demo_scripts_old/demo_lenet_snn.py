import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import snntorch.functional as sf
from byzfl import Client, Server, DataDistributor, ByzantineClient
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
        # Repeat the static image across the time steps
        temporal_img = img.unsqueeze(0).repeat(self.time_steps, 1, 1, 1)
        return temporal_img, target

TIME_STEPS = 25

print("Downloading and preparing real MNIST dataset...")
os.makedirs('./data', exist_ok=True)
raw_mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
raw_mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

train_dataset = TemporalMNIST(raw_mnist_train, time_steps=TIME_STEPS)
test_dataset = TemporalMNIST(raw_mnist_test, time_steps=TIME_STEPS)

# Configurations (17 honest clients, 1 Byzantine client, iid data distribution)
nb_honest_clients = 17
nb_byz_clients = 5
attack_name = "InnerProductManipulation"  # Choices: "SignFlipping", "InnerProductManipulation", "ALittleIsEnough", "NoAttack"

batch_size = 128  # local batch size for clients
dataloader_batch_size = 256  # batch size for dataloader

# Calculate training steps dynamically based on Epochs
NUM_EPOCHS = 5
dataset_length = len(train_dataset)
samples_per_client = dataset_length / nb_honest_clients
steps_per_epoch = int(samples_per_client / batch_size)
nb_training_steps = NUM_EPOCHS * steps_per_epoch

print(f"Dataset: MNIST ({dataset_length} train samples, {len(test_dataset)} test samples)")
print(f"Training for {NUM_EPOCHS} epochs ({nb_training_steps} total communication steps)...")

# 3. Create DataLoaders
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

# LeNet-5 SNN Model parameters
model_params = {
    "in_channels": 1,
    "input_height": 28,
    "input_width": 28,
    "output_dim": 10,
    "beta": 0.95,
    "surrogate_gradient": "atan",
}

# 5. Initialize Server, Clients, and Byzantine Client
print("Initializing 17 Client instances with 'lenet_snn' model...")
honest_clients = [
    Client({
        "model_name": "lenet_snn",
        "model_params": model_params,
        "device": device,
        "loss_name": "ce_rate_loss",
        "loss_params": {},
        "accuracy_name": "accuracy_rate",
        "LabelFlipping": False,
        "training_dataloader": client_dataloaders[i],
        "momentum": 0.9,
        "nb_labels": 10,
        "store_per_client_metrics": False,
    }) for i in range(nb_honest_clients)
]

# Instantiate Byzantine client dynamically
if nb_byz_clients > 0:
    print(f"Initializing {nb_byz_clients} Byzantine Client instance(s) with '{attack_name}' attack...")
    byz_worker = ByzantineClient({
        "name": attack_name,
        "f": nb_byz_clients,
        "parameters": {"tau": 3.0}, # Used if attack_name is InnerProductManipulation or ALittleIsEnough
    })
else:
    print("No Byzantine clients configured.")
    byz_worker = ByzantineClient({
        "name": "NoAttack",
        "f": 0,
        "parameters": {},
    })

# Select aggregator dynamically based on presence of Byzantine clients
if nb_byz_clients == 0:
    aggregator_info = {"name": "Average", "parameters": {}}
    print("Initializing Server instance with Average (FedAvg) aggregator...")
else:
    aggregator_info = {"name": "TrMean", "parameters": {"f": nb_byz_clients}}
    print(f"Initializing Server instance with robust TrMean (f={nb_byz_clients}) aggregator...")

server = Server({
    "device": device,
    "model_name": "lenet_snn",
    "model_params": model_params,
    "accuracy_name": "accuracy_rate",
    "test_loader": test_loader,
    "optimizer_name": "SGD",
    "learning_rate": 0.05,
    "weight_decay": 0.0001,
    "milestones": [int(NUM_EPOCHS * 0.4) * steps_per_epoch, int(NUM_EPOCHS * 0.8) * steps_per_epoch],
    "learning_rate_decay": 0.5,
    "aggregator_info": aggregator_info,
    "pre_agg_list": []
})

step_list = []
loss_history = []
acc_history = []

print("\nStarting federated training loop...")
# 6. Training loop
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

    # Aggregate (robustly) and Update
    honest_gradients = [client.get_flat_gradients_with_momentum() for client in honest_clients]
    byzantine_gradients = byz_worker.apply_attack(honest_gradients)
    all_gradients = honest_gradients + byzantine_gradients
    server.update_model_with_gradients(all_gradients)

    # Evaluate
    test_acc = server.compute_test_accuracy()
    if step % 5 == 0 or step == 1 or step == nb_training_steps:
        print(f"Step {step:02d}/{nb_training_steps} | Avg Loss: {mean_loss:.4f} | Test Acc: {test_acc:.4f}")

    step_list.append(step)
    loss_history.append(mean_loss)
    acc_history.append(test_acc)

print("\nTraining completed successfully!")

# 7. Plot Results
print("Generating evaluation plots...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Construct dynamic titles and filenames based on configuration
if nb_byz_clients == 0:
    setup_suffix = "No Attack"
    file_suffix = "no_attack"
else:
    setup_suffix = f"f={nb_byz_clients}, {attack_name}"
    file_suffix = f"f{nb_byz_clients}_{attack_name.lower()}"

ax1.plot(step_list, loss_history, color="#3b82f6", linewidth=2, label="CE Rate Loss")
ax1.set_title(f"Spiking LeNet-5 ({setup_suffix}) - Average Training Loss", fontsize=13, fontweight='bold')
ax1.set_xlabel("Training Step", fontsize=11)
ax1.set_ylabel("Loss", fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(fontsize=9, loc="upper right")

ax2.plot(step_list, acc_history, color="#10b981", linewidth=2, label="Accuracy Rate")
ax2.set_title(f"Spiking LeNet-5 ({setup_suffix}) - Test Accuracy", fontsize=13, fontweight='bold')
ax2.set_xlabel("Training Step", fontsize=11)
ax2.set_ylabel("Accuracy", fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(fontsize=9, loc="lower right")

plt.tight_layout()
plot_dir = "experiments"
os.makedirs(plot_dir, exist_ok=True)
plot_path = os.path.join(plot_dir, f"spiking_lenet5_{file_suffix}.png")
plt.savefig(plot_path, dpi=150)
print(f"Plot saved successfully as '{os.path.abspath(plot_path)}'")
