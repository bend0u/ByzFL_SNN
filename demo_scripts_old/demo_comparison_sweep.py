import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import snntorch.functional as sf
from byzfl import Client, Server, DataDistributor
from byzfl.utils.misc import set_random_seed
import matplotlib.pyplot as plt

# Set seed for reproducibility
SEED = 42
set_random_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Define a temporal wrapper for static MNIST
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
mnist_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.0,), (1.0,))
])

raw_mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transform)
raw_mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=mnist_transform)

train_dataset = TemporalMNIST(raw_mnist_train, time_steps=TIME_STEPS)
test_dataset = TemporalMNIST(raw_mnist_test, time_steps=TIME_STEPS)

# Configurations
batch_size = 128  # Power of 2 (divisible cleanly by 1, 8, and 16)
dataloader_batch_size = batch_size  # Set to be equal to training batch_size for simplicity

# Equivalent of 4 epochs:
steps_per_epoch = int(len(train_dataset) / batch_size)
nb_training_steps = 4 * steps_per_epoch

train_loader = DataLoader(train_dataset, batch_size=dataloader_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=dataloader_batch_size, shuffle=False)

model_params = {
    "in_channels": 1,
    "input_height": 28,
    "input_width": 28,
    "output_dim": 10,
    "beta": 0.95,
    "surrogate_gradient": "atan",
}

client_counts = [1, 16]
histories = {}

for num_clients in client_counts:
    print(f"\n=========================================")
    print(f"RUNNING SWEEP FOR {num_clients} HONEST CLIENTS")
    print(f"=========================================")
    
    # Calculate client-specific local batch size to keep global batch size constant
    current_client_batch_size = batch_size // num_clients
    print(f"Client count: {num_clients} | Local batch size per client: {current_client_batch_size} (Total global batch size = {current_client_batch_size * num_clients})")
    
    # 4. Partition data
    print(f"Partitioning data among {num_clients} clients using IID distribution...")
    data_distributor = DataDistributor({
        "data_distribution_name": "iid",
        "distribution_parameter": 0.5,
        "nb_honest": num_clients,
        "data_loader": train_loader,
        "batch_size": current_client_batch_size,
    })
    client_dataloaders = data_distributor.split_data()
    
    # 5. Initialize Server and Clients
    honest_clients = [
        Client({
            "model_name": "convnet_snn",
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
        }) for i in range(num_clients)
    ]
    
    server = Server({
        "device": device,
        "model_name": "convnet_snn",
        "model_params": model_params,
        "accuracy_name": "accuracy_rate",
        "test_loader": test_loader,
        "optimizer_name": "SGD",
        "learning_rate": 0.05,
        "weight_decay": 0.0001,
        # Milestones: decay learning rate at 40% and 80% of steps
        "milestones": [int(nb_training_steps * 0.4), int(nb_training_steps * 0.8)],
        "learning_rate_decay": 0.5,
        "aggregator_info": {"name": "Average", "parameters": {}},
        "pre_agg_list": []
    })
    
    step_list = []
    loss_history = []
    acc_history = []
    
    for step in range(1, nb_training_steps + 1):
        server_model_state = server.get_dict_parameters()
        for client in honest_clients:
            client.set_model_state(server_model_state)
            
        losses = []
        for client in honest_clients:
            loss = client.compute_gradients()
            losses.append(loss)
        mean_loss = sum(losses) / len(losses)
        
        honest_gradients = [client.get_flat_gradients_with_momentum() for client in honest_clients]
        server.update_model_with_gradients(honest_gradients)
        
        test_acc = server.compute_test_accuracy()
        if step % 20 == 0 or step == 1 or step == nb_training_steps:
            print(f"Step {step:04d}/{nb_training_steps} | Avg Loss: {mean_loss:.4f} | Test Acc: {test_acc:.4f}")
            
        step_list.append(step)
        loss_history.append(mean_loss)
        acc_history.append(test_acc)
        
    histories[num_clients] = {
        "steps": step_list,
        "losses": loss_history,
        "accuracies": acc_history,
        "final_loss": loss_history[-1],
        "final_accuracy": acc_history[-1]
    }

# 6. Save results text summary
plot_dir = "experiments"
os.makedirs(plot_dir, exist_ok=True)
summary_path = os.path.join(plot_dir, "honest_clients_sweep_summary_correct.txt")
with open(summary_path, "w") as f:
    f.write("==================================================\n")
    f.write("Honest Clients Federated Sweep Summary (1872 Steps)\n")
    f.write("==================================================\n\n")
    for num_clients, data in histories.items():
        f.write(f"Honest Clients: {num_clients}\n")
        f.write(f"  - Final Average Loss: {data['final_loss']:.4f}\n")
        f.write(f"  - Final Test Accuracy: {data['final_accuracy']:.4f}\n")
        f.write("-" * 40 + "\n")
print(f"\nSummary saved successfully to {summary_path}")

# 7. Generate comparison plots
print("Generating final comparison plots...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

colors = {1: "#3b82f6", 8: "#f59e0b", 16: "#10b981"}

for num_clients, data in histories.items():
    ax1.plot(data["steps"], data["losses"], label=f"{num_clients} Client(s)", color=colors[num_clients], linewidth=2)
    ax2.plot(data["steps"], data["accuracies"], label=f"{num_clients} Client(s)", color=colors[num_clients], linewidth=2)

ax1.set_title("Average Training Loss Comparison", fontsize=12, fontweight='bold')
ax1.set_xlabel("Training Step")
ax1.set_ylabel("Loss")
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend()

ax2.set_title("Global Server Test Accuracy Comparison", fontsize=12, fontweight='bold')
ax2.set_xlabel("Training Step")
ax2.set_ylabel("Accuracy")
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend()

plt.tight_layout()
plot_path = os.path.join(plot_dir, "honest_clients_comparison_correct.png")
plt.savefig(plot_path, dpi=150)
print(f"Comparison plot saved successfully to {plot_path}")
