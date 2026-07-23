import os
import glob
import matplotlib.pyplot as plt

# Path to the downloaded results
results_dir = r"c:\Users\7430\Desktop\SummerIntheLab\ByzFL_SNN\results\cnn_cifar_tuning"

# Look for directories matching the pattern
dirs = glob.glob(os.path.join(results_dir, "cifar10_*"))

plt.figure(figsize=(10, 6))

# We will group by model type and plot different learning rates
# Model types: cnn_cifar (ReLU), cnn_cifar_tanh (Tanh)
# LRs: 0.01, 0.05, 0.1

for d in dirs:
    folder_name = os.path.basename(d)
    
    # Parse model type
    if "cnn_cifar_tanh" in folder_name:
        model_type = "CNN Tanh"
        style = "--"
    else:
        model_type = "CNN ReLU"
        style = "-"
        
    # Parse learning rate
    lr = "0.01"
    if "lr_0.05" in folder_name:
        lr = "0.05"
    elif "lr_0.1" in folder_name:
        lr = "0.1"
    elif "lr_0.01" in folder_name:
        lr = "0.01"
        
    # Find the test accuracy file
    acc_files = glob.glob(os.path.join(d, "test_accuracy_*.txt"))
    if not acc_files:
        print(f"No test accuracy file found in {folder_name}")
        continue
        
    acc_file = acc_files[0]
    
    with open(acc_file, "r") as f:
        content = f.read().strip()
        
    # Parse comma separated values
    try:
        accuracies = [float(x) for x in content.split(",") if x.strip()]
        steps = list(range(len(accuracies)))
        
        label = f"{model_type} (LR={lr})"
        plt.plot(steps, accuracies, label=label, linestyle=style, marker='o', markersize=3)
    except Exception as e:
        print(f"Error parsing file {acc_file}: {e}")

plt.title("CIFAR-10: Test Accuracy vs Epochs (Learning Rate Tuning)", fontsize=14)
plt.xlabel("Evaluation Step / Epoch", fontsize=12)
plt.ylabel("Test Accuracy", fontsize=12)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(loc="lower right")

# Save the plot
output_plot = r"c:\Users\7430\Desktop\SummerIntheLab\ByzFL_SNN\plots\cifar_lr_tuning.png"
os.makedirs(os.path.dirname(output_plot), exist_ok=True)
plt.savefig(output_plot, dpi=300, bbox_inches="tight")
print(f"Plot saved successfully to: {output_plot}")
# plt.show()
