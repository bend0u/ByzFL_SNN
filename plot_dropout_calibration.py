import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import re

def get_best_test_accuracy(base_dir, pattern):
    dirs = glob.glob(os.path.join(base_dir, pattern))
    if not dirs:
        return 0.0
    
    accs = []
    for d in dirs:
        test_files = sorted(glob.glob(os.path.join(d, 'test_accuracy_*.txt')))
        val_files = sorted(glob.glob(os.path.join(d, 'val_accuracy_*.txt')))
        
        for t_f, v_f in zip(test_files, val_files):
            try:
                t_data = np.loadtxt(t_f, delimiter=',')
                v_data = np.loadtxt(v_f, delimiter=',')
                if t_data.ndim == 0: t_data = np.array([t_data])
                if v_data.ndim == 0: v_data = np.array([v_data])
                
                if v_data.size > 0 and t_data.size > 0:
                    best_idx = np.argmax(v_data)
                    accs.append(t_data[best_idx].item())
            except Exception:
                continue
    
    if accs:
        return np.mean(accs) * 100
    return 0.0

dropouts = [60, 70, 80, 93]
accuracies = []

base_dir = 'results/'

for dp in dropouts:
    pattern = f'*cnn_mnist_tanh_dropout_dense_{dp}_*'
    acc = get_best_test_accuracy(base_dir, pattern)
    accuracies.append(acc)

plt.figure(figsize=(10, 6))
plt.plot(dropouts, accuracies, marker='o', linewidth=2, markersize=8)
plt.axhline(y=96.0, color='r', linestyle='--', alpha=0.7, label='Target Threshold (96%)')

for i, acc in enumerate(accuracies):
    plt.annotate(f'{acc:.1f}%', (dropouts[i], accuracies[i]), 
                 textcoords="offset points", xytext=(0,10), ha='center')

plt.title('Honest CNN Tanh Baseline vs Dense Dropout Rate (MNIST)', fontsize=14)
plt.xlabel('Dropout Rate (%)', fontsize=12)
plt.ylabel('Test Accuracy (%)', fontsize=12)
plt.xticks(dropouts)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()

save_path = '/localhome/bendouro/.gemini/antigravity-ide/brain/881c0a4f-bb2b-4e36-be90-1f4adf0775dd/dropout_calibration_plot.png'
plt.savefig(save_path, dpi=300)
print(f'Plot generated successfully! Saved to {save_path}')
