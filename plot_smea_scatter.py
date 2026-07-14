import matplotlib.pyplot as plt

f_values = []
acc_values = []

with open('acc_clusters.txt', 'r') as f:
    current_f = None
    for line in f:
        line = line.strip()
        if line.startswith('f='):
            current_f = int(line.split('=')[1].split(' ')[0])
        elif line:
            acc = float(line)
            f_values.append(current_f)
            acc_values.append(acc * 100)  # Convert to percentage

# Jitter the f_values slightly for better visualization
import numpy as np
jitter = np.random.uniform(-0.15, 0.15, len(f_values))
f_values_jittered = np.array(f_values) + jitter

fig, ax = plt.subplots(figsize=(10, 6))

# Plot all points as semi-transparent scatter points
ax.scatter(f_values_jittered, acc_values, color='#1f77b4', alpha=0.6, s=60, edgecolors='none')

ax.set_title('SMEA Final Test Accuracy across all $\\tau$ values (ALIE & IPM)', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Number of Byzantine Workers ($f$)', fontsize=12)
ax.set_ylabel('Final Test Accuracy (%)', fontsize=12)
ax.set_xticks(range(6))
ax.set_ylim(-5, 105)

ax.grid(True, linestyle=':', alpha=0.6)

# Highlight the Baseline
ax.axhline(y=96.7, color='grey', linestyle='--', alpha=0.8, label='Honest Baseline (~96.7%)')
# Highlight Random Chance
ax.axhline(y=10.0, color='red', linestyle=':', alpha=0.8, label='Random Guessing (10%)')

ax.legend(fontsize=11, loc='center right')

plt.tight_layout()

save_path = '/localhome/bendouro/.gemini/antigravity-ide/brain/881c0a4f-bb2b-4e36-be90-1f4adf0775dd/smea_scatter_clusters.png'
plt.savefig(save_path, dpi=300)
print(f'Scatter plot saved to {save_path}')
