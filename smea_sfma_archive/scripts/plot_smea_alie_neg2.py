import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def get_mean_test_acc(base_dir, pattern):
    dirs = glob.glob(os.path.join(base_dir, pattern))
    if not dirs:
        return None
    target_dir = dirs[0]
    files = glob.glob(os.path.join(target_dir, 'test_accuracy_*.txt'))
    if not files:
        return None
        
    curves = []
    for f in files:
        data = np.loadtxt(f, delimiter=',')
        if data.size > 0:
            if data.ndim == 0:
                data = np.array([data.item()])
            curves.append(data)
    
    if curves:
        min_len = min(len(c) for c in curves)
        curves = [c[:min_len] for c in curves]
        return np.mean(curves, axis=0) * 100
    return None

attacks = [
    ('ALittleIsEnough (tau=1.5)', 'results/smea_experiment/snn_atan/', '*_SMEA_NNM_ARC_ALittleIsEnough_*alpha_1.2*'),
    ('ALittleIsEnough (tau=-2.0)', 'results/smea_experiment/snn_atan_alie_neg2/', '*_SMEA_NNM_ARC_ALittleIsEnough_*alpha_1.2*'),
    ('SignFlipping', 'results/smea_experiment/snn_atan/', '*_SMEA_NNM_ARC_SignFlipping_*alpha_1.2*'),
    ('InnerProductManipulation', 'results/smea_experiment/snn_atan/', '*_SMEA_NNM_ARC_InnerProductManipulation_*alpha_1.2*')
]
colors = ['#1f77b4', '#9467bd', '#ff7f0e', '#d62728']

fig, ax = plt.subplots(figsize=(10, 6))

for idx, (label, base_dir, pattern) in enumerate(attacks):
    curve = get_mean_test_acc(base_dir, pattern)
    
    if curve is not None:
        x_vals = np.arange(len(curve)) * 50
        ax.plot(x_vals, curve, label=label, color=colors[idx], linewidth=2.5, marker='o', markersize=6)

ax.set_title('SMEA Test Accuracy Evolution (f=4, gamma=0.33, alpha=1.2)', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Training Steps', fontsize=12)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)

ax.set_ylim(0, 100)
ax.axhline(y=95.0, color='grey', linestyle='--', alpha=0.8, label='Honest Baseline (~95%)')
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(fontsize=11, loc='center right')

plt.tight_layout()

save_path = '/localhome/bendouro/.gemini/antigravity-ide/brain/881c0a4f-bb2b-4e36-be90-1f4adf0775dd/smea_alie_neg2_accuracy.png'
plt.savefig(save_path, dpi=300)
print(f'SMEA plot saved to {save_path}')
