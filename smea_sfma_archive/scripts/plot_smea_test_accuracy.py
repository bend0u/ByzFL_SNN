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

attacks_display = ['ALittleIsEnough', 'SignFlipping', 'InnerProductManipulation']
other_attacks = ['Optimal_ALittleIsEnough_neg1', 'SignFlipping', 'Optimal_InnerProductManipulation']
aggregators = ['GeometricMedian', 'CenteredClipping', 'SMEA']

curves_data = {attack: {agg: None for agg in aggregators} for attack in attacks_display}

# Fetch SMEA Data
for attack in attacks_display:
    pattern = f'*_SMEA_NNM_ARC_{attack}_*alpha_1.2*'
    curves_data[attack]['SMEA'] = get_mean_test_acc('results/smea_experiment/snn_atan/', pattern)

# Fetch GM and CC Data matching exactly f=4 and gamma=0.33 and alpha=1.2 (or 1.25)
for attack_disp, attack_other in zip(attacks_display, other_attacks):
    for agg in ['GeometricMedian', 'CenteredClipping']:
        pattern = f'*_n_14_f_4_d_4_gamma_similarity_niid_0.33_{agg}_NNM_ARC_{attack_other}_*alpha_1.2*'
        curves_data[attack_disp][agg] = get_mean_test_acc('results/snn/robust_new_atan_sweep/', pattern)

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for idx, attack in enumerate(attacks_display):
    ax = axes[idx]
    ax.set_title(f'{attack}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Evaluation Steps', fontsize=12)
    if idx == 0:
        ax.set_ylabel('Test Accuracy (%)', fontsize=12)
        
    has_data = False
    for c_idx, agg in enumerate(aggregators):
        curve = curves_data[attack][agg]
        if curve is not None:
            has_data = True
            x_vals = np.arange(len(curve)) * 50
            ax.plot(x_vals, curve, label=agg, color=colors[c_idx], linewidth=2.5, marker='o', markersize=4)
            
    if not has_data:
        ax.text(0.5, 0.5, 'No Data Available', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12)
    
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(0, 100)
    ax.axhline(y=95.0, color='r', linestyle=':', alpha=0.6, label='Honest Baseline (~95%)')
    
    if has_data:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='lower right')

plt.suptitle('Test Accuracy Evolution: SMEA vs Other Aggregators (f=4, gamma=0.33, alpha=1.2)', fontsize=16, y=1.05)
plt.tight_layout()

save_path = '/localhome/bendouro/.gemini/antigravity-ide/brain/881c0a4f-bb2b-4e36-be90-1f4adf0775dd/smea_vs_others_test_accuracy.png'
plt.savefig(save_path, dpi=300)
print(f'Test Accuracy plot generated successfully! Saved to {save_path}')
