import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def get_mean_train_loss(base_dir, pattern):
    dirs = glob.glob(os.path.join(base_dir, pattern))
    if not dirs:
        return None
    target_dir = dirs[0]
    files = glob.glob(os.path.join(target_dir, 'train_loss_*.txt'))
    if not files:
        return None
        
    curves = []
    for f in files:
        data = np.loadtxt(f, delimiter=',')
        if data.size > 0:
            curves.append(data)
    
    if curves:
        min_len = min(len(c) for c in curves)
        curves = [c[:min_len] for c in curves]
        return np.mean(curves, axis=0)
    return None

def moving_average(a, n=20):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

attacks_display = ['ALittleIsEnough', 'SignFlipping', 'InnerProductManipulation']
other_attacks = ['Optimal_ALittleIsEnough_neg1', 'SignFlipping', 'Optimal_InnerProductManipulation']
aggregators = ['GeometricMedian', 'CenteredClipping', 'SMEA']

curves_data = {attack: {agg: None for agg in aggregators} for attack in attacks_display}

# Fetch SMEA Data
for attack in attacks_display:
    pattern = f'*_SMEA_NNM_ARC_{attack}_*alpha_1.2*'
    curves_data[attack]['SMEA'] = get_mean_train_loss('results/smea_experiment/snn_atan/', pattern)

# Fetch GM and CC Data
for attack_disp, attack_other in zip(attacks_display, other_attacks):
    for agg in ['GeometricMedian', 'CenteredClipping']:
        pattern = f'*_n_15_f_5_d_5_gamma_similarity_niid_0.33_{agg}_NNM_ARC_{attack_other}_*alpha_1.*'
        curves_data[attack_disp][agg] = get_mean_train_loss('results/snn/robust_new_atan_sweep/', pattern)

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for idx, attack in enumerate(attacks_display):
    ax = axes[idx]
    ax.set_title(f'{attack}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Training Steps', fontsize=12)
    if idx == 0:
        ax.set_ylabel('Training Loss', fontsize=12)
        
    has_data = False
    for c_idx, agg in enumerate(aggregators):
        curve = curves_data[attack][agg]
        if curve is not None:
            has_data = True
            # Plot raw
            ax.plot(curve, color=colors[c_idx], alpha=0.2, linewidth=1)
            # Plot smoothed
            smoothed = moving_average(curve, n=20)
            ax.plot(np.arange(19, len(curve)), smoothed, label=agg, color=colors[c_idx], linewidth=2.5)
            
    if not has_data:
        ax.text(0.5, 0.5, 'No Data Available', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12)
    
    ax.grid(True, linestyle='--', alpha=0.6)
    if has_data:
        ax.legend(loc='upper right')

plt.suptitle('Training Loss Evolution: SMEA vs Other Aggregators (gamma=0.33)', fontsize=16, y=1.05)
plt.tight_layout()

save_path = '/localhome/bendouro/.gemini/antigravity-ide/brain/881c0a4f-bb2b-4e36-be90-1f4adf0775dd/smea_vs_others_loss.png'
plt.savefig(save_path, dpi=300)
print(f'Loss plot generated successfully! Saved to {save_path}')
