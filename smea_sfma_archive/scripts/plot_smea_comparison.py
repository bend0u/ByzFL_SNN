import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def get_mean_accuracy(base_dir, pattern):
    dirs = glob.glob(os.path.join(base_dir, pattern))
    if not dirs:
        return 0.0
    
    target_dir = dirs[0]
    files = glob.glob(os.path.join(target_dir, 'test_accuracy_*.txt'))
    if not files:
        return 0.0
        
    accs = []
    for f in files:
        data = np.loadtxt(f, delimiter=',')
        if data.size > 0:
            accs.append(data[-1] if data.ndim > 0 else data.item())
    
    if accs:
        return np.mean(accs) * 100
    return 0.0

attacks_display = ['ALittleIsEnough', 'SignFlipping', 'InnerProductManipulation']
aggregators = ['GeometricMedian', 'CenteredClipping', 'SMEA']

accuracies = {agg: [] for agg in aggregators}

# 1. Fetch SMEA Data dynamically (from results/smea_experiment/snn_atan)
smea_attacks = ['ALittleIsEnough', 'SignFlipping', 'InnerProductManipulation']
for attack in smea_attacks:
    pattern = f'*_SMEA_NNM_ARC_{attack}_*alpha_1.2*'
    acc = get_mean_accuracy('results/smea_experiment/snn_atan/', pattern)
    accuracies['SMEA'].append(acc)

# 2. Fetch GeometricMedian and CenteredClipping Data
for agg in ['GeometricMedian', 'CenteredClipping']:
    
    # ALIE from old robust sweep (n=14, f=4)
    pattern_alie = f'*_n_14_f_4_d_4_gamma_similarity_niid_0.33_{agg}_NNM_ARC_Optimal_ALittleIsEnough_neg1_*alpha_1.*'
    acc_alie = get_mean_accuracy('results/snn/robust_new_atan_sweep/', pattern_alie)
    accuracies[agg].append(acc_alie)
    
    # SF from targeted sweep
    pattern_sf = f'*_n_14_f_4_d_4_gamma_similarity_niid_0.33_{agg}_NNM_ARC_SignFlipping_*alpha_1.2*'
    acc_sf = get_mean_accuracy('results/snn/robust_targeted_comparison/', pattern_sf)
    accuracies[agg].append(acc_sf)
    
    # IPM from targeted sweep
    pattern_ipm = f'*_n_14_f_4_d_4_gamma_similarity_niid_0.33_{agg}_NNM_ARC_InnerProductManipulation_*alpha_1.2*'
    acc_ipm = get_mean_accuracy('results/snn/robust_targeted_comparison/', pattern_ipm)
    accuracies[agg].append(acc_ipm)

# Prepare plotting
x = np.arange(len(attacks_display))
width = 0.25
multiplier = 0

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i, (attribute, measurement) in enumerate(accuracies.items()):
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute, color=colors[i])
    
    # Add labels, mark missing data as 'N/A' if 0.0
    labels = [f'{val:.1f}%' if val > 0 else 'N/A' for val in measurement]
    ax.bar_label(rects, labels=labels, padding=3)
    multiplier += 1

ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('SMEA vs Other Aggregators on SNN Atan (gamma=0.33)', fontsize=14, pad=20)
ax.set_xticks(x + width)
ax.set_xticklabels(attacks_display, fontsize=11)
ax.legend(loc='upper right', bbox_to_anchor=(1, 1.1), ncol=3)
ax.set_ylim(0, 100)

ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

ax.axhline(y=95.0, color='r', linestyle=':', alpha=0.6, label='Honest Baseline (~95%)')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(1, 1.1), ncol=4)

plt.tight_layout()

save_path = '/localhome/bendouro/.gemini/antigravity-ide/brain/881c0a4f-bb2b-4e36-be90-1f4adf0775dd/smea_vs_others.png'
plt.savefig(save_path, dpi=300)
print(f'Plot generated successfully dynamically from results! Saved to {save_path}')
