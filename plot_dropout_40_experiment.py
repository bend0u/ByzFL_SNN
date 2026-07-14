import matplotlib.pyplot as plt
import numpy as np
import os
import glob

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

def plot_for_gamma(gamma, ax):
    attacks_display = ['Baseline (f=0)', 'ALittleIsEnough (f=5)', 'SignFlipping (f=5)', 'InnerProductManipulation (f=5)']
    attack_patterns = {
        'Baseline (f=0)': ('_f_0_', '*'), 
        'ALittleIsEnough (f=5)': ('_f_5_', '*Optimal_ALittleIsEnough_neg1*'),
        'SignFlipping (f=5)': ('_f_5_', '*SignFlipping*'),
        'InnerProductManipulation (f=5)': ('_f_5_', '*Optimal_InnerProductManipulation*')
    }
    
    aggregators = ['Average', 'GeometricMedian', 'CenteredClipping', 'TrMean', 'MultiKrum']
    accuracies = {agg: [] for agg in aggregators}
    
    base_dir = 'results/cnn_dropout_40_experiment/'
    
    for attack_name in attacks_display:
        f_pat, att_pat = attack_patterns[attack_name]
        for agg in aggregators:
            pattern = f'*{f_pat}*gamma_similarity_niid_{gamma}_{agg}_{att_pat}*'
            acc = get_best_test_accuracy(base_dir, pattern)
            accuracies[agg].append(acc)
            
    x = np.arange(len(attacks_display))
    width = 0.15
    multiplier = 0
    
    colors = ['#7f7f7f', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, agg in enumerate(aggregators):
        measurement = accuracies[agg]
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=agg, color=colors[i])
        
        labels = [f'{val:.1f}' if val > 0 else '' for val in measurement]
        ax.bar_label(rects, labels=labels, padding=3, fontsize=8, rotation=90)
        multiplier += 1

    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title(f'CNN 40% Dropout Robustness (gamma={gamma})', fontsize=14)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(attacks_display, fontsize=10)
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

plot_for_gamma('0.66', ax1)
plot_for_gamma('1.0', ax2)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=5, fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.93])

save_path = '/localhome/bendouro/.gemini/antigravity-ide/brain/881c0a4f-bb2b-4e36-be90-1f4adf0775dd/cnn_dropout_40_results.png'
plt.savefig(save_path, dpi=300)
print(f'Plot generated successfully! Saved to {save_path}')
