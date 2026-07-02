import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def get_final_accuracy(results_dir, surrogate_type, param_name, param_value):
    # Pattern to match the directory for the specific parameter value and f=0
    # e.g., ..._surrogate_gradient_box_beta_0.5_...
    pattern = os.path.join(
        results_dir, 
        f"*_f_0_*_surrogate_gradient_{surrogate_type}_{param_name}_{param_value}_*"
    )
    matching_dirs = glob.glob(pattern)
    
    accuracies = []
    for directory in matching_dirs:
        # Find all test accuracy files in this directory
        acc_files = glob.glob(os.path.join(directory, "test_accuracy_tr_seed_*.txt"))
        for acc_file in acc_files:
            try:
                data = np.genfromtxt(acc_file, delimiter=',')
                if data.ndim == 0:
                    continue
                # Extract the final accuracy at step 500 (last element)
                final_acc = data[-1]
                accuracies.append(final_acc)
            except Exception as e:
                pass
                
    if not accuracies:
        return None
    return np.mean(accuracies), np.std(accuracies)

def main():
    plots_dir = "./plots/snn"
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Atan Sweep alphas
    atan_dir = "./results/snn/robust_atan_sweep"
    alphas = [1.0, 1.5, 2.0, 3.0, 4.0]
    atan_accs = []
    atan_stds = []
    for alpha in alphas:
        res = get_final_accuracy(atan_dir, "atan", "alpha", alpha)
        if res:
            atan_accs.append(res[0])
            atan_stds.append(res[1])
        else:
            atan_accs.append(None)
            atan_stds.append(None)
            
    # 2. Box Sweep betas
    box_dir = "./results/snn/robust_box_sweep"
    betas = [0.5, 0.75, 1.0, 1.25, 1.5]
    box_accs = []
    box_stds = []
    for beta in betas:
        res = get_final_accuracy(box_dir, "box", "beta", beta)
        if res:
            box_accs.append(res[0])
            box_stds.append(res[1])
        else:
            box_accs.append(None)
            box_stds.append(None)
            
    # 3. Tri Sweep betas
    tri_dir = "./results/snn/robust_tri_sweep"
    tri_accs = []
    tri_stds = []
    for beta in betas:
        res = get_final_accuracy(tri_dir, "tri", "beta", beta)
        if res:
            tri_accs.append(res[0])
            tri_stds.append(res[1])
        else:
            tri_accs.append(None)
            tri_stds.append(None)

    print("Atan Alphas:", alphas, "Accs:", atan_accs)
    print("Box Betas:", betas, "Accs:", box_accs)
    print("Tri Betas:", betas, "Accs:", tri_accs)

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    # Plot Atan
    valid_atan = [i for i, acc in enumerate(atan_accs) if acc is not None]
    if valid_atan:
        x_atan = [alphas[i] for i in valid_atan]
        y_atan = [atan_accs[i] for i in valid_atan]
        err_atan = [atan_stds[i] for i in valid_atan]
        axes[0].errorbar(x_atan, y_atan, yerr=err_atan, fmt='-o', color='blue', capsize=5, elinewidth=1.5, label='Atan')
    axes[0].set_title("Atan Surrogate")
    axes[0].set_xlabel("Alpha (Stiffness)")
    axes[0].set_ylabel("Final Test Accuracy (f=0)")
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot Box
    valid_box = [i for i, acc in enumerate(box_accs) if acc is not None]
    if valid_box:
        x_box = [betas[i] for i in valid_box]
        y_box = [box_accs[i] for i in valid_box]
        err_box = [box_stds[i] for i in valid_box]
        axes[1].errorbar(x_box, y_box, yerr=err_box, fmt='-s', color='green', capsize=5, elinewidth=1.5, label='Box')
    axes[1].set_title("Box (Rectangular) Surrogate")
    axes[1].set_xlabel("Beta (Width)")
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # Plot Tri
    valid_tri = [i for i, acc in enumerate(tri_accs) if acc is not None]
    if valid_tri:
        x_tri = [betas[i] for i in valid_tri]
        y_tri = [tri_accs[i] for i in valid_tri]
        err_tri = [tri_stds[i] for i in valid_tri]
        axes[2].errorbar(x_tri, y_tri, yerr=err_tri, fmt='-^', color='red', capsize=5, elinewidth=1.5, label='Tri')
    axes[2].set_title("Triangular Surrogate")
    axes[2].set_xlabel("Beta (Width)")
    axes[2].grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle("SNN Surrogate Hyperparameter Comparison (f=0, T=10, LR=0.10)", fontsize=14, y=0.98)
    plt.tight_layout()
    
    plot_path = os.path.join(plots_dir, "f0_surrogate_comparison.png")
    plt.savefig(plot_path, dpi=300)
    print(f"--> Saved comparison plot: {plot_path}")

if __name__ == "__main__":
    main()
