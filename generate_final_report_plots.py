import os
import shutil
from byzfl.benchmark.evaluate_results import aggregated_test_heatmap

out_dir = "plots/final_report"
os.makedirs(out_dir, exist_ok=True)

models = {
    "SNN_Atan": "results/snn/robust_new_atan_sweep/alpha_1.2",
    "SNN_Tri": "results/snn/robust_new_tri_sweep/beta_2.5",
    "SNN_Box": "results/snn/robust_new_box_sweep/beta_1.5",
    "CNN_Tanh": "results/cnn/tanh_heatmap_sweep",
    "CNN_Clipped": "results/cnn/clipped_heatmap_sweep",
    "CNN_ReLU": "results/cnn/robust_comparison_sweep"
}

for name, path in models.items():
    if os.path.exists(path):
        try:
            print(f"Generating for {name}...")
            # Clear directory of any lingering 'best_test' files to avoid confusion
            for existing in os.listdir(out_dir):
                if existing.startswith("best_test_"):
                    os.remove(os.path.join(out_dir, existing))
                    
            aggregated_test_heatmap(path, out_dir, target_attack="SignFlipping")
            
            # Find the newly generated file
            found = False
            for f in os.listdir(out_dir):
                if f.startswith("best_test_SignFlipping") and f.endswith(".png"):
                    new_name = f"{name}.png"
                    shutil.move(os.path.join(out_dir, f), os.path.join(out_dir, new_name))
                    found = True
                    break
            if not found:
                print(f"  Warning: No PNG generated for {name}.")
        except Exception as e:
            print(f"Failed for {name}: {e}")
    else:
        print(f"Path does not exist: {path}")
