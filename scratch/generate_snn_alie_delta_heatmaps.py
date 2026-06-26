import os
import sys

# Ensure workspace root is in Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from byzfl.benchmark.evaluate_results import test_heatmap, aggregated_test_heatmap

def main():
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Map from results directory to output plots directory
    runs = [
        ("results/cnn/simplest", "cnn_simplest_plots"),
        ("results/snn/simplest", "snn_simplest_plots")
    ]
    
    # Delta variants to plot
    attacks = [
        "Optimal_ALittleIsEnough",       # delta = 10
        "Optimal_ALittleIsEnough_pos1",  # delta = 1
        "Optimal_ALittleIsEnough_neg1",  # delta = -1
        "Optimal_ALittleIsEnough_neg10"  # delta = -10
    ]
    
    for relative_res_dir, relative_plots_dir in runs:
        results_dir = os.path.join(workspace_dir, relative_res_dir)
        plots_dir = os.path.join(workspace_dir, relative_plots_dir)
        
        print("=======================================================")
        print(f"Generating heatmaps for {relative_res_dir} -> {relative_plots_dir}...")
        print("=======================================================")
        
        os.makedirs(plots_dir, exist_ok=True)
        
        for attack in attacks:
            print(f"  Generating plots for: {attack}")
            try:
                test_heatmap(results_dir, plots_dir, target_attack=attack, metric="best_step")
                aggregated_test_heatmap(results_dir, plots_dir, target_attack=attack, metric="best_step")
            except Exception as e:
                print(f"  [ERROR] Failed to generate heatmaps for {attack}: {e}")
                
    print("\nHeatmap generation completed!")

if __name__ == "__main__":
    main()
