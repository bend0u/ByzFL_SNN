import os
from byzfl.benchmark.evaluate_results import test_heatmap, loss_heatmap, aggregated_test_heatmap

def main():
    experiments = [
        ("results/cnn/complete", "plots/cnn/complete_extended"),
        ("results/snn/complete_direct", "plots/snn/complete_direct_extended"),
        ("results/snn/complete_nmnist", "plots/snn/complete_nmnist_extended"),
        ("results/snn/complete_rate", "plots/snn/complete_rate_extended")
    ]
    
    attacks = [
        "Optimal_ALittleIsEnough", 
        "Optimal_InnerProductManipulation", 
        "SignFlipping"
    ]
    
    for results_dir, plots_dir in experiments:
        if not os.path.exists(results_dir):
            print(f"Results directory '{results_dir}' does not exist. Skipping.")
            continue
            
        print(f"\n=======================================================")
        print(f"Generating heatmaps for: {results_dir}")
        print(f"=======================================================")
        os.makedirs(plots_dir, exist_ok=True)
        
        for attack in attacks:
            print(f"Generating plots for fixed attack: {attack}...")
            test_heatmap(results_dir, plots_dir, target_attack=attack)
            aggregated_test_heatmap(results_dir, plots_dir, target_attack=attack)
            
    print("\nAll fixed-attack heatmaps successfully generated!")

if __name__ == "__main__":
    main()
