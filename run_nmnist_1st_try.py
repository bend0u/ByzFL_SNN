import os
from byzfl import run_benchmark

def main():
    print("==========================================================")
    print("STARTING BYZANTINE N-MNIST PARALLEL SWEEP (4 COMBINATIONS)")
    print("==========================================================")
    
    # Run the configuration sequentially with nb_jobs=1 on GPU 1
    run_benchmark("snn_nmnist_1st_try.json", nb_jobs=1)
    
    print("\n==========================================================")
    print("RUN COMPLETED SUCCESSFULLY!")
    print("==========================================================")

if __name__ == "__main__":
    main()
