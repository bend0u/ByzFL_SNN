# Commands to run Box and Tri beta sweeps in the background using nohup on specific GPUs.
# Make sure your virtual environment is activated before running these commands.

# 1. Run Box Beta Sweep on CUDA 0
CUDA_VISIBLE_DEVICES=0 nohup python run_box_beta_sweep.py > box_sweep.log 2>&1 &

# 2. Run Tri Beta Sweep on CUDA 1
CUDA_VISIBLE_DEVICES=1 nohup python run_tri_beta_sweep.py > tri_sweep.log 2>&1 &
