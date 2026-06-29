import torch
from byzfl.benchmark.managers import ParamsManager
from byzfl.benchmark.train import start_training
import json

with open("configs/cnn_cifar_baseline.json") as f:
    config = json.load(f)

# Override just to test f=1, SignFlipping
config["benchmark_config"]["f"] = [1]
config["aggregator"] = [{"name": "Average", "parameters": {}}]
config["benchmark_config"]["nb_steps"] = 10

pm = ParamsManager(config)
pm.update_state(f=1, agg_name="Average", attack_name="SignFlipping")
start_training(pm)
