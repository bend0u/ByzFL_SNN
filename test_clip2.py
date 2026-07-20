import json
import byzfl.benchmark.benchmark as b
from byzfl.benchmark.managers import ParamsManager
with open('configs/interarch/config_cnn_relu_clip21.json') as f:
    config = json.load(f)
dict_list = []
b.generate_all_combinations_aux(dict_list, config, {}, ["evaluation_delta", "milestones", "f", "distribution_parameter"])
setting = dict_list[0]
pm = ParamsManager(setting)
print("gradient_clip_val:", pm.get_honest_clients_gradient_clip_val())
print("setting:", setting)
