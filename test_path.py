import json
import byzfl.benchmark.benchmark as b
from byzfl.benchmark.managers import ParamsManager

with open('configs/interarch/config_cnn_relu_clip21.json') as f:
    config = json.load(f)

dict_list = b.unpack_parameters(config)
setting = dict_list[0]
pm = ParamsManager(setting)
clip_val = pm.get_honest_clients_gradient_clip_val()
print("clip_val directly from pm:", clip_val)
