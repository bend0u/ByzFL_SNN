import torch
from torchvision import datasets, transforms
from byzfl.benchmark.data import SharedTensorDataset, pre_load_shared_dataset, transforms_cifar_train

shared_cache = pre_load_shared_dataset('cifar10', './data')
if shared_cache is not None:
    train_ds = SharedTensorDataset(
        data_tensor=shared_cache["train_data"],
        targets_tensor=shared_cache["train_targets"],
        transform=transforms_cifar_train
    )
    img, target = train_ds[0]
    print(f"Success! Image shape: {img.shape}, target: {target}")
else:
    print("Failed to load cache")
