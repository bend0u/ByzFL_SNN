import os
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import functional as F_t
from snntorch import spikegen
import tonic
import tonic.transforms as tonic_transforms


# =====================================================================
# SNN Temporal Encoding Transforms
# =====================================================================

class TemporalEncodingTransform:
    """
    Transforms standard vision dataset images (like MNIST/CIFAR) into temporal spike trains.
    Supports Constant, Rate (Poisson), Latency, and Delta encoding types.
    """
    def __init__(self, time_steps=25, encoding_type="constant", encoding_params=None):
        self.time_steps = time_steps
        self.encoding_type = encoding_type.lower()
        self.encoding_params = encoding_params if encoding_params is not None else {}

    def __call__(self, img):
        # Ensure input is a PyTorch tensor
        if not isinstance(img, torch.Tensor):
            img = F_t.to_tensor(img)
            
        # Select spike generator based on encoding type
        if self.encoding_type == "rate":
            img_clamped = torch.clamp(img, 0.0, 1.0)
            return spikegen.rate(img_clamped, num_steps=self.time_steps, **self.encoding_params)
        elif self.encoding_type == "latency":
            img_clamped = torch.clamp(img, 0.0, 1.0)
            return spikegen.latency(img_clamped, num_steps=self.time_steps, **self.encoding_params)
            
        else:  # 'constant' coding: replicate input image across time steps
            return img.unsqueeze(0).repeat(self.time_steps, *([1] * img.ndim))


# =====================================================================
# Neuromorphic Dataset Wrapper
# =====================================================================

class TonicDatasetWrapper(torch.utils.data.Dataset):
    """
    Wraps event-based Neuromorphic datasets loaded via Tonic, converting
    events to PyTorch tensors and exposing targets for splitting/partitioning.
    """
    def __init__(self, tonic_dataset, time_steps=25):
        self.tonic_dataset = tonic_dataset
        self.time_steps = time_steps
        # Safely extract and cache targets/labels for dataset partitioning
        if hasattr(tonic_dataset, "targets"):
            self.targets = torch.tensor(tonic_dataset.targets, dtype=torch.long)
        elif hasattr(tonic_dataset, "labels"):
            self.targets = torch.tensor(tonic_dataset.labels, dtype=torch.long)
        else:
            try:
                self.targets = torch.tensor([y for _, y in tonic_dataset], dtype=torch.long)
            except Exception:
                raise ValueError("Could not find targets or labels in Tonic dataset.")

    def __len__(self):
        return len(self.tonic_dataset)

    def __getitem__(self, idx):
        try:
            x, y = self.tonic_dataset[idx]
            x_tensor = torch.tensor(x, dtype=torch.float32)
            
            # Ensure time step dimension is consistently equal to self.time_steps
            t_size = x_tensor.size(0)
            if t_size < self.time_steps:
                # Pad with zeros at the end
                pad_size = self.time_steps - t_size
                padding = torch.zeros((pad_size,) + x_tensor.shape[1:], dtype=x_tensor.dtype)
                x_tensor = torch.cat([x_tensor, padding], dim=0)
            elif t_size > self.time_steps:
                # Truncate
                x_tensor = x_tensor[:self.time_steps]
                
            return x_tensor, y
        except Exception as e:
            print(f"\n[ERROR] TonicDatasetWrapper failed to retrieve/transform index {idx}: {e}")
            import traceback
            traceback.print_exc()
            raise e


# =====================================================================
# Base Transforms Constants (from original codebase)
# =====================================================================

transforms_hflip = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

transforms_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

transforms_cifar_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transforms_cifar_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Supported datasets mapping: key -> (ClassName, TrainTransform, TestTransform)
dict_datasets = {
    "mnist":        ("MNIST", transforms_mnist, transforms_mnist),
    "fashionmnist": ("FashionMNIST", transforms_hflip, transforms_hflip),
    "emnist":       ("EMNIST", transforms_mnist, transforms_mnist),
    "cifar10":      ("CIFAR10", transforms_cifar_train, transforms_cifar_test),
    "cifar100":     ("CIFAR100", transforms_cifar_train, transforms_cifar_test),
    "imagenet":     ("ImageNet", transforms_hflip, transforms_hflip)
}


# =====================================================================
# Transform Generation
# =====================================================================

def get_dataset_transforms(key_dataset_name, is_train, is_snn, params_manager):
    """
    Retrieves standard dataset transforms. If model is an SNN, appends
    a TemporalEncodingTransform to the transform chain.
    """
    # Retrieve standard base transforms (Compose objects) from dict_datasets
    base_info = dict_datasets[key_dataset_name]
    base_transform = base_info[1] if is_train else base_info[2]
    
    # Extract the sub-transforms list
    transform_list = list(base_transform.transforms)
    
    # Append SNN temporal encoding if target model is SNN
    if is_snn:
        enc_type = params_manager.get_encoding_type()
        # If using a spike-based encoding (not direct/constant), remove torchvision Normalize
        if enc_type and enc_type.lower() != "constant":
            transform_list = [t for t in transform_list if not isinstance(t, transforms.Normalize)]
            
        temporal_transform = TemporalEncodingTransform(
            time_steps=params_manager.get_time_steps(),
            encoding_type=enc_type,
            encoding_params=params_manager.get_encoding_params()
        )
        transform_list.append(temporal_transform)
        
    return transforms.Compose(transform_list)


# =====================================================================
# Dataset Loaders & Splits
# =====================================================================

def load_nmnist_dataset(data_folder, params_manager):
    """
    Helper function to load and wrap the neuromorphic N-MNIST dataset.
    """
    time_steps = params_manager.get_time_steps()
    
    # Define cache paths in data_folder (admin freed space, now 57 GB available)
    cache_dir = os.path.join(data_folder, f"nmnist_cache_ts_{time_steps}")
    train_cache_path = os.path.join(cache_dir, "train")
    test_cache_path = os.path.join(cache_dir, "test")
    
    # Check cache status and log details
    print(f"\n[ByzFL-SNN] N-MNIST Cache Location: {cache_dir}")
    if os.path.exists(cache_dir):
        import glob
        num_train = len(glob.glob(os.path.join(train_cache_path, "*.hdf5")))
        num_test = len(glob.glob(os.path.join(test_cache_path, "*.hdf5")))
        print(f"[ByzFL-SNN]   Found existing disk cache.")
        print(f"[ByzFL-SNN]   Cached files: Train: {num_train}/60000 | Test: {num_test}/10000")
        if num_train == 60000:
            print(f"[ByzFL-SNN]   Train dataset fully cached! Will load directly from memory/disk (Instant).")
    else:
        print(f"[ByzFL-SNN]   No existing cache found. A new preprocessed cache will be created on-the-fly.")
            
    time_window = int(300000 / time_steps)
    sensor_size = tonic.datasets.NMNIST.sensor_size
    
    # Apply denoising on raw events first, then accumulate to frames
    frame_transform = tonic_transforms.Compose([
        tonic_transforms.Denoise(filter_time=time_window),
        tonic_transforms.ToFrame(sensor_size=sensor_size, time_window=time_window),
    ])
    
    # Load raw datasets using Tonic (raw events are read from data_folder on /localhome)
    raw_train = tonic.datasets.NMNIST(save_to=data_folder, train=True, transform=frame_transform)
    raw_test = tonic.datasets.NMNIST(save_to=data_folder, train=False, transform=frame_transform)
    
    # Wrap in Tonic's DiskCachedDataset, caching to /tmp
    disk_cached_train = tonic.DiskCachedDataset(raw_train, cache_path=train_cache_path)
    disk_cached_test = tonic.DiskCachedDataset(raw_test, cache_path=test_cache_path)
    
    # Wrap the DiskCachedDataset in MemoryCachedDataset for RAM caching
    cached_raw_train = tonic.MemoryCachedDataset(disk_cached_train)
    cached_raw_train.targets = raw_train.targets
    
    cached_raw_test = tonic.MemoryCachedDataset(disk_cached_test)
    cached_raw_test.targets = raw_test.targets
    
    # Wrap in PyTorch Datasets
    train_ds = TonicDatasetWrapper(cached_raw_train, time_steps)
    test_ds = TonicDatasetWrapper(cached_raw_test, time_steps)
    return train_ds, test_ds


def load_and_split_data(params_manager):
    """
    Main function to load datasets, split training/validation partitions,
    and build standard PyTorch DataLoaders.
    """
    key_dataset_name = params_manager.get_dataset_name().lower()
    is_snn = params_manager.is_snn()
    data_folder = params_manager.get_data_folder()
    
    # 1. Neuromorphic Dataset (N-MNIST)
    if key_dataset_name == "nmnist":
        dataset, test_dataset = load_nmnist_dataset(data_folder, params_manager)
        
        # Split the dataset into training and validation sets
        train_size = int(params_manager.get_size_train_set() * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
    # 2. Standard Vision Datasets
    else:
        dataset_name = dict_datasets[key_dataset_name][0]
        
        # Build training, validation, and test transform chains
        train_transform = get_dataset_transforms(
            key_dataset_name, is_train=True,
            is_snn=is_snn, params_manager=params_manager
        )
        val_transform = get_dataset_transforms(
            key_dataset_name, is_train=False,
            is_snn=is_snn, params_manager=params_manager
        )
        test_transform = get_dataset_transforms(
            key_dataset_name, is_train=False,
            is_snn=is_snn, params_manager=params_manager
        )
        
        # Instantiate separate dataset objects for train and val splits to avoid Shared State Hazard
        train_base = getattr(datasets, dataset_name)(
            root=data_folder, 
            train=True, 
            download=True,
            transform=train_transform
        )
        train_base.targets = Tensor(train_base.targets).long()
        
        val_base = getattr(datasets, dataset_name)(
            root=data_folder, 
            train=True, 
            download=True,
            transform=val_transform
        )
        val_base.targets = Tensor(val_base.targets).long()
        
        test_dataset = getattr(datasets, dataset_name)(
            root=data_folder,
            train=False, 
            download=True,
            transform=test_transform
        )
        
        # Split the dataset into training and validation sets
        train_size = int(params_manager.get_size_train_set() * len(train_base))
        val_size = len(train_base) - train_size
        
        # Split using random_split on train_base to get indices partition
        train_dummy, val_dummy = random_split(train_base, [train_size, val_size])
        
        # Wrap the respective base datasets with the partitioned indices
        train_dataset = torch.utils.data.Subset(train_base, train_dummy.indices)
        val_dataset = torch.utils.data.Subset(val_base, val_dummy.indices)
        
    # Create final PyTorch DataLoaders
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset, 
            batch_size=params_manager.get_batch_size_evaluation(), 
            shuffle=False
        )
    else:
        val_loader = None
        
    test_loader = DataLoader(
        test_dataset, 
        batch_size=params_manager.get_batch_size_evaluation(), 
        shuffle=False
    )
    
    return train_dataset, val_loader, test_loader
