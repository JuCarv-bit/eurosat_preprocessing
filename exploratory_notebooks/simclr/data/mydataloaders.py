import numpy as np
import torch
from torchvision import transforms, datasets
# import config
from torch.utils.data import DataLoader, Subset
from simclr.config import CONFIG
from simclr.data.datamodule import get_split_indexes

def get_data_loaders_train_test_linear_probe(data_dir, batch_size):
    NUM_WORKERS = CONFIG["NUM_WORKERS"]
    SEED = CONFIG["SEED"]
    EUROSAT_IMAGE_SIZE = CONFIG["EUROSAT_IMAGE_SIZE"]

   # get meand and std from the file where we saved it
    with open("models/mean_std.txt", "r") as f:
        lines = f.readlines()
        mean = [float(x) for x in lines[0].strip().split(": ")[1][1:-1].split(",")]
        std = [float(x) for x in lines[1].strip().split(": ")[1][1:-1].split(",")]
    
    print(f"Using mean: {mean}")
    print(f"Using std: {std}")

    normalize = transforms.Normalize(mean=mean, std=std)
    color_jitter = transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1
    )
    gaussian_blur = transforms.GaussianBlur(
        kernel_size=7,
        sigma=(0.1, 2.0)
    )
    augment_transform = transforms.Compose([
        transforms.RandomResizedCrop(EUROSAT_IMAGE_SIZE[0], scale=(0.5, 1.0)),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomApply([gaussian_blur], p=0.5),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
            transforms.Resize(EUROSAT_IMAGE_SIZE),   
            transforms.ToTensor(),
            normalize,
    ])
    
    dataset_for_stats = datasets.ImageFolder(
        root=data_dir,
        transform=transforms.ToTensor()
    )
    total_len = len(dataset_for_stats)
    labels = np.array(dataset_for_stats.targets)
    print(f"Total samples in folder: {total_len}, classes: {dataset_for_stats.classes}")

    train_indices, val_indices, test_indices = get_split_indexes(labels, total_len)
    dataset_eval = datasets.ImageFolder(
        root=data_dir,
        transform=eval_transform
    )
    test_subset = Subset(dataset_eval, test_indices)
    train_val_indices = np.concatenate((train_indices, val_indices))

    # get dataloaders
    dataset_train_val = datasets.ImageFolder(
        root=data_dir,
        transform=augment_transform
    )
    train_val_subset = Subset(dataset_train_val, train_val_indices.tolist())
    train_loader = DataLoader(
        train_val_subset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        generator=torch.Generator().manual_seed(SEED)
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        generator=torch.Generator().manual_seed(SEED)
    )
    print(f"Train/Test loaders: {len(train_loader)}/{len(test_loader)} batches")
    assert total_len == len(train_val_subset) + len(test_subset), "Total length mismatch after split."
    return train_loader, test_loader, len(dataset_eval.classes)