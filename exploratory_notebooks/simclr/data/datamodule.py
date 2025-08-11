# import config
import os
import ssl
import urllib.request
import zipfile
from simclr.config import CONFIG

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision import datasets

import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit


from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import seaborn as sns
from utils.version_utils import print_versions, configure_gpu_device, set_seed
from torch.utils.data import ConcatDataset, DataLoader
import torch.nn.functional as F
from transfer.knn import WeightedKNNClassifier
import joblib

from simclr.data.transforms import get_transforms, TwoCropsTransform

class SimCLRDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, _ = self.dataset[idx]
        x1, x2 = self.transform(x)
        return x1, x2

def prepare_data():
    if CONFIG["LOCAL_OR_COLAB"] == "LOCAL":
        return CONFIG["DATA_DIR_EUROSAT_RGB"]

    if not os.path.exists(CONFIG["DATA_DIR_COLAB"]):
        print("Downloading EuroSAT RGB...")
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib.request.urlretrieve(CONFIG["EUROSAT_URL"], CONFIG["ZIP_PATH"])
        with zipfile.ZipFile(CONFIG["ZIP_PATH"], 'r') as zip_ref:
            zip_ref.extractall("/content")
        os.rename("/content/2750", CONFIG["DATA_DIR_COLAB"])
        print("EuroSAT RGB dataset downloaded and extracted.")
    return CONFIG["DATA_DIR_COLAB"]


def compute_mean_std(dataset, batch_size, yaware=False):
    SEED = CONFIG["SEED"]
    loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=CONFIG["NUM_WORKERS"], generator=torch.Generator().manual_seed(SEED))
    mean = 0.0
    std = 0.0
    n_samples = 0
    
    with torch.no_grad():
        for sample_loader in loader:
            if yaware:
                data, _, _ = sample_loader
            else:
                data, _ = sample_loader

            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)  # (B, C, H*W)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            n_samples += batch_samples

        mean /= n_samples
        std /= n_samples
        return mean.tolist(), std.tolist()

def combine_train_val_loaders(train_loader, val_loader):
    train_ds = train_loader.dataset
    val_ds   = val_loader.dataset

    combined_ds = ConcatDataset([train_ds, val_ds])
    SEED = CONFIG["SEED"]
    NUM_WORKERS = CONFIG["NUM_WORKERS"]
    train_val_loader = DataLoader(
        combined_ds,
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        generator=torch.Generator().manual_seed(SEED)
    )
    
    return train_val_loader

def get_split_indexes(labels, total_count):
    TRAIN_FRAC = CONFIG["TRAIN_FRAC"]
    VAL_FRAC = CONFIG["VAL_FRAC"]
    SEED = CONFIG["SEED"]

    n_train = int(np.floor(TRAIN_FRAC * total_count))
    n_temp = total_count - n_train   # this is val + test

    sss1 = StratifiedShuffleSplit(
        n_splits=1,
        train_size=n_train,
        test_size=n_temp,
        random_state=SEED
    )
    # Train and temp(val+test) indices
    train_idx, temp_idx = next(sss1.split(np.zeros(total_count), labels))

    n_val = int(np.floor(VAL_FRAC * total_count))
    n_test = total_count - n_train - n_val
    assert n_temp == n_val + n_test, "Fractions must sum to 1."

    labels_temp = labels[temp_idx]

    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        train_size=n_val,
        test_size=n_test,
        random_state=SEED
    )
    val_idx_in_temp, test_idx_in_temp = next(sss2.split(np.zeros(len(temp_idx)), labels_temp))

    val_idx = temp_idx[val_idx_in_temp]
    test_idx = temp_idx[test_idx_in_temp]

    assert len(train_idx) == n_train
    assert len(val_idx) == n_val
    assert len(test_idx) == n_test

    print(f"Stratified split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    return train_idx, val_idx, test_idx


def get_data_loaders(data_dir, batch_size):

    dataset_for_stats = datasets.ImageFolder(
        root=data_dir,
        transform=transforms.ToTensor()
    )
    total_len = len(dataset_for_stats)
    labels = np.array(dataset_for_stats.targets)
    num_classes = len(dataset_for_stats.classes)
    print(f"Total samples in folder: {total_len}, classes: {dataset_for_stats.classes}")

    train_indices, val_indices, test_indices = get_split_indexes(labels, total_len)

    train_for_stats_subset = Subset(dataset_for_stats, train_indices)
    mean, std = compute_mean_std(train_for_stats_subset, batch_size)
    print(f"Computed mean: {mean}")
    print(f"Computed std:  {std}")
    # save the mean and std to a file
    os.makedirs("models", exist_ok=True)

    with open("models/mean_std.txt", "w") as f:
        f.write(f"mean: {mean}\n")
        f.write(f"std: {std}\n")
    print("Mean and std saved to models/mean_std.txt")

    dataset_train_no_transform = datasets.ImageFolder(
        root=data_dir,
        transform=None
    )
    train_subset_no_transform = Subset(dataset_train_no_transform, train_indices)

    dataset_val_no_transform = datasets.ImageFolder(root=data_dir, transform=None)
    val_subset_no_transform  = Subset(dataset_val_no_transform, val_indices)


    eval_transform, augment_transform = get_transforms(mean, std)

    dataset_eval = datasets.ImageFolder(
        root=data_dir,
        transform=eval_transform
    )
    val_subset = Subset(dataset_eval, val_indices)
    test_subset = Subset(dataset_eval, test_indices)
    simclr_transform = TwoCropsTransform(augment_transform)
    train_ds_simclr = SimCLRDataset(train_subset_no_transform, simclr_transform)
    NUM_WORKERS = CONFIG["NUM_WORKERS"]
    SEED = CONFIG["SEED"]

    train_loader = DataLoader(
        train_ds_simclr,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        generator=torch.Generator().manual_seed(SEED)
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
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

    print(f"Train/Val/Test loaders: {len(train_loader)}/{len(val_loader)}/{len(test_loader)} batches")

    return train_loader, val_loader, test_loader, val_subset_no_transform, num_classes


class LabeledEvalDataset(Dataset):
    def __init__(self, subset, transform):
        self.subset    = subset      # e.g. Subset(ImageFolder, train_indices)
        self.transform = transform   # eval_transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label