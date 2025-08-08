import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import rasterio
from rasterio.transform import xy
from pyproj import Transformer
# import get_split_indexes from simclr.data.datamodule import get_split_indexes
from simclr.data.datamodule import get_split_indexes, compute_mean_std, get_transforms, TwoCropsTransform, SimCLRDataset
from tqdm import tqdm
from functools import lru_cache

CONFIG = {
    "LOCAL_OR_COLAB": "LOCAL",
    "DATA_DIR_LOCAL": "/share/DEEPLEARNING/carvalhj/EuroSAT_RGB/",
    "DATA_DIR_COLAB": "/content/EuroSAT_RGB",
    "ZIP_PATH": "/content/EuroSAT.zip",
    "EUROSAT_URL": "https://madm.dfki.de/files/sentinel/EuroSAT.zip",
    "SEED": 42,  
    "BATCH_SIZE": 256,
    "LR": 3.75e-4,
    "WD": 0.5,
    "LR_LINEAR": 3.75e-4,
    "EPOCHS_SIMCLR": 2,
    "TEMPERATURE": 0.2,
    "PROJ_DIM": 64,
    "MODEL": "Resnet18",  # Options: "ResNet18", "ResNet50"
    "NUM_CLASSES": 10,  # EuroSAT has 10 classes
    "FEATURE_DIM_RESNET50": 2048, # ResNet50 feature dimension = 2048
    "FEATURE_DIM_RESNET18": 512, # ResNet18 feature dimension = 512
    "MEAN":  [0.3441457152366638, 0.3800985515117645, 0.40766361355781555],
    "STD":   [0.09299741685390472, 0.06464490294456482, 0.05413917079567909],
    "NUM_WORKERS": 8,
    "K": 5,
    "EPOCH_SAVE_INTERVAL": 10,
    "TRAIN_FRAC": 0.8,  
    "VAL_FRAC": 0.1,    
    "TEST_FRAC": 0.1,
    "EUROSAT_IMAGE_SIZE": (64, 64),
    "TARGET_GPU_INDEX": 0,
    "PROJ_DIM": 128,
}

CACHE_FILE = "eurosat_metadata_cache.pkl"
@lru_cache(maxsize=1)
def extract_metadata(ms_path, rgb_path, use_cache=True):
    if use_cache and os.path.exists(CACHE_FILE):
        print(f"Loading metadata from cache ({CACHE_FILE})")
        return pd.read_pickle(CACHE_FILE)
    
    tif_recs = []
    classes = sorted([d for d in os.listdir(ms_path) if os.path.isdir(os.path.join(ms_path, d))])
    for class_id, class_name in enumerate(tqdm(classes, desc="Classes", unit="class")):
        class_dir = os.path.join(ms_path, class_name)
        tif_files = [f for f in os.listdir(class_dir) if f.endswith(".tif")]
        for tif in tqdm(tif_files, desc=f"  {class_name}", unit="tif", leave=False):
            bname = tif[:-4]
            tif_path = os.path.join(class_dir, tif)
            with rasterio.open(tif_path) as src:
                r, c = src.height // 2, src.width // 2
                x, y = xy(src.transform, r, c)
                transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                lon, lat = transformer.transform(x, y)
            tif_recs.append({
                "id":         bname,
                "class":      class_name,
                "label_id":   class_id,
                "latitude":   lat,
                "longitude":  lon
            })
    df_tif = pd.DataFrame(tif_recs)

    jpg_recs = []
    for class_name in tqdm(classes, desc="Finding JPGs", unit="class"):
        class_dir = os.path.join(rgb_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        for jpg in os.listdir(class_dir):
            if not jpg.endswith(".jpg"):
                continue
            bname = jpg[:-4]
            rel = os.path.join(class_name, jpg)
            jpg_recs.append({"id": bname, "filepath": rel})
    df_jpg = pd.DataFrame(jpg_recs)

    df = pd.merge(df_tif, df_jpg, on=["id"], how="inner")
    df = df[["filepath", "class", "label_id", "latitude", "longitude"]]
    if use_cache:
        print(f"Saving metadata to cache ({CACHE_FILE})")
        df.to_pickle(CACHE_FILE)
    return df

# 2) custom Dataset that returns image, metadata, label
class EuroSATDataset(Dataset):
    def __init__(self, rgb_root, metadata_df, transform=None):
        """
        rgb_root:   root directory of your JPEGs (EUROSAT_RGB_PATH)
        metadata_df: as returned by extract_metadata
        """
        self.rgb_root = rgb_root
        # reset index so .iloc works cleanly
        self.df = metadata_df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.rgb_root, row["filepath"])
        img  = Image.open(path)
        if self.transform:
            img = self.transform(img)

        meta  = torch.tensor([row["latitude"], row["longitude"]], dtype=torch.float32)
        label = int(row["label_id"])
        return img, meta, label

class SimCLRWithMetaDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, meta, label = self.dataset[idx]
        x1, x2 = self.transform(img)
        return x1, x2, meta

def get_data_loaders(ms_path, rgb_path, batch_size):
    df = extract_metadata(ms_path, rgb_path)
    labels = df["label_id"].values
    total = len(df)
    num_classes = df["label_id"].nunique()
    train_idx, val_idx, test_idx = get_split_indexes(labels, total)

    tensor_ds = EuroSATDataset(rgb_path, df, transform=transforms.ToTensor())
    train_stats = Subset(tensor_ds, train_idx)

    if os.path.exists("models/mean_std.txt"):
        with open("models/mean_std.txt", "r") as f:
            lines = f.readlines() 
            mean_list = lines[0].strip().split(": ")[1][1:-1].split(", ")
            std_list = lines[1].strip().split(": ")[1][1:-1].split(", ")
        mean = [float(m) for m in mean_list]
        std = [float(s) for s in std_list]
    else:
        mean, std = compute_mean_std(train_stats, batch_size, yaware=True)
        os.makedirs("models", exist_ok=True)
        with open("models/mean_std.txt", "w") as f:
            f.write(f"mean: {mean}\n")
            f.write(f"std: {std}\n")
        print("Mean and std saved to models/mean_std.txt")

    eval_tf, aug_tf = get_transforms(mean, std)

    # for SimCLR pre‐training
    train_no_tf        = EuroSATDataset(rgb_path, df.iloc[train_idx], transform=None)
    train_simclr_ds    = SimCLRWithMetaDataset(train_no_tf, TwoCropsTransform(aug_tf))

    # for validation/testing
    val_ds             = EuroSATDataset(rgb_path, df.iloc[val_idx], transform=eval_tf)
    test_ds            = EuroSATDataset(rgb_path, df.iloc[test_idx], transform=eval_tf)

    val_no_transform_ds = EuroSATDataset(rgb_path, df.iloc[val_idx], transform=None)

    g = torch.Generator().manual_seed(CONFIG["SEED"])
    train_loader = DataLoader(
        train_simclr_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=CONFIG["NUM_WORKERS"], generator=g
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=CONFIG["NUM_WORKERS"], generator=g
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=CONFIG["NUM_WORKERS"], generator=g
    )

    print(f"Loaders: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)} batches")
    return train_loader, val_loader, test_loader, val_no_transform_ds, num_classes
