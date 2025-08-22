# eurosat_datasets.py
import os
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Dict, Any
import rasterio
from rasterio.transform import xy
from pyproj import Transformer
from tqdm import tqdm
from functools import lru_cache
from sklearn.model_selection import StratifiedShuffleSplit

from simclr.data.datamodule import compute_mean_std, get_transforms, TwoCropsTransform
from simclr.config import CONFIG


import os
from typing import Tuple, Any
import torch
from torch.utils.data import Dataset
from PIL import Image

CACHE_FILE = "/share/homes/carvalhj/projects/eurosat_preprocessing/exploratory_notebooks/yaware/eurosat_metadata_cache.pkl"

@lru_cache(maxsize=1)
def extract_metadata(ms_path, rgb_path, use_cache=True):
    """
    Build a dataframe mapping RGB JPG filepaths to:
      class, label_id, latitude, longitude.
    Keeps full float precision (repr when writing).
    """
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
                "latitude":   float(lat),
                "longitude":  float(lon),
            })
    df_tif = pd.DataFrame(tif_recs)

    jpg_recs = []
    for class_name in tqdm(classes, desc="Finding JPGs", unit="class"):
        class_dir = os.path.join(rgb_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        for jpg in os.listdir(class_dir):
            if jpg.endswith(".jpg"):
                bname = jpg[:-4]
                rel = os.path.join(class_name, jpg)  # relative path from rgb root
                jpg_recs.append({"id": bname, "filepath": rel})
    df_jpg = pd.DataFrame(jpg_recs)

    df = pd.merge(df_tif, df_jpg, on=["id"], how="inner")
    df = df[["filepath", "class", "label_id", "latitude", "longitude"]].reset_index(drop=True)

    if use_cache:
        print(f"Saving metadata to cache ({CACHE_FILE})")
        df.to_pickle(CACHE_FILE)
    return df

def _stratified_indices(labels, train_frac=CONFIG["TRAIN_FRAC"], val_frac=CONFIG["VAL_FRAC"], test_frac=CONFIG["TEST_FRAC"], seed=CONFIG["SEED"]):
    labels = np.asarray(labels)
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "fractions must sum to 1.0"
    n = len(labels)

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - train_frac), random_state=seed)
    train_idx, temp_idx = next(sss1.split(np.zeros(n), labels))

    temp_labels = labels[temp_idx]
    val_ratio_in_temp = val_frac / (val_frac + test_frac)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - val_ratio_in_temp), random_state=seed)
    val_rel_idx, test_rel_idx = next(sss2.split(np.zeros(len(temp_idx)), temp_labels))
    val_idx = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]
    return train_idx, val_idx, test_idx

def _write_split_txts(df, train_idx, val_idx, test_idx,
                      out_dir="splits", meta_out_dir="splits_meta"):
    """
    Writes paths & aligned metadata with full precision via repr(float).
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(meta_out_dir, exist_ok=True)

    def write_pair(indices, name):
        sub = df.iloc[indices]
        with open(os.path.join(out_dir, f"{name}.txt"), "w") as f_paths, \
             open(os.path.join(meta_out_dir, f"{name}.txt"), "w") as f_meta:
            for p, lat, lon in zip(sub["filepath"], sub["latitude"], sub["longitude"]):
                f_paths.write(f"{p}\n")
                f_meta.write(f"{repr(float(lat))},{repr(float(lon))}\n")

    write_pair(train_idx, "train")
    write_pair(val_idx,   "val")
    write_pair(test_idx,  "test")
    print(f"Wrote split files to '{out_dir}' and metadata to '{meta_out_dir}'")

def _try_load_split_txts(df, splits_dir="splits", meta_dir="splits_meta"):
    """
    Try loading existing split files; validate against df lat/lon (tolerance 1e-5).
    Returns (train_idx, val_idx, test_idx) or None.
    """
    def read_lines(path):
        if not os.path.exists(path): return None
        with open(path, "r") as f: return [ln.strip() for ln in f if ln.strip()]

    expected_splits = ["train", "val", "test"]
    file_to_idx = {p: i for i, p in enumerate(df["filepath"].tolist())}
    loaded = []

    for split in expected_splits:
        fp_txt = os.path.join(splits_dir, f"{split}.txt")
        md_txt = os.path.join(meta_dir,   f"{split}.txt")
        fps = read_lines(fp_txt); mds = read_lines(md_txt)
        if fps is None or mds is None: return None
        if len(fps) != len(mds):
            warnings.warn(f"{split}: count mismatch between {fp_txt} and {md_txt}")
            return None

        idxs = []
        for i, rel in enumerate(fps):
            if rel not in file_to_idx:
                warnings.warn(f"{split}: path not in df: {rel}")
                return None
            idx = file_to_idx[rel]
            try:
                lat_s, lon_s = mds[i].split(",")
                lat = float(lat_s.strip()); lon = float(lon_s.strip())
            except Exception:
                warnings.warn(f"{split}: bad metadata line '{mds[i]}'")
                return None
            drow = df.iloc[idx]
            if not (abs(drow["latitude"] - lat) < 1e-5 and abs(drow["longitude"] - lon) < 1e-5):
                warnings.warn(f"{split}: metadata mismatch for {rel}")
                return None
            idxs.append(idx)
        loaded.append(np.array(idxs, dtype=int))

    return tuple(loaded)

def load_or_create_splits(ms_path, rgb_path,
                          train_frac=0.8, val_frac=0.1, test_frac=0.1,
                          seed=42, splits_dir="splits", meta_dir="splits_meta",
                          use_cache=True):
    """
    1) Build df (filepath, class, label_id, lat, lon).
    2) If split + meta txts exist and are consistent -> read indices from them.
       Else -> compute stratified split and write BOTH sets of txts.
    Returns: df, (train_idx, val_idx, test_idx)
    """
    df = extract_metadata(ms_path, rgb_path, use_cache=use_cache)
    got = _try_load_split_txts(df, splits_dir=splits_dir, meta_dir=meta_dir)
    if got is not None:
        print(f"Loaded existing splits from '{splits_dir}' and '{meta_dir}'.")
        return df, got

    labels = df["label_id"].values
    train_idx, val_idx, test_idx = _stratified_indices(
        labels, train_frac, val_frac, test_frac, seed
    )
    _write_split_txts(df, train_idx, val_idx, test_idx, out_dir=splits_dir, meta_out_dir=meta_dir)
    print("Created new stratified splits and metadata files.")
    return df, (train_idx, val_idx, test_idx)

def _read_or_compute_mean_std(train_df, rgb_path, batch_size):
    """
    Read mean/std if models/mean_std.txt exists, else compute and save.
    """
    os.makedirs("models", exist_ok=True)
    mean_std_txt = "models/mean_std.txt"
    if os.path.exists(mean_std_txt):
        with open(mean_std_txt, "r") as f:
            lines = f.readlines()
        mean = [float(x) for x in lines[0].strip().split(": ")[1][1:-1].split(", ")]
        std  = [float(x) for x in lines[1].strip().split(": ")[1][1:-1].split(", ")]
        print(f"Loaded mean/std from {mean_std_txt}")
        return mean, std

    print("Computing mean/std from training subset...")
    # lightweight dataset just for mean/std
    class _MeanStdDS(Dataset):
        __slots__ = ("rgb_root", "paths", "transform")
        def __init__(self, rgb_root, filepaths, transform):
            self.rgb_root = rgb_root
            self.paths = list(filepaths)
            self.transform = transform
        def __len__(self): return len(self.paths)
        def __getitem__(self, idx):
            img = Image.open(os.path.join(self.rgb_root, self.paths[idx])).convert("RGB")
            return self.transform(img)
    meanstd_ds = _MeanStdDS(rgb_path, train_df["filepath"].tolist(), transforms.ToTensor())
    mean, std = compute_mean_std(meanstd_ds, batch_size, yaware=False)
    with open(mean_std_txt, "w") as f:
        f.write(f"mean: {mean}\n")
        f.write(f"std: {std}\n")
    print(f"Saved mean/std to {mean_std_txt}")
    return mean, std

def _slice_arrays(df_subset):
    paths  = df_subset["filepath"].tolist()
    labels = df_subset["label_id"].to_numpy(dtype=np.int64, copy=True)
    metas  = df_subset[["latitude", "longitude"]].to_numpy(dtype=np.float32, copy=True)
    return paths, labels, metas

class _BasePaths(torch.utils.data.Dataset):
    __slots__ = ("rgb_root", "paths")
    def __init__(self, rgb_root, paths):
        self.rgb_root = rgb_root
        self.paths = list(paths)
    def __len__(self): return len(self.paths)
    def _open(self, idx):
        from PIL import Image
        return Image.open(os.path.join(self.rgb_root, self.paths[idx])).convert("RGB")


class EuroSATBase(Dataset):
    """
    Returns (image, label) by default; if return_meta=True -> (image, label, metadata)
      - image: PIL.Image (transforms applied later)
      - label: int
      - metadata: torch.float32 tensor [2] (lat, lon)
    No two-crops here. `transform` is a callable that takes the returned tuple
    and returns a (possibly transformed) tuple.
    """
    __slots__ = ("rgb_root","paths","labels","metas","return_meta","_transform")

    def __init__(self, rgb_root, metadata_df, *, return_meta: bool = False, transform=None):
        self.rgb_root    = rgb_root
        self.paths       = metadata_df["filepath"].tolist()
        self.labels      = metadata_df["label_id"].to_numpy(dtype=int, copy=True)
        self.metas       = metadata_df[["latitude","longitude"]].to_numpy(dtype="float32", copy=True)
        self.return_meta = bool(return_meta)
        self._transform  = transform  # callable: tuple -> tuple
        self.class_to_idx = metadata_df[['class', 'label_id']].drop_duplicates().set_index('class').to_dict()['label_id']
        sorted_items = sorted(self.class_to_idx.items(), key=lambda item: item[1])
        self.classes = [item[0] for item in sorted_items]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[Any, ...]:
        path  = os.path.join(self.rgb_root, self.paths[index])
        img   = Image.open(path).convert("RGB")
        label = int(self.labels[index])

        out = (img, label) if not self.return_meta else (img, label, torch.from_numpy(self.metas[index]))
        if self._transform is not None:
            out = self._transform(out)
        return out

# get_pretrain_loaders (always num_workers > 0), returns four loaders
import torch
from torch.utils.data import DataLoader
from simclr.config import CONFIG

def get_pretrain_loaders(
    ms_path: str,
    rgb_path: str,
    batch_size: int,
    *,
    task: str = "simclr",              # "simclr" or "yaware"
    build_eval_loaders: bool = True,   # also return (train_eval, eval_eval) using TwoCrops(EVAL)
    use_test_as_eval: bool = False,    # if True: use TEST split instead of VAL as the eval split
    splits_dir: str = "splits",
    meta_dir: str = "splits_meta",
    use_cache: bool = True,
    train_frac: float | None = None,
    val_frac: float | None = None,
    seed: int | None = None,
):
    if train_frac is None: train_frac = CONFIG.get("TRAIN_FRAC", 0.8)
    if val_frac   is None: val_frac   = CONFIG.get("VAL_FRAC", 0.1)
    if seed       is None: seed       = CONFIG.get("SEED", 42)

    # 1) splits
    df, (train_idx, val_idx, test_idx) = load_or_create_splits(
        ms_path, rgb_path,
        train_frac=train_frac, val_frac=val_frac, test_frac=1.0 - train_frac - val_frac,
        seed=seed, splits_dir=splits_dir, meta_dir=meta_dir, use_cache=use_cache
    )
    eval_idx = test_idx if use_test_as_eval else val_idx

    # 2) stats + transforms
    mean, std = _read_or_compute_mean_std(df.iloc[train_idx], rgb_path, batch_size)
    eval_tf, aug_tf = get_transforms(mean, std)
    two_train = TwoCropsTransform(aug_tf)
    two_eval  = TwoCropsTransform(eval_tf)

    # 3) tuple-level wrappers (no dicts)
    #   simclr outputs: (x1, x2, label)
    #   yaware aug/eval: (x1, x2, meta, label)   (meta before label)
    def two_crops_no_meta(two):
        def _t(sample):
            img, label = sample
            x1, x2 = two(img)
            return (x1, x2, label)
        return _t

    def two_crops_with_meta(two):
        def _t(sample):
            img, label, meta = sample
            x1, x2 = two(img)
            return (x1, x2, meta, label)
        return _t

    # 4) datasets
    if task == "yaware":
        train_aug_ds = EuroSATBase(rgb_path, df.iloc[train_idx], return_meta=True,  transform=two_crops_with_meta(two_train))
        eval_aug_ds  = EuroSATBase(rgb_path, df.iloc[eval_idx],  return_meta=True,  transform=two_crops_with_meta(two_train))
        if build_eval_loaders:
            train_eval_ds = EuroSATBase(rgb_path, df.iloc[train_idx], return_meta=False, transform=two_crops_no_meta(two_eval))
            eval_eval_ds  = EuroSATBase(rgb_path, df.iloc[eval_idx],  return_meta=False, transform=two_crops_no_meta(two_eval))
        else:
            train_eval_ds = None; eval_eval_ds = None
    elif task == "simclr":
        train_aug_ds = EuroSATBase(rgb_path, df.iloc[train_idx], return_meta=False, transform=two_crops_no_meta(two_train))
        eval_aug_ds  = EuroSATBase(rgb_path, df.iloc[eval_idx],  return_meta=False, transform=two_crops_no_meta(two_train))
        if build_eval_loaders:
            train_eval_ds = EuroSATBase(rgb_path, df.iloc[train_idx], return_meta=False, transform=two_crops_no_meta(two_eval))
            eval_eval_ds  = EuroSATBase(rgb_path, df.iloc[eval_idx],  return_meta=False, transform=two_crops_no_meta(two_eval))
        else:
            train_eval_ds = None; eval_eval_ds = None
    else:
        raise ValueError(f"Unknown task: {task}")

    # 5) DataLoaders (num_workers > 0 assumed)
    cuda = torch.cuda.is_available()
    nw = int(CONFIG.get("NUM_WORKERS", 8))

    train_aug_loader = DataLoader(
        train_aug_ds, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=nw, pin_memory=cuda,
        persistent_workers=True, prefetch_factor=4,
        generator=torch.Generator().manual_seed(seed)
    )
    eval_aug_loader = DataLoader(
        eval_aug_ds, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=cuda, persistent_workers=True,
        generator=torch.Generator().manual_seed(seed)
    )

    if build_eval_loaders:
        train_eval_loader = DataLoader(
            train_eval_ds, batch_size=batch_size, shuffle=True,
            drop_last=False, num_workers=nw, pin_memory=cuda,
            persistent_workers=True, prefetch_factor=4,
            generator=torch.Generator().manual_seed(seed)
        )
        eval_eval_loader = DataLoader(
            eval_eval_ds, batch_size=batch_size, shuffle=False,
            num_workers=nw, pin_memory=cuda, persistent_workers=True,
            generator=torch.Generator().manual_seed(seed)
        )
    else:
        train_eval_loader, eval_eval_loader = None, None

    return train_aug_loader, eval_aug_loader, train_eval_loader, eval_eval_loader
