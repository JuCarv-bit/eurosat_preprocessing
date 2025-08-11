from simclr.data import eurosat_datasets

import os
import sys
import tempfile
import contextlib
import importlib
from typing import Optional, Tuple

import numpy as np
import torch
from simclr.config import CONFIG


@contextlib.contextmanager
def _temp_cwd():
    prev = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="eurosat_tests_") as td:
        os.chdir(td)
        try:
            yield td
        finally:
            os.chdir(prev)


def _resolve_paths() -> Optional[Tuple[str, str]]:
    ms_path = CONFIG.get("DATA_DIR_EUROSAT_MS") 
    rgb_path = CONFIG.get("DATA_DIR_EUROSAT_RGB")
    if not ms_path or not rgb_path or not os.path.isdir(ms_path) or not os.path.isdir(rgb_path):
        print("EUROSAT_MS and/or EUROSAT_RGB not set or not found; skipping tests.")
        return None
    return ms_path, rgb_path


def _patch_fast_mean_std(M):
    # Replace heavy mean/std with a fast subset-based computation (no disk writes).
    def fast_mean_std(train_df, rgb_root, batch_size):
        from PIL import Image
        from torchvision import transforms

        files = train_df["filepath"].tolist()
        k = min(len(files), 128)  # small subset for speed
        xs = []
        to_tensor = transforms.ToTensor()
        for fp in files[:k]:
            img = Image.open(os.path.join(rgb_root, fp)).convert("RGB")
            xs.append(to_tensor(img))
        x = torch.stack(xs, 0)  # [k,3,H,W], float in [0,1]
        mean = x.mean(dim=(0, 2, 3)).tolist()
        std = x.std(dim=(0, 2, 3), unbiased=False).tolist()
        return mean, std

    M._read_or_compute_mean_std = fast_mean_std


def _check_two_view_batch_simclr(batch):
    x1, x2, y = batch
    assert isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor)
    assert x1.shape == x2.shape and x1.dim() == 4
    assert isinstance(y, torch.Tensor) and y.dim() == 1 and y.shape[0] == x1.shape[0]


def _check_two_view_batch_yaware_aug(batch):
    x1, x2, meta, y = batch
    assert isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor)
    assert x1.shape == x2.shape and x1.dim() == 4
    assert isinstance(meta, torch.Tensor) and meta.shape == (x1.shape[0], 2)
    assert isinstance(y, torch.Tensor) and y.shape[0] == x1.shape[0]


def run_tests():
    
    paths = _resolve_paths()
    if paths is None:
        return 0  # skip gracefully
    ms_path, rgb_path = paths

    with _temp_cwd() as td:
        # temporary, non-persistent artifacts
        splits_dir = "/share/homes/carvalhj/projects/eurosat_preprocessing/splits"
        meta_dir = "/share/homes/carvalhj/projects/eurosat_preprocessing/splits_meta"

        # speed patch: fast, in-memory mean/std; keep real images
        _patch_fast_mean_std(eurosat_datasets)

        # ensure workers>0 per design
        eurosat_datasets.CONFIG["NUM_WORKERS"] = max(2, int(eurosat_datasets.CONFIG.get("NUM_WORKERS", 2)))
        eurosat_datasets.CONFIG["EUROSAT_IMAGE_SIZE"] = eurosat_datasets.CONFIG.get("EUROSAT_IMAGE_SIZE", (64, 64))

        # EuroSATBase __getitem__ on real metadata
        df = eurosat_datasets.extract_metadata(ms_path, rgb_path, use_cache=True)

        ds_plain = eurosat_datasets.EuroSATBase(rgb_path, df.iloc[:16], return_meta=False)
        img0, y0 = ds_plain[0]
        assert hasattr(img0, "size") and isinstance(y0, int)

        ds_meta = eurosat_datasets.EuroSATBase(rgb_path, df.iloc[:16], return_meta=True)
        img1, y1, m1 = ds_meta[1]
        assert hasattr(img1, "size") and isinstance(y1, int)
        assert isinstance(m1, torch.Tensor) and m1.shape == (2,), "metadata must be [2] (lat, lon)"

        # SimCLR
        train_aug, eval_aug, train_eval, eval_eval = eurosat_datasets.get_pretrain_loaders(
            ms_path, rgb_path, batch_size=8,
            task="simclr",
            build_eval_loaders=True,
            use_test_as_eval=False,
            splits_dir=splits_dir,
            meta_dir=meta_dir,
            use_cache=True,
            train_frac=eurosat_datasets.CONFIG.get("TRAIN_FRAC", 0.8),
            val_frac=eurosat_datasets.CONFIG.get("VAL_FRAC", 0.1),
            seed=eurosat_datasets.CONFIG.get("SEED", 42),
        )
        _check_two_view_batch_simclr(next(iter(train_aug)))
        _check_two_view_batch_simclr(next(iter(eval_aug)))
        _check_two_view_batch_simclr(next(iter(train_eval)))
        _check_two_view_batch_simclr(next(iter(eval_eval)))

        # YAware
        train_aug, eval_aug, train_eval, eval_eval = eurosat_datasets.get_pretrain_loaders(
            ms_path, rgb_path, batch_size=8,
            task="yaware",
            build_eval_loaders=True,
            use_test_as_eval=False,
            splits_dir=splits_dir,
            meta_dir=meta_dir,
            use_cache=True,
            train_frac=eurosat_datasets.CONFIG.get("TRAIN_FRAC", 0.8),
            val_frac=eurosat_datasets.CONFIG.get("VAL_FRAC", 0.1),
            seed=eurosat_datasets.CONFIG.get("SEED", 42),
        )
        _check_two_view_batch_yaware_aug(next(iter(train_aug)))
        _check_two_view_batch_yaware_aug(next(iter(eval_aug)))
        _check_two_view_batch_simclr(next(iter(train_eval)))  # eval loaders exclude meta by design
        _check_two_view_batch_simclr(next(iter(eval_eval)))

        # capture splits to know test length
        df2, (tr_idx, va_idx, te_idx) = eurosat_datasets.load_or_create_splits(
            ms_path, rgb_path,
            train_frac=eurosat_datasets.CONFIG.get("TRAIN_FRAC", 0.8),
            val_frac=eurosat_datasets.CONFIG.get("VAL_FRAC", 0.1),
            test_frac=1.0 - eurosat_datasets.CONFIG.get("TRAIN_FRAC", 0.8) - eurosat_datasets.CONFIG.get("VAL_FRAC", 0.1),
            seed=eurosat_datasets.CONFIG.get("SEED", 42),
            splits_dir=splits_dir, meta_dir=meta_dir, use_cache=True
        )
        train_aug, eval_aug, tr_ev, ev_ev = eurosat_datasets.get_pretrain_loaders(
            ms_path, rgb_path, batch_size=8,
            task="simclr",
            build_eval_loaders=False,
            use_test_as_eval=True,
            splits_dir=splits_dir,
            meta_dir=meta_dir,
            use_cache=True,
            train_frac=eurosat_datasets.CONFIG.get("TRAIN_FRAC", 0.8),
            val_frac=eurosat_datasets.CONFIG.get("VAL_FRAC", 0.1),
            seed=eurosat_datasets.CONFIG.get("SEED", 42),
        )
        assert len(eval_aug.dataset) == len(te_idx)
        assert tr_ev is None and ev_ev is None

        print("All tests passed.")
        return 0


if __name__ == "__main__":
    sys.exit(run_tests())
