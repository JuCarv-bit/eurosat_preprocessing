from simclr.config import CONFIG
from simclr.data.datamodule import LabeledEvalDataset
from torch.utils.data import DataLoader
import torch

import wandb
import joblib
from utils.version_utils import print_versions, configure_gpu_device
from transfer.logistic_regrssion import  SklearnLogisticProbe

from torch.utils.data import DataLoader, Dataset



def get_probe_loaders(train_loader, val_loader, eval_transform, probe_batch_size, yaware=False):
    NUM_WORKERS = CONFIG["NUM_WORKERS"]
    SEED        = CONFIG["SEED"]

    simclr_ds  = train_loader.dataset                 # either SimCLRDataset or SimCLRWithMetaDataset
    raw_subset = getattr(simclr_ds, "dataset", simclr_ds)  # Subset(...) pointing at base Dataset

    class DropMetaDataset(Dataset):
        def __init__(self, ds):
            self.ds = ds
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            item = self.ds[idx]
            # old behaviour: (img, label)
            if len(item) == 2:
                img, label = item
            # new behaviour: (img, meta, label)
            elif len(item) == 3:
                img, _meta, label = item
            else:
                raise RuntimeError(f"Expected 2‑ or 3‑tuple, got {len(item)} elements")
            return img, label

    probe_raw_ds = DropMetaDataset(raw_subset) if yaware else raw_subset

    probe_train_ds = LabeledEvalDataset(probe_raw_ds, eval_transform)
    probe_train_loader = DataLoader(
        probe_train_ds,
        batch_size=probe_batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        generator=torch.Generator().manual_seed(SEED)
    )

    if yaware:
        val_raw_ds = val_loader.dataset  # Subset(EuroSATDataset)
        val_for_probe = DropMetaDataset(val_raw_ds)
        probe_val_loader = DataLoader(
            val_for_probe,
            batch_size=getattr(val_loader, "batch_size", probe_batch_size),
            shuffle=False,
            num_workers=NUM_WORKERS,
            generator=torch.Generator().manual_seed(SEED)
        )
    else:
        probe_val_loader = val_loader

    return probe_train_loader, probe_val_loader



def run_logistic_probe_experiment(
    seed,
    train_loader,
    val_loader,
    test_loader,
    num_classes,
    simclr_model,
    bs
):
    
    TEMPERATURE = CONFIG["TEMPERATURE"]
    TARGET_GPU_INDEX = CONFIG.get("TARGET_GPU_INDEX", 0)  # Default to 0 if not set
    DEVICE = configure_gpu_device(TARGET_GPU_INDEX)
    # 1) prepare wandb & combined loader
    wandb.init(
        project="logistic_probe_eurosat-simclr",
        name=f"logistic_probe_seed{seed}_temperature{TEMPERATURE}_bs{bs}",
        config={
            "seed": seed,
            "temperature": TEMPERATURE,
            "batch_size": bs,
            "num_classes": num_classes,
            "C": 1.0,      # repurpose LR as inverse‐reg strength
            "max_iter": 500
        }
    )

    if val_loader is not None:
        train_val_loader = combine_train_val_loaders(train_loader, val_loader)
    else:
        train_val_loader = train_loader

    print(f"[Data] train+val loader has {len(train_val_loader)} batches")

    # 2) wrap frozen encoder in the sklearn probe
    probe = SklearnLogisticProbe(
        encoder=simclr_model.encoder,
        device=DEVICE,
        scale_features="standard",
        C=wandb.config.C,
        max_iter=wandb.config.max_iter,
        multi_class="multinomial",
        solver="lbfgs"
    )

    # 3) fit on train+val
    print("Fitting logistic regression probe…")
    probe.fit(train_val_loader)

    # 4) evaluate on train+val and test
    train_acc = probe.score(train_val_loader) * 100.0
    test_acc  = probe.score(test_loader)      * 100.0

    print(f"[Probe] Train+Val Acc: {train_acc:.2f}%,  Test Acc: {test_acc:.2f}%")

    wandb.log({
        "probe_trainval_accuracy": train_acc,
        "probe_test_accuracy": test_acc
    })

    # 5) save sklearn classifier (and optionally scaler & encoder weights)
    model_path = f"models/logistic_probe_seed{seed}_bs{bs}.pkl"
    joblib.dump({
        "clf": probe.clf,
        "scaler": probe.scaler,
        "encoder_state_dict": simclr_model.encoder.state_dict(),
        "config": CONFIG,
        "seed": seed
    }, model_path)
    print(f"Saved probe + encoder to {model_path}")

    return train_acc / 100.0, test_acc / 100.0