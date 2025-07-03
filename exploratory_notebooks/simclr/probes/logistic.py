from simclr.config import CONFIG
from simclr.data.datamodule import LabeledEvalDataset
from torch.utils.data import DataLoader
import torch

import wandb
import joblib
from utils.version_utils import print_versions, configure_gpu_device, set_seed


def get_probe_loaders(train_loader, val_loader, eval_transform, probe_batch_size):

    NUM_WORKERS = CONFIG["NUM_WORKERS"]

    simclr_ds   = train_loader.dataset         # SimCLRDataset instance
    raw_subset  = simclr_ds.dataset            # e.g. Subset(ImageFolder, train_indices)

    # labeled Dataset for probe‐training
    probe_train_ds = LabeledEvalDataset(raw_subset, eval_transform)

    SEED = CONFIG["SEED"]

    probe_train_loader = DataLoader(
        probe_train_ds,
        batch_size=probe_batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        generator=torch.Generator().manual_seed(SEED)
    )

    # use existing val_loader as the probe‐validation loader
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