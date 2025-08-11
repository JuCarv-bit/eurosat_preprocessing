#!/usr/bin/env python
# coding: utf-8

from dotenv import load_dotenv
load_dotenv()       # reads .env and sets os.environ
import wandb
wandb.login()

import sys
sys.path.insert(0, "/share/homes/carvalhj/projects/eurosat_preprocessing")
from yaware import information_extraction

import os
import torch
from torchvision.models import resnet18
import time
import argparse


from yaware.haversine_loss import HaversineRBFNTXenLoss
from yaware.losses import GeneralizedSupervisedNTXenLoss
import simclr.data.datamodule as simclr_datamodule
from utils.version_utils import print_versions, configure_gpu_device, set_seed
from simclr.data.transforms import  get_transforms
from simclr.models.loss import NTXentLoss
from simclr.models.simclr import build_simclr_network
from simclr.probes.logistic import get_probe_loaders, run_logistic_probe_experiment
from simclr.utils.scheduler import make_optimizer_scheduler
from simclr.data.mydataloaders import get_data_loaders_train_test_linear_probe
from simclr.train_simclr_v2  import train_simclr_v2_function
from simclr.config import CONFIG
import argparse
from simclr.data.eurosat_datasets import get_pretrain_loaders
import os

print_versions()
set_seed(seed=42)
TARGET_GPU_INDEX = CONFIG["TARGET_GPU_INDEX"] if "TARGET_GPU_INDEX" in CONFIG else 0  # Default to 0 if not set
DEVICE = configure_gpu_device(TARGET_GPU_INDEX)


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# split fractions
TRAIN_FRAC = CONFIG["TRAIN_FRAC"]
VAL_FRAC   = CONFIG["VAL_FRAC"]
TEST_FRAC  = CONFIG["TEST_FRAC"]

SEED = CONFIG["SEED"]

PRETRAINED = False

TEMPERATURE = CONFIG["TEMPERATURE"]

BETAS=(0.9,0.98)
EPS = 1e-8

GLOBAL_SEED = CONFIG["SEED"]
NUM_WORKERS = CONFIG["NUM_WORKERS"]

EUROSAT_IMAGE_SIZE = (64, 64)
MODEL_INPUT_SIZE = [224, 224]
EPOCH_SAVE_INTERVAL = CONFIG["EPOCH_SAVE_INTERVAL"]

MS_PATH  = "/users/c/carvalhj/datasets/eurosat/EuroSAT_MS/"
RGB_PATH = "/users/c/carvalhj/datasets/eurosat/EuroSAT_RGB/"

BATCH_SIZE = CONFIG["BATCH_SIZE"]
YAWARE = CONFIG["Y_AWARE"] if "Y_AWARE" in CONFIG else False


parser = argparse.ArgumentParser("SimCLR EuroSAT")
parser.add_argument("--dataset",    type=str,   default="eurosat",
                    help="dataset name (controls CIFAR‑stem in network.py)")
parser.add_argument("--model",      type=str,   default="resnet18",
                    choices=["resnet18","resnet34","resnet50","resnet101","resnet152"],
                    help="which ResNet depth to use")
parser.add_argument("--n_classes",  type=int,   default=10,
                    help="# of EuroSAT semantic classes")
parser.add_argument("--feature_dim",type=int,   default=512,
                    help="backbone output dim (for SimCLR we set fc→feature_dim)")
parser.add_argument("--proj_dim",   type=int,   default=CONFIG["PROJ_DIM"],
                    help="projection MLP output dim (usually 128)")

args = parser.parse_args([])

print(f"Arguments: {args}")

simclr_model = build_simclr_network(DEVICE, args)


seeds = [GLOBAL_SEED]
for seed in seeds:
    print(f"\n=== Starting run with seed {seed} ===")
    set_seed(seed)
    if YAWARE:
        loaders = information_extraction.get_data_loaders(MS_PATH, RGB_PATH, batch_size=BATCH_SIZE)
    else:
        loaders = simclr_datamodule.get_data_loaders(RGB_PATH, BATCH_SIZE)
    train_loader, val_loader, test_loader, val_subset_no_transform, num_classes = loaders

    wd =  0.5 
    optimizer, scheduler = make_optimizer_scheduler(
        simclr_model.parameters(),
        CONFIG["LR"],
        CONFIG["WD"],
        len(train_loader),
        CONFIG["EPOCHS_SIMCLR"]
        )
    
    bs = CONFIG["BATCH_SIZE"]
    if YAWARE:
        if CONFIG["ORIGINAL_Y_AWARE"]:
            loss_fn = GeneralizedSupervisedNTXenLoss(
                temperature=TEMPERATURE,
                return_logits=True,
                sigma=0.8
            ).to(DEVICE)
        else:
            loss_fn = HaversineRBFNTXenLoss(temperature=0.9, sigma=0.003).to(DEVICE)
    else:
        loss_fn = NTXentLoss(
            batch_size=bs,
            temperature=TEMPERATURE,
        ).to(DEVICE)

    print("Starting SimCLR training...")
    epochs_simclr = CONFIG["EPOCHS_SIMCLR"]
    lr = CONFIG["LR"]
    wandb_run = wandb.init(
        project="eurosat-contrastive-scratch",
        name=f"BS{bs}_LR{lr:.0e}_SEED{seed}_TEMPERATURE{TEMPERATURE}_EPOCHS{epochs_simclr}",
        tags=["SimCLR", "EuroSAT", "Contrastive Learning"],
        config=CONFIG
    )

    wandb.log({"model_summary": str(simclr_model)})

    eval_transform, augment_transform = get_transforms(
        mean = CONFIG["MEAN"],
        std = CONFIG["STD"]
    )  # these must match the transforms used in test_loader


    probe_train_loader, probe_val_loader = get_probe_loaders(
        train_loader,
        val_loader,
        eval_transform,               # must match transforms used in test_loader
        probe_batch_size=CONFIG["BATCH_SIZE"],
        yaware=YAWARE
    )

    start = time.time()
    print(f"Starting SimCLR training at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))}")
    filename_pretrained_weights = train_simclr_v2_function(
        simclr_model,
        optimizer, 
        loss_fn, 
        DEVICE,
        simclr_epochs=CONFIG["EPOCHS_SIMCLR"],
        feature_dim=CONFIG["FEATURE_DIM"],
        num_classes=num_classes,
        wandb_run=wandb_run,
        scheduler=scheduler,
        seed=seed,
        yaware=YAWARE 
    )
    end = time.time()
    print(f"SimCLR training completed in {end - start:.2f} seconds.")
    wandb_run.log({
        "training_time_seconds": end - start,
    })

    wandb_run.finish()

print("All runs completed.")
wandb.finish()


# get the saved model and run linear probe
seed = CONFIG["SEED"]
bs = CONFIG["BATCH_SIZE"]
epochs_simclr = CONFIG["EPOCHS_SIMCLR"]
simclr_lr = CONFIG["LR"]
lr_str = f"{simclr_lr:.0e}" if simclr_lr < 0.0001 else f"{simclr_lr:.6f}"
print(f"Using model path: {filename_pretrained_weights}")

if not os.path.exists(filename_pretrained_weights):
    print(f"Model {filename_pretrained_weights} does not exist. Please run the SimCLR pretraining first.")

state_dict = torch.load(filename_pretrained_weights, map_location=torch.device(DEVICE), weights_only=True)
simclr_model.load_state_dict(state_dict)

# Perform linear probe on train+val as train set, and test as test set
train_loader, test_loader, num_classes = get_data_loaders_train_test_linear_probe(CONFIG["DATA_DIR_EUROSAT_RGB"], CONFIG["BATCH_SIZE"])
_, _, train_loader_eval, test_loader_eval = get_pretrain_loaders(
    CONFIG["DATA_DIR_EUROSAT_MS"],
    CONFIG["DATA_DIR_EUROSAT_RGB"],
    batch_size=CONFIG["BATCH_SIZE"],
    task="simclr",
    build_eval_loaders=True,
    use_test_as_eval=True,
)

run_logistic_probe_experiment(
    CONFIG["SEED"],
    train_loader,
    test_loader,
    num_classes,
    simclr_model,
    bs,
    save_dir=os.path.dirname(filename_pretrained_weights)
)
