#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# # Implementation of the SIMCLR with resnet18 backbone

from dotenv import load_dotenv
load_dotenv()       # reads .env and sets os.environ
import wandb
wandb.login()



# In[2]:


import os
import torch
import torch.nn as nn
from torchvision.models import resnet18
import numpy as np
import time
import argparse


# In[3]:


from yaware import information_extraction
from new_architecture_simclr.network import resnet18, projection_MLP
from yaware.haversine_loss import HaversineRBFNTXenLoss
from yaware.losses import GeneralizedSupervisedNTXenLoss
import simclr.data.datamodule as simclr_datamodule


from simclr.data.datamodule import compute_mean_std, prepare_data, combine_train_val_loaders, SimCLRDataset, get_split_indexes
from utils.version_utils import print_versions, configure_gpu_device, set_seed

from simclr.data.transforms import  get_transforms
from simclr.models.loss import NTXentLoss
from simclr.probes.logistic import get_probe_loaders, run_logistic_probe_experiment
from simclr.utils.scheduler import make_optimizer_scheduler
from simclr.utils.misc import evaluate
from simclr.data.mydataloaders import get_data_loaders_train_test_linear_probe
from simclr.config import CONFIG
from simclr.train import train_simclr




# In[4]:


print_versions()
set_seed(seed=42)
TARGET_GPU_INDEX = CONFIG["TARGET_GPU_INDEX"] if "TARGET_GPU_INDEX" in CONFIG else 0  # Default to 0 if not set
DEVICE = configure_gpu_device(TARGET_GPU_INDEX)


# In[5]:


# Prevent nondeterminism
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
print(f"Y_AWARE: {YAWARE}")


# In[6]:


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
parser.add_argument("--proj_dim",   type=int,   default=CONFIG["PROJ_   DIM"],
                    help="projection MLP output dim (usually 128)")

args = parser.parse_args([])

def build_model(DEVICE, args):
    base_encoder = resnet18(
        args,
        num_classes=args.feature_dim,     # make fc output = feature_dim
        zero_init_residual=False
    )
    proj_head = projection_MLP(args)

    class SimCLRModel(nn.Module):
        def __init__(self, base_encoder, proj_head):
            super().__init__()
            self.encoder = base_encoder
            self.encoder.fc = nn.Identity()
            self.projection_head = proj_head

        def forward(self, x):
            feat = self.encoder(x)
            proj = self.projection_head(feat)
            return feat, proj
    
    simclr_model = SimCLRModel(base_encoder, proj_head).to(DEVICE)
    return simclr_model

def make_training(batch_size, temperature, sigma):
    BATCH_SIZE = batch_size
    TEMPERATURE = temperature
    SIGMA = sigma

    simclr_model = build_model(DEVICE, args)
    seed = GLOBAL_SEED
    
    print(f"\n=== Starting run with seed {seed} ===")
    set_seed(seed)
    if YAWARE:
        loaders = information_extraction.get_data_loaders(MS_PATH, RGB_PATH, batch_size=BATCH_SIZE)
    else:
        loaders = simclr_datamodule.get_data_loaders(RGB_PATH, BATCH_SIZE)
    train_loader, val_loader, test_loader, val_subset_no_transform, num_classes = loaders

    optimizer, scheduler = make_optimizer_scheduler(
        simclr_model.parameters(),
        CONFIG["LR"],
        CONFIG["WD"],
        len(train_loader),
        CONFIG["EPOCHS_SIMCLR"]
    )

    if YAWARE:
        if CONFIG["ORIGINAL_Y_AWARE"]:
            loss_fn = GeneralizedSupervisedNTXenLoss(
                temperature=TEMPERATURE,
                return_logits=True,
                sigma=SIGMA
            ).to(DEVICE)
        else:
            loss_fn = HaversineRBFNTXenLoss(temperature=TEMPERATURE, sigma=SIGMA).to(DEVICE)
    else:
        loss_fn = NTXentLoss(
            batch_size=BATCH_SIZE,
            temperature=TEMPERATURE,
        ).to(DEVICE)

    print("Starting SimCLR training...")
    epochs_simclr = CONFIG["EPOCHS_SIMCLR"]
    lr = CONFIG["LR"]
    wandb_run = wandb.init(
        project="eurosat-contrastive-scratch",
        name=f"BS{BATCH_SIZE}_LR{lr:.0e}_SEED{seed}_TEMPERATURE{TEMPERATURE}_EPOCHS{epochs_simclr}",
        tags=["SimCLR", "EuroSAT", "Contrastive Learning"],
        config=CONFIG
    )

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
    train_simclr(
        simclr_model,
        train_loader, 
        val_loader,
        probe_train_loader, 
        probe_val_loader,
        optimizer, 
        loss_fn, 
        DEVICE,
        simclr_epochs=CONFIG["EPOCHS_SIMCLR"],
        probe_lr=CONFIG["LR_LINEAR"],
        probe_epochs=1,            # 1 pass per epoch is typical
        feature_dim=CONFIG["FEATURE_DIM"],
        num_classes=num_classes,
        augment_transform=augment_transform,
        val_subset_no_transform=val_subset_no_transform,
        wandb_run=wandb_run,
        scheduler=scheduler,
        seed=seed,
        yaware=YAWARE  # Set to True for Yaware model
    )
    end = time.time()
    print(f"SimCLR training completed in {end - start:.2f} seconds.")
    wandb_run.log({
        "training_time_seconds": end - start,
    })

    wandb_run.finish()

    print("All runs completed.")
    wandb.finish()
    return simclr_model
        



# In[8]:




# In[9]:

def make_linear_probe(simclr_model):
    # get the saved model and run linear probe
    seed = GLOBAL_SEED
    epochs_simclr = CONFIG["EPOCHS_SIMCLR"]
    simclr_lr = CONFIG["LR"]
    lr_str = f"{simclr_lr:.0e}" if simclr_lr < 0.0001 else f"{simclr_lr:.6f}"
    model_path = f"models/simclr_seed{seed}_bs{BATCH_SIZE}_temp{TEMPERATURE}_Tepochs{epochs_simclr}_lr{lr_str}.pth"

    if not os.path.exists(model_path):
        print(f"Model {model_path} does not exist. Please run the SimCLR pretraining first.")

    checkpoint_path = model_path
    state_dict = torch.load(checkpoint_path, map_location=torch.device(DEVICE), weights_only=True)
    simclr_model.load_state_dict(state_dict)

    # Perform linear probe on train+val as train set, and test as test set
    train_loader, test_loader, num_classes = get_data_loaders_train_test_linear_probe(CONFIG["DATA_DIR_LOCAL"], CONFIG["BATCH_SIZE"])
    train_acc, test_acc = run_logistic_probe_experiment(
        42,
        train_loader,
        None,  # No validation loader for linear probe
        test_loader,
        num_classes,
        simclr_model,
        BATCH_SIZE,
    )
    return train_acc, test_acc

    



# In[10]:


# grid search for best hyperparameters

def run_grid_search():
    batch_sizes = [64, 128, 256]
    temperatures = [0.1, 0.2, 0.5]
    sigmas = [0.1, 0.2, 0.5]

    best_model = None
    best_accuracy = 0

    for batch_size in batch_sizes:
        for temperature in temperatures:
            for sigma in sigmas:
                print(f"Running training with batch_size={batch_size}, temperature={temperature}, sigma={sigma}")
                simclr_model = make_training(batch_size, temperature, sigma)
                make_linear_probe(simclr_model)

                # Evaluate the model and get accuracy
                train_acc, test_accuracy = make_linear_probe(simclr_model)  # Assuming evaluate returns accuracy
                print(f"Accuracy: {test_accuracy}")

                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    best_model = simclr_model

    print(f"Best model achieved accuracy: {best_accuracy}")
    return best_model

# run it
if __name__ == "__main__":
    best_model = run_grid_search()
    print("Best model training completed.")
    # Save the best model if needed
    torch.save(best_model.state_dict(), "best_simclr_model.pth")

