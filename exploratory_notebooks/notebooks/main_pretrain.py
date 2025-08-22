#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Pre-train

import os
import time
import argparse
import yaml
from dotenv import load_dotenv
import torch

from utils.version_utils import print_versions, configure_gpu_device, set_seed
from simclr.config import CONFIG
from simclr.models.simclr import build_simclr_network
from simclr.models.loss import NTXentLoss
from simclr.utils.scheduler import make_optimizer_scheduler
from simclr.data.transforms import get_transforms
from simclr.data.eurosat_datasets import get_pretrain_loaders
from simclr.train_simclr_v2 import train_simclr_v2_function

from yaware.losses import GeneralizedSupervisedNTXenLoss
from yaware.haversine_loss import HaversineRBFNTXenLoss

try:
    import wandb
except Exception:
    wandb = None

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("SimCLR / yAware on EuroSAT")
    p.add_argument("--data_rgb", type=str,
                   default=CONFIG.get("DATA_DIR_EUROSAT_RGB", "/users/c/carvalhj/datasets/eurosat/EuroSAT_RGB/"))
    p.add_argument("--data_ms", type=str,
                   default=CONFIG.get("DATA_DIR_EUROSAT_MS", "/users/c/carvalhj/datasets/eurosat/EuroSAT_MS/"))
    p.add_argument("--use_test_as_eval", action="store_true", default=False)
    p.add_argument("--model", type=str, default="resnet18",
                   choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
    p.add_argument("--feature_dim", type=int, default=CONFIG.get("FEATURE_DIM", 512))
    p.add_argument("--proj_dim", type=int, default=CONFIG.get("PROJ_DIM", 128))
    p.add_argument("--batch_size", type=int, default=CONFIG.get("BATCH_SIZE", 256))
    p.add_argument("--epochs", type=int, default=CONFIG.get("EPOCHS_SIMCLR", 100))
    p.add_argument("--lr", type=float, default=CONFIG.get("LR", 1e-3))
    p.add_argument("--wd", type=float, default=CONFIG.get("WD", 1e-4))
    p.add_argument("--temperature", type=float, default=CONFIG.get("TEMPERATURE", 0.5))
    p.add_argument("--seed", type=int, default=CONFIG.get("SEED", 42))
    p.add_argument("--num_workers", type=int, default=CONFIG.get("NUM_WORKERS", 8))
    p.add_argument("--yaware", action="store_true", default=CONFIG.get("Y_AWARE", False))
    p.add_argument("--original_yaware", action="store_true", default=CONFIG.get("ORIGINAL_Y_AWARE", False))
    p.add_argument("--gpu", type=int, default=CONFIG.get("TARGET_GPU_INDEX", 0))
    p.add_argument("--cudnn_deterministic", action="store_true", default=False)
    p.add_argument("--disable_cudnn", action="store_true", default=False)
    p.add_argument("--wandb", action="store_true", default=CONFIG.get("ACTIVATE_WANDB", True))
    p.add_argument("--wandb_project", type=str, default="eurosat-contrastive-scratch")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--epoch_save_interval", type=int, default=CONFIG.get("EPOCH_SAVE_INTERVAL", 20))
    p.add_argument("--dataset", type=str, default="eurosat")
    p.add_argument("--perform_eval", action="store_true", default=CONFIG.get("PERFORM_EVAL", True))
    return p


def setup_env_and_device(gpu_index: int, seed: int, disable_cudnn: bool, cudnn_deterministic: bool):
    load_dotenv()
    print_versions()
    set_seed(seed)
    device = configure_gpu_device(gpu_index)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_index}")
    else:
        device = torch.device("cpu")
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.backends.cudnn.enabled = not disable_cudnn
    torch.backends.cudnn.benchmark = not cudnn_deterministic
    torch.backends.cudnn.deterministic = cudnn_deterministic
    return device


def maybe_init_wandb(enable: bool, project: str, name: str, config: dict):
    if not enable:
        return None
    if wandb is None:
        return None
    try:
        wandb.login()
    except Exception:
        return None
    run = wandb.init(project=project, name=name, tags=["SimCLR", "EuroSAT"], config=config)
    return run


def get_loaders(args, task: str, build_eval_loaders: bool):
    loaders = get_pretrain_loaders(
        args.data_ms,
        args.data_rgb,
        batch_size=args.batch_size,
        task=task,
        build_eval_loaders=build_eval_loaders,
        use_test_as_eval=args.use_test_as_eval,
        # num_workers=args.num_workers,
    )
    train_loader_pretrain, val_loader_pretrain, train_loader_eval, val_loader_eval = loaders
    return train_loader_pretrain, val_loader_pretrain, train_loader_eval, val_loader_eval


def build_model(device, args):
    return build_simclr_network(device, args)


def make_loss(args, device):
    if args.yaware:
        if args.original_yaware:
            loss_fn = GeneralizedSupervisedNTXenLoss(
                temperature=args.temperature,
                return_logits=True,
                sigma=0.8,
            ).to(device)
        else:
            loss_fn = HaversineRBFNTXenLoss(
                temperature=args.temperature if args.temperature is not None else 0.9,
                sigma=0.003,
            ).to(device)
    else:
        loss_fn = NTXentLoss(
            batch_size=args.batch_size,
            temperature=args.temperature,
        ).to(device)
    return loss_fn


def train_once(args) -> str:
    device = setup_env_and_device(
        gpu_index=args.gpu,
        seed=args.seed,
        disable_cudnn=args.disable_cudnn,
        cudnn_deterministic=args.cudnn_deterministic,
    )
    task = "yaware" if args.yaware else "simclr"
    train_loader_pretrain, val_loader_pretrain, train_loader_eval, val_loader_eval = get_loaders(args, task, build_eval_loaders=True)
    del val_loader_pretrain, train_loader_eval, val_loader_eval
    simclr_model = build_model(device, args)
    run_name = args.run_name or f"{task.upper()}_BS{args.batch_size}_LR{args.lr:.0e}_SEED{args.seed}_T{args.temperature}_E{args.epochs}"
    wandb_run = maybe_init_wandb(args.wandb, args.wandb_project, run_name, CONFIG)
    if wandb_run is not None:
        wandb_run.log({"model_summary": str(simclr_model)})
    optimizer, scheduler = make_optimizer_scheduler(
        simclr_model.parameters(),
        args.lr,
        args.wd,
        len(train_loader_pretrain),
        args.epochs,
    )
    loss_fn = make_loss(args, device)

    start = time.time()
    print(f"Starting {task} training at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))}")
    num_classes = 10 if args.data_rgb else 1000
    ckpt_path = train_simclr_v2_function(
        simclr_model,
        optimizer,
        loss_fn,
        device,
        simclr_epochs=args.epochs,
        feature_dim=args.feature_dim,
        num_classes=num_classes,
        wandb_run=wandb_run,
        scheduler=scheduler,
        seed=args.seed,
        yaware=args.yaware,
        perform_eval=args.perform_eval,
    )
    end = time.time()
    print(f"Training completed in {end - start:.2f} seconds.")
    if wandb_run is not None:
        wandb_run.log({"training_time_seconds": end - start})
        wandb_run.finish()
    if isinstance(ckpt_path, str):
        cfg_out = os.path.join(os.path.dirname(ckpt_path), "config.yaml")
        with open(cfg_out, "w") as f:
            f.write(f"run_id: {wandb_run.id}\n")
            yaml.dump(CONFIG, f)
        print(f"Saved config to: {cfg_out}")
    else:
        print("[WARN] train_simclr_v2_function did not return a path; config not saved.")
    return ckpt_path if isinstance(ckpt_path, str) else ""


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    _ = train_once(args)
    print("All runs completed.")


if __name__ == "__main__":
    main()
