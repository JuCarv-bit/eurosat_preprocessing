#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import time
import json
import argparse
from typing import Optional, Iterable, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import trange
from torchmetrics import (
    Accuracy, Precision, Recall, F1Score, ConfusionMatrix, MetricCollection
)

from simclr.config import CONFIG
from simclr.data.eurosat_datasets import get_pretrain_loaders
from simclr.utils.scheduler import make_optimizer_scheduler
from new_architecture_simclr.network import resnet18


# --------------- AMP ---------------

class AMPManager:
    def __init__(self, device: torch.device, enabled: Optional[bool] = None):
        self.device_type = "cuda" if device.type == "cuda" else "cpu"
        self.enabled = (device.type == "cuda") if enabled is None else enabled
        self.dtype = torch.bfloat16 if (self.enabled and torch.cuda.is_bf16_supported()) else torch.float16
        self.scaler = torch.amp.GradScaler(self.device_type, enabled=self.enabled)

    def autocast(self):
        return torch.amp.autocast(device_type=self.device_type, dtype=self.dtype, enabled=self.enabled)


# --------------- Data adapters ---------------

def iter_single_view_train(loader: DataLoader, device: torch.device, yaware: bool = False):
    for batch in loader:
        if yaware:
            x1, x2, meta, y = batch
        else:
            x1, x2, y = batch
        yield x1.to(device, non_blocking=True), y.to(device, non_blocking=True)

def iter_single_view_eval(loader: DataLoader, device: torch.device):
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                x, y = batch
            elif len(batch) >= 3:
                x, y = batch[0], batch[-1]
            else:
                raise ValueError(f"Unexpected eval batch length: {len(batch)}")
        else:
            raise ValueError(f"Unexpected eval batch type: {type(batch)}")
        yield x.to(device, non_blocking=True), y.to(device, non_blocking=True)


# --------------- Train / Eval steps ---------------

def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iterator: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    amp: AMPManager,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_correct, n = 0.0, 0, 0
    for x, y in iterator:
        with amp.autocast():
            logits = model(x)
            loss = loss_fn(logits, y)
        optimizer.zero_grad(set_to_none=True)
        amp.scaler.scale(loss).backward()
        amp.scaler.unscale_(optimizer)
        amp.scaler.step(optimizer)
        amp.scaler.update()
        if scheduler is not None:
            scheduler.step()
        bs = y.size(0)
        total_loss += float(loss.detach().item()) * bs
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        n += bs
    return total_loss / max(n, 1), total_correct / max(n, 1)

@torch.no_grad()
def evaluate_loss_acc(model: nn.Module, iterator: Iterable[Tuple[torch.Tensor, torch.Tensor]]):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_correct, n = 0.0, 0, 0
    for x, y in iterator:
        logits = model(x)
        loss = loss_fn(logits, y)
        bs = y.size(0)
        total_loss += float(loss.detach().item()) * bs
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        n += bs
    return total_loss / max(n, 1), total_correct / max(n, 1)

@torch.no_grad()
def evaluate_full_metrics(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int):
    model.eval()
    metric_kwargs = dict(task="multiclass", num_classes=num_classes)
    # metrics = MetricCollection({
    #     "accuracy_macro": Accuracy(average="macro", **metric_kwargs),
    #     "accuracy_micro": Accuracy(average="micro", **metric_kwargs),
    #     "precision_macro": Precision(average="macro", **metric_kwargs),
    #     "precision_per_class": Precision(average=None, **metric_kwargs),
    #     "recall_macro": Recall(average="macro", **metric_kwargs),
    #     "recall_per_class": Recall(average=None, **metric_kwargs),
    #     "f1_macro": F1Score(average="macro", **metric_kwargs),
    #     "f1_per_class": F1Score(average=None, **metric_kwargs),
    #     "confusion_matrix": ConfusionMatrix(**metric_kwargs),
    # }).to(device)

    metrics = MetricCollection(
        {
            "accuracy": Accuracy(top_k=1, **metric_kwargs),
            "accuracy_top5": Accuracy(top_k=5, **metric_kwargs),
            "precision_macro": Precision(average="macro", **metric_kwargs),
            "precision_per_class": Precision(average=None, **metric_kwargs),
            "recall_macro": Recall(average="macro", **metric_kwargs),
            "recall_per_class": Recall(average=None, **metric_kwargs),
            "f1_macro": F1Score(average="macro", **metric_kwargs),
            "f1_per_class": F1Score(average=None, **metric_kwargs),
            "confusion_matrix": ConfusionMatrix(**metric_kwargs),
        }
    )

    logits_list, targets_list = [], []
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                x, y = batch
            elif len(batch) >= 3:
                x, y = batch[0], batch[-1]
            else:
                raise ValueError(f"Unexpected eval batch length: {len(batch)}")
        else:
            raise ValueError(f"Unexpected eval batch type: {type(batch)}")
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        logits_list.append(logits)
        targets_list.append(y)

    preds = torch.cat(logits_list, dim=0).softmax(dim=1)
    targets = torch.cat(targets_list, dim=0)
    out = metrics(preds, targets)
    return {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in out.items()}


# --------------- Saving / Loading ---------------

def make_run_dir(base: str = "models_supervised") -> str:
    run_dir = os.path.join(base, time.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def save_weights(model: nn.Module, run_dir: str, filename: str = "model.pth") -> str:
    path = os.path.join(run_dir, filename)
    torch.save(model.state_dict(), path)
    return path

def load_model_from_run(run_dir: str, model: nn.Module, device: torch.device) -> nn.Module:
    path = os.path.join(run_dir, "model.pth")
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


# --------------- Trainer ---------------

def train_supervised_resnet18(
    device: torch.device,
    *,
    num_classes: int,
    feature_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    batch_size: int,
    yaware: bool = False,
    ckpt_root: str = "models_supervised",
    args
):
    torch.manual_seed(seed)

    train_aug_loader, eval_aug_loader, train_eval_loader, eval_eval_loader = get_pretrain_loaders(
        CONFIG["DATA_DIR_EUROSAT_MS"],
        CONFIG["DATA_DIR_EUROSAT_RGB"],
        batch_size=batch_size,
        task="yaware" if yaware else "simclr",
        build_eval_loaders=True,
        use_test_as_eval=False,
        splits_dir=CONFIG["SPLITS_DIR"],
        meta_dir=CONFIG["SPLITS_META_DIR"],
        use_cache=True,
        seed=seed,
    )

    model = resnet18(args=args,
                       num_classes=args.num_classes,
                       zero_init_residual=False
                     ).to(device)
    optimizer, scheduler = make_optimizer_scheduler(
        model.parameters(), lr, weight_decay, len(train_aug_loader), epochs
    )
    amp = AMPManager(device)

    run_dir = make_run_dir(ckpt_root)

    SAVE_EPOCHS_INTERVAL = CONFIG.get("EPOCH_SAVE_INTERVAL", 10)
    for epoch in trange(1, epochs + 1, desc="Supervised Epochs"):
        train_iter = iter_single_view_train(train_aug_loader, device, yaware=yaware)
        train_loss, train_acc = train_one_epoch(model, optimizer, train_iter, amp, scheduler=scheduler)

        val_iter = iter_single_view_eval(eval_eval_loader, device)
        val_loss, val_acc = evaluate_loss_acc(model, val_iter)

        print(
            f"Epoch {epoch:03d}: "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f}"
        )
        if epoch % SAVE_EPOCHS_INTERVAL == 0:
            save_weights(model, run_dir, f"model_epoch_{epoch}.pth")

    weights_path = save_weights(model, run_dir, "model.pth")

    metrics_val = evaluate_full_metrics(model, eval_eval_loader, device, num_classes=num_classes)
    metrics_json = {k: (v.tolist() if hasattr(v, "tolist") else float(v)) for k, v in metrics_val.items()}
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics_json, f, indent=2)

    print(f"Saved to {run_dir}")
    return {"run_dir": run_dir, "weights": weights_path, "val_metrics": metrics_val}

# --------------- CLI ---------------

def parse_args():
    ap = argparse.ArgumentParser("Supervised ResNet18 on EuroSAT (same loaders/augs)")
    ap.add_argument("--num-classes", type=int, default=10)
    ap.add_argument("--feature-dim", type=int, default=CONFIG.get("FEATURE_DIM", 512))
    ap.add_argument("--epochs", type=int, default=CONFIG.get("EPOCHS_SUPERVISED", CONFIG.get("EPOCHS_SIMCLR", 100)))
    ap.add_argument("--lr", type=float, default=CONFIG.get("LR", 1e-3))
    ap.add_argument("--wd", type=float, default=CONFIG.get("WD", 1e-4))
    ap.add_argument("--seed", type=int, default=CONFIG.get("SEED", 42))
    ap.add_argument("--batch-size", type=int, default=CONFIG.get("BATCH_SIZE", 256))
    ap.add_argument("--yaware", action="store_true", default=False)
    ap.add_argument("--gpu", type=int, default=CONFIG.get("TARGET_GPU_INDEX", 0))
    ap.add_argument("--ckpt-root", type=str, default="models_supervised")
    ap.add_argument("--dataset", type=str, default="eurosat")
    return ap.parse_args()

def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out = train_supervised_resnet18(
        device,
        num_classes=args.num_classes,
        feature_dim=args.feature_dim,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.wd,
        seed=args.seed,
        batch_size=args.batch_size,
        yaware=args.yaware,
        ckpt_root=args.ckpt_root,
        args=args
    )
    print("Training complete. Results:")
    for key, value in out.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
