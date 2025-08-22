from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
from tqdm.auto import trange

from simclr.config import CONFIG
from simclr.data.eurosat_datasets import get_pretrain_loaders
from simclr.models.loss import compute_contrastive_accuracy

from transfer.new_knn import NNClassifier
from transfer.new_logistic import SklearnLogisticClassifier
import torch.nn.functional as F


from notebooks.main_evaluation import extract_features, evaluate_classification


INTERVAL_EPOCHS_LINEAR_PROBE = CONFIG["INTERVAL_EPOCHS_LINEAR_PROBE"]
INTERVAL_EPOCHS_KNN = CONFIG["INTERVAL_EPOCHS_KNN"]
INTERVAL_CONTRASTIVE_ACC = CONFIG["INTERVAL_CONTRASTIVE_ACC"]
EPOCH_SAVE_INTERVAL = CONFIG["EPOCH_SAVE_INTERVAL"]
TEMPERATURE = CONFIG["TEMPERATURE"]

@torch.no_grad()
def run_knn_probe_function(
    model: torch.nn.Module,
    train_loader_eval,
    val_loader_eval,
    device: torch.device,
    num_classes: int,
    *,
    k: int = 5,
    l2norm: bool = True,
) -> tuple[float, dict]:
    """
    Extract features on eval loaders and run KNN (NNClassifier).
    Returns (top1_accuracy, full_metrics_dict).
    """
    model.eval()
    X_train, y_train = extract_features(model, train_loader_eval, device, l2norm=l2norm, desc="KNN: train feats")
    X_val,   y_val   = extract_features(model, val_loader_eval,   device, l2norm=l2norm, desc="KNN: val feats")

    clf = NNClassifier(num_classes=num_classes, k=k)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_val).cpu()

    metrics = evaluate_classification(proba, y_val, num_classes)
    top1 = metrics["accuracy"].item() if hasattr(metrics["accuracy"], "item") else float(metrics["accuracy"])
    return top1, metrics


@torch.no_grad()
def run_logistic_probe_function(
    model: torch.nn.Module,
    train_loader_eval,
    val_loader_eval,
    device: torch.device,
    num_classes: int,
    *,
    random_state: int = 42,
    penalty: str | None = None,
    C: float = 1.0,
    solver: str = "lbfgs",
    tol: float = 1e-4,
    max_iter: int = 200,
    l2norm: bool = True,
    verbose: int = 0,
) -> tuple[float, dict]:
    """
    Extract features on eval loaders and run sklearn logistic regression.
    Returns (top1_accuracy, full_metrics_dict).
    """
    model.eval()
    X_train, y_train = extract_features(model, train_loader_eval, device, l2norm=l2norm, desc="LOG: train feats")
    X_val,   y_val   = extract_features(model, val_loader_eval,   device, l2norm=l2norm, desc="LOG: val feats")

    clf = SklearnLogisticClassifier(
        random_state=random_state,
        penalty=penalty,
        C=C,
        solver=solver,
        tol=tol,
        max_iter=max_iter,
        verbose=verbose,
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_val).cpu()

    metrics = evaluate_classification(proba, y_val, num_classes)
    top1 = metrics["accuracy"].item() if hasattr(metrics["accuracy"], "item") else float(metrics["accuracy"])
    return top1, metrics


# Utilities
@dataclass
class Timings:
    stages: Dict[str, float] = field(default_factory=lambda: {
        'load_batch': 0.0,
        'forward': 0.0,
        'loss+backward+opt': 0.0,
        'scheduler': 0.0,
        'val_forward': 0.0,
        'contrastive_acc': 0.0,
        'linear_probe': 0.0,
        'knn': 0.0,
        'checkpoint': 0.0,
        'logging': 0.0,
    })

    def add(self, key: str, dt: float):
        if key in self.stages:
            self.stages[key] += dt
        else:
            self.stages[key] = dt

    def dump(self, epochs: int):
        print("\n=== Timing Breakdown ===")
        for stage, t in self.stages.items():
            per_epoch = t / max(epochs, 1)
            print(f"{stage:15s}: {t:.1f}s ({per_epoch:.1f}s/epoch)")


class AMPManager:
    """Handle autocast and GradScaler consistently."""
    def __init__(self, device: torch.device, enabled: Optional[bool] = None):
        self.device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        self.enabled = (device.type == 'cuda') if enabled is None else enabled
        self.dtype = torch.bfloat16 if (self.enabled and torch.cuda.is_bf16_supported()) else torch.float16
        self.scaler = torch.amp.GradScaler(self.device_type, enabled=self.enabled)

    def autocast(self):
        return torch.amp.autocast(device_type=self.device_type, dtype=self.dtype, enabled=self.enabled)


@dataclass
class EvalResults:
    contrastive_acc_train: float = 0.0
    contrastive_acc_val: float = 0.0
    logistic_acc_train: float = 0.0
    logistic_acc_val: float = 0.0
    knn_acc_val: float = 0.0


class Checkpointer:
    def __init__(self, base_dir: str = "models"):
        self.run_dir = os.path.join(base_dir, time.strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(self.run_dir, exist_ok=True)

    def save(self, model: torch.nn.Module, fname: str) -> str:
        path = os.path.join(self.run_dir, fname)
        torch.save(model.state_dict(), path)
        return path

def build_dataloaders(seed: int, yaware: bool, perform_eval: bool):
    return get_pretrain_loaders(
        CONFIG["DATA_DIR_EUROSAT_MS"],
        CONFIG["DATA_DIR_EUROSAT_RGB"],
        CONFIG["BATCH_SIZE"],
        task="yaware" if yaware else "simclr",
        build_eval_loaders=perform_eval,
        use_test_as_eval=False,
        splits_dir=CONFIG["SPLITS_DIR"],
        meta_dir=CONFIG["SPLITS_META_DIR"],
        use_cache=True,
        seed=seed,
    )

# Core training/eval steps
def training_step(model, batch, device, loss_fn, yaware, amp: AMPManager):
    if yaware:
        x1, x2, meta, *_ = batch
    else:
        x1, x2, *_ = batch
    x1 = x1.to(device, non_blocking=True)
    x2 = x2.to(device, non_blocking=True)
    meta = meta.to(device, non_blocking=True) if yaware else None

    with amp.autocast():
        x_cat = torch.cat([x1, x2], dim=0)
        _, z_cat = model(x_cat)
        z1, z2 = torch.chunk(z_cat, 2, dim=0)
        if yaware:
            loss = loss_fn(z1, z2, meta)
            loss_val = loss[0] if isinstance(loss, tuple) else loss
        else:
            loss_val = loss_fn(z1, z2)
    return loss_val


def evaluate_loss(model, loader, device, loss_fn, yaware) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            if yaware:
                v1, v2, meta, *_ = batch
            else:
                v1, v2, *_ = batch
            v1, v2 = v1.to(device), v2.to(device)
            meta = meta.to(device) if yaware else None 
            v_cat = torch.cat([v1, v2], dim=0)
            _, zv_cat = model(v_cat)
            zv1, zv2 = torch.chunk(zv_cat, 2, dim=0)
            if yaware:
                res = loss_fn(zv1, zv2, meta)
                l = res[0] if isinstance(res, tuple) else res
            else:
                l = loss_fn(zv1, zv2)
            bs = v1.size(0)
            total += l.item() * bs
            n += bs
    return total / max(n, 1)


def maybe_run_evaluations(
    epoch: int,
    model: torch.nn.Module,
    device: torch.device,
    *,
    # loaders
    train_aug_loader,          # for contrastive acc (train)
    eval_aug_loader,           # for contrastive acc (val)
    train_eval_loader,         # for probes: feature extraction (train split, eval transforms)
    eval_eval_loader,          # for probes: feature extraction (val/test split, eval transforms)
    # config
    num_classes: int,
    seed: int,
    timings: Timings,
    yaware: bool
) -> EvalResults:
    """
    Run periodic evaluations:
      - Contrastive accuracy (aug loaders)
      - Logistic probe (new pipeline, eval loaders)
      - kNN (new pipeline, eval loaders)
    Returns aggregated EvalResults with top-1 accuracies.
    """
    results = EvalResults()


    if epoch % INTERVAL_CONTRASTIVE_ACC == 0:
        t0 = time.perf_counter()
        results.contrastive_acc_val = compute_contrastive_accuracy(
            model, eval_aug_loader, device, yaware=yaware
        )
        results.contrastive_acc_train = compute_contrastive_accuracy(
            model, train_aug_loader, device, yaware=yaware
        )
        timings.add("contrastive_acc", time.perf_counter() - t0)        

    if epoch % INTERVAL_EPOCHS_LINEAR_PROBE == 0:
        t0 = time.perf_counter()
        # val-on-val
        results.logistic_acc_val, _ = run_logistic_probe_function(
            model,
            train_eval_loader,
            eval_eval_loader,
            device,
            num_classes=num_classes,
            random_state=seed,
            penalty=CONFIG.get("LOGISTIC_PENALTY", None),
            C=CONFIG.get("C_LIN_PROBE", 1.0),
            solver=CONFIG.get("LOGISTIC_SOLVER", "lbfgs"),
            tol=CONFIG.get("LOGISTIC_TOL", 1e-4),
            max_iter=CONFIG.get("MAX_ITER_LIN_PROBE", 200),
            l2norm=True,
            verbose=0,
        )
        results.logistic_acc_train, _ = run_logistic_probe_function(
            model,
            train_eval_loader,
            train_eval_loader,
            device,
            num_classes=num_classes,
            random_state=seed,
            penalty=CONFIG.get("LOGISTIC_PENALTY", None),
            C=CONFIG.get("C_LIN_PROBE", 1.0),
            solver=CONFIG.get("LOGISTIC_SOLVER", "lbfgs"),
            tol=CONFIG.get("LOGISTIC_TOL", 1e-4),
            max_iter=CONFIG.get("MAX_ITER_LIN_PROBE", 200),
            l2norm=True,
            verbose=0,
        )
        timings.add("linear_probe", time.perf_counter() - t0)

    if epoch % INTERVAL_EPOCHS_KNN == 0:
        t0 = time.perf_counter()
        results.knn_acc_val, _ = run_knn_probe_function(
            model,
            train_eval_loader,
            eval_eval_loader,
            device,
            num_classes=num_classes,
            k=CONFIG.get("K", 5),
            l2norm=True,
        )
        timings.add("knn", time.perf_counter() - t0)

    return results


# Public API
def train_simclr_v2_function(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device: torch.device,
    simclr_epochs: int,
    feature_dim: int,
    num_classes: int,
    wandb_run=None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    seed: int = CONFIG["SEED"],
    yaware: bool = False,
    perform_eval: bool = True,
):
    """Train SimCLR v2 with modular components, preserving your original behavior."""
    # Data
    train_aug_loader, eval_aug_loader, train_eval_loader, eval_eval_loader = build_dataloaders(seed, yaware, perform_eval)

    # Setup
    model.to(device)
    model.train()
    amp = AMPManager(device)
    timings = Timings()

    # Filenames
    bs = CONFIG["BATCH_SIZE"]
    lr = CONFIG["LR"]
    lr_str = f"{lr:.0e}" if lr < 0.0001 else f"{lr:.6f}"
    model_base_filename = f"simclr_seed{seed}_bs{bs}_temp{TEMPERATURE}_Tepochs{simclr_epochs}_lr{lr_str}"

    checkpointer = Checkpointer(base_dir="models")

    # Training loop
    for epoch in trange(1, simclr_epochs + 1, desc="Epochs"):
        model.train()
        total_loss = 0.0
        n_samples = 0

        end = time.perf_counter()
        data_time = compute_time = 0.0

        for batch in train_aug_loader:
            data_time += time.perf_counter() - end

            # forward + loss
            t0 = time.perf_counter()
            loss_val = training_step(model, batch, device, loss_fn, yaware, amp)

            optimizer.zero_grad(set_to_none=True)
            amp.scaler.scale(loss_val).backward()
            amp.scaler.unscale_(optimizer)
            amp.scaler.step(optimizer)
            amp.scaler.update()

            if scheduler is not None:
                t_s = time.perf_counter()
                scheduler.step()
                timings.add('scheduler', time.perf_counter() - t_s)

            compute_time += time.perf_counter() - t0

            # accounting
            bs_cur = batch[0].size(0)
            total_loss += float(loss_val.detach().item()) * bs_cur
            n_samples += bs_cur

            end = time.perf_counter()

        train_loss = total_loss / max(n_samples, 1)

        # Validation loss
        val_loss = evaluate_loss(model, eval_aug_loader, device, loss_fn, yaware)

        # Interval evaluations
        eval_res = EvalResults()
        if perform_eval:
            eval_res = maybe_run_evaluations(
                epoch,
                model,
                device,
                train_aug_loader=train_aug_loader,
                eval_aug_loader=eval_aug_loader,
                train_eval_loader=train_eval_loader,
                eval_eval_loader=eval_eval_loader,
                num_classes=num_classes,
                seed=seed,
                timings=timings,
                yaware=yaware
            )

        # Logging (stdout)
        if perform_eval and (epoch % max(1, min(INTERVAL_CONTRASTIVE_ACC, INTERVAL_EPOCHS_LINEAR_PROBE, INTERVAL_EPOCHS_KNN)) == 0):
            msg = (
                f"Epoch {epoch:02d}/{simclr_epochs} | "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | "
                f"Logistic Probe Acc (Val): {eval_res.logistic_acc_val:.3f}, "
                f"Logistic Probe Acc (Train): {eval_res.logistic_acc_train:.3f} | "
                f"Contrastive Acc (Train): {eval_res.contrastive_acc_train:.3f}, "
                f"Contrastive Acc (Val): {eval_res.contrastive_acc_val:.3f} | "
                f"KNN Acc (Val): {eval_res.knn_acc_val:.3f}"
            )
            print("\n" + msg)

        if wandb_run is not None:
            t0 = time.perf_counter()
            log_payload = {
                "epoch": epoch,
                "simclr_train_loss": train_loss,
                "simclr_val_loss": val_loss,
            }
            if perform_eval:
                log_payload.update({
                    "logistic_probe_acc": eval_res.logistic_acc_val,
                    "logistic_probe_train_acc": eval_res.logistic_acc_train,
                    "contrastive_val_acc": eval_res.contrastive_acc_val,
                    "contrastive_train_acc": eval_res.contrastive_acc_train,
                    "knn_val_acc": eval_res.knn_acc_val,
                })
            wandb_run.log(log_payload)
            timings.add('logging', time.perf_counter() - t0)

        # Checkpoint
        if epoch % EPOCH_SAVE_INTERVAL == 0:
            t0 = time.perf_counter()
            _ = checkpointer.save(model, f"{model_base_filename}_epoch_{epoch:03d}.pth")
            timings.add('checkpoint', time.perf_counter() - t0)

        print(f"===\ndata: {data_time:.2f}s, compute: {compute_time:.2f}s")

    # Final save
    final_ckpt = checkpointer.save(model, f"{model_base_filename}_epoch_{simclr_epochs:03d}.pth")

    # Final evaluations
    with torch.no_grad():
        final_contrastive_train = compute_contrastive_accuracy(model, train_aug_loader, device, yaware=yaware)
        print(f"Final contrastive accuracy on train split: {final_contrastive_train*100:.2f}%")
        final_contrastive_val = compute_contrastive_accuracy(model, eval_aug_loader, device, yaware=yaware)
        print(f"Final contrastive accuracy on val split: {final_contrastive_val*100:.2f}%")
        knn_train_acc, final_knn_acc, knn_train_acc_k1, final_knn_acc_k1 = get_knn_metrics(
            model,
            device,
            train_eval_loader,   # single-view eval transform
            eval_eval_loader,    # single-view eval transform
            num_classes=num_classes,
            k=CONFIG.get("K", 5),
            l2norm=True,
        )

    # Artifact logging
    if wandb_run is not None:
        t0 = time.perf_counter()
        try:
            import wandb  # type: ignore
            artifact_name = get_model_name()
            artifact = wandb.Artifact(name=artifact_name, type="model", description="SimCLR model trained on Eurosat dataset")
            artifact.add_file(final_ckpt)
            wandb_run.log_artifact(artifact)
        except Exception as e:
            print(f"wandb artifact upload failed: {e}")
        finally:
            wandb_run.log({
                "final_contrastive_accuracy": final_contrastive_val,
                "final_contrastive_accuracy_train": final_contrastive_train,
                "final_knn_acc": final_knn_acc,
                "final_knn_acc_k1": final_knn_acc_k1,
                "final_knn_train_acc": knn_train_acc,
                "final_knn_train_acc_k1": knn_train_acc_k1,
            })
            timings.add('logging', time.perf_counter() - t0)

    timings.dump(simclr_epochs)
    return final_ckpt


def get_model_name():
    if CONFIG.get("ORIGINAL_Y_AWARE", False):
        return "original_yaware"
    if CONFIG.get("Y_AWARE", False):
        return "yaware"
    return "simclr"


@torch.no_grad()
def get_knn_metrics(
    model: torch.nn.Module,
    device: torch.device,
    train_eval_loader,
    eval_eval_loader,
    num_classes: int,
    k: int | None = None,
    l2norm: bool = True,
):
    """
    New KNN metrics using the NNClassifier (feature extraction -> predict_proba).
    Returns: (knn_train_acc@k, knn_val_acc@k, knn_train_acc@1, knn_val_acc@1)
    """
    model.eval()
    if k is None:
        k = CONFIG.get("K", 5)

    # 1) Extract features once
    X_train, y_train = extract_features(model, train_eval_loader, device, l2norm=l2norm, desc="KNN(final): train feats")
    X_val,   y_val   = extract_features(model, eval_eval_loader,  device, l2norm=l2norm, desc="KNN(final): val feats")

    def top1_from_metrics(metrics: dict) -> float:
        acc = metrics["accuracy"]
        return acc.item() if hasattr(acc, "item") else float(acc)

    # 2) Evaluate at k
    knn_k = NNClassifier(num_classes=num_classes, k=k)
    knn_k.fit(X_train, y_train)
    proba_train_k = knn_k.predict_proba(X_train).cpu()
    proba_val_k   = knn_k.predict_proba(X_val).cpu()
    m_train_k = evaluate_classification(proba_train_k, y_train, num_classes)
    m_val_k   = evaluate_classification(proba_val_k,   y_val,   num_classes)
    knn_train_acc_k  = top1_from_metrics(m_train_k)
    knn_val_acc_k    = top1_from_metrics(m_val_k)

    # 3) Evaluate at k=1
    knn_1 = NNClassifier(num_classes=num_classes, k=1)
    knn_1.fit(X_train, y_train)
    proba_train_1 = knn_1.predict_proba(X_train).cpu()
    proba_val_1   = knn_1.predict_proba(X_val).cpu()
    m_train_1 = evaluate_classification(proba_train_1, y_train, num_classes)
    m_val_1   = evaluate_classification(proba_val_1,   y_val,   num_classes)
    knn_train_acc_1 = top1_from_metrics(m_train_1)
    knn_val_acc_1   = top1_from_metrics(m_val_1)

    print(f"Final kNN (k={k}) on train: {knn_train_acc_k*100:.2f}%")
    print(f"Final kNN (k={k}) on val  : {knn_val_acc_k*100:.2f}%")
    print(f"Final kNN (k=1) on train  : {knn_train_acc_1*100:.2f}%")
    print(f"Final kNN (k=1) on val    : {knn_val_acc_1*100:.2f}%")

    return knn_train_acc_k, knn_val_acc_k, knn_train_acc_1, knn_val_acc_1
