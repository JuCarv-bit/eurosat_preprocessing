import os
import re
import argparse
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from simclr.config import CONFIG
from simclr.data.eurosat_datasets import get_pretrain_loaders
from new_architecture_simclr.network import resnet18
import json
from torchmetrics import (
    Accuracy, Precision, Recall, F1Score, ConfusionMatrix, MetricCollection
)
from notebooks.main_evaluation import save_metrics
import utils.plot_metrics as plotter

from notebooks.main_pretrain import setup_env_and_device


@torch.no_grad()
def evaluate_full_metrics(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int):
    model.eval()
    metric_kwargs = dict(task="multiclass", num_classes=num_classes)
    metrics = MetricCollection(
        {
            "accuracy": Accuracy(top_k=1, **metric_kwargs),
            **({"accuracy_top5": Accuracy(top_k=5, **metric_kwargs)} if num_classes >= 5 else {}),
            "precision_macro": Precision(average="macro", **metric_kwargs),
            "precision_per_class": Precision(average=None, **metric_kwargs),
            "recall_macro": Recall(average="macro", **metric_kwargs),
            "recall_per_class": Recall(average=None, **metric_kwargs),
            "f1_macro": F1Score(average="macro", **metric_kwargs),
            "f1_per_class": F1Score(average=None, **metric_kwargs),
            "confusion_matrix": ConfusionMatrix(**metric_kwargs),
        }
    ).to(device)
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


def evaluate_from_weights(
    args: argparse.Namespace
):
    device = setup_env_and_device(
        gpu_index=args.gpu,
        seed=args.seed,
        disable_cudnn=args.disable_cudnn,
        cudnn_deterministic=args.cudnn_deterministic,
    )
    weights_uri = args.weights_path
    batch_size = args.batch_size
    yaware = args.yaware
    num_classes = args.num_classes
    dataset = args.dataset

    if os.path.isdir(weights_uri):
        weights_path = os.path.join(weights_uri, "model.pth")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Directory provided but no 'model.pth' inside: {weights_uri}")
        state = torch.load(weights_path, map_location=device)
    elif os.path.isfile(weights_uri):
        weights_path = weights_uri
        state = torch.load(weights_path, map_location=device)
    else:
        raise FileNotFoundError(f"weights_uri not found and not a valid URL: {weights_uri}")

    _, _, _, eval_eval_loader = get_pretrain_loaders(
        CONFIG["DATA_DIR_EUROSAT_MS"],
        CONFIG["DATA_DIR_EUROSAT_RGB"],
        batch_size=batch_size,
        task="yaware" if yaware else "simclr",
        build_eval_loaders=True,
        use_test_as_eval=False,
        splits_dir=CONFIG["SPLITS_DIR"],
        meta_dir=CONFIG["SPLITS_META_DIR"],
        use_cache=True,
        seed=int(CONFIG.get("SEED", 42)),
    )

    default_args = argparse.Namespace(
        num_classes=num_classes,
        feature_dim=args.feature_dim,
        yaware=yaware,
        dataset=dataset,
    )

    model = resnet18(
        args=default_args,
        num_classes=num_classes,
        zero_init_residual=False,
    ).to(device).eval()

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        missing = [k for k in model.state_dict().keys() if k not in state]
        extra = [k for k in state.keys() if k not in model.state_dict()]
        raise RuntimeError(
            f"Failed to load state_dict: {e}\n"
            f"Missing keys (in checkpoint): {len(missing)} e.g. {missing[:5]}\n"
            f"Unexpected keys (in checkpoint): {len(extra)} e.g. {extra[:5]}"
        )

    metrics = evaluate_full_metrics(model, eval_eval_loader, device, num_classes=num_classes)
    # save  metrics  on the same directory as the weights
    class_names = eval_eval_loader.dataset.classes
    return {
        "device": str(device),
        "weights_path": weights_path,
        "metrics": metrics,
        "metrics_jsonable": metrics,
        "class_names": class_names,
    }

def parse_args():
    ap = argparse.ArgumentParser("Supervised ResNet18 on EuroSAT (same loaders/augs)")
    ap.add_argument("--num-classes", type=int, default=10)
    ap.add_argument("--feature-dim", type=int, default=CONFIG.get("FEATURE_DIM", 512))
    ap.add_argument("--seed", type=int, default=CONFIG.get("SEED", 42))
    ap.add_argument("--batch-size", type=int, default=CONFIG.get("BATCH_SIZE", 256))
    ap.add_argument("--yaware", action="store_true", default=False)
    ap.add_argument("--gpu", type=int, default=CONFIG.get("TARGET_GPU_INDEX", 0))
    ap.add_argument("--ckpt-root", type=str, default="models_supervised")
    ap.add_argument("--dataset", type=str, default="eurosat")
    ap.add_argument("--cudnn_deterministic", action="store_true", default=False)
    ap.add_argument("--disable_cudnn", action="store_true", default=False)
    ap.add_argument("--weights-path", type=str, default="/share/homes/carvalhj/projects/eurosat_preprocessing/models_supervised/2025-08-22_18-22-57/model_epoch_200.pth")
    return ap.parse_args()


def main():
    args = parse_args()
    out = evaluate_from_weights(
        args
    )
    weights_path = out["weights_path"]
    metrics = out["metrics"]
    outfile = save_metrics(metrics, weights_path, "evaluation")

    class_names = out["class_names"]
    plotter.main(outfile, "supervised", class_names)

    print(out["metrics"])

if __name__ == "__main__":
    main()