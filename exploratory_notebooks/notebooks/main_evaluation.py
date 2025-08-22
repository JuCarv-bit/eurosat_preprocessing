#!/usr/bin/env python
# coding: utf-8

# Evaluation

from __future__ import annotations
import os
import argparse
from typing import Tuple
import torch
import torch.nn.functional as F
from tqdm import tqdm
from dotenv import load_dotenv
from torch import Tensor
import json
from pathlib import Path
import datetime
from transfer.new_knn import NNClassifier
from transfer.new_logistic import SklearnLogisticClassifier
from torchmetrics import (
    Accuracy,
    AveragePrecision,
    ConfusionMatrix,
    F1Score,
    MetricCollection,
    Precision,
    Recall,
)
import utils.plot_metrics as plotter
try:
    from utils.version_utils import print_versions, configure_gpu_device, set_seed
except Exception:
    def print_versions(): pass
    def configure_gpu_device(_idx: int): 
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def set_seed(seed: int = 42):
        import random, numpy as np
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

from simclr.config import CONFIG
from simclr.models.simclr import build_simclr_network
from simclr.data.eurosat_datasets import get_pretrain_loaders




def _none_or_str(value: str) -> str | None:
    if value == "None":
        return None
    return value

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Feature extraction + KNN on EuroSAT (SimCLR backbone)")
    parser.add_argument("--model",      type=str, default="resnet18",
                        choices=["resnet18","resnet34","resnet50","resnet101","resnet152"])
    parser.add_argument("--n-classes",  type=int, default=10)
    parser.add_argument("--feature-dim",type=int, default=512)
    parser.add_argument("--proj-dim",   type=int, default=CONFIG.get("PROJ_DIM", 128))
    parser.add_argument("--weights", type=str,  default=None, required=False,
                        help="Optional path to pretrained weights (.pth). If not provided, the model starts with random weights.")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save output files", required=False)
    parser.add_argument("--batch-size", type=int, default=CONFIG.get("BATCH_SIZE", 256))
    parser.add_argument("--k",          type=int, default=5, help="k for KNN")
    parser.add_argument("--l2norm",     action="store_true", help="L2-normalize features")
    parser.add_argument("--seed",       type=int, default=CONFIG.get("SEED", 42))
    parser.add_argument("--gpu-index",  type=int, default=CONFIG.get("TARGET_GPU_INDEX", 0))
    parser.add_argument("--data-ms",    type=str, default=CONFIG.get("DATA_DIR_EUROSAT_MS"))
    parser.add_argument("--data-rgb",   type=str, default=CONFIG.get("DATA_DIR_EUROSAT_RGB"))
    parser.add_argument("--use-test-as-eval", action="store_true", default=True)
    parser.add_argument("--cudnn-benchmark", action="store_true", default=True)
    parser.add_argument("--no-cudnn", action="store_true", help="Disable cuDNN (debug)")
    parser.add_argument("--dataset",    type=str, default="eurosat")
    parser.add_argument("--logistic-penalty", type=_none_or_str, default=None, help="Type of penalty (l1, l2, etc.)")
    parser.add_argument("--logistic-c", type=float, default=1.0, help="Inverse regularization strength")
    parser.add_argument("--logistic-solver", type=str, default="lbfgs", help="Solver to use for optimization")
    parser.add_argument("--logistic-tol", type=float, default=1e-4, help="Tolerance for stopping criteria")
    parser.add_argument("--logistic-max-iter", type=int, default=200, help="Maximum number of iterations for optimization")
    return parser.parse_args()


def setup_environment(args: argparse.Namespace) -> torch.device:
    load_dotenv()
    print_versions()
    set_seed(args.seed)

    device = configure_gpu_device(args.gpu_index)

    # CUDA debug/perf toggles
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" if args.no_cudnn else os.environ.get("CUDA_LAUNCH_BLOCKING","0")
    torch.backends.cudnn.enabled = not args.no_cudnn
    torch.backends.cudnn.benchmark = args.cudnn_benchmark and not args.no_cudnn
    return device


def build_model(device: torch.device, args: argparse.Namespace) -> torch.nn.Module:
    model = build_simclr_network(device, args)
    if args.weights is not None:
        if not os.path.exists(args.weights):
            raise FileNotFoundError(f"Weights not found: {args.weights}")
        state = torch.load(args.weights, map_location=device, weights_only=True)
        model.load_state_dict(state)
    else:
        print("No weights provided. The model is initialized with random weights.")

    model.eval()
    return model


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    l2norm: bool = False,
    desc: str = "Extracting features"
) -> Tuple[torch.Tensor, torch.Tensor]:
    feats_list, labels_list = [], []
    for sample in tqdm(dataloader, desc=desc, leave=True):
        images, lbls = sample[0], sample[-1]
        images = images.to(device, non_blocking=True)
        feats, _ = model(images)
        if l2norm:
            feats = F.normalize(feats, dim=-1)
        feats_list.append(feats.detach().to(device))
        labels_list.append(lbls.detach().to(device))
    return torch.cat(feats_list, dim=0), torch.cat(labels_list, dim=0)
    

def get_eval_loaders(args: argparse.Namespace):
    loaders = get_pretrain_loaders(
        args.data_ms,
        args.data_rgb,
        batch_size=args.batch_size,
        task="simclr",
        build_eval_loaders=True,
        use_test_as_eval=args.use_test_as_eval,
    )
    _train_pre, _test_pre, train_eval, test_eval = loaders
    # free pretraining loaders to save memory
    del _train_pre, _test_pre
    return train_eval, test_eval

def evaluate_classification(
    preds: Tensor, targets: Tensor, num_classes: int
) -> dict[str, Tensor]:
    """Evaluate predictions for classification.

    Args:
        preds (Tensor): Predictions from the model.
        targets (Tensor): Ground truth labels.
        num_classes (int): Number of classes for classification.

    Returns:
        dict[str, Tensor]: Dictionary containing evaluation metrics.
    """
    metric_kwargs = dict(task="multiclass", num_classes=num_classes)
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
    return metrics(preds.cpu(), targets.cpu())


def _to_serializable(val):
    """Convert tensors or numpy arrays into Python scalars/lists for JSON."""
    if isinstance(val, torch.Tensor):
        if val.numel() == 1:
            return val.item()
        return val.tolist()
    if isinstance(val, (list, tuple)):
        return [_to_serializable(v) for v in val]
    if isinstance(val, dict):
        return {k: _to_serializable(v) for k, v in val.items()}
    return val

def save_metrics(metrics: dict, weights_path: str | Path, name: str):
    """Save metrics dictionary to a JSON file in the same directory as weights."""
    serializable = _to_serializable(metrics)
    weights_path = Path(weights_path)
    # if it ends in .pth, use the parent directory, else it is the dir itself
    if weights_path.suffix == ".pth":
        outdir = weights_path.parent
    else:
        outdir = weights_path

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = outdir / f"{name}_metrics_{timestamp}.json"
    with open(outfile, "w") as f:
        json.dump(serializable, f, indent=4)
    print(f"Metrics saved to {outfile}")
    return outfile

def main():
    args = parse_args()
    device = setup_environment(args)
    print(f"Device: {device}")
    print(f"Using weights: {args.weights}")

    train_loader_eval, test_loader_eval = get_eval_loaders(args)

    model = build_model(device, args)

    X_train, y_train = extract_features(model, train_loader_eval, device, l2norm=args.l2norm, desc="Train features")
    X_test,  y_test  = extract_features(model, test_loader_eval,  device, l2norm=args.l2norm, desc="Test features")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # KNN evaluation
    k = args.k
    num_classes=args.n_classes
    knn = NNClassifier(num_classes=num_classes, k=k)
    knn.fit(X_train, y_train)
    proba_knn = knn.predict_proba(X_test).cpu()
    metrics_knn = evaluate_classification(proba_knn, y_test, num_classes)
    print(f"KNN Classifier Metrics (k={k}):")
    print(metrics_knn)
    save_dir = args.output_dir if args.weights is None else args.weights
    outfile_knn = save_metrics(metrics_knn, save_dir, "knn")

    # Logistic probe evaluation
    classifier_log = SklearnLogisticClassifier(
            random_state=args.seed,
            penalty=args.logistic_penalty,
            C=args.logistic_c,
            solver=args.logistic_solver,
            tol=args.logistic_tol,
            max_iter=args.logistic_max_iter,
            verbose=1,
        )

    classifier_log.fit(X_train, y_train)

    proba_log = classifier_log.predict_proba(X_test).cpu()
    metrics_log = evaluate_classification(proba_log, y_test, num_classes)
    print("Logistic Regression Classifier Metrics:")
    print(metrics_log)
    outfile_log_prob = save_metrics(metrics_log, save_dir, "logistic")
    
    class_names = train_loader_eval.dataset.classes
    plotter.main(outfile_knn, "knn", class_names)
    plotter.main(outfile_log_prob, "logistic", class_names)

    
if __name__ == "__main__":
    main()