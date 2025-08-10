"""Module and entrypoint for evaluation."""

import json

import wandb
from torch import Tensor
from torchmetrics import (
    Accuracy,
    AveragePrecision,
    ConfusionMatrix,
    F1Score,
    MetricCollection,
    Precision,
    Recall,
)


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


def evaluate(
    preds: Tensor,
    targets: Tensor,
    num_classes: int,
    enable_wandb: bool = False,
    output_file: str = "evaluation_metrics.json",
    task_type: str = "multiclass",
) -> None:
    """Evaluate predictions and log metrics.

    Args:
        preds (Tensor): Predictions from the model.
        targets (Tensor): Ground truth labels.
        num_classes (int): Number of classes for classification.
        enable_wandb (bool, optional): Whether to log metrics to wandb. Defaults to False.
        output_file (FilePath, optional): Path to save the metrics as JSON. Defaults to None.
        task_type (str, optional): Type of task ('multiclass' or 'multilabel'). Defaults to 'multiclass'.
    """
    if task_type == "multiclass":
        metrics_dict = evaluate_classification(preds, targets, num_classes)
    else:  # multilabel classification
        raise NotImplementedError(
            "Multilabel classification evaluation is not implemented yet."
        )

    def _make_json_serializable(dict):
        """Makes a dict, with values possibly being torch tensors, JSON-serializable."""
        out_dict = {}
        for key, value in dict.items():
            if isinstance(value, Tensor):
                out_dict[key] = value.tolist()
            else:
                out_dict[key] = value
        return out_dict

    if enable_wandb:
        # log scalar metrics to wandb
        scalars_dict: dict[str, float] = dict()
        for name, value in metrics_dict.items():
            try:
                scalars_dict[name] = value.item()
            except RuntimeError:
                pass
        wandb.log(scalars_dict)

    metrics_dict = _make_json_serializable(metrics_dict)

    if output_file is not None:
        with open(output_file, "w") as fp:
            json.dump(metrics_dict, fp, indent=4)
            print(f"Evaluation metrics saved at {output_file!r}")
    else:
        print(json.dumps(metrics_dict, indent=4))
