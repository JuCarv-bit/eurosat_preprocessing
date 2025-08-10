import time
from typing import Any, Literal

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn


class SklearnLogisticClassifier:
    """Logistic classifier wrapping scikit-learn's logistic regression.

    Args:
        scale_features (str, optional): Feature scaling method. Options are "standard" for standard scaling, or None for no scaling. Default is "standard".
        **kwargs: Additional arguments for the logistic regression model.
    """

    def __init__(
        self, scale_features: Literal["standard", None] = "standard", **kwargs: Any
    ):
        self.scale_features = scale_features
        use_sgd = kwargs["solver"] == "sgd"
        if use_sgd:
            kwargs.pop("solver")
            alpha = 1.0 / kwargs.pop("C", 1e-4)
            kwargs["alpha"] = alpha
            self.clf = SGDClassifier(loss="log_loss", **kwargs)
        else:
            self.clf = LogisticRegression(**kwargs)
        self.scaler = None
        if scale_features == "standard":
            self.scaler = StandardScaler()

    def _scale_features(self, inpt: np.ndarray, is_train: bool = False) -> np.ndarray:
        if self.scale_features == "standard":
            if is_train:
                return self.scaler.fit_transform(inpt)
            else:
                return self.scaler.transform(inpt)
        return inpt  # None: no scaling

    def fit(self, inpt: Tensor, labels: Tensor) -> None:
        inpt = inpt.cpu().numpy()
        labels = labels.cpu().numpy()
        inpt = self._scale_features(inpt, is_train=True)
        self.clf.fit(inpt, labels)

    def predict(self, inpt: Tensor) -> Tensor:
        device = inpt.device
        dtype = inpt.dtype
        inpt = inpt.cpu().numpy()
        inpt = self._scale_features(inpt)
        preds = self.clf.predict(inpt)
        return torch.from_numpy(preds).to(device=device, dtype=dtype)

    def predict_proba(self, inpt: Tensor) -> Tensor:
        device = inpt.device
        dtype = inpt.dtype
        inpt = inpt.cpu().numpy()
        inpt = self._scale_features(inpt)
        preds = self.clf.predict_proba(inpt)
        return torch.from_numpy(preds).to(device=device, dtype=dtype)

    def predict_log_proba(self, inpt: Tensor) -> Tensor:
        device = inpt.device
        dtype = inpt.dtype
        inpt = inpt.cpu().numpy()
        inpt = self._scale_features(inpt)
        preds = self.clf.predict_log_proba(inpt)
        return torch.from_numpy(preds).to(device=device, dtype=dtype)


class LogisticClassifier:
    """Logistic classifier in PyTorch with gradient descent optimizer.

    Args:
        in_features (int): Number of input features.
        num_classes (int): Number of output classes.
        task (str, optional): Task type. Options are "multiclass" or "multilabel". Default is "multiclass".
        lr (float, optional): Learning rate. Default is 0.001.
        weight_decay (float, optional): Weight decay for the optimizer. Default is 0.0.
        epochs (int, optional): Number of training epochs. Default is 200.
        scale_features (str, optional): Feature scaling method. Options are "standard" for standard scaling, "norm" for normalization, or None for no scaling. Default is "norm".
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        task: str = "multiclass",
        lr: float = 0.001,
        weight_decay: float = 0.0,
        epochs: int = 200,
        scale_features: Literal["standard", "norm", None] = "norm",
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.linear = nn.Linear(in_features, num_classes)
        self.task = task
        if task == "multiclass":
            self.criterion = nn.CrossEntropyLoss()
        elif task == "multilabel":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"unsupported task: {task!r}")
        self.initialize()

        self.scale_features = scale_features
        self.scaler = None
        if scale_features == "standard":
            self.scaler = StandardScaler()

    def initialize(self) -> None:
        self.linear.reset_parameters()
        self.optimizer = torch.optim.SGD(
            self.linear.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self._epoch = 0
        self._fitted = False

    def _scale_features(self, inpt: Tensor, is_train: bool = False) -> Tensor:
        if self.scale_features == "norm":
            return nn.functional.normalize(inpt, dim=1)
        elif self.scale_features == "standard":
            inpt_numpy = inpt.cpu().numpy()
            if is_train:
                output_numpy = self.scaler.fit_transform(inpt_numpy)
            else:
                output_numpy = self.scaler.transform(inpt_numpy)
            return torch.from_numpy(output_numpy).to(
                device=inpt.device, dtype=inpt.dtype
            )
        return inpt  # None: no scaling

    def _move_to_device(self, device: torch.device) -> None:
        self.linear.to(device)
        self.criterion.to(device)

    def _train_one_epoch(self, inpt: Tensor, labels: Tensor) -> Tensor:
        self.optimizer.zero_grad(set_to_none=True)

        output = self.linear(inpt)
        loss = self.criterion(output, labels)
        loss.backward()
        self.optimizer.step()
        self._epoch += 1
        return loss

    def fit(self, inpt: Tensor, labels: Tensor, print_freq: int | None = 10) -> None:
        # inpt = self._scale_features(inpt, is_train=True)

        if inpt.size(0) != labels.size(0):
            raise ValueError(
                f"input features length ({inpt.size(0)}) not equal to output labels length ({labels.size(0)})"
            )

        inpt = self._scale_features(inpt, is_train=True)
        self._move_to_device(inpt.device)

        epoch_time = AverageMeter("epoch_time", ":6.2f")
        losses = AverageMeter("loss", ":6.2e")

        progress = ProgressMeter(self.epochs, [losses, epoch_time], prefix="Train ")

        if isinstance(self.criterion, nn.BCEWithLogitsLoss):
            # we need to set dtype of labels to dtype on input
            labels = labels.to(dtype=inpt.dtype)

        for epoch in range(self.epochs):
            end = time.time()

            loss = self._train_one_epoch(inpt, labels)

            epoch_time.update(time.time() - end)
            losses.update(loss)
            if print_freq is not None and epoch % print_freq == 0:
                progress.display(epoch)

        self._fitted = True

    def predict_proba(self, inpt: Tensor) -> Tensor:
        norm_func = (
            nn.functional.softmax
            if self.task == "multiclass"
            else nn.functional.sigmoid
        )
        return norm_func(self.linear(inpt))

    def predict(self, inpt: Tensor) -> Tensor:
        output = self.predict_proba(inpt)
        if self.task == "multiclass":
            return output.argmax(dim=-1).long()
        else:
            return torch.round(output).long()
