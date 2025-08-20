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
