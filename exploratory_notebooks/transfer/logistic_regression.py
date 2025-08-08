import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Literal, Any


class SklearnLogisticProbe:
    """
    Simple linear probe over a frozen encoder using sklearn's LogisticRegression.

    Args:
        encoder (nn.Module): feature extractor (frozen).
        device (torch.device): device for running encoder.
        scale_features ("standard" or None): whether to apply StandardScaler.
        **kwargs: passed directly to LogisticRegression.
                  (e.g. C=1.0, max_iter=500, multi_class="multinomial", solver="lbfgs")
    """
    def __init__(
        self,
        encoder: torch.nn.Module,
        device: torch.device,
        scale_features: Literal["standard", None] = "standard",
        yaware: bool = False,
        **kwargs: Any
    ):
        self.encoder = encoder.eval().to(device)
        self.device = device
        self.clf = LogisticRegression(**kwargs)
        self.scaler = StandardScaler() if scale_features == "standard" else None
        self.yaware = yaware

    def _extract(self, loader: torch.utils.data.DataLoader):
        feats, labs = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(self.device)
                f = self.encoder(imgs)
                feats.append(f.cpu().numpy())
                labs.append(labels.numpy())
        X = np.concatenate(feats, axis=0)
        y = np.concatenate(labs, axis=0)
        return X, y

    def _scale(self, X: np.ndarray, train: bool = False):
        if self.scaler is None:
            return X
        return self.scaler.fit_transform(X) if train else self.scaler.transform(X)

    def fit(self, loader: torch.utils.data.DataLoader) -> None:
        """Extract features + labels from `loader`, scale, then fit."""
        X, y = self._extract(loader)
        X = self._scale(X, train=True)
        self.clf.fit(X, y)

    def predict(self, loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """Return class indices for all samples in `loader`."""
        X, _ = self._extract(loader)
        X = self._scale(X, train=False)
        preds = self.clf.predict(X)
        return torch.from_numpy(preds).to(self.device)

    def predict_proba(self, loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """Return probability vectors for all samples."""
        X, _ = self._extract(loader)
        X = self._scale(X, train=False)
        prob = self.clf.predict_proba(X)
        return torch.from_numpy(prob).to(self.device)

    def score(self, loader: torch.utils.data.DataLoader) -> float:
        """Convenience: accuracy on `loader`."""
        X, y = self._extract(loader)
        X = self._scale(X, train=False)
        return self.clf.score(X, y)

def run_logistic_probe(
    model,
    probe_train_loader,
    probe_val_loader,
    feature_dim,       
    num_classes,       
    device,
    C=1.0,
    max_iter=200,
    scale_features="standard",
    yaware=False
):
    """
    1) Wraps encoder in SklearnLogisticProbe
    2) .fit() on train loader
    3) .score() on val loader
    Returns float accuracy in [0,1].
    """

    # Build the probe
    probe = SklearnLogisticProbe(
        encoder=model.encoder,
        device=device,
        scale_features=scale_features,
        C=C,
        max_iter=max_iter,
        multi_class="multinomial",
        solver="lbfgs",
        yaware=yaware
    )

    # Fit
    probe.fit(probe_train_loader)

    # Evaluate
    acc = probe.score(probe_val_loader)
    print(f"[probe] val accuracy = {acc*100:.2f}%")
    return acc

