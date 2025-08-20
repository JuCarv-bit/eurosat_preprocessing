import math
import time
from typing import Literal

import torch
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn

class NNClassifier:
    """A weighted k nearest neighbors (k-NN) classifier.

    Args:
        num_classes (int): Number of classes in dataset.
        k (int): Number of nearest neighbors.
        tau (float): Temperature hyperparameter used for weighting the neighbors contributions by softmax scaling.
        distance_fn (str): Distance function for computing similarities.
        weighted (bool): Wheter or not to weight neighbors contributions by softmax scaling.
        scale_features (str): Whether to scale features by standardization or normalization.
            - "standard": Standardize features
            - "norm": Normalize features
            - None: No scaling
    """

    def __init__(
        self,
        num_classes: int,
        k: int = 5,
        tau: float = 0.07,
        distance_fn: Literal["cosine", "euclidean"] = "cosine",
        weighted: bool = True,
        scale_features: Literal["standard", "norm", None] = "norm",
    ):
        self.num_classes = num_classes
        self.k = k
        self.tau = tau
        self.distance_fn = distance_fn
        self.weighted = weighted
        self.scale_features = scale_features
        self.scaler = None
        if scale_features == "standard":
            self.scaler = StandardScaler()

        self.device = None
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

    def fit(self, train_features, train_labels) -> None:
        """
        Fit model.

        Args:
            train_features (float tensor): features of the training set examples, with shape `(train_length, dim)`
            train_labels (float tensor): the target labels of the training set, with shape `(train_length,)`
        """
        if train_features.size(0) != train_labels.size(0):
            raise ValueError(
                f"train_features length ({train_features.size(0)}) not equal to train_labels length ({train_labels.size(0)})"
            )
        train_features = self._scale_features(train_features, is_train=True)

        self.train_features = train_features
        self.train_labels = train_labels
        self.device = self.train_features.device
        self._fitted = True
        
        self._retrieval_one_hot = torch.zeros(self.k, self.num_classes).to(self.device)

    def forward(
        self,
        cur_feats,
        output_as="probs",
        enable_amp=False,
    ):
        """Predict over a batch."""

        if not self._fitted:
            raise AttributeError("the model must be fitted before calling predict")

        if cur_feats.size(1) != self.train_features.size(1):
            raise ValueError(
                f"test_features dimension ({cur_feats.size(1)}) not equal to train_features dimension ({self.train_features.size(1)})"
            )

        cur_size = cur_feats.size(0)  # current batch size

        with torch.cuda.amp.autocast(enabled=enable_amp):
            # calculate similarities: (batch_size, train_length)
            if self.distance_fn == "cosine":
                sim = torch.mm(cur_feats, self.train_features.t()).float()
            else:
                sim = 1 / (torch.cdist(cur_feats, self.train_features) + 1e-5)

        # get top-k neighbors: (cur_size, k)
        nns_sim, nns_indices = sim.topk(self.k, dim=1, largest=True, sorted=True)

        # retrieve neighbors labels:
        candidates = self.train_labels.view(1, -1).expand(cur_size, -1)
        retrieval = torch.gather(
            candidates, 1, nns_indices
        )  # (cur_size, k): class of each neighbor
        self._retrieval_one_hot.resize_(cur_size * self.k, self.num_classes).zero_()
        self._retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
        if self.weighted:
            # scale similarities by temperature
            nns_scores = nns_sim.div(self.tau).exp()
        else:
            nns_scores = torch.ones_like(nns_sim)

        # get affinity scores: (cur_size, num_classes)
        probs = torch.sum(
            torch.mul(
                self._retrieval_one_hot.view(cur_size, -1, self.num_classes),
                nns_scores.view(cur_size, -1, 1),
            ),
            1,
        )
        if output_as == "probs":
            return probs / probs.sum(-1, keepdims=True)
        else:
            return probs.argmax(-1)

    @torch.inference_mode()
    def _predict(
        self,
        test_features: Tensor,
        output_as: Literal["probs", "preds"] = "probs",
        batch_size: int = 256,
        print_freq: int | None = 10,
        enable_amp: bool = False,
        log_prefix: str = "predict ",
    ) -> Tensor:

        test_length = test_features.size(0)
        num_batches = math.ceil(test_length / batch_size)
        test_features = self._scale_features(test_features)


        def _yield_preds():
            end = time.time()
            for batch_idx, cur_feats in enumerate(
                torch.split(test_features, batch_size)
            ):
                # cur_size = cur_feats.size(0)  # current batch size
                cur_preds = self.forward(
                    cur_feats,
                    output_as=output_as,
                    enable_amp=enable_amp,
                )

                # measure elapsed time
                end = time.time()
                torch.cuda.reset_peak_memory_stats(self.device)

                yield cur_preds

        test_preds = torch.cat(tuple(_yield_preds()))
        return test_preds

    def predict(
        self,
        test_features: Tensor,
        batch_size: int = 256,
        print_freq: int | None = 10,
        enable_amp: bool = False,
        log_prefix: str = "predict ",
    ):
        """Predict top-1 class labels.

        Args:
            test_features (float tensor): features of the testing set examples, with shape `(test_length, dim)`
            batch_size (int): divide the test batches into chunks of this size to run k-NN.
            print_freq (int): frequency for printing progress to stdout
            enable_amp (bool): if True, will use autocast for the forward passes and distance matrices, if using cuda
                computations.
            log_prefix: prefix for progress meter, default is "predict"

        Returns:
            (int tensor): predicted top-1 class labels for all samples, with shape `(batch_size,)`.
        """
        return self._predict(
            test_features,
            output_as="preds",
            batch_size=batch_size,
            print_freq=print_freq,
            enable_amp=enable_amp,
            log_prefix=log_prefix,
        )

    def predict_proba(
        self,
        test_features: Tensor,
        batch_size: int = 256,
        print_freq: int | None = 10,
        enable_amp: bool = False,
        log_prefix: str = "predict ",
    ):
        """Predict class probability scores.

        Args:
            test_features (float tensor): features of the testing set examples, with shape `(test_length, dim)`
            batch_size (int): divide the test batches into chunks of this size to run k-NN.
            print_freq (int): frequency for printing progress to stdout
            enable_amp (bool): if True, will use autocast for the forward passes and distance matrices, if using cuda
                computations.
            log_prefix: prefix for progress meter, default is "predict"

        Returns:
            (float tensor): predicted class probability scores for all samples, with shape `(batch_size, num_classes)`.
        """
        return self._predict(
            test_features,
            output_as="probs",
            batch_size=batch_size,
            print_freq=print_freq,
            enable_amp=enable_amp,
            log_prefix=log_prefix,
        )


class NNMultilabelClassifier(NNClassifier):
    """A weighted k nearest neighbors (k-NN) classifier for multilabel classification."""

    def fit(self, train_features, train_labels):
        def _is_multilabel(labels):
            unique_values = torch.unique(labels)
            return labels.ndim == 2 and torch.allclose(
                unique_values, torch.tensor([0, 1], device=labels.device)
            )

        if not _is_multilabel(train_labels):
            raise ValueError("labels should be multi-label")

        super().fit(train_features, train_labels)

    def forward(
        self,
        cur_feats,
        output_as="probs",
        enable_amp=False,
    ):
        """Predict over a batch."""

        if not self._fitted:
            raise AttributeError("the model must be fitted before calling predict")

        if cur_feats.size(1) != self.train_features.size(1):
            raise ValueError(
                f"test_features dimension ({cur_feats.size(1)}) not equal to train_features dimension ({self.train_features.size(1)})"
            )

        cur_size = cur_feats.size(0)  # current batch size

        with torch.cuda.amp.autocast(enabled=enable_amp):
            # calculate similarities: (batch_size, train_length)
            if self.distance_fn == "cosine":
                sim = torch.mm(cur_feats, self.train_features.t()).float()
            else:
                sim = 1 / (torch.cdist(cur_feats, self.train_features) + 1e-5)

        # get top-k neighbors: (cur_size, k)
        nns_sim, nns_indices = sim.topk(self.k, dim=1, largest=True, sorted=True)

        retrieval = self.train_labels[
            nns_indices
        ]  # (cur_size, n_neighbors, num_classes)
        # scale similarities score for weighting
        if self.weighted:
            nns_scores = nns_sim.div(self.tau).exp()
        else:
            nns_scores = torch.ones_like(nns_sim)

        # compute affinity scores weighted by similarity: (cur_size, num_classes)
        probs = torch.zeros(cur_size, self.num_classes).to(self.device)
        for c in range(self.num_classes):
            probs[:, c] = torch.div(
                torch.mul(retrieval[:, :, c], nns_scores).sum(1), nns_scores.sum(1)
            )

        if output_as == "probs":
            return probs
        else:
            # threshold probabilities at 0.5
            return (probs >= 0.5).to(torch.long)
