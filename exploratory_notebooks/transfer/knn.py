import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.neighbors import KNeighborsClassifier

class WeightedKNNClassifier:
    def __init__(self, model, device, k=5, normalize=True):
        """
        Cosine-distance k-NN with distance-weighted voting.
        
        Args:
            model      : Your SimCLR model; must have `model.encoder(imgs)` -> features
            device     : 'cpu' or 'cuda'
            k          : Number of neighbors
            normalize  : Whether to L2-normalize features before indexing
        """
        self.device    = device
        self.encoder   = model.encoder.to(device)
        self.k          = k
        self.normalize = normalize
        # `metric='cosine'` + `weights='distance'` -> weighted by cosine similarity
        self.knn        = KNeighborsClassifier(
                              n_neighbors=k,
                              metric='cosine',
                              weights='distance'
                           )

    def _extract(self, loader: DataLoader):
        """
        Runs the encoder on `loader` and returns (features, labels) as numpy arrays.
        """
        self.encoder.eval()
        feats, labs = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(self.device)
                f    = self.encoder(imgs)              # (B, D)
                if self.normalize:
                    f = F.normalize(f, dim=1)
                feats.append(f.cpu().numpy())
                labs.append(labels.cpu().numpy())
        return np.concatenate(feats, axis=0), np.concatenate(labs, axis=0)

    def fit(self, gallery_loader: DataLoader):
        """
        Build the k-NN index on gallery_loader (imgs + labels).
        """
        Xg, yg = self._extract(gallery_loader)
        self.knn.fit(Xg, yg)

    def score(self, query_loader: DataLoader) -> float:
        """
        Returns Top-1 accuracy of the weighted k-NN on the query_loader.
        """
        Xq, yq = self._extract(query_loader)
        return self.knn.score(Xq, yq)


def sweep_knn_k(model,
                device,
                gallery_loader: DataLoader,
                query_loader: DataLoader,
                ks=(1, 5, 20, 100),
                normalize=True):
    """
    Runs WeightedKNNClassifier for each k in `ks` and returns a dict {k: accuracy}.
    """
    results = {}
    for k in ks:
        knn = WeightedKNNClassifier(
            model=model,
            device=device,
            k=k,
            normalize=normalize
        )
        knn.fit(gallery_loader)
        acc = knn.score(query_loader)
        results[k] = acc
        print(f"k = {k:3d} → Top-1 accuracy = {acc*100:5.2f}%")
    return results
