import torch
import torch.nn as nn
import torch.nn.functional as F

DISTS = {"min_distance": 0.02212141876755244, "max_distance": 5043.998712454391}
MIN_DISTANCE = DISTS["min_distance"]
MAX_DISTANCE = DISTS["max_distance"]
R = 6371.0  # Earth radius in kilometers

import torch
import torch.nn as nn
import torch.nn.functional as F

class HaversineRBFNTXenLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.1,
        return_logits: bool = False,
        sigma: float = 0.05,
        min_d: float = MIN_DISTANCE,
        max_d: float = MAX_DISTANCE,
        R: float = R,  # e.g. 6371.0 km
    ):
        super().__init__()
        self.temperature = temperature
        self.return_logits = return_logits
        self.sigma = sigma
        self.min_d = min_d
        self.max_d = max_d
        self.R = R

    def haversine_matrix(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: (N,2) in degrees
        returns: (2N, 2N) pairwise great‐circle distances
        """
        # duplicate coords to (2N,2)
        coords2 = torch.cat([coords, coords], dim=0).to(coords.device)  # (2N,2)

        # convert lat/lon to radians
        lat = torch.deg2rad(coords2[:, 0]).unsqueeze(1)  # (2N,1)
        lon = torch.deg2rad(coords2[:, 1]).unsqueeze(1)  # (2N,1)

        # pairwise diffs
        dlat = lat - lat.T                                # (2N,2N)
        dlon = lon - lon.T                                # (2N,2N)

        a = (torch.sin(dlat / 2) ** 2
             + torch.cos(lat) * torch.cos(lat.T) * torch.sin(dlon / 2) ** 2)
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

        return self.R * c  # (2N,2N)

    def forward(self,
                z_i: torch.Tensor,   # (N,D)
                z_j: torch.Tensor,   # (N,D)
                coords: torch.Tensor # (N,2) lat/lon in degrees
               ):
        N = z_i.size(0)
        coords = coords.to(z_i.device)

        # 1) normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # 2) build standard SimCLR logits (2N × 2N−1)
        sim_ii = (z_i @ z_i.T) / self.temperature
        sim_jj = (z_j @ z_j.T) / self.temperature
        sim_ij = (z_i @ z_j.T) / self.temperature

        mask = ~torch.eye(N, device=z_i.device).bool()
        sim_ii = sim_ii[mask].view(N, N-1)
        sim_jj = sim_jj[mask].view(N, N-1)

        top    = torch.cat([sim_ii,   sim_ij],   dim=1)  # N × (2N−1)
        bottom = torch.cat([sim_ij.T, sim_jj],   dim=1)  # N × (2N−1)
        logits = torch.cat([top, bottom],       dim=0)  # (2N × 2N−1)

        # 3) compute RBF‐Haversine weights (no grad)
        with torch.no_grad():
            d_mat = self.haversine_matrix(coords)        # (2N,2N)
            sim_mat = torch.exp(-((d_mat / self.max_d)**2) /
                                (2 * self.sigma**2))
            mask2 = ~torch.eye(2*N, device=z_i.device).bool()
            w = sim_mat[mask2].view(2*N, 2*N-1)
            w = w / (w.sum(dim=1, keepdim=True) + 1e-12)

        # 4) weighted NT‐Xent
        logp = F.log_softmax(logits, dim=1)            # (2N × 2N−1)
        loss = - (w * logp).sum(dim=1).mean()

        if self.return_logits:
            idx = torch.arange(N, device=z_i.device)
            return loss, sim_ij, idx, w
        else:
            return loss
