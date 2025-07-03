import torch
import torch.nn as nn
import torch.nn.functional as F
from simclr.data.transforms import TwoCropsTransform

class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5, device='cuda'):
        super().__init__()
        self.temperature = temperature
        self.batch_size = batch_size
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, zis, zjs):
        N = zis.size(0)
        z = F.normalize(torch.cat([zis, zjs], dim=0), dim=1)
        sim = torch.matmul(z, z.T) / self.temperature
        mask = torch.eye(2 * N, dtype=torch.bool).to(self.device)
        sim = sim.masked_fill(mask, -1e9)
        labels = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(self.device)
        return self.criterion(sim, labels)

@torch.no_grad()
def compute_contrastive_val_loss(model, val_loader, criterion,
                                 two_crop: TwoCropsTransform,
                                 device):
    model.eval()
    total, acc_loss = 0, 0.0
    for imgs, _labels in val_loader:                 # unlabeled for loss
        imgs = imgs.to(device)
        x1, x2 = two_crop(imgs)                      # produce two views
        _, z1 = model(x1)
        _, z2 = model(x2)
        loss = criterion(z1, z2)
        batch_size = imgs.size(0)
        acc_loss += loss.item() * batch_size
        total += batch_size
    return acc_loss / total


    

@torch.no_grad()
def compute_contrastive_accuracy(model, loader, device):
    """
    Returns fraction of times the positive pair is the nearest neighbor
    across the batch of size 2N.
    """
    model.eval()
    correct, total = 0, 0

    for x1, x2 in loader:
        x1, x2 = x1.to(device), x2.to(device)
        _, z1 = model(x1)   # proj head output
        _, z2 = model(x2)
        z = torch.cat([z1, z2], dim=0)           # (2N, dim)
        sim = F.cosine_similarity(
            z.unsqueeze(1), z.unsqueeze(0), dim=2
        )                                        # (2N, 2N)

        N = x1.size(0)
        # mask out self-similarity
        sim.fill_diagonal_(-9e15)

        # for each i in [0..2N), its positive index is:
        #    if i < N -> i + N  else -> i - N
        pos_idx = torch.arange(2*N, device=device)
        pos_idx = (pos_idx + N) % (2*N)

        # find the index of the max similarity for each row
        nbr_idx = sim.argmax(dim=1)
        correct += (nbr_idx == pos_idx).sum().item()
        total   += 2 * N

    return correct / total