import os
import ssl
import zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from torchvision.models import resnet50
from PIL import Image
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import seaborn as sns
from utils.version_utils import print_versions, configure_gpu_device, set_seed
from torch.utils.data import ConcatDataset, DataLoader
import torch.nn.functional as F
from transfer.knn import WeightedKNNClassifier
from transfer.logistic_regrssion import  SklearnLogisticProbe, run_logistic_probe
import joblib
from simclr.config import CONFIG

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, proj_dim=128, hidden_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, x):
        return self.net(x)

class SimCLRModel(nn.Module):
    def __init__(self, base_encoder, proj_dim=128):
        super().__init__()
        self.encoder = base_encoder
        self.encoder.fc = nn.Identity()
        self.projection_head = ProjectionHead(input_dim=CONFIG["FEATURE_DIM"], proj_dim=proj_dim)

    def forward(self, x):
        feat = self.encoder(x)
        proj = self.projection_head(feat)
        return feat, proj