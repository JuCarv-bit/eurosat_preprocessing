import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import time

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.enabled = False


# ===========
#  Utilities
# ===========

# TwoCropsTransform creates two differently augmented versions of each image.
class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]


# ===========
#  Data Preparation
# ===========

# Set paths for your EuroSat dataset.
# This should be the root folder containing subfolders for each class
data_dir = "/home/juliana/internship_LINUX/datasets/EuroSAT_RGB"

# Standard normalization parameters for ImageNet (can be adapted if needed)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Augmentations for SimCLR training (contrastive)
simclr_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=64, scale=(0.5, 1.0)),   # EuroSat images are 64x64
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # brightness, contrast, saturation, hue
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    # transforms.GaussianBlur(kernel_size=3),  # a small gaussian blur
    transforms.ToTensor(),
    normalize,
])

# For evaluation / linear probing, use a deterministic transform:
eval_transform = transforms.Compose([
    transforms.Resize(72),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    normalize,
])

# Create a dataset for contrastive training using TwoCropsTransform.
contrastive_dataset = datasets.ImageFolder(
    root=data_dir,
    transform=TwoCropsTransform(simclr_transform)
)

contrastive_loader = DataLoader(
    contrastive_dataset,
    batch_size=128,  # adjust depending on your hardware
    shuffle=True,
    # num_workers=4,
    drop_last=True
)

# Create datasets for linear probing. Here, we use the standard ImageFolder with evaluation transforms.
# We'll split the dataset (you can use your own train/val split as needed).
all_dataset = datasets.ImageFolder(
    root=data_dir,
    transform=eval_transform
)

# For simplicity, we split manually here (80% train, 20% val)
train_size = int(0.2 * len(all_dataset))
val_size = len(all_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(all_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)


# ===========
#  Model: Backbone + Projection Head for SimCLR
# ===========

class ProjectionHead(nn.Module):
    """
    A small MLP with one hidden layer (and batch normalization) to map
    the backbone representations to the space where contrastive loss is computed.
    """
    def __init__(self, input_dim, proj_dim=128, hidden_dim=2048):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim)
        )
    
    def forward(self, x):
        return self.net(x)

    
class SimCLRModel(nn.Module):
    def __init__(self, base_encoder, proj_dim=128):
        super(SimCLRModel, self).__init__()
        # Use a pretrained model if available (or train from scratch)
        self.encoder = base_encoder
        # Get the dimension of the last layer's features
        # For ResNet18, the penultimate layer typically has 512 features.
        if hasattr(self.encoder, 'fc'):
            feat_dim = self.encoder.fc.in_features
            # remove the original fc layer
            self.encoder.fc = nn.Identity()
        else:
            raise ValueError("Base encoder does not have an attribute 'fc'")
        
        self.projection_head = ProjectionHead(input_dim=feat_dim, proj_dim=proj_dim)

    def forward(self, x):
        feat = self.encoder(x)
        proj = self.projection_head(feat)
        return feat, proj  # return both for later use in linear probing


# ===========
#  NT-Xent Loss (Normalized Temperature-scaled Cross Entropy Loss)
# ===========

class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5, device='cuda'):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, zis, zjs):
        N = zis.size(0)
        z = torch.cat([zis, zjs], dim=0)  # shape [2N, D]
        z = F.normalize(z, dim=1)

        sim_matrix = torch.matmul(z, z.T) / self.temperature
        # Remove self-similarity
        mask = torch.eye(2 * N, dtype=torch.bool).to(self.device)
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)

        # Positive pairs are (i, i + N) and (i + N, i)
        positives = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(self.device)
        labels = positives

        logits = sim_matrix
        return self.criterion(logits, labels)

# ===========
#  Training Functions
# ===========

def train_simclr(model, dataloader, optimizer, criterion, device, epochs=100):
    model.train()
    model.to(device)
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0.0
        start_time = time.time()
        for (images, _) in dataloader:
            print("Batch size:", images[0].size())
            # images is a list of two augmented images
            images1 = images[0].to(device)
            images2 = images[1].to(device)
            optimizer.zero_grad()

            # Forward pass for both views:
            _, proj1 = model(images1)
            _, proj2 = model(images2)
            print("Projection shape:", proj1.size(), proj2.size())

            loss = criterion(proj1, proj2)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Time: {elapsed:.2f}s")
    print("SimCLR training complete.")


def train_linear_probe(backbone, train_loader, val_loader, device, epochs=20, lr=0.001):
    # Freeze backbone weights
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # Create a simple linear classifier that takes features from the backbone and maps to class logits.
    # Assuming EuroSat has 10 classes (adjust num_classes accordingly).
    num_classes = len(train_loader.dataset.dataset.classes) if hasattr(train_loader.dataset, 'dataset') else len(train_loader.dataset.classes)
    
    # Get feature dimension from the backbone (assuming resnet18: 512)
    feature_dim = 512
    classifier = nn.Linear(feature_dim, num_classes).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        classifier.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # Extract features from the frozen backbone
            features = backbone(images)
            # Ensure features are flattened (for ResNet, they are already [batch, feature_dim])
            logits = classifier(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            total += labels.size(0)
            _, predicted = torch.max(logits.data, 1)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        train_acc = correct / total

        # Evaluation
        classifier.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                features = backbone(images)
                logits = classifier(features)
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)
                total_val += labels.size(0)
                _, predicted = torch.max(logits.data, 1)
                correct_val += (predicted == labels).sum().item()
        val_loss /= total_val
        val_acc = correct_val / total_val

        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    print("Linear probing training complete.")


# ===========
#  Main Training Execution
# ===========

def main():
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Prepare the base encoder (ResNet18) for SimCLR.
    base_encoder = models.resnet18(pretrained=False)
    simclr_model = SimCLRModel(base_encoder=base_encoder, proj_dim=128)

    # Optimizer and criterion for SimCLR stage
    optimizer = optim.Adam(simclr_model.parameters(), lr=0.001)
    batch_size = 128  # make sure this matches your DataLoader batch_size
    contrastive_criterion = NTXentLoss(batch_size=batch_size, temperature=0.5, device=device)

    # 1. Pretrain with SimCLR (contrastive learning)
    print("Starting SimCLR pretraining...")
    train_simclr(simclr_model, contrastive_loader, optimizer, contrastive_criterion, device, epochs=1)

    # Save or extract the backbone encoder. We “freeze” the projection head.
    backbone = simclr_model.encoder  # This is our feature extractor

    # 2. Linear Probing: train a linear classifier on frozen features.
    print("Starting linear probe training...")
    train_linear_probe(backbone, train_loader, val_loader, device, epochs=5, lr=0.001)

if __name__ == "__main__":
    main()


