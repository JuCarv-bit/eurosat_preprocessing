# =========== GLOBAL CONFIGURATION ===========
import os
import ssl
import zipfile
import urllib.request
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18
from PIL import Image
import numpy as np
import random

# Prevent nondeterminism
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.enabled = False

CONFIG = {
    "LOCAL_OR_COLAB": "COLAB",
    "DATA_DIR_LOCAL": "/home/juliana/internship_LINUX/datasets/EuroSAT_RGB",
    "DATA_DIR_COLAB": "/content/EuroSAT_RGB",
    "ZIP_PATH": "/content/EuroSAT.zip",
    "EUROSAT_URL": "https://madm.dfki.de/files/sentinel/EuroSAT.zip",
    "SEED": 42,  # Default seed (will be overridden per run)
    "BATCH_SIZE": 128,
    "LR": 0.001,
    "EPOCHS_SIMCLR": 2,
    "EPOCHS_LINEAR": 2,
    "PROJ_DIM": 128,
    "FEATURE_DIM": 512,
}

# =========== SETUP ===========
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def prepare_data():
    if CONFIG["LOCAL_OR_COLAB"] == "LOCAL":
        return CONFIG["DATA_DIR_LOCAL"]

    if not os.path.exists(CONFIG["DATA_DIR_COLAB"]):
        print("Downloading EuroSAT RGB...")
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib.request.urlretrieve(CONFIG["EUROSAT_URL"], CONFIG["ZIP_PATH"])
        with zipfile.ZipFile(CONFIG["ZIP_PATH"], 'r') as zip_ref:
            zip_ref.extractall("/content")
        os.rename("/content/2750", CONFIG["DATA_DIR_COLAB"])
        print("EuroSAT RGB dataset downloaded and extracted.")
    return CONFIG["DATA_DIR_COLAB"]

# =========== TRANSFORMS ===========
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

simclr_transform = transforms.Compose([
    transforms.RandomResizedCrop(64, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    normalize,
])

eval_transform = transforms.Compose([
    transforms.Resize(72),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    normalize,
])

class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]

# =========== MODEL COMPONENTS ===========
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

# =========== TRAINING ===========
def train_simclr(model, loader, optimizer, criterion, device, epochs):
    model.train()
    model.to(device)
    for epoch in range(epochs):
        total_loss = 0
        for (x1, x2), _ in loader:
            x1, x2 = x1.to(device), x2.to(device)
            _, z1 = model(x1)
            _, z2 = model(x2)
            loss = criterion(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        print(f"[SimCLR] Epoch {epoch+1}/{epochs} - Loss: {avg:.4f}")
    print("Finished SimCLR pretraining.")

def train_linear_probe(backbone, train_loader, val_loader, device, epochs, lr, run_id):
    # Freeze backbone parameters
    for p in backbone.parameters():
        p.requires_grad = False
    # Create a classifier on top of the frozen features
    classifier = nn.Linear(CONFIG["FEATURE_DIM"], len(train_loader.dataset.dataset.classes)).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        classifier.train()
        correct, total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            features = backbone(images)
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = correct / total * 100
        val_acc = evaluate(classifier, backbone, val_loader, device)
        print(f"[Linear] Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    # Save the classifier weights uniquely for each run
    torch.save(classifier.state_dict(), f"linear_probe_seed{run_id}.pth")
    # Return the final validation accuracy
    return val_acc

def evaluate(classifier, backbone, loader, device):
    classifier.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            features = backbone(images)
            outputs = classifier(features)
            total += labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
    return correct / total * 100

# =========== RUN EVERYTHING ===========
if __name__ == "__main__":
    # Define the list of seeds for each run
    seeds = [42, 43, 44]
    results = []  # Will store the final linear probe validation accuracies

    for seed in seeds:
        print(f"\n=== Starting run with seed {seed} ===")
        set_seed(seed)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        data_dir = prepare_data()

        # Prepare datasets and dataloaders for contrastive and evaluation
        contrastive_dataset = datasets.ImageFolder(data_dir, transform=TwoCropsTransform(simclr_transform))
        contrastive_loader = DataLoader(contrastive_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, drop_last=True)

        full_dataset = datasets.ImageFolder(data_dir, transform=eval_transform)
        train_len = int(0.8 * len(full_dataset))
        val_len = len(full_dataset) - train_len
        train_set, val_set = random_split(full_dataset, [train_len, val_len])
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=32)

        # Initialize base encoder and SimCLR model
        pretrained = True
        base_encoder = resnet18(weights=None if not pretrained else "DEFAULT")
        simclr_model = SimCLRModel(base_encoder, proj_dim=CONFIG["PROJ_DIM"])
        optimizer = optim.Adam(simclr_model.parameters(), lr=CONFIG["LR"])
        loss_fn = NTXentLoss(CONFIG["BATCH_SIZE"], temperature=0.5, device=device)

        print("Starting SimCLR training...")
        train_simclr(simclr_model, contrastive_loader, optimizer, loss_fn, device, CONFIG["EPOCHS_SIMCLR"])

        print("Saving encoder...")
        torch.save(simclr_model.state_dict(), f"simclr_model_seed{seed}.pth")

        print("Starting linear probe training...")
        final_val_acc = train_linear_probe(simclr_model.encoder, train_loader, val_loader, device, CONFIG["EPOCHS_LINEAR"], CONFIG["LR"], seed)
        results.append(final_val_acc)
        print(f"Run with seed {seed} finished with final Val Acc: {final_val_acc:.2f}%")

    # Compute and print overall mean and standard deviation of the final validation accuracies
    mean_acc = np.mean(results)
    std_acc = np.std(results)
    print("\n=== Summary over runs ===")
    print(f"Final Linear Probe Validation Accuracies: {results}")
    print(f"Mean Accuracy: {mean_acc:.2f}%")
    print(f"Standard Deviation Accuracy: {std_acc:.2f}%")
