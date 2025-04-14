import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet18
import numpy as np
import random
import os
import urllib.request
import zipfile
import ssl


# Global config
LOCAL_OR_COLAB = "LOCAL" # Change to "COLAB" if running on Google Colab Or "LOCAL" if running locally

SEED = 42
if LOCAL_OR_COLAB == "LOCAL":
    DATA_DIR = "/home/juliana/internship_LINUX/datasets/EuroSAT_RGB"
else:

  # Set paths
  data_root = "/content/EuroSAT_RGB"
  zip_path = "/content/EuroSAT.zip"

  # Download and extract EuroSAT RGB dataset
  if not os.path.exists(data_root):
      print("Downloading EuroSAT RGB...")
      url = "https://madm.dfki.de/files/sentinel/EuroSAT.zip"
      
      # Create an unverified SSL context
      ssl._create_default_https_context = ssl._create_unverified_context

      urllib.request.urlretrieve(url, zip_path)

      print("Unzipping...")
      with zipfile.ZipFile(zip_path, 'r') as zip_ref:
          zip_ref.extractall("/content")
      os.rename("/content/2750", data_root)

  DATA_DIR = data_root
  print("EuroSAT RGB dataset downloaded and extracted.")


BATCH_SIZE = 32
NUM_EPOCHS = 2
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_data_loaders(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    return train_loader, test_loader, len(dataset.classes)

def build_model(num_classes, pretrained=False):
    model = resnet18(weights=None if not pretrained else "DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)

def train(model, loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss, correct, total = 0, 0, 0
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss:.4f} - Train Accuracy: {acc:.2f}%")
    return model

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    return acc

def run_experiment(seed=SEED, freeze=False, pretrained=False):
    set_seed(seed)
    train_loader, test_loader, num_classes = get_data_loaders(DATA_DIR, BATCH_SIZE)

    model = build_model(num_classes, pretrained=pretrained)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("Training full model...")
    model = train(model, train_loader, optimizer, criterion, NUM_EPOCHS)
    torch.save(model.state_dict(), "resnet18_eurosat_from_scratch.pt")

    acc = evaluate(model, test_loader)

    if freeze:
        print("\nRunning linear probe...")
        model = build_model(num_classes)
        model.load_state_dict(torch.load("resnet18_eurosat_from_scratch.pt"))

        for param in model.parameters():
            param.requires_grad = False

        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(DEVICE)
        optimizer = optim.Adam(model.fc.parameters(), lr=LR)

        model = train(model, train_loader, optimizer, criterion, NUM_EPOCHS)
        acc = evaluate(model, test_loader)

    return acc

def run_multiple_experiments(n_runs=3, freeze=False, pretrained=False):
    accs = []
    for run in range(n_runs):
        print(f"\n========== Run {run + 1} ==========")
        acc = run_experiment(seed=SEED + run, freeze=freeze, pretrained=pretrained)
        accs.append(acc)
    print(f"\nMean Accuracy over {n_runs} runs: {np.mean(accs):.2f}%")
    print(f"Std Deviation: {np.std(accs):.2f}%")

# Run the code
if __name__ == "__main__":
    run_multiple_experiments(n_runs=3, freeze=True, pretrained=True)
