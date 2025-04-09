import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import resnet18, ResNet18_Weights

import os

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
base_dir = "/home/juliana/internship_LINUX/datasets/EuroSAT_RGB"
batch_size = 32
num_epochs = 2
learning_rate = 1e-3

# Image transforms
transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                         std=[0.229, 0.224, 0.225])
])

# Load dataset using ImageFolder
dataset = datasets.ImageFolder(root=base_dir, transform=transform)

# Train/test split (80/20)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size)

# Load pre-trained ResNet18 model
weights = ResNet18_Weights.DEFAULT  # or .IMAGENET1K_V1 if you want the older version explicitly
model = resnet18(weights=weights)

model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))  # 10 classes
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {running_loss:.4f} - Train Accuracy: {acc:.2f}%")

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

test_acc = 100 * correct / total
print(f"\nTest Accuracy: {test_acc:.2f}%")

# Using device: cuda

# Epoch 1/2
# Epoch [1/2] - Loss: 332.8913 - Train Accuracy: 84.73%

# Epoch 2/2
# Epoch [2/2] - Loss: 185.9975 - Train Accuracy: 91.27%

# Test Accuracy: 89.26%