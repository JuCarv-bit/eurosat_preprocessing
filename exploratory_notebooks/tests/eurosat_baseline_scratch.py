#!/usr/bin/env python
# coding: utf-8

# In[1]:


import wandb
wandb.login()  # Opens a browser once to authenticate
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms, models
from torchvision.models import resnet50
from itertools import product
import numpy as np
import os, ssl, zipfile, urllib
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm import tqdm
import ssl


# In[2]:


TARGET_GPU_INDEX = 1

if torch.cuda.is_available():
    if TARGET_GPU_INDEX < torch.cuda.device_count():
        DEVICE = torch.device(f"cuda:{TARGET_GPU_INDEX}")
        print(f"Successfully set to use GPU: {TARGET_GPU_INDEX} ({torch.cuda.get_device_name(TARGET_GPU_INDEX)})")
    else:
        print(f"Error: Physical GPU {TARGET_GPU_INDEX} is not available. There are only {torch.cuda.device_count()} GPUs (0 to {torch.cuda.device_count() - 1}).")
        print("Falling back to CPU.")
        DEVICE = torch.device("CPU")
else:
    print("CUDA is not available. Falling back to CPU.")
    DEVICE = torch.device("CPU")

print(f"Final DEVICE variable is set to: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"Current PyTorch default device: {torch.cuda.current_device()}")
    torch.cuda.set_device(TARGET_GPU_INDEX)
    print(f"Current PyTorch default device (after set_device): {torch.cuda.current_device()}")


dummy_tensor = torch.randn(2, 2)
dummy_tensor_on_gpu = dummy_tensor.to(DEVICE)
print(f"Dummy tensor is on device: {dummy_tensor_on_gpu.device}")


# In[ ]:


LOCAL_OR_COLAB = "LOCAL"
SEED           = 42
NUM_EPOCHS     = 34

TRAIN_FRAC = 0.8
VAL_FRAC   = 0.1
TEST_FRAC  = 0.1

# hyperparameter grid
# BATCH_SIZES = [64, 128, 256]
BATCH_SIZES = [64, 128]  # Using a single batch size for simplicity
LRS = [1e-4, 3e-4]

GRID        = [
    (3.75e-4, 0.5  ),
]

WEIGHT_DECAY = 0.5

BETAS=(0.9,0.98)
EPS = 1e-8

if LOCAL_OR_COLAB == "LOCAL":
    DATA_DIR = "/share/DEEPLEARNING/carvalhj/EuroSAT_RGB/"
else:
    data_root = "/content/EuroSAT_RGB"
    zip_path  = "/content/EuroSAT.zip"
    if not os.path.exists(data_root):
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib.request.urlretrieve(
            "https://madm.dfki.de/files/sentinel/EuroSAT.zip", zip_path
        )
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall("/content")
        os.rename("/content/2750", data_root)
    DATA_DIR = data_root

NUM_WORKERS = 4 


# In[4]:


def compute_mean_std(dataset, batch_size):
    loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=2)
    mean = 0.0
    std = 0.0
    n_samples = 0

    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)  # (B, C, H*W)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        n_samples += batch_samples

    mean /= n_samples
    std /= n_samples
    return mean.tolist(), std.tolist()

def get_data_loaders(data_dir, batch_size):

    base_tf = transforms.ToTensor()
    ds_all = datasets.ImageFolder(root=data_dir, transform=base_tf)
    labels = np.array(ds_all.targets)   # numpy array of shape (N,)
    num_classes = len(ds_all.classes)
    total_count = len(ds_all)
    print(f"Total samples in folder: {total_count}, classes: {ds_all.classes}")

    train_idx, val_idx, test_idx = get_split_indexes(labels, total_count)

    train_subset_for_stats = Subset(ds_all, train_idx)
    mean, std = compute_mean_std(train_subset_for_stats, batch_size)
    print(f"Computed mean: {mean}")
    print(f"Computed std:  {std}")

    tf_final = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    #  full ImageFolder but now with normalization baked in
    ds_all_norm = datasets.ImageFolder(root=data_dir, transform=tf_final)

    train_ds = Subset(ds_all_norm, train_idx)
    val_ds   = Subset(ds_all_norm, val_idx)
    test_ds  = Subset(ds_all_norm, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=NUM_WORKERS, generator=torch.Generator().manual_seed(SEED))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, generator=torch.Generator().manual_seed(SEED))
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, generator=torch.Generator().manual_seed(SEED))

    print(f"Train/Val/Test splits: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    return train_loader, val_loader, test_loader, num_classes

def get_proportion(num_classes, dataset):
    return np.bincount(np.array(dataset.dataset.targets)[dataset.indices], minlength=num_classes) / len(dataset)

def get_split_indexes(labels, total_count):
    n_train = int(np.floor(TRAIN_FRAC * total_count))
    n_temp = total_count - n_train   # this is val + test

    sss1 = StratifiedShuffleSplit(
        n_splits=1,
        train_size=n_train,
        test_size=n_temp,
        random_state=SEED
    )
    # Train and temp(val+test) indices
    train_idx, temp_idx = next(sss1.split(np.zeros(total_count), labels))

    n_val = int(np.floor(VAL_FRAC * total_count))
    n_test = total_count - n_train - n_val
    assert n_temp == n_val + n_test, "Fractions must sum to 1."

    labels_temp = labels[temp_idx]

    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        train_size=n_val,
        test_size=n_test,
        random_state=SEED
    )
    val_idx_in_temp, test_idx_in_temp = next(sss2.split(np.zeros(len(temp_idx)), labels_temp))

    val_idx = temp_idx[val_idx_in_temp]
    test_idx = temp_idx[test_idx_in_temp]

    assert len(train_idx) == n_train
    assert len(val_idx) == n_val
    assert len(test_idx) == n_test

    print(f"Stratified split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    return train_idx,val_idx,test_idx



# # Logistic regresssion with Scikit-learn for comparing linear probing

# In[5]:


BATCH_SIZE = BATCH_SIZES[0]
LEARNING_RATE, WEIGHT_DECAY = GRID[0]


# In[6]:


def get_common_feature_extractor_model():

    base_model = models.resnet50(weights=None) # No pre-trained ImageNet weights
    feature_extractor = nn.Sequential(*list(base_model.children())[:-1]) # Exclude the last fc layer
    for param in feature_extractor.parameters():
        param.requires_grad = False
    feature_extractor.eval() 
    feature_extractor.to(DEVICE)
    print("Common ResNet50 feature extractor (randomly initialized and frozen) created.")
    return feature_extractor

def extract_features(dataloader, model):
    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(DEVICE)
            features = model(images)
            features = features.squeeze(-1).squeeze(-1) # Flatten (batch_size, 2048, 1, 1) to (batch_size, 2048)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return np.vstack(all_features), np.concatenate(all_labels)

class PyTorchLinearProbingModel(nn.Module):

    def __init__(self, shared_feature_extractor, num_classes):
        super().__init__()
        self.backbone = shared_feature_extractor
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        feature_dim = 2048 
        self.linear_head = nn.Linear(feature_dim, num_classes)
        self.linear_head.to(DEVICE)

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
            features = features.squeeze(-1).squeeze(-1)
        logits = self.linear_head(features)
        return logits

def train_epoch(model, dataloader, criterion, optimizer, scheduler, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs} (Train)")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct_predictions/total_samples:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}") # Added LR to postfix
    return running_loss / total_samples, correct_predictions / total_samples

def evaluate_test_set_pytorch(model, dataloader, num_classes):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating Test Set (PyTorch)"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    print(classification_report(all_labels, all_preds, target_names=[f'class_{i}' for i in range(num_classes)]))
    test_accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
    return test_accuracy

def make_optimizer_scheduler(params, lr, wd, steps_per_epoch, epochs):
    total_steps  = epochs * steps_per_epoch
    warmup_steps = steps_per_epoch
    opt = optim.Adam(params, lr=lr, betas=(0.9,0.98), eps=1e-8, weight_decay=wd)
    sched = SequentialLR(
        opt,
        schedulers=[
            LinearLR(opt,  start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps),
            CosineAnnealingLR(opt, T_max=total_steps - warmup_steps)
        ],
        milestones=[warmup_steps]
    )
    return opt, sched

def validate_epoch(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    return running_loss / total_samples, correct_predictions / total_samples

def hyperparameter_search_pytorch(train_loader, val_loader, num_classes, common_feature_extractor):
    best_val_accuracy = -1.0
    best_params = {}

    print("\n--- Starting PyTorch Hyperparameter Search ---")
    
    for bs, (lr, wd) in product(BATCH_SIZES, GRID):
        epochs = NUM_EPOCHS

        print(f"\n--- Trying config: LR={lr}, WD={wd}, Epochs={epochs} ---")

        model = PyTorchLinearProbingModel(common_feature_extractor, num_classes=num_classes)
        criterion = nn.CrossEntropyLoss()

        steps_per_epoch = len(train_loader) 
        optimizer, scheduler = make_optimizer_scheduler(
            model.linear_head.parameters(),
            lr,
            wd,
            steps_per_epoch,
            epochs
        )

        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, epochs)
            val_loss, val_acc = validate_epoch(model, val_loader, criterion)

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_params = {
                'batch_size': bs,
                'learning_rate': lr,
                'weight_decay': wd,
                'epochs': epochs
            }
            print(f"New best validation accuracy: {best_val_accuracy:.4f} with params: {best_params}")

    print("\n--- PyTorch Hyperparameter Search Complete ---")
    print(f"Best validation accuracy found: {best_val_accuracy:.4f}")
    print(f"Best parameters: {best_params}")
    
    return best_params, best_val_accuracy 


# In[ ]:


train_loader, val_loader, test_loader, num_classes = get_data_loaders(DATA_DIR, BATCH_SIZE)

common_feature_extractor = get_common_feature_extractor_model()

X_train_features, y_train_labels = extract_features(train_loader, common_feature_extractor)
X_val_features, y_val_labels = extract_features(val_loader, common_feature_extractor)
X_test_features, y_test_labels = extract_features(test_loader, common_feature_extractor)

X_train_val_features = np.vstack((X_train_features, X_val_features))
y_train_val_labels = np.concatenate((y_train_labels, y_val_labels))

print(f"Train features shape: {X_train_features.shape}, labels shape: {y_train_labels.shape}")
print(f"Validation features shape: {X_val_features.shape}, labels shape: {y_val_labels.shape}")
print(f"Test features shape: {X_test_features.shape}, labels shape: {y_test_labels.shape}")

param_grid = {
    'C': [0.1, 1.0],
    'solver': ['lbfgs'],
    'max_iter': [1000]
}

logistic_classifier = LogisticRegression(random_state=SEED)
grid_search = GridSearchCV(
    logistic_classifier,
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0 
)
grid_search.fit(X_train_features, y_train_labels)

print(f"\nBest parameters found for Logistic Regression: {grid_search.best_params_}")
print(f"Best cross-validation accuracy for Logistic Regression: {grid_search.best_score_:.4f}")

best_logistic_model = grid_search.best_estimator_
y_pred_val_lr = best_logistic_model.predict(X_val_features)
val_accuracy_lr = accuracy_score(y_val_labels, y_pred_val_lr)
print(f"Logistic Regression Validation Accuracy: {val_accuracy_lr:.4f}")

final_logistic_model = LogisticRegression(**grid_search.best_params_, random_state=SEED)
final_logistic_model.fit(X_train_val_features, y_train_val_labels)

y_pred_test_lr = final_logistic_model.predict(X_test_features)
test_accuracy_lr = accuracy_score(y_test_labels, y_pred_test_lr)
print(f"Logistic Regression Test Accuracy: {test_accuracy_lr:.4f}")

print("\n--- Running PyTorch Linear Probing ---")

best_pytorch_params, best_pytorch_val_accuracy = hyperparameter_search_pytorch(
        train_loader, val_loader, num_classes, common_feature_extractor
    )

print(f"\nPyTorch Best Hyperparameters: {best_pytorch_params}")
print(f"PyTorch Best Validation Accuracy: {best_pytorch_val_accuracy:.4f}")

print("\n--- Training Final PyTorch Linear Probing Model on combined Train+Val set ---")
final_pytorch_model = PyTorchLinearProbingModel(common_feature_extractor, num_classes=num_classes)
criterion_final = nn.CrossEntropyLoss()


combined_train_val_dataset = ConcatDataset([train_loader.dataset, val_loader.dataset])
combined_train_val_loader = DataLoader(combined_train_val_dataset,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True,
                                        num_workers=NUM_WORKERS,
                                        generator=torch.Generator().manual_seed(SEED),
                                        pin_memory=True)

steps_per_epoch = len(combined_train_val_loader)
optimizer, scheduler = make_optimizer_scheduler(
    final_pytorch_model.linear_head.parameters(),
    LEARNING_RATE,
    WEIGHT_DECAY,
    steps_per_epoch,
    NUM_EPOCHS
)

print(f"Starting PyTorch Linear Probing Training for {NUM_EPOCHS} epochs on combined Train+Val set.")
for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train_epoch(final_pytorch_model, combined_train_val_loader, criterion_final, optimizer, scheduler, epoch, NUM_EPOCHS)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

test_accuracy_pytorch = evaluate_test_set_pytorch(final_pytorch_model, test_loader, num_classes)


print(f"Logistic Regression Test Accuracy: {test_accuracy_lr:.4f}")
print(f"PyTorch Linear Probing Test Accuracy: {test_accuracy_pytorch:.4f}")

