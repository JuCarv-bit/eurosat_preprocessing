#!/usr/bin/env python
# coding: utf-8

# # Implementation of the SIMCLR with resnet50 backbone

# In[1]:


from dotenv import load_dotenv
load_dotenv()       # reads .env and sets os.environ
import wandb
wandb.login()


# In[2]:


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



# In[3]:


print_versions()
set_seed(seed=42)

TARGET_GPU_INDEX = 2

DEVICE = configure_gpu_device(TARGET_GPU_INDEX)


# In[ ]:


# Prevent nondeterminism
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.enabled = False

CONFIG = {
    "LOCAL_OR_COLAB": "LOCAL",
    "DATA_DIR_LOCAL": "/share/DEEPLEARNING/carvalhj/EuroSAT_RGB/",
    "DATA_DIR_COLAB": "/content/EuroSAT_RGB",
    "ZIP_PATH": "/content/EuroSAT.zip",
    "EUROSAT_URL": "https://madm.dfki.de/files/sentinel/EuroSAT.zip",
    "SEED": 42,  
    "BATCH_SIZE": 128,
    "LR": 3.75e-4,
    "WD": 0.5,
    "LR_LINEAR": 3.75e-4,
    "EPOCHS_SIMCLR": 50,
    "EPOCHS_LINEAR": 30,
    "PROJ_DIM": 128,
    "FEATURE_DIM": 2048, # ResNet50 feature dimension = 2048
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# split fractions
TRAIN_FRAC = 0.8
VAL_FRAC   = 0.1
TEST_FRAC  = 0.1

SEED = CONFIG["SEED"]

PRETRAINED = False

TEMPERATURE = 0.2

BETAS=(0.9,0.98)
EPS = 1e-8

LINEAR_PROB_TRAIN_SPLIT = 0.75

GLOBAL_SEED = CONFIG["SEED"]
NUM_WORKERS = 4


# In[5]:


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


class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]
    
class SimCLRDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, _ = self.dataset[idx]
        x1, x2 = self.transform(x)
        return x1, x2


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

def get_data_loaders(data_dir, batch_size):

    dataset_for_stats = datasets.ImageFolder(
        root=data_dir,
        transform=transforms.ToTensor()
    )
    total_len = len(dataset_for_stats)
    labels = np.array(dataset_for_stats.targets)
    num_classes = len(dataset_for_stats.classes)
    print(f"Total samples in folder: {total_len}, classes: {dataset_for_stats.classes}")

    train_indices, val_indices, test_indices = get_split_indexes(labels, total_len)

    train_for_stats_subset = Subset(dataset_for_stats, train_indices)
    mean, std = compute_mean_std(train_for_stats_subset, batch_size)
    print(f"Computed mean: {mean}")
    print(f"Computed std:  {std}")
    # save the mean and std to a file
    os.makedirs("models", exist_ok=True)
    with open("models/mean_std.txt", "w") as f:
        f.write(f"mean: {mean}\n")
        f.write(f"std: {std}\n")
    print("Mean and std saved to models/mean_std.txt")

    dataset_train_no_transform = datasets.ImageFolder(
        root=data_dir,
        transform=None
    )
    train_subset_no_transform = Subset(dataset_train_no_transform, train_indices)

    eval_transform = transforms.Compose([
        transforms.Resize(72),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    dataset_eval = datasets.ImageFolder(
        root=data_dir,
        transform=eval_transform
    )
    val_subset = Subset(dataset_eval, val_indices)
    test_subset = Subset(dataset_eval, test_indices)

    normalize = transforms.Normalize(mean=mean, std=std)
    color_jitter = transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1
    )
    gaussian_blur = transforms.GaussianBlur(
        kernel_size=7,
        sigma=(0.1, 2.0)
    )
    augment_transform = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.5, 1.0)),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomApply([gaussian_blur], p=0.5),
        transforms.ToTensor(),
        normalize,
    ])
    simclr_transform = TwoCropsTransform(augment_transform)
    train_ds_simclr = SimCLRDataset(train_subset_no_transform, simclr_transform)

    train_loader = DataLoader(
        train_ds_simclr,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        generator=torch.Generator().manual_seed(SEED)
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        generator=torch.Generator().manual_seed(SEED)
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        generator=torch.Generator().manual_seed(SEED)
    )

    print(f"Train/Val/Test loaders: {len(train_loader)}/{len(val_loader)}/{len(test_loader)} batches")

    return train_loader, val_loader, test_loader, num_classes

    

 
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

def train_simclr(model, train_loader, val_loader, optimizer, criterion, device, epochs, wandb_run=None, scheduler=None, seed=SEED, bs=CONFIG["BATCH_SIZE"]):
    model.train()
    model.to(device)
    print("Starting SimCLR pretraining...")
   
    for epoch in range(epochs):
        total_loss = 0
        # Train loss
        for (x1, x2) in train_loader:
            x1, x2 = x1.to(device), x2.to(device)
            _, z1 = model(x1)
            _, z2 = model(x2)
            loss = criterion(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            total_loss += loss.item()
     
        avg_train_loss = total_loss / len(train_loader)

        # # Validation loss
        avg_val_loss = -1  # Placeholder, as validation loss is not computed in this example

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        if wandb_run:
            wandb_run.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss
            })

    print("Finished SimCLR pretraining.")
    print("Saving encoder...")
    torch.save(model.state_dict(), f"models/simclr_model_seed{seed}_temperature{TEMPERATURE}_bs{bs}.pth")
    

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

def make_optimizer_scheduler(params, lr, wd, steps_per_epoch, epochs):
    total_steps  = epochs * steps_per_epoch
    warmup_steps = steps_per_epoch
    opt = optim.AdamW(params, lr=lr, betas=(0.9,0.98), eps=1e-8, weight_decay=wd)
    sched = SequentialLR(
        opt,
        schedulers=[
            LinearLR(opt,  start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps),
            CosineAnnealingLR(opt, T_max=total_steps - warmup_steps)
        ],
        milestones=[warmup_steps]
    )
    return opt, sched


# In[6]:

def get_data_loaders_train_test_linear_probe(data_dir, batch_size):

   # get meand and std from the file where we saved it
    with open("models/mean_std.txt", "r") as f:
        lines = f.readlines()
        mean = [float(x) for x in lines[0].strip().split(": ")[1][1:-1].split(",")]
        std = [float(x) for x in lines[1].strip().split(": ")[1][1:-1].split(",")]
    
    print(f"Using mean: {mean}")
    print(f"Using std: {std}")

    
    dataset_for_stats = datasets.ImageFolder(
        root=data_dir,
        transform=transforms.ToTensor()
    )
    total_len = len(dataset_for_stats)
    labels = np.array(dataset_for_stats.targets)
    print(f"Total samples in folder: {total_len}, classes: {dataset_for_stats.classes}")

    train_indices, val_indices, test_indices = get_split_indexes(labels, total_len)
    dataset_eval = datasets.ImageFolder(
        root=data_dir,
        transform=eval_transform
    )
    test_subset = Subset(dataset_eval, test_indices)
    train_val_indices = np.concatenate((train_indices, val_indices))

    # get dataloaders
    dataset_train_val = datasets.ImageFolder(
        root=data_dir,
        transform=augment_transform
    )
    train_val_subset = Subset(dataset_train_val, train_val_indices.tolist())
    train_loader = DataLoader(
        train_val_subset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        generator=torch.Generator().manual_seed(SEED)
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        generator=torch.Generator().manual_seed(SEED)
    )
    print(f"Train/Test loaders: {len(train_loader)}/{len(test_loader)} batches")
    assert total_len == len(train_val_subset) + len(test_subset), "Total length mismatch after split."
    return train_loader, test_loader, len(dataset_eval.classes)


# In[7]:


def run_linear_probe_experiment(seed, train_loader, val_loader, test_loader, num_classes, simclr_model, bs):
    
    linear_probe_layer = nn.Linear(CONFIG["FEATURE_DIM"], num_classes).to(DEVICE)
    linear_optimizer = optim.Adam(linear_probe_layer.parameters(), lr=CONFIG["LR_LINEAR"], weight_decay=CONFIG["WD"])
    linear_criterion = nn.CrossEntropyLoss()
    steps_per_epoch = len(train_loader)
    linear_scheduler = SequentialLR(
        linear_optimizer,
        schedulers=[
            LinearLR(linear_optimizer, start_factor=1e-6, end_factor=1.0, total_iters=steps_per_epoch),
            CosineAnnealingLR(linear_optimizer, T_max=CONFIG["EPOCHS_LINEAR"] * steps_per_epoch - steps_per_epoch)
        ],
        milestones=[steps_per_epoch]
    )

    if val_loader is not None:
        train_val_loader = combine_train_val_loaders(train_loader, val_loader)
    else:
        train_val_loader = train_loader
    
    print(f"Train + Val loader: {len(train_val_loader)} batches")
    simclr_model.eval()
    # freeze the encoder
    with torch.no_grad():
        for param in simclr_model.encoder.parameters():
            param.requires_grad = False

    simclr_model.to(DEVICE)
    wandb.init(
        project="linear_probe_eurosat-simclr",
        name=f"linear_probe_seed{seed}_temperature{TEMPERATURE}_bs{bs}",
        config={
            "seed": seed,
            "temperature": TEMPERATURE,
            "batch_size": bs,
            "epochs_linear": CONFIG["EPOCHS_LINEAR"],
            "learning_rate_linear": CONFIG["LR_LINEAR"],
            "weight_decay_linear": CONFIG["WD"],
            "num_classes": num_classes
        }
    )
    wandb.watch(simclr_model.encoder, log="all", log_freq=100)
    wandb.watch(linear_probe_layer, log="all", log_freq=100)
    print("Starting linear probe training...")

    linear_probe_layer.train()
    for epoch in range(CONFIG["EPOCHS_LINEAR"]):
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in train_val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            features, _ = simclr_model(images)
            outputs = linear_probe_layer(features)
            loss = linear_criterion(outputs, labels)

            linear_optimizer.zero_grad()
            loss.backward()
            linear_optimizer.step()
            linear_scheduler.step()

            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total * 100
        total_epochs = CONFIG["EPOCHS_LINEAR"]
        print(f"[Linear Probe] Epoch {epoch+1}/{total_epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        wandb.log({
            "linear_probe_loss": avg_loss,
            "linear_probe_accuracy": accuracy,
            "epoch": epoch + 1
        })
        linear_prob_acc = accuracy
        linear_prob_loss = avg_loss
    test_accuracy = evaluate(linear_probe_layer, simclr_model.encoder, test_loader, DEVICE)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    wandb.log({
        "test_accuracy": test_accuracy,
        "linear_probe_loss": linear_prob_loss,
        "linear_probe_accuracy": linear_prob_acc
    })
    print(f"Linear probe accuracy: {linear_prob_acc:.2f}%, Test accuracy: {test_accuracy:.2f}%")
    # Save the linear probe model
    torch.save({
        'linear_probe_state_dict': linear_probe_layer.state_dict(),
        'simclr_encoder_state_dict': simclr_model.encoder.state_dict(),
        'config': CONFIG,
        'seed': seed
    }, f"models/linear_probe_seed{seed}_temperature{TEMPERATURE}_bs{bs}.pth")

def combine_train_val_loaders(train_loader, val_loader):
    train_ds = train_loader.dataset
    val_ds   = val_loader.dataset

    combined_ds = ConcatDataset([train_ds, val_ds])
    train_val_loader = DataLoader(
        combined_ds,
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        generator=torch.Generator().manual_seed(SEED)
    )
    
    return train_val_loader


# In[8]:


seeds = [GLOBAL_SEED]
for seed in seeds:
    print(f"\n=== Starting run with seed {seed} ===")
    set_seed(seed)
    
    data_dir = prepare_data()
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(data_dir, CONFIG["BATCH_SIZE"])

    # Initialize base encoder and SimCLR model
    base_encoder = resnet50(weights=None)
    simclr_model = SimCLRModel(base_encoder, proj_dim=CONFIG["PROJ_DIM"])
    # optimizer = optim.Adam(simclr_model.parameters(), lr=CONFIG["LR"])
    wd =  0.5 
    optimizer, scheduler = make_optimizer_scheduler(
        simclr_model.parameters(),
        CONFIG["LR"],
        CONFIG["WD"],
        len(train_loader),
        CONFIG["EPOCHS_SIMCLR"]
        )
    
    bs = CONFIG["BATCH_SIZE"]
    loss_fn = NTXentLoss(bs, temperature=TEMPERATURE, device=DEVICE)

    print("Starting SimCLR training...")
    lr = CONFIG["LR"]
    wandb_run = wandb.init(
        project="eurosat-contrastive-scratch-grid-search",
        name=f"BS{bs}_LR{lr:.0e}_SEED{seed}_TEMPERATURE{TEMPERATURE}",
        config={
            "seed": seed,
            "temperature": TEMPERATURE,
            "model": "SimCLR",
            "dataset": "EuroSAT",
            "batch_size": bs,
            "learning_rate": CONFIG["LR"],
            "epochs": CONFIG["EPOCHS_SIMCLR"],
            "proj_dim": CONFIG["PROJ_DIM"],
            "feature_dim": CONFIG["FEATURE_DIM"],
            "pretrained": PRETRAINED,
        }
    )
    train_simclr(simclr_model, train_loader, val_loader, optimizer, loss_fn, DEVICE, CONFIG["EPOCHS_SIMCLR"], wandb_run=wandb_run, scheduler=scheduler, seed=seed, bs=bs)
    wandb_run.finish()

print("All runs completed.")
wandb.finish()
   

# In[9]:


# get the saved model and run linear probe
seed = CONFIG["SEED"]
bs = CONFIG["BATCH_SIZE"]
model_path = f"models/simclr_model_seed{seed}_temperature{TEMPERATURE}_bs{bs}.pth"

if not os.path.exists(model_path):
    print(f"Model {model_path} does not exist. Please run the SimCLR pretraining first.")

base_encoder = resnet50(weights=None)
simclr_model = SimCLRModel(base_encoder, proj_dim=CONFIG["PROJ_DIM"])
checkpoint_path = f"models/simclr_model_seed{seed}_temperature{TEMPERATURE}_bs{bs}.pth"
state_dict = torch.load(checkpoint_path, map_location=torch.device(DEVICE))
simclr_model.load_state_dict(state_dict)


train_loader, test_loader, num_classes = get_data_loaders_train_test_linear_probe(CONFIG["DATA_DIR_LOCAL"], CONFIG["BATCH_SIZE"])
# Run linear probe experiment
run_linear_probe_experiment(
    42,
    train_loader,
    None,  # No validation loader for linear probe
    test_loader,
    num_classes,
    simclr_model,
    bs
)

