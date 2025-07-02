#!/usr/bin/env python
# coding: utf-8

# # Implementation of the SIMCLR with resnet18 backbone

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
from torchvision.models import resnet18
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


# In[3]:



print_versions()
set_seed(seed=42)

TARGET_GPU_INDEX = 0

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
    "BATCH_SIZE": 256,
    "LR": 3.75e-4,
    "WD": 0.5,
    "LR_LINEAR": 3.75e-4,
    "EPOCHS_SIMCLR": 60,
    "EPOCHS_LINEAR": 30,
    "TEMPERATURE": 0.2,
    "PROJ_DIM": 64,
    # "FEATURE_DIM": 2048, # resnet18 feature dimension = 2048 
    "FEATURE_DIM": 512, # ResNet18 feature dimension = 512
    "MEAN":  [0.3441457152366638, 0.3800985515117645, 0.40766361355781555],
    "STD":   [0.09299741685390472, 0.06464490294456482, 0.05413917079567909],
    "NUM_WORKERS": 8,
    "K": 5,
    "EPOCH_SAVE_INTERVAL": 10
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# split fractions
TRAIN_FRAC = 0.8
VAL_FRAC   = 0.1
TEST_FRAC  = 0.1

SEED = CONFIG["SEED"]

PRETRAINED = False

TEMPERATURE = CONFIG["TEMPERATURE"]

BETAS=(0.9,0.98)
EPS = 1e-8

LINEAR_PROB_TRAIN_SPLIT = 0.75

GLOBAL_SEED = CONFIG["SEED"]
NUM_WORKERS = CONFIG["NUM_WORKERS"]

EUROSAT_IMAGE_SIZE = (64, 64)
MODEL_INPUT_SIZE = [224, 224]
EPOCH_SAVE_INTERVAL = CONFIG["EPOCH_SAVE_INTERVAL"]


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
    loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=CONFIG["NUM_WORKERS"], generator=torch.Generator().manual_seed(SEED))
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

    dataset_val_no_transform = datasets.ImageFolder(root=data_dir, transform=None)
    val_subset_no_transform  = Subset(dataset_val_no_transform, val_indices)


    eval_transform, augment_transform = get_transforms(mean, std)

    dataset_eval = datasets.ImageFolder(
        root=data_dir,
        transform=eval_transform
    )
    val_subset = Subset(dataset_eval, val_indices)
    test_subset = Subset(dataset_eval, test_indices)
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

    return train_loader, val_loader, test_loader, val_subset_no_transform, num_classes

def get_transforms(mean, std):

    normalize = transforms.Normalize(mean=mean, std=std)
    # image_size = 224
    image_size = EUROSAT_IMAGE_SIZE[0]  # Use the defined EuroSAT image size 64

    eval_transform = transforms.Compose([
        transforms.Resize(image_size), # Use tuple for Resize
        transforms.CenterCrop(image_size), # Use tuple for CenterCrop
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    def get_color_distortion(s=1.0):
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
        return color_distort
        
    # Gaussian Blur Kernel size is 10% of image size, and must be odd
    k = int(0.1 * image_size) // 2 * 2 + 1
    gaussian_blur = transforms.GaussianBlur(kernel_size=k, sigma=(0.1, 2.0))

    augment_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),  
        transforms.RandomHorizontalFlip(p=0.5),
        get_color_distortion(1.0),                                
        transforms.RandomApply([gaussian_blur], p=0.5), 
        transforms.ToTensor(),
        normalize,
    ])
    
    return eval_transform, augment_transform

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
    
def run_linear_probe(model,
                     probe_train_loader,
                     probe_val_loader,
                     feature_dim,
                     num_classes,
                     device,
                     lr,
                     epochs):

    head = nn.Linear(feature_dim, num_classes).to(device)
    opt  = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9)
    crit = nn.CrossEntropyLoss()

    head.train()
    for _ in range(epochs):
        for imgs, labels in probe_train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            # extract features without touching the encoder
            with torch.no_grad():
                feats = model.encoder(imgs)
            # forward + backward only on the head
            logits = head(feats)
            loss   = crit(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

    head.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in probe_val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feats  = model.encoder(imgs)     
            preds  = head(feats).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    return correct / total


class LabeledEvalDataset(Dataset):
    def __init__(self, subset, transform):
        self.subset    = subset      # e.g. Subset(ImageFolder, train_indices)
        self.transform = transform   # eval_transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label


def get_probe_loaders(train_loader, val_loader, eval_transform, probe_batch_size):

    simclr_ds   = train_loader.dataset         # SimCLRDataset instance
    raw_subset  = simclr_ds.dataset            # e.g. Subset(ImageFolder, train_indices)

    # labeled Dataset for probe‐training
    probe_train_ds = LabeledEvalDataset(raw_subset, eval_transform)

    probe_train_loader = DataLoader(
        probe_train_ds,
        batch_size=probe_batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        generator=torch.Generator().manual_seed(SEED)
    )

    # use existing val_loader as the probe‐validation loader
    probe_val_loader = val_loader

    return probe_train_loader, probe_val_loader

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


# In[6]:


INTERVAL_EPOCHS_LINEAR_PROBE = 20
INTERVAL_EPOCHS_KNN = 10
INTERVAL_CONTRASTIVE_ACC = 10


def train_simclr(model,
                 train_loader,        # yields (x1, x2)
                 val_loader,          # labeled loader: yields (img, label)
                 probe_train_loader,  # labeled loader for probe head train
                 probe_val_loader,    # labeled loader for probe head val
                 optimizer,
                 criterion,
                 device,
                 simclr_epochs,
                 probe_lr,
                 probe_epochs,
                 feature_dim,
                 num_classes,
                 augment_transform,   # the same augment in SimCLRDataset
                 val_subset_no_transform,   # always PIL, TwoCrops works
                 wandb_run=None,
                 scheduler=None,
                 seed=SEED):
    model.to(device)

    bs = train_loader.batch_size
    temp = TEMPERATURE 
    lr = CONFIG["LR"]
    lr_str = f"{lr:.0e}" if lr < 0.0001 else f"{lr:.6f}"
    model_base_filename = f"simclr_seed{seed}_bs{bs}_temp{temp}_Tepochs{simclr_epochs}_lr{lr_str}"

    two_crop = TwoCropsTransform(augment_transform)
    raw_val_subset = val_subset_no_transform 
    contrastive_val_ds = SimCLRDataset(raw_val_subset, two_crop)
    contrastive_val_loader = torch.utils.data.DataLoader(
        contrastive_val_ds,
        batch_size=val_loader.batch_size,
        shuffle=False,
        num_workers=val_loader.num_workers,
        pin_memory=True
    )

    # model.train()
    contrast_acc = 0.0
    contrastive_acc_train = 0.0
    logistic_accuracy = 0.0
    logistic_accuracy_train = 0.0
    knn_acc = 0.0
    knn_train_acc = 0.0
    for epoch in range(1, simclr_epochs+1):
        # contrastive training
        model.train()
        total_loss = 0.0
        for x1, x2 in train_loader:
            x1, x2 = x1.to(device), x2.to(device)
            _, z1 = model(x1)
            _, z2 = model(x2)
            loss = criterion(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item() * x1.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for v1, v2 in contrastive_val_loader:
                v1, v2 = v1.to(device), v2.to(device)
                _, zv1 = model(v1)
                _, zv2 = model(v2)
                l = criterion(zv1, zv2)
                val_loss += l.item() * v1.size(0)
        val_loss /= len(contrastive_val_loader.dataset)


        if epoch % INTERVAL_CONTRASTIVE_ACC == 0:
            contrast_acc = compute_contrastive_accuracy(
                model, contrastive_val_loader, device
            )

            contrastive_acc_train = compute_contrastive_accuracy(
                model, train_loader, device
            )

        if epoch % INTERVAL_EPOCHS_LINEAR_PROBE == 0:
            logistic_accuracy = run_logistic_probe(
                model,
                probe_train_loader,
                probe_val_loader,
                feature_dim,       # e.g. 512
                num_classes,       # e.g. 10
                device,
                C=0.1,             # stronger L2
                max_iter=200,      # increase if not converging
                scale_features="standard"
            )

            logistic_accuracy_train = run_logistic_probe(
                model,
                probe_train_loader,
                probe_train_loader,  # use train loader for training
                feature_dim,          # e.g. 512
                num_classes,          # e.g. 10
                device,
                C=0.1,                # stronger L2
                max_iter=200,         # increase if not converging
                scale_features="standard"
            )

        if epoch % INTERVAL_EPOCHS_KNN == 0:

            # fit on probe_train_loader, eval on probe_val_loader
            knn = WeightedKNNClassifier(
                model=model,
                device=device,
                k=CONFIG["K"],             
                normalize=True
            )
            knn.fit(probe_train_loader)
            knn_acc = knn.score(probe_val_loader)
            # knn_train_acc = knn.score(probe_train_loader)

    
        

        msg = (f"Epoch {epoch:02d}/{simclr_epochs} | "
               f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | ")
        print(msg)
        if epoch % INTERVAL_EPOCHS_LINEAR_PROBE == 0:
            msg2 =(f"Logistic Probe Acc (Val): {logistic_accuracy:.3f}, Logistic Probe Acc (Train): {logistic_accuracy_train:.3f} | "
                f"Contrastive Acc (Train): {contrastive_acc_train:.3f}, Contrastive Acc (Val): {contrast_acc:.3f}"
                f" | KNN Acc (Val): {knn_acc:.3f}")
        
        if wandb_run:
            wandb_run.log({
                "epoch": epoch,
                "simclr_train_loss": train_loss,
                "simclr_val_loss": val_loss,
                "logistic_probe_acc": logistic_accuracy,
                "logistic_probe_train_acc": logistic_accuracy_train,
                "contrastive_val_acc": contrast_acc,
                "contrastive_train_acc": contrastive_acc_train,
                "knn_val_acc": knn_acc,
            })

 
        if epoch % EPOCH_SAVE_INTERVAL  == 0:
            checkpoint_path = os.path.join("models", f"{model_base_filename}_epoch_{epoch:03d}.pth")
            torch.save(model.state_dict(), checkpoint_path)

    seed = CONFIG["SEED"]
    bs = train_loader.batch_size
    epochs_simclr = CONFIG["EPOCHS_SIMCLR"]
    simclr_lr = CONFIG["LR"]
    lr_str = f"{simclr_lr:.0e}" if simclr_lr < 0.0001 else f"{simclr_lr:.6f}"
    model_path = f"models/simclr_seed{seed}_bs{bs}_temp{TEMPERATURE}_Tepochs{epochs_simclr}_lr{lr_str}.pth"
    if wandb_run:
        wandb_run.save("models/simclr_seed{seed}_bs{bs}_temp{TEMPERATURE}_Tepochs{epochs_simclr}_lr{simclr_lr}.pth")

    

    final_contrast_acc = compute_contrastive_accuracy(
        model, contrastive_val_loader, device
    )
    final_contrast_acc_train = compute_contrastive_accuracy(
        model, train_loader, device
    )
    print(f"Final contrastive accuracy on val split: {final_contrast_acc*100:.2f}%")
    print(f"Final contrastive accuracy on train split: {final_contrast_acc_train*100:.2f}%")
       
    if wandb_run:
        wandb_run.log({"final_contrastive_accuracy": final_contrast_acc})
        wandb_run.log({"final_contrastive_accuracy_train": final_contrast_acc_train})

    knn = WeightedKNNClassifier(
                model=model,
                device=device,
                k=CONFIG["K"],             
                normalize=True
            )
    
    knn.fit(probe_train_loader)
    knn_train_acc = knn.score(probe_train_loader)
    print(f"Final kNN (k={knn.k}) on train: {knn_train_acc*100:.2f}%")
    
    final_knn_acc = knn.score(probe_val_loader)
    print(f"Final kNN (k={knn.k}) on val: {final_knn_acc*100:.2f}%")

    if wandb_run:
        wandb_run.log({"final_knn_acc": final_knn_acc})

    
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")



# In[7]:



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


# In[8]:



def get_data_loaders_train_test_linear_probe(data_dir, batch_size):

   # get meand and std from the file where we saved it
    with open("models/mean_std.txt", "r") as f:
        lines = f.readlines()
        mean = [float(x) for x in lines[0].strip().split(": ")[1][1:-1].split(",")]
        std = [float(x) for x in lines[1].strip().split(": ")[1][1:-1].split(",")]
    
    print(f"Using mean: {mean}")
    print(f"Using std: {std}")

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
        transforms.RandomResizedCrop(EUROSAT_IMAGE_SIZE[0], scale=(0.5, 1.0)),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomApply([gaussian_blur], p=0.5),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
            transforms.Resize(EUROSAT_IMAGE_SIZE),   
            transforms.ToTensor(),
            normalize,
    ])
    
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


# In[9]:




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


# In[10]:



seeds = [GLOBAL_SEED]
for seed in seeds:
    print(f"\n=== Starting run with seed {seed} ===")
    set_seed(seed)
    
    data_dir = prepare_data()
    train_loader, val_loader, test_loader, val_subset_no_transform, num_classes = get_data_loaders(data_dir, CONFIG["BATCH_SIZE"])


# In[ ]:



seeds = [GLOBAL_SEED]
for seed in seeds:
    print(f"\n=== Starting run with seed {seed} ===")
    set_seed(seed)
    
    data_dir = prepare_data()
    train_loader, val_loader, test_loader, val_subset_no_transform, num_classes = get_data_loaders(data_dir, CONFIG["BATCH_SIZE"])

    base_encoder = resnet18(weights=None)
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
    epochs_simclr = CONFIG["EPOCHS_SIMCLR"]
    lr = CONFIG["LR"]
    wandb_run = wandb.init(
        project="eurosat-contrastive-scratch",
        name=f"BS{bs}_LR{lr:.0e}_SEED{seed}_TEMPERATURE{TEMPERATURE}_EPOCHS{epochs_simclr}",
        tags=["SimCLR", "EuroSAT", "Contrastive Learning"],
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

    eval_transform, augment_transform = get_transforms(
        mean =CONFIG["MEAN"],
        std = CONFIG["STD"]
    )  # these must match the transforms used in test_loader

    probe_train_loader, probe_val_loader = get_probe_loaders(
        train_loader,
        val_loader,
        eval_transform,               # must match transforms used in test_loader
        probe_batch_size=CONFIG["BATCH_SIZE"]
    )

    train_simclr(
        simclr_model,
        train_loader, val_loader,
        probe_train_loader, probe_val_loader,
        optimizer, loss_fn, DEVICE,
        simclr_epochs=CONFIG["EPOCHS_SIMCLR"],
        probe_lr=CONFIG["LR_LINEAR"],
        probe_epochs=1,            # 1 pass per epoch is typical
        feature_dim=CONFIG["FEATURE_DIM"],
        num_classes=num_classes,
        augment_transform=augment_transform,
        val_subset_no_transform=val_subset_no_transform,
        wandb_run=wandb_run,
        scheduler=scheduler,
        seed=seed
    )

    wandb_run.finish()



print("All runs completed.")
wandb.finish()
   


# In[ ]:




def run_logistic_probe_experiment(
    seed,
    train_loader,
    val_loader,
    test_loader,
    num_classes,
    simclr_model,
    bs
):
    # 1) prepare wandb & combined loader
    wandb.init(
        project="logistic_probe_eurosat-simclr",
        name=f"logistic_probe_seed{seed}_temperature{TEMPERATURE}_bs{bs}",
        config={
            "seed": seed,
            "temperature": TEMPERATURE,
            "batch_size": bs,
            "num_classes": num_classes,
            "C": 1.0,      # repurpose LR as inverse‐reg strength
            "max_iter": 500
        }
    )

    if val_loader is not None:
        train_val_loader = combine_train_val_loaders(train_loader, val_loader)
    else:
        train_val_loader = train_loader

    print(f"[Data] train+val loader has {len(train_val_loader)} batches")

    # 2) wrap frozen encoder in the sklearn probe
    probe = SklearnLogisticProbe(
        encoder=simclr_model.encoder,
        device=DEVICE,
        scale_features="standard",
        C=wandb.config.C,
        max_iter=wandb.config.max_iter,
        multi_class="multinomial",
        solver="lbfgs"
    )

    # 3) fit on train+val
    print("Fitting logistic regression probe…")
    probe.fit(train_val_loader)

    # 4) evaluate on train+val and test
    train_acc = probe.score(train_val_loader) * 100.0
    test_acc  = probe.score(test_loader)      * 100.0

    print(f"[Probe] Train+Val Acc: {train_acc:.2f}%,  Test Acc: {test_acc:.2f}%")

    wandb.log({
        "probe_trainval_accuracy": train_acc,
        "probe_test_accuracy": test_acc
    })

    # 5) save sklearn classifier (and optionally scaler & encoder weights)
    model_path = f"models/logistic_probe_seed{seed}_bs{bs}.pkl"
    joblib.dump({
        "clf": probe.clf,
        "scaler": probe.scaler,
        "encoder_state_dict": simclr_model.encoder.state_dict(),
        "config": CONFIG,
        "seed": seed
    }, model_path)
    print(f"Saved probe + encoder to {model_path}")

    return train_acc / 100.0, test_acc / 100.0


# In[ ]:


# get the saved model and run linear probe
seed = CONFIG["SEED"]
bs = CONFIG["BATCH_SIZE"]
epochs_simclr = CONFIG["EPOCHS_SIMCLR"]
simclr_lr = CONFIG["LR"]
lr_str = f"{simclr_lr:.0e}" if simclr_lr < 0.0001 else f"{simclr_lr:.6f}"
model_path = f"models/simclr_seed{seed}_bs{bs}_temp{TEMPERATURE}_Tepochs{epochs_simclr}_lr{lr_str}.pth"

if not os.path.exists(model_path):
    print(f"Model {model_path} does not exist. Please run the SimCLR pretraining first.")

base_encoder = resnet18(weights=None)
simclr_model = SimCLRModel(base_encoder, proj_dim=CONFIG["PROJ_DIM"])
checkpoint_path = model_path
state_dict = torch.load(checkpoint_path, map_location=torch.device(DEVICE), weights_only=True)
simclr_model.load_state_dict(state_dict)

# Perform linear probe on train+val as train set, and test as test set
train_loader, test_loader, num_classes = get_data_loaders_train_test_linear_probe(CONFIG["DATA_DIR_LOCAL"], CONFIG["BATCH_SIZE"])
run_logistic_probe_experiment(
    42,
    train_loader,
    None,  # No validation loader for linear probe
    test_loader,
    num_classes,
    simclr_model,
    bs
)


# In[ ]:


# grid search for best hyperparameters

batch_sizes_epochs = [
    (64, 35),
    (128, 40),
    (256, 100),
    (512, 100),
    (1024, 150),
]

learning_rates = [
    1e-3,
    3.75e-4,
    1e-4,
    3.75e-5,
    1e-5,
]

# use linspace for computing the temperature
temperatures = np.linspace(0.05, 0.5, 5).tolist() # [0.05, 0.1625, 0.275, 0.3875, 0.5]
temperatures.append(0.2)  # add the original temperature

gpu_indexes = [0, 1]
# put half of the experiments on each GPU
gpu_experiments = {0: [], 1: []}
all_acc = []

# train simclr with different hyperparameters and apply linear probe

