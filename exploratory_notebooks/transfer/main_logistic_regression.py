from simclr.config import CONFIG
from simclr.train import train_simclr
from simclr.data.mydataloaders import get_data_loaders_train_test_linear_probe

from utils.version_utils import configure_gpu_device
import os
import torch
from simclr.models.simclr import build_simclr_network
import argparse
import torch.nn.functional as F

from transfer.new_knn import NNClassifier                       
from transfer.new_logistic import SklearnLogisticClassifier    
from transfer.eval import evaluate                             


@torch.inference_mode()
def _extract_features(encoder: torch.nn.Module,
                      loader: torch.utils.data.DataLoader,
                      device: torch.device,
                      l2norm: bool = True):
    """
    Runs the frozen encoder on a loader and returns:
      X: torch.Tensor [N, D]  (features)
      y: torch.Tensor [N]     (labels)
    If l2norm=True, applies F.normalize to features (row-wise).
    """
    encoder.eval().to(device)
    feats, labs = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            f = encoder(imgs)
            if l2norm:
                f = F.normalize(f, dim=1)
            feats.append(f.cpu())
            labs.append(labels.cpu())
    X = torch.cat(feats, dim=0)
    y = torch.cat(labs, dim=0).long()
    return X, y


# Set up device
TARGET_GPU_INDEX = CONFIG["TARGET_GPU_INDEX"] if "TARGET_GPU_INDEX" in CONFIG else 0  # Default to 0 if not set
DEVICE = configure_gpu_device(TARGET_GPU_INDEX)

parser = argparse.ArgumentParser("SimCLR EuroSAT")
parser.add_argument("--dataset",    type=str,   default="eurosat",
                    help="dataset name (controls CIFAR‑stem in network.py)")
parser.add_argument("--model",      type=str,   default="resnet18",
                    choices=["resnet18","resnet34","resnet50","resnet101","resnet152"],
                    help="which ResNet depth to use")
parser.add_argument("--n_classes",  type=int,   default=10,
                    help="# of EuroSAT semantic classes")
parser.add_argument("--feature_dim",type=int,   default=512,
                    help="backbone output dim (for SimCLR we set fc→feature_dim)")
parser.add_argument("--proj_dim",   type=int,   default=CONFIG["PROJ_DIM"],
                    help="projection MLP output dim (usually 128)")

args = parser.parse_args([])

print(f"Arguments: {args}")


simclr_model = build_simclr_network(DEVICE, args)
seed = CONFIG["SEED"]
bs = CONFIG["BATCH_SIZE"]
epochs_simclr = CONFIG["EPOCHS_SIMCLR"]
simclr_lr = CONFIG["LR"]
lr_str = f"{simclr_lr:.0e}" if simclr_lr < 0.0001 else f"{simclr_lr:.6f}"
checkpoint_path = "/share/homes/carvalhj/projects/eurosat_preprocessing/exploratory_notebooks/notebooks/models/2025-07-29_19-41-31/simclr_seed42_bs256_temp0.2_Tepochs200_lr0.000375_epoch_200.pth"
folder_name = os.path.dirname(checkpoint_path)
print(f"Using model path: {checkpoint_path}")

if not os.path.exists(checkpoint_path):
    print(f"Model {checkpoint_path} does not exist. Please run the SimCLR pretraining first.")

state_dict = torch.load(checkpoint_path, map_location=torch.device(DEVICE), weights_only=True)
simclr_model.load_state_dict(state_dict)

encoder = getattr(simclr_model, "encoder", simclr_model)
encoder.eval().to(DEVICE)

train_loader, test_loader, num_classes = get_data_loaders_train_test_linear_probe(
    CONFIG["DATA_DIR_LOCAL"], CONFIG["BATCH_SIZE"]
)

Xtr_knn, ytr = _extract_features(encoder, train_loader, DEVICE, l2norm=True)
Xte_knn, yte = _extract_features(encoder, test_loader, DEVICE, l2norm=True)

Xtr_lr, _ = _extract_features(encoder, train_loader, DEVICE, l2norm=False)
Xte_lr, _ = _extract_features(encoder, test_loader, DEVICE, l2norm=False)

logreg = SklearnLogisticClassifier(
    scale_features="standard",
    solver="lbfgs",
    multi_class="multinomial",
    C=1.0,               
    max_iter=200,
    tol=1e-4,
    verbose=1,
)

logreg.fit(Xtr_lr.to(DEVICE), ytr.to(DEVICE))
probs_lr = logreg.predict_proba(Xte_lr.to(DEVICE)).cpu()  # [N, C]

evaluate(probs_lr, yte, num_classes, enable_wandb=False, output_file=os.path.join(folder_name, "logistic_regression_results.json"), task_type="multiclass")

knn = NNClassifier(
    num_classes=num_classes,
    k=5,
    tau=0.07,
    distance_fn="cosine",
    weighted=True,
    scale_features="norm",   
)
knn.fit(Xtr_knn.to(DEVICE), ytr.to(DEVICE))
probs_knn = knn.predict_proba(
    Xte_knn.to(DEVICE),
    batch_size=256,
    print_freq=10,
    enable_amp=(DEVICE.type == "cuda"),
).cpu()

evaluate(probs_knn, yte, num_classes, enable_wandb=False, output_file=os.path.join(folder_name, "knn_results.json"), task_type="multiclass")
