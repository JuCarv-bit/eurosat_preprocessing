from simclr.data.datamodule import SimCLRDataset, TwoCropsTransform
from transfer.logistic_regrssion import  SklearnLogisticProbe, run_logistic_probe
from transfer.knn import WeightedKNNClassifier
from simclr.config import CONFIG
from simclr.utils.scheduler import make_optimizer_scheduler
from simclr.utils.misc import evaluate
import torch
import os
import wandb
from simclr.models.loss import NTXentLoss, compute_contrastive_accuracy
from simclr.data.mydataloaders import get_data_loaders_train_test_linear_probe

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
                 seed=CONFIG["SEED"]):
    model.to(device)

    EPOCH_SAVE_INTERVAL = CONFIG["EPOCH_SAVE_INTERVAL"]
    TEMPERATURE = CONFIG["TEMPERATURE"]

    bs = train_loader.batch_size
    lr = CONFIG["LR"]
    lr_str = f"{lr:.0e}" if lr < 0.0001 else f"{lr:.6f}"
    model_base_filename = f"simclr_seed{seed}_bs{bs}_temp{TEMPERATURE}_Tepochs{simclr_epochs}_lr{lr_str}"

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

        contrast_acc = compute_contrastive_accuracy(
            model, contrastive_val_loader, device
        )

        contrastive_acc_train = compute_contrastive_accuracy(
            model, train_loader, device
        )

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
               f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | "
                f"Logistic Probe Acc (Val): {logistic_accuracy:.3f}, Logistic Probe Acc (Train): {logistic_accuracy_train:.3f} | "
               f"Contrastive Acc (Train): {contrastive_acc_train:.3f}, Contrastive Acc (Val): {contrast_acc:.3f}"
               f" | KNN Acc (Val): {knn_acc:.3f}")
        print(msg)
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
    
    final_knn_acc = knn.score(probe_val_loader)
    print(f"Final kNN (k={knn.k}) on val: {final_knn_acc*100:.2f}%")

    if wandb_run:
        wandb_run.log({"final_knn_acc": final_knn_acc})

    
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


