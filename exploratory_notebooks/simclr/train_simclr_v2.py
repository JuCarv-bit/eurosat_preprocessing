from transfer.logistic_regression import run_logistic_probe
from transfer.knn import WeightedKNNClassifier
from simclr.config import CONFIG
import torch
import os
import wandb
from simclr.models.loss import compute_contrastive_accuracy
import time
from tqdm.notebook import trange
from simclr.data.eurosat_datasets import get_pretrain_loaders

INTERVAL_EPOCHS_LINEAR_PROBE = CONFIG["INTERVAL_EPOCHS_LINEAR_PROBE"]
INTERVAL_EPOCHS_KNN = CONFIG["INTERVAL_EPOCHS_KNN"]
INTERVAL_CONTRASTIVE_ACC = CONFIG["INTERVAL_CONTRASTIVE_ACC"]

def train_simclr_v2_function(
                model,
                optimizer,
                loss_fn,
                device,
                simclr_epochs,
                feature_dim,
                num_classes,
                wandb_run=None,
                scheduler=None,
                seed=CONFIG["SEED"],
                yaware=False):
    
    train_aug_loader, eval_aug_loader, train_eval_loader, eval_eval_loader = get_pretrain_loaders(
        CONFIG["DATA_DIR_EUROSAT_MS"],
        CONFIG["DATA_DIR_EUROSAT_RGB"],
        CONFIG["BATCH_SIZE"],
        task="yaware" if yaware else "simclr",
        build_eval_loaders=True,
        use_test_as_eval=False,
        splits_dir=CONFIG["SPLITS_DIR"],
        meta_dir=CONFIG["SPLITS_META_DIR"],
        use_cache=True,
        seed=seed
     )
    
    model.to(device)
    model.train()
    print("Measuring time...")
    timings = {
        'load_batch': 0.0,
        'forward': 0.0,
        'loss+backward+opt': 0.0,
        'scheduler': 0.0,
        'val_forward': 0.0,
        'contrastive_acc': 0.0,
        'linear_probe': 0.0,
        'knn': 0.0,
        'checkpoint': 0.0,
        'logging': 0.0,
    }

    # create a YYYY-MM-DD_HH-MM-SS directory for saving models
    dirname = time.strftime("%Y-%m-%d_%H-%M-%S")
    full_model_path = os.path.join("models", dirname)
    if not os.path.exists(full_model_path):
        os.makedirs(full_model_path)

    EPOCH_SAVE_INTERVAL = CONFIG["EPOCH_SAVE_INTERVAL"]
    TEMPERATURE = CONFIG["TEMPERATURE"]

    bs = CONFIG["BATCH_SIZE"]
    lr = CONFIG["LR"]
    lr_str = f"{lr:.0e}" if lr < 0.0001 else f"{lr:.6f}"
    model_base_filename = f"simclr_seed{seed}_bs{bs}_temp{TEMPERATURE}_Tepochs{simclr_epochs}_lr{lr_str}"
    
    run_contrast_acc_val = 0.0
    contrastive_acc_train = 0.0
    logistic_accuracy = 0.0
    logistic_accuracy_train = 0.0
    knn_acc = 0.0
    knn_train_acc = 0.0

    original_yaware = CONFIG["ORIGINAL_Y_AWARE"]
    for epoch in trange(1, simclr_epochs + 1, desc="Epochs"):

        # contrastive training
        model.train()
        total_loss = 0.0
        for sample_train in train_aug_loader:
            t0 = time.perf_counter()

            if yaware:
                x1, x2, meta, _ = sample_train
            else:
                x1, x2, _ = sample_train
        
            x1, x2 = x1.to(device), x2.to(device)
            t1 = time.perf_counter()
            timings['load_batch'] += (t1 - t0)

            _, z1 = model(x1)
            _, z2 = model(x2)
            t2 = time.perf_counter()
            timings['forward'] += (t2 - t1)
            if yaware:
                # meta is  tensor([[ 48.4232,   7.7490], ...]) # e.g. [latitude, longitude]
                if original_yaware:
                    meta_second = meta[:, 1]  # e.g. [7.7490, 7.7490, ...] longitude
                    res = loss_fn(z1, z2, meta_second)
                else:
                    res = loss_fn(z1, z2, meta)
                if isinstance(res, tuple):
                    loss = res[0]
                else:
                    loss = res
            else:
                loss = loss_fn(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t3 = time.perf_counter()
            timings['loss+backward+opt'] += (t3 - t2)


            if scheduler is not None:
                scheduler.step()
            t4 = time.perf_counter()
            timings['scheduler'] += (t4 - t3)
            total_loss += loss.item() * x1.size(0)
        train_loss = total_loss / len(train_aug_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for v in eval_aug_loader:
                t0 = time.perf_counter()

                if yaware:
                    v1, v2, meta, y = v
                else:
                    v1, v2, y = v
                
                t1 = time.perf_counter()
                timings['val_forward'] += (t1 - t0)

                v1, v2 = v1.to(device), v2.to(device)
                _, zv1 = model(v1)
                _, zv2 = model(v2)
                if yaware:
                    res = loss_fn(zv1, zv2, meta)
                    l   = res[0] if isinstance(res, tuple) else res
                else:
                    l   = loss_fn(zv1, zv2)
                t2 = time.perf_counter()
                timings['val_forward'] += (t2 - t1)

                val_loss += l.item() * v1.size(0)
            val_loss /= len(eval_aug_loader.dataset)

        if epoch % INTERVAL_CONTRASTIVE_ACC == 0:
            t0 = time.perf_counter()

            run_contrast_acc_val = compute_contrastive_accuracy(
                model, eval_aug_loader, device, yaware=yaware
            )

            contrastive_acc_train = compute_contrastive_accuracy(
                model, train_aug_loader, device, yaware=yaware
            )
            timings['contrastive_acc'] += (time.perf_counter() - t0)


        if epoch % INTERVAL_EPOCHS_LINEAR_PROBE == 0:
            t0 = time.perf_counter()

            logistic_accuracy = run_logistic_probe(
                model,
                train_eval_loader,
                eval_eval_loader,
                device,
                yaware,
                C=CONFIG["C_LIN_PROBE"],             # stronger L2
                max_iter=CONFIG["MAX_ITER_LIN_PROBE"],      # increase if not converging
                scale_features="standard",
            )

            logistic_accuracy_train = run_logistic_probe(
                model,
                train_eval_loader,
                train_eval_loader,  # use train loader for training
                device,
                yaware,
                C=CONFIG["C_LIN_PROBE"],                # stronger L2
                max_iter=CONFIG["MAX_ITER_LIN_PROBE"],      # increase if not converging
                scale_features="standard",
            )
            timings['linear_probe'] += (time.perf_counter() - t0)

        
        if epoch % INTERVAL_EPOCHS_KNN == 0:
            t0 = time.perf_counter()

            # fit on probe_train_loader, eval on probe_val_loader
            knn = WeightedKNNClassifier(
                model=model,
                device=device,
                k=CONFIG["K"],             
                normalize=True
            )
            knn.fit(train_eval_loader)
            knn_acc = knn.score(eval_eval_loader)
            # knn_train_acc = knn.score(probe_train_loader)
            timings['knn'] += (time.perf_counter() - t0)


        msg = (f"Epoch {epoch:02d}/{simclr_epochs} | "
               f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | "
                f"Logistic Probe Acc (Val): {logistic_accuracy:.3f}, Logistic Probe Acc (Train): {logistic_accuracy_train:.3f} | "
               f"Contrastive Acc (Train): {contrastive_acc_train:.3f}, Contrastive Acc (Val): {run_contrast_acc_val:.3f}"
               f" | KNN Acc (Val): {knn_acc:.3f}")
        if epoch % INTERVAL_EPOCHS_LINEAR_PROBE == 0:
            print(msg)
        if wandb_run:
            wandb_run.log({
                "epoch": epoch,
                "simclr_train_loss": train_loss,
                "simclr_val_loss": val_loss,
                "logistic_probe_acc": logistic_accuracy,
                "logistic_probe_train_acc": logistic_accuracy_train,
                "contrastive_val_acc": run_contrast_acc_val,
                "contrastive_train_acc": contrastive_acc_train,
                "knn_val_acc": knn_acc,
            })

 
        if epoch % EPOCH_SAVE_INTERVAL  == 0:
            t0 = time.perf_counter()

            checkpoint_path = os.path.join(full_model_path, f"{model_base_filename}_epoch_{epoch:03d}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            timings['checkpoint'] += (time.perf_counter() - t0)


    simclr_lr = CONFIG["LR"]
    lr_str = f"{simclr_lr:.0e}" if simclr_lr < 0.0001 else f"{simclr_lr:.6f}"

    def get_model_name():
        if CONFIG["ORIGINAL_Y_AWARE"]:
            return f"original_yaware"
        if CONFIG["Y_AWARE"]:
            return f"yaware"
        return "simclr"

    artifact = wandb.Artifact(
        name=get_model_name(),
        type="model",
        description="SimCLR model trained on Eurosat dataset"
    )
    checkpoint_path = os.path.join(full_model_path, f"{model_base_filename}_epoch_{simclr_epochs:03d}.pth")

    artifact.add_file(checkpoint_path)
    wandb_run.log_artifact(artifact)


    final_contrast_acc = compute_contrastive_accuracy(
        model, eval_aug_loader, device, yaware=yaware
    )
    final_contrast_acc_train = compute_contrastive_accuracy(
        model, train_aug_loader, device, yaware=yaware
    )
    print(f"Final contrastive accuracy on val split: {final_contrast_acc*100:.2f}%")
    print(f"Final contrastive accuracy on train split: {final_contrast_acc_train*100:.2f}%")
       
    if wandb_run:
        t0 = time.perf_counter()
        wandb_run.log({"final_contrastive_accuracy": final_contrast_acc})
        wandb_run.log({"final_contrastive_accuracy_train": final_contrast_acc_train})
        timings['logging'] += (time.perf_counter() - t0)
    
    knn = WeightedKNNClassifier(
                model=model,
                device=device,
                k=CONFIG["K"],             
                normalize=True
            )
    
    knn.fit(train_eval_loader)
    knn_train_acc = knn.score(train_eval_loader)
    print(f"Final kNN (k={knn.k}) on train: {knn_train_acc*100:.2f}%")

    final_knn_acc = knn.score(eval_eval_loader)
    print(f"Final kNN (k={knn.k}) on val: {final_knn_acc*100:.2f}%")

    # Compute knn with k=1 too
    knn.k = 1
    knn.fit(train_eval_loader)
    knn_train_acc_k1 = knn.score(train_eval_loader)
    print(f"Final kNN (k=1) on train: {knn_train_acc_k1*100:.2f}%")

    final_knn_acc_k1 = knn.score(eval_eval_loader)
    print(f"Final kNN (k=1) on val: {final_knn_acc_k1*100:.2f}%")

    if wandb_run:
        wandb_run.log({"final_knn_acc": final_knn_acc})
        wandb_run.log({"final_knn_acc_k1": final_knn_acc_k1})
    if wandb_run:
        wandb_run.log({"final_knn_train_acc": knn_train_acc})
        wandb_run.log({"final_knn_train_acc_k1": knn_train_acc_k1})
    print("\n=== Timing Breakdown ===")
    for stage, t in timings.items():
        print(f"{stage:15s}: {t:.1f}s ({t/simclr_epochs:.1f}s/epoch)")
    # return the filename of the last checkpoint    
    return checkpoint_path
