from xml.parsers.expat import model
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

import time


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
                yaware=False,
                perform_eval=True):
    
    train_aug_loader, eval_aug_loader, train_eval_loader, eval_eval_loader = get_pretrain_loaders(
        CONFIG["DATA_DIR_EUROSAT_MS"],
        CONFIG["DATA_DIR_EUROSAT_RGB"],
        CONFIG["BATCH_SIZE"],
        task="yaware" if yaware else "simclr",
        build_eval_loaders=perform_eval,
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
    use_amp = (torch.device.type == "cuda")
    device_type = "cuda" if use_amp else "cpu"
    autocast_dtype = (
        torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    )
    scaler = torch.amp.GradScaler(device_type, enabled=use_amp)

    for epoch in trange(1, simclr_epochs + 1, desc="Epochs"):
        print(epoch,"/",simclr_epochs)

        # contrastive training
        model.train()
        total_loss = 0.0
        data_time = compute_time = 0.0
        end = time.perf_counter()
        for sample_train in train_aug_loader:
            data_time += time.perf_counter() - end

            x1, x2, *rest = sample_train
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            if rest: rest = [r.to(device, non_blocking=True) if hasattr(r,'to') else r for r in rest]

            t0 = time.perf_counter()
            x_cat = torch.cat([x1, x2], dim=0)
            # print("before amp")
            with torch.amp.autocast(device_type=device_type, dtype=autocast_dtype, enabled=use_amp):
                x_cat = torch.cat([x1, x2], 0)
                _, z_cat = model(x_cat)
                z1, z2 = torch.chunk(z_cat, 2, 0)
                # loss = loss_fn(z1, z2, coords) if yaware else loss_fn(z1, z2)
                loss = loss_fn(z1, z2, rest[0]) if yaware else loss_fn(z1, z2)


            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
                
            # z1, z2 = torch.chunk(z_cat, 2, dim=0)
            # loss = loss_fn(z1, z2, rest[0]) if yaware else loss_fn(z1, z2)
            # optimizer.zero_grad(set_to_none=True)
            # loss.backward()
            # optimizer.step()
            if scheduler is not None: scheduler.step()

            compute_time += time.perf_counter() - t0
            end = time.perf_counter()

        print(f"\n\n\n===\ndata: {data_time:.2f}s, compute: {compute_time:.2f}s")

        train_loss = total_loss / len(train_aug_loader.dataset)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for v in eval_aug_loader:

                if yaware:
                    v1, v2, meta, y = v
                else:
                    v1, v2, y = v
                
                v1, v2 = v1.to(device), v2.to(device)
                v_cat = torch.cat([v1, v2], dim=0)
                _, zv_cat = model(v_cat)
                zv1, zv2 = torch.chunk(zv_cat, 2, dim=0)
                if yaware:
                    res = loss_fn(zv1, zv2, meta)
                    l   = res[0] if isinstance(res, tuple) else res
                else:
                    l   = loss_fn(zv1, zv2)

                val_loss += l.item() * v1.size(0)
            val_loss /= len(eval_aug_loader.dataset)

            if perform_eval:
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
                        train_aug_loader,
                        eval_eval_loader,
                        device,
                        yaware,
                        C=CONFIG["C_LIN_PROBE"],             # stronger L2
                        max_iter=CONFIG["MAX_ITER_LIN_PROBE"],      # increase if not converging
                        scale_features="standard",
                    )

                    logistic_accuracy_train = run_logistic_probe(
                        model,
                        train_aug_loader,
                        train_aug_loader,  # use train loader for training
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
                    knn.fit(train_aug_loader)
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
            if perform_eval:
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
            else:
                wandb_run.log({
                    "epoch": epoch,
                    "simclr_train_loss": train_loss,
                    "simclr_val_loss": val_loss,
                })
 
        if epoch % EPOCH_SAVE_INTERVAL  == 0:
            t0 = time.perf_counter()

            checkpoint_path = os.path.join(full_model_path, f"{model_base_filename}_epoch_{epoch:03d}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            timings['checkpoint'] += (time.perf_counter() - t0)
    # evaluate the model at the end of training
    model.eval()
    def get_model_name():
        if CONFIG["ORIGINAL_Y_AWARE"]:
            return f"original_yaware"
        if CONFIG["Y_AWARE"]:
            return f"yaware"
        return "simclr"
    
    checkpoint_path = os.path.join(full_model_path, f"{model_base_filename}_epoch_{simclr_epochs:03d}.pth")
    torch.save(model.state_dict(), checkpoint_path)

    if wandb_run:
        artifact = wandb.Artifact(
            name=get_model_name(),
            type="model",
            description="SimCLR model trained on Eurosat dataset"
        )

        artifact.add_file(checkpoint_path)
        wandb_run.log_artifact(artifact)

    
    with torch.no_grad():
        final_contrast_acc_train = compute_contrastive_accuracy(
            model, train_aug_loader, device, yaware=yaware
        )
        print(f"Final contrastive accuracy on train split: {final_contrast_acc_train*100:.2f}%")

        final_contrast_acc = compute_contrastive_accuracy(
            model, eval_aug_loader, device, yaware=yaware
        )
        print(f"Final contrastive accuracy on val split: {final_contrast_acc*100:.2f}%")
        knn_train_acc, final_knn_acc, knn_train_acc_k1, final_knn_acc_k1 = get_knn_metrics(model, device, train_aug_loader, eval_aug_loader)

    if wandb_run:
        t0 = time.perf_counter()
        wandb_run.log({"final_contrastive_accuracy": final_contrast_acc})
        wandb_run.log({"final_contrastive_accuracy_train": final_contrast_acc_train})
        timings['logging'] += (time.perf_counter() - t0)
        wandb_run.log({"final_knn_acc": final_knn_acc})
        wandb_run.log({"final_knn_acc_k1": final_knn_acc_k1})
        wandb_run.log({"final_knn_train_acc": knn_train_acc})
        wandb_run.log({"final_knn_train_acc_k1": knn_train_acc_k1})
    print("\n=== Timing Breakdown ===")
    for stage, t in timings.items():
        print(f"{stage:15s}: {t:.1f}s ({t/simclr_epochs:.1f}s/epoch)")
    # return the filename of the last checkpoint    
    return checkpoint_path

@torch.no_grad()
def get_knn_metrics(model, device, train_aug_loader, eval_aug_loader):
    knn = WeightedKNNClassifier(
                model=model,
                device=device,
                k=CONFIG["K"],             
                normalize=True
            )
    
    knn.fit(train_aug_loader)
    knn_train_acc = knn.score(train_aug_loader)
    print(f"Final kNN (k={knn.k}) on train: {knn_train_acc*100:.2f}%")

    final_knn_acc = knn.score(eval_aug_loader)
    print(f"Final kNN (k={knn.k}) on val: {final_knn_acc*100:.2f}%")

    # Compute knn with k=1 too
    knn.k = 1
    knn.fit(train_aug_loader)
    knn_train_acc_k1 = knn.score(train_aug_loader)
    print(f"Final kNN (k=1) on train: {knn_train_acc_k1*100:.2f}%")

    final_knn_acc_k1 = knn.score(eval_aug_loader)
    print(f"Final kNN (k=1) on val: {final_knn_acc_k1*100:.2f}%")
    return knn_train_acc,final_knn_acc,knn_train_acc_k1,final_knn_acc_k1
