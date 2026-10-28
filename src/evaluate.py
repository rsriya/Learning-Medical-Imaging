from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
import time

def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(
        Path(checkpoint_path),
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print("Loaded epoch:", checkpoint["epoch"])

    return model

def evaluate_model(model, loader, device, pathology_names):
    model.eval()

    all_labels = []
    all_probs = []

    total_batches = len(loader)
    start_time = time.perf_counter()
    last_print_time = start_time

    with torch.inference_mode():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking=True)

            logits = model(images)
            probs = torch.sigmoid(logits)

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

            completed = batch_idx + 1

            if completed % 100 == 0 or completed == total_batches:
                now = time.perf_counter()

                elapsed = now - start_time
                interval = now - last_print_time
                avg_batch_time = elapsed / completed
                remaining = (total_batches - completed) * avg_batch_time

                print(
                    f"Processed {completed}/{total_batches} batches | "
                    f"Interval: {interval:.1f} s | "
                    f"Elapsed: {elapsed/60:.1f} min | "
                    f"ETA: {remaining/60:.1f} min",
                    flush=True
                )

                last_print_time = now

    y_true = np.concatenate(all_labels, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)

    auc_scores = []

    for i in range(len(pathology_names)):
        if np.unique(y_true[:, i]).size < 2:
            auc_scores.append(np.nan)
        else:
            auc_scores.append(
                roc_auc_score(y_true[:, i], y_prob[:, i])
            )

    df = pd.DataFrame({
        "Name": pathology_names,
        "ROC AUC": auc_scores,
    })

    macro_auc = np.nanmean(auc_scores)

    total_time = time.perf_counter() - start_time

    print(f"\nMacro ROC-AUC: {macro_auc:.4f}")
    print(f"Total evaluation time: {total_time/60:.2f} min")

    return df, macro_auc