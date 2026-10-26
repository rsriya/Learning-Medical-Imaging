from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

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

    with torch.inference_mode():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Processed {batch_idx + 1}/{len(loader)} batches"
                )

    y_true = np.concatenate(all_labels, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)

    auc_scores = []

    for i in range(len(pathology_names)):
        # ROC-AUC requires both positive and negative examples.
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

    print(f"Macro ROC-AUC: {macro_auc:.4f}")

    return df, macro_auc