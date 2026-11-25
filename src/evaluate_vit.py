"""Evaluate a ViT-S/16 ChestX-ray14 checkpoint across Poisson noise levels.

From the project root:
    python -m src.evaluate_vit --epoch 3 --num-workers 4
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import timm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import ChestXrayDataset, PoissonNoise, RepeatChannels
from src.evaluate import evaluate_model

PATHOLOGY_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural Thickening", "Hernia",
]
PEAKS = ("Clean", 10000, 5000, 2000, 1000, 250, 25)


def build_transform(peak):
    steps = [
        transforms.Resize(
            (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
    ]
    if peak != "Clean":
        steps.append(PoissonNoise(peak=peak))
    steps.extend([
        RepeatChannels(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    return transforms.Compose(steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = (
        project_dir / "checkpoints" /
        f"vit_s16_adamw_clean_epoch_{args.epoch}.pth"
    )
    output_dir = project_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    macro_path = (
        output_dir /
        f"vit_clean_epoch{args.epoch}_macro_auroc.csv"
    )
    
    pathology_path = (
        output_dir /
        f"vit_clean_epoch{args.epoch}_pathology_auroc.csv"
    )
    print(f"Device: {device}", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    val_df = pd.read_csv(project_dir / "metadata" / "val.csv")
    data_dir = project_dir / "data" / "data"

    model = timm.create_model(
        "vit_small_patch16_224.augreg_in1k",
        pretrained=False,
        num_classes=len(PATHOLOGY_NAMES),
    )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded completed epoch: {checkpoint.get('epoch', 'unknown')}", flush=True)

    macro_rows = []
    pathology_frames = []
    model_name = f"ViT-S/16 clean Epoch {args.epoch}"
    for peak in PEAKS:
        print(f"\nEvaluating peak={peak}", flush=True)
        dataset = ChestXrayDataset(
            dataframe=val_df,
            data_dir=str(data_dir),
            transform=build_transform(peak),
            image_mode="L",
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=False,
        )

        # For repeatable sampling with a fixed loader/worker configuration.
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        start = time.perf_counter()
        with torch.inference_mode():
            per_pathology, macro_auc = evaluate_model(
                model, loader, device, PATHOLOGY_NAMES
            )
        elapsed = (time.perf_counter() - start) / 60
        
        macro_rows.append({
            "Model": model_name,
            "Peak": str(peak),
            "Macro_AUROC": float(macro_auc),
            "Evaluation_Minutes": elapsed,
        })
        
        pathology_result = per_pathology.copy()
        pathology_result.insert(0, "Peak", str(peak))
        pathology_result.insert(0, "Model", model_name)
        pathology_frames.append(pathology_result)

        # Save progressively so results survive a later interruption.
        pd.DataFrame(macro_rows).to_csv(macro_path, index=False)
        pd.concat(pathology_frames, ignore_index=True).to_csv(
            pathology_path, index=False
        )
        print(f"Peak {peak}: macro AUROC={macro_auc:.6f}; {elapsed:.2f} min", flush=True)
        del loader, dataset

    print(f"\nSaved: {macro_path}\nSaved: {pathology_path}", flush=True)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
