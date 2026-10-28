
from pathlib import Path
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18

from src.dataset import ChestXrayDataset, RepeatChannels
from src.evaluate import evaluate_model


PATHOLOGY_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural Thickening", "Hernia"
]

CHECKPOINTS = {
    #"Original": "resnet18_adam_epoch_3.pth",
    #"Poisson_Epoch4": "resnet18_poisson250_epoch_4.pth",
    #"Poisson_Epoch5": "resnet18_poisson250_epoch_5.pth",
    "Poisson25_Epoch4": "resnet18_poisson25_epoch_4.pth",    
}

NOISE_LEVELS = [
    None, 10000, 5000, 2000, 1000, 250, 25
]

SEED = 42


class DeterministicPoissonNoise:
    """
    Generate reproducible Poisson noise per validation sample.

    Uses a fixed seed based on the sample's image tensor
    position in the sorted validation dataset, via an
    index-aware dataset wrapper below.
    """

    def __init__(self, peak):
        self.peak = peak

    def __call__(self, image, index):
        generator = torch.Generator()
        generator.manual_seed(SEED + index)

        # torch.poisson supports a generator argument.
        noisy = torch.poisson(
            image * self.peak,
            generator=generator
        ) / self.peak

        return noisy.clamp(0, 1)


class IndexedNoiseDataset(torch.utils.data.Dataset):
    """
    Applies deterministic noise before channel repetition
    and ImageNet normalization.
    """

    def __init__(self, base_dataset, peak):
        self.base_dataset = base_dataset
        self.peak = peak
        self.noise = (
            DeterministicPoissonNoise(peak)
            if peak is not None
            else None
        )

        self.post_transform = transforms.Compose([
            RepeatChannels(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, label = self.base_dataset[index]

        if self.noise is not None:
            image = self.noise(image, index)

        image = self.post_transform(image)

        return image, label


def load_model(checkpoint_path, device):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 14)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model


def main():
    project_dir = Path(__file__).resolve().parents[1]
    data_dir = project_dir / "data" / "data"

    output_dir = project_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device, flush=True)

    # Read validation metadata once.
    val_df = pd.read_csv(
        project_dir / "metadata" / "val.csv"
    )

    val_df = val_df.sort_values(
        ["shard", "row_index"]
    ).reset_index(drop=True)

    # Base dataset returns single-channel tensors in [0, 1].
    base_dataset = ChestXrayDataset(
        val_df,
        data_dir=str(data_dir),
        transform=transforms.ToTensor(),
        image_mode="L"
    )

    macro_results = []
    pathology_results = []

    # Evaluate one noise condition at a time.
    # All checkpoints receive identical images at that level.
    for peak in NOISE_LEVELS:

        noise_label = "Clean" if peak is None else str(peak)

        print(
            f"\n{'=' * 55}\n"
            f"Noise condition: {noise_label}\n"
            f"{'=' * 55}",
            flush=True
        )

        dataset = IndexedNoiseDataset(
            base_dataset,
            peak=peak
        )

        loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=6,
            pin_memory=True,
            persistent_workers=True
        )

        for model_name, filename in CHECKPOINTS.items():

            checkpoint_path = (
                project_dir / "checkpoints" / filename
            )

            print(
                f"\nEvaluating {model_name} at {noise_label}",
                flush=True
            )

            model = load_model(
                checkpoint_path,
                device
            )

            start = time.perf_counter()

            df, macro_auc = evaluate_model(
                model,
                loader,
                device,
                PATHOLOGY_NAMES
            )

            elapsed = time.perf_counter() - start

            print(
                f"Macro AUROC: {macro_auc:.4f} "
                f"| Time: {elapsed / 60:.2f} min",
                flush=True
            )

            macro_results.append({
                "Model": model_name,
                "Peak": noise_label,
                "Macro_AUROC": macro_auc,
                "Evaluation_Minutes": elapsed / 60
            })

            for _, row in df.iterrows():
                pathology_results.append({
                    "Model": model_name,
                    "Peak": noise_label,
                    "Pathology": row["Name"],
                    "AUROC": row["ROC AUC"]
                })

            # Save incrementally to avoid losing completed runs.
            pd.DataFrame(macro_results).to_csv(
                output_dir / "macro_auroc_poisson25.csv",
                index=False
            )

            pd.DataFrame(pathology_results).to_csv(
                output_dir / "pathology_auroc_poisson25.csv",
                index=False
            )

            del model

        # Shut down workers before creating the next loader.
        del loader

    # Display summary table.
    results_df = pd.DataFrame(macro_results)

    table = results_df.pivot(
        index="Peak",
        columns="Model",
        values="Macro_AUROC"
    )

    order = [
        "Clean", "10000", "5000", "2000",
        "1000", "250", "25"
    ]

    table = table.reindex(order)

    print("\nFinal AUROC comparison:")
    print(table.round(4).to_string())

    table.to_csv(
        output_dir / "auroc_comparison.csv"
    )

    # Plot robustness curves.
    fig, ax = plt.subplots(figsize=(9, 5.5))

    x = np.arange(len(order))

    for model_name in CHECKPOINTS:
        ax.plot(
            x,
            table[model_name],
            marker="o",
            linewidth=2,
            label=model_name
        )

    ax.set_xticks(x)
    ax.set_xticklabels(order)

    ax.set_xlabel("Synthetic Poisson peak")
    ax.set_ylabel("Macro ROC-AUC")
    ax.set_title(
        "ResNet-18 robustness to synthetic Poisson noise"
    )

    ax.set_ylim(0.45, 0.90)
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir / "robustness_curves.png",
        dpi=200
    )

    plt.close(fig)

    print(
        f"\nResults saved to: {output_dir}",
        flush=True
    )


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
