from pathlib import Path
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18

from src.dataset import ChestXrayDataset, PoissonNoise, RepeatChannels
from src.train import train


def main():
    project_dir = Path(__file__).resolve().parents[1]

    data_dir = project_dir / "data" / "data"
    checkpoint_path = (
        project_dir / "checkpoints" / "resnet18_adam_epoch_3.pth"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device, flush=True)

    train_df = pd.read_csv("metadata/train.csv")
    val_df = pd.read_csv("metadata/val.csv")

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        PoissonNoise(peak=25),
        RepeatChannels(),
        normalize
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        RepeatChannels(),
        normalize
    ])
    
    train_dataset = ChestXrayDataset(
        train_df,
        data_dir=str(data_dir),
        transform=train_transform,
        image_mode="L"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True
    )

    val_dataset = ChestXrayDataset(
        val_df,
        data_dir=str(data_dir),
        transform=val_transform,
        image_mode="L"
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True
    )

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 14)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print("Loaded epoch:", checkpoint["epoch"], flush=True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-5
    )

    criterion = nn.BCEWithLogitsLoss()

    train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=2,
        start_epoch=3,
        checkpoint_dir=str(project_dir / "checkpoints"),
        run_name="resnet18_poisson25",
        print_every=5
    )


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()