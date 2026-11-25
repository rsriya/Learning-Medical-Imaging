from pathlib import Path
import pandas as pd
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import ChestXrayDataset, RepeatChannels, PoissonNoise
from src.train import train, RowGroupBatchSampler


def main():
    seed = 42

    torch.manual_seed(seed)
    project_dir = Path(__file__).resolve().parents[1]

    data_dir = project_dir / "data" / "data"
    checkpoint_path = (
        project_dir / "checkpoints" / "vit_s16_adamw_clean_epoch_3.pth"
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device, flush=True)

    train_df = pd.read_csv(project_dir / "metadata" / "train.csv")
    val_df = pd.read_csv(project_dir / "metadata" / "val.csv")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )
    
    clean_transform = transforms.Compose([
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        RepeatChannels(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    
    train_dataset = ChestXrayDataset(
        dataframe=train_df,
        data_dir=str(data_dir),
        transform=clean_transform,
        image_mode="L"
    )
    batch_sampler = RowGroupBatchSampler(
        train_dataset,
        batch_size=32,
        seed=seed
    )
    batch_sampler.epoch = checkpoint["epoch"]
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True
    )

    val_dataset = ChestXrayDataset(
        val_df,
        data_dir=str(data_dir),
        transform=clean_transform,
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

    model = timm.create_model(
        "vit_small_patch16_224.augreg_in1k",
        pretrained=False,
        num_classes=14 #automatically replace the pretrained classification head with a new linear layer producing 14 outputs
    )
    model = model.to(device)


    model.load_state_dict(checkpoint["model_state_dict"])

    print("Loaded epoch:", checkpoint["epoch"], flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.05
    )

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = 1e-5
    criterion = nn.BCEWithLogitsLoss()

    model = train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=2,
        start_epoch=checkpoint["epoch"],
        scaler_state_dict=checkpoint.get("scaler_state_dict"),
        print_every=100,
        checkpoint_dir=str(project_dir / "checkpoints"),
        run_name="vit_s16_adamw_clean"
    )

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()