import os
import time
import torch

def evaluate_loss(loader, model, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device, dtype=torch.float32)

            scores = model(images)
            loss = criterion(scores, labels)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def train(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    device,
    epochs=1,
    print_every=100,
    checkpoint_dir="checkpoints",
    run_name="resnet18_adamw",
    start_epoch=0,
):
    model = model.to(device)
    os.makedirs(checkpoint_dir, exist_ok=True)

    overall_start = time.perf_counter()

    for epoch in range(epochs):
        epoch_number = start_epoch + epoch + 1
        epoch_start = time.perf_counter()
        last_print_time = epoch_start

        model.train()
        total_loss = 0.0
        total_samples = 0
        total_batches = len(train_loader)

        for step, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device, dtype=torch.float32)

            scores = model(images)
            loss = criterion(scores, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            completed = step + 1

            if completed % print_every == 0 or completed == total_batches:
                now = time.perf_counter()
                elapsed = now - epoch_start
                interval = now - last_print_time

                avg_loss = total_loss / total_samples
                avg_batch_time = elapsed / completed
                remaining = (total_batches - completed) * avg_batch_time

                print(
                    f"Epoch {epoch_number} | "
                    f"Batch {completed}/{total_batches} | "
                    f"Train loss: {avg_loss:.4f} | "
                    f"Interval: {interval / 60:.1f} min | "
                    f"Elapsed: {elapsed / 60:.1f} min | "
                    f"ETA: {remaining / 60:.1f} min"
                )

                last_print_time = now

        train_loss = total_loss / total_samples

        val_loss = evaluate_loss(
            val_loader, model, criterion, device
        )

        epoch_time = time.perf_counter() - epoch_start

        checkpoint = {
            "epoch": epoch_number,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"{run_name}_epoch_{epoch_number}.pth",
        )
        torch.save(checkpoint, checkpoint_path)

        print(
            f"\nEpoch {epoch_number} complete | "
            f"Train loss: {train_loss:.4f} | "
            f"Validation loss: {val_loss:.4f} | "
            f"Epoch time: {epoch_time / 60:.2f} min"
        )
        print(f"Checkpoint saved: {checkpoint_path}\n")

    total_time = time.perf_counter() - overall_start
    print(f"Total training time: {total_time / 60:.2f} min")

    return model