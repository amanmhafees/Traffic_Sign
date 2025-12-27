import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm


def build_model(num_classes: int) -> nn.Module:
    # ResNet-18 with ImageNet weights
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
    except Exception:
        weights = None
    model = models.resnet18(weights=weights)

    # Replace final fully-connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
    return model


def get_transforms(img_size: int = 224):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),  # mild only
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    return train_tf, val_tf


def train(
    data_root: Path,
    out_path: Path,
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_split: float = 0.2,
    num_workers: int = 2,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    out_path.mkdir(parents=True, exist_ok=True)
    train_tf, val_tf = get_transforms(224)

    full_dataset = datasets.ImageFolder(root=str(data_root), transform=train_tf)
    class_to_idx = full_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Persist class mapping for inference
    (out_path / "cnn_classes.json").write_text(json.dumps(idx_to_class, indent=2))

    n_total = len(full_dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])
    # Assign val transform to validation subset
    val_ds.dataset.transform = val_tf

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_model(num_classes=len(class_to_idx)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_path = out_path / "cnn_classifier.pt"

    epoch_bar = tqdm(range(1, epochs + 1), desc="Epochs", position=0)
    for epoch in epoch_bar:
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", position=1, leave=False)
        for inputs, labels in train_pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            running_corrects += (preds == labels).sum().item()
            total += inputs.size(0)

            batch_loss = loss.item()
            batch_acc = (preds == labels).float().mean().item()
            train_pbar.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{batch_acc:.4f}")

        epoch_loss = running_loss / max(1, total)
        epoch_acc = running_corrects / max(1, total)

        # Validation
        model.eval()
        val_corrects = 0
        val_total = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]  ", position=1, leave=False)
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                val_corrects += (preds == labels).sum().item()
                val_total += inputs.size(0)

                val_batch_acc = (preds == labels).float().mean().item()
                val_pbar.set_postfix(acc=f"{val_batch_acc:.4f}")

        val_acc = val_corrects / max(1, val_total)
        epoch_bar.set_postfix(loss=f"{epoch_loss:.4f}", acc=f"{epoch_acc:.4f}", val_acc=f"{val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": len(class_to_idx),
                "idx_to_class": idx_to_class,
                "arch": "mobilenet_v3_large",
            }, str(best_path))
            tqdm.write(f"Saved best model to {best_path} (val_acc={best_val_acc:.4f})")

    print("Training complete.")
    return best_path


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV3-Large CNN classifier on cropped dataset")
    parser.add_argument("--data", type=str, default="cnn_dataset", help="Cropped CNN dataset root")
    parser.add_argument("--out", type=str, default="output", help="Output directory")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    data_root = Path(args.data)
    out_path = Path(args.out)
    train(data_root, out_path, args.epochs, args.batch_size, args.lr, args.val_split, args.num_workers)


if __name__ == "__main__":
    main()
