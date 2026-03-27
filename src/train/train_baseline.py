import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.data.chexpert_dataset import CheXpertDataset, CheXpertConfig
from src.models.densenet_baseline import DenseNetBaseline
from src.utils.metrics import compute_auc


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    all_preds = []
    all_targets = []
    total_loss = 0.0

    with torch.set_grad_enabled(train):
        pbar = tqdm(loader, leave=False)
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            preds = torch.sigmoid(logits)

            total_loss += loss.item()
            all_preds.append(preds.detach())
            all_targets.append(targets.detach())

            pbar.set_postfix(loss=loss.item())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    aucs = compute_auc(all_targets, all_preds)
    mean_auc = torch.tensor(aucs).nanmean().item()

    return total_loss / len(loader), aucs, mean_auc


def main():
    device = torch.device("cpu")

    # -------------------------
    # Config
    # -------------------------
    batch_size = 8
    num_epochs = 5
    lr = 1e-4

    images_dir = "data/images_30k_final"
    train_csv = "data/labels/train_30k_patient_train.csv"
    valid_csv = "data/labels/train_30k_patient_valid.csv"

    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # -------------------------
    # Transforms
    # -------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # -------------------------
    # Datasets & Loaders
    # -------------------------
    train_cfg = CheXpertConfig(images_dir=images_dir, csv_path=train_csv)
    val_cfg   = CheXpertConfig(images_dir=images_dir, csv_path=valid_csv)

    train_ds = CheXpertDataset(train_cfg, transform=transform)
    val_ds   = CheXpertDataset(val_cfg, transform=transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # -------------------------
    # Model
    # -------------------------
    model = DenseNetBaseline(num_classes=5, pretrained=True)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_auc = 0.0

    # -------------------------
    # Training Loop
    # -------------------------
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss, train_aucs, train_mean = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )

        val_loss, val_aucs, val_mean = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )

        print(f"Train Mean AUC: {train_mean:.4f}")
        print(f"Val   Mean AUC: {val_mean:.4f}")

        if val_mean > best_val_auc:
            best_val_auc = val_mean
            ckpt = os.path.join(save_dir, "densenet_baseline_best.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"🔥 New best model saved: {ckpt}")

    print(f"\nBest Validation AUC: {best_val_auc:.4f}")


if __name__ == "__main__":
    main()
