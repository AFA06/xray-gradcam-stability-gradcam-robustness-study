import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.data.chexpert_dataset import CheXpertDataset, default_train_transform
from src.models.resnet18 import CheXpertResNet18


# -----------------------------
# Device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -----------------------------
# Dataset (TEST ONLY)
# -----------------------------
dataset = CheXpertDataset(
    csv_file="data/labels/test_split.csv",
    image_dir="data/images_30k_final",
    transform=default_train_transform(),
)

loader = DataLoader(dataset, batch_size=8, shuffle=False)

# -----------------------------
# Model
# -----------------------------
model = CheXpertResNet18(
    num_classes=5,
    pretrained=False
)

checkpoint_path = "checkpoints/resnet18_epoch3.pt"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()

all_probs = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probs = torch.sigmoid(outputs)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

all_probs = np.vstack(all_probs)
all_labels = np.vstack(all_labels)

# -----------------------------
# Compute AUROC per class
# -----------------------------
num_classes = all_labels.shape[1]
aucs = []

for i in range(num_classes):
    try:
        auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
        aucs.append(auc)
    except:
        aucs.append(np.nan)

macro_auc = np.nanmean(aucs)
micro_auc = roc_auc_score(all_labels, all_probs, average="micro")

# -----------------------------
# Print Results
# -----------------------------
print("\n==== Test Evaluation Results ====")
for i, auc in enumerate(aucs):
    print(f"Class {i} AUROC: {auc:.4f}")

print(f"\nMacro AUROC: {macro_auc:.4f}")
print(f"Micro AUROC: {micro_auc:.4f}")

# -----------------------------
# Save Results
# -----------------------------
os.makedirs("results", exist_ok=True)

with open("results/test_metrics.txt", "w") as f:
    for i, auc in enumerate(aucs):
        f.write(f"Class {i} AUROC: {auc:.4f}\n")
    f.write(f"\nMacro AUROC: {macro_auc:.4f}\n")
    f.write(f"Micro AUROC: {micro_auc:.4f}\n")

print("\nResults saved to results/test_metrics.txt")
