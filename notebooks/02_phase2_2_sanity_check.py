import os
import math
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.data.chexpert_dataset import CheXpertDataset, CheXpertConfig, DEFAULT_TARGETS
from src.data.transforms import get_transform, IMAGENET_MEAN, IMAGENET_STD

def unnormalize(x):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    x = x.cpu() * std + mean
    return torch.clamp(x, 0, 1)

def main():
    root = os.path.dirname(os.path.dirname(__file__))
    images_dir = os.path.join(root, "data", "images_30k_final")
    train_csv = os.path.join(root, "data", "labels", "train_30k_final.csv")

    ds = CheXpertDataset(
        CheXpertConfig(
            images_dir=images_dir,
            csv_path=train_csv,
            targets=DEFAULT_TARGETS,
            uncertain_policy="zeros",
        ),
        transform=get_transform(224),
        return_path=True,
    )

    dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    imgs, labels, paths = next(iter(dl))

    print("Images:", imgs.shape, imgs.dtype)
    print("Labels:", labels.shape, labels.dtype)
    print("Example path:", paths[0])
    print("Example label row:", labels[0].tolist())

    n = min(16, imgs.size(0))
    cols = 4
    rows = math.ceil(n / cols)

    fig = plt.figure(figsize=(cols * 4.2, rows * 4.6))
    for i in range(n):
        ax = plt.subplot(rows, cols, i + 1)
        img = unnormalize(imgs[i]).permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.axis("off")

        y = labels[i].cpu().numpy().tolist()
        present = [DEFAULT_TARGETS[j] for j, v in enumerate(y) if v >= 0.5]
        base = os.path.basename(paths[i])
        subtitle = " | ".join(present) if present else "No positive labels"
        ax.set_title(f"{base}\n{subtitle}", fontsize=9)

    out_dir = os.path.join(root, "results", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "phase2_2_sanity_batch.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", out_path)

if __name__ == "__main__":
    torch.manual_seed(42)
    main()
