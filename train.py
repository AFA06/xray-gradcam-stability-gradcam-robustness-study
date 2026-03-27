import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.chexpert_dataset import CheXpertDataset, default_train_transform
from src.models.resnet18 import CheXpertResNet18


def main():
    
    # Device
  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

 
    # Dataset & Loader
   
    dataset = CheXpertDataset(
        csv_file="data/labels/train_30k_final.csv",
        image_dir="data/images_30k_final",
        transform=default_train_transform(),
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=(device == "cuda"),
    )

  
    # Model
   
    model = CheXpertResNet18(
        num_classes=5,
        pretrained=True,
    ).to(device)

  
    # Loss & Optimizer
   
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    
    # Checkpoint folder
  
    os.makedirs("checkpoints", exist_ok=True)

    
    # Training loop
   
    num_epochs = 5
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        progress_bar = tqdm(
            loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            leave=True,
        )

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f}")

        # Save checkpoint EACH epoch
 
        ckpt_path = f"checkpoints/resnet18_epoch{epoch + 1}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
