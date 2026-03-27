import os
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.models.densenet_baseline import get_densenet121
from scripts.gradcam import GradCAM

# ------------------------------
# CONFIG
# ------------------------------
IMG_PATH = "/home/azureuser/chexpert/data/images_30k_final/patient00001_study1_view1_frontal.jpg"
SAVE_DIR = "/home/azureuser/chexpert/gradcam_results/phase4_instability"
CLASS_IDX = 0  # Atelectasis

os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Load model
# ------------------------------
model = get_densenet121(pretrained=True)
model.eval()
model.to(device)

target_layer = model.model.features.denseblock4
gradcam = GradCAM(model, target_layer)

# ------------------------------
# Transforms
# ------------------------------
base_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# ------------------------------
# Load base image
# ------------------------------
img = Image.open(IMG_PATH).convert("RGB")

variants = {
    "original": img,
    "bright_5pct": T.functional.adjust_brightness(img, 1.05),
    "rotate_2deg": T.functional.rotate(img, 2)
}

# ------------------------------
# Grad-CAM generation
# ------------------------------
for name, variant_img in variants.items():
    input_tensor = base_transform(variant_img).unsqueeze(0).to(device)

    cam = gradcam.generate(input_tensor, CLASS_IDX)

    img_np = np.array(variant_img.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.5 * img_np + 0.5 * heatmap).astype(np.uint8)

    out_path = os.path.join(SAVE_DIR, f"{name}_overlay.png")
    plt.imsave(out_path, overlay)

    print(f"Saved: {out_path}")
