import os
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.models.resnet18 import CheXpertResNet18
from scripts.gradcam import GradCAM

# ==============================
# CONFIGURATION
# ==============================

IMAGE_PATH = "/home/azureuser/chexpert/data/images_30k_final/patient00001_study1_view1_frontal.jpg"
CHECKPOINT_PATH = "/home/azureuser/chexpert/checkpoints/resnet18_epoch3.pt"

OUTPUT_PATH = "results/gradcam_comparison.png"

CLASS_IDX = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# LOAD MODEL
# ==============================

model = CheXpertResNet18(num_classes=5, pretrained=False)

model.load_state_dict(
    torch.load(CHECKPOINT_PATH, map_location=device)
)

model.to(device)
model.eval()

target_layer = model.model.layer4[-1]

gradcam = GradCAM(model, target_layer)

# ==============================
# TRANSFORM (same as training)
# ==============================

transform = T.Compose([
    T.Resize((224,224)),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ==============================
# LOAD ORIGINAL X-RAY
# ==============================

img = Image.open(IMAGE_PATH).convert("L")

# ==============================
# CREATE PERTURBATIONS
# ==============================

variants = {
    "Original": img,
    "Brightness +5%": T.functional.adjust_brightness(img,1.05),
    "Contrast +5%": T.functional.adjust_contrast(img,1.05),
    "Rotation +2°": T.functional.rotate(img,2)
}

cams = {}

# ==============================
# GENERATE GRAD-CAM MAPS
# ==============================

for name, variant in variants.items():

    input_tensor = transform(variant).unsqueeze(0).to(device)

    cam = gradcam.generate(input_tensor, CLASS_IDX)

    cam = np.array(cam)

    # normalize heatmap safely
    cam_min = cam.min()
    cam_max = cam.max()

    if cam_max - cam_min > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)

    cams[name] = cam

# ==============================
# CREATE VISUALIZATION
# ==============================

fig, axes = plt.subplots(1,4, figsize=(16,4))

for ax, (name, variant) in zip(axes, variants.items()):

    cam = cams[name]

    # raw image for display
    raw = np.array(variant.resize((224,224)))

    # normalize for visualization
    raw = (raw - raw.min()) / (raw.max() - raw.min())

    ax.imshow(raw, cmap="gray")

    ax.imshow(
        cam,
        cmap="jet",
        alpha=0.30,
        interpolation="bilinear"
    )

    ax.set_title(name, fontsize=12)

    ax.axis("off")

plt.tight_layout()

os.makedirs("results", exist_ok=True)

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

print("Grad-CAM comparison saved to:", OUTPUT_PATH)