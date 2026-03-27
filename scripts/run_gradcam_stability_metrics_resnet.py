import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim

from src.models.resnet18 import CheXpertResNet18
from scripts.gradcam import GradCAM


# ==============================
# CONFIG
# ==============================
IMG_PATH = "/home/azureuser/chexpert/data/images_30k_final/patient00001_study1_view1_frontal.jpg"
CHECKPOINT_PATH = "/home/azureuser/chexpert/checkpoints/resnet18_epoch3.pt"
CLASS_IDX = None  # AUTO SELECT PREDICTED CLASS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================
# Load Model
# ==============================
model = CheXpertResNet18(num_classes=5, pretrained=False)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.to(device)
model.eval()

target_layer = model.model.layer4[-1]
gradcam = GradCAM(model, target_layer)


# ==============================
# Transform
# ==============================
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])


# ==============================
# Load Image
# ==============================
img = Image.open(IMG_PATH).convert("RGB")

variants = {
    "original": img,
    "brightness_5pct": T.functional.adjust_brightness(img, 1.05),
    "rotate_2deg": T.functional.rotate(img, 2),
    "contrast_5pct": T.functional.adjust_contrast(img, 1.05),
}


cams = {}

print("\n===== PREDICTIONS & CAM STATS =====\n")

for name, variant_img in variants.items():

    input_tensor = transform(variant_img).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output)

    # Auto-select predicted class if not fixed
    if CLASS_IDX is None:
        class_idx = torch.argmax(probs).item()
    else:
        class_idx = CLASS_IDX

    print(f"{name} probabilities: {probs.cpu().numpy()}")
    print(f"{name} predicted class: {class_idx}")

    # Generate GradCAM
    cam = gradcam.generate(input_tensor, class_idx)

    print(f"{name} CAM stats -> min: {cam.min():.6f}, "
          f"max: {cam.max():.6f}, "
          f"mean: {cam.mean():.6f}, "
          f"std: {cam.std():.6f}")
    print("---------------------------------------------------")

    cams[name] = cam


# ==============================
# Stability Metrics
# ==============================
print("\n===== STABILITY METRICS =====\n")

original_cam = cams["original"]

for name, cam in cams.items():
    if name == "original":
        continue

    mad = np.mean(np.abs(original_cam - cam))

    cos_sim = cosine_similarity(
        original_cam.flatten().reshape(1, -1),
        cam.flatten().reshape(1, -1)
    )[0][0]

    ssim_score = ssim(original_cam, cam, data_range=1.0)

    print(f"Original vs {name}")
    print(f"Mean Absolute Difference: {mad:.6f}")
    print(f"Cosine Similarity: {cos_sim:.6f}")
    print(f"SSIM: {ssim_score:.6f}")
    print("----------------------------------------")
