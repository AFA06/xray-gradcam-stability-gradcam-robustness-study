import os
import random
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim

from src.models.resnet18 import CheXpertResNet18
from scripts.gradcam import GradCAM

# ------------------------------
# CONFIG
# ------------------------------
IMAGE_FOLDER = "/home/azureuser/chexpert/data/images_30k_final"
CHECKPOINT_PATH = "/home/azureuser/chexpert/checkpoints/resnet18_epoch3.pt"
NUM_IMAGES = 30
CLASS_IDX = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Load model
# ------------------------------
model = CheXpertResNet18(num_classes=5, pretrained=False)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.to(device)
model.eval()

target_layer = model.model.layer4[-1]
gradcam = GradCAM(model, target_layer)

# ------------------------------
# Transform
# ------------------------------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# ------------------------------
# Get random images
# ------------------------------
all_images = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(".jpg")]
selected_images = random.sample(all_images, NUM_IMAGES)

# ------------------------------
# Storage
# ------------------------------
metrics = {
    "brightness": [],
    "rotate": [],
    "contrast": []
}

print("\nProcessing images...\n")

for img_name in tqdm(selected_images):

    img_path = os.path.join(IMAGE_FOLDER, img_name)
    img = Image.open(img_path).convert("RGB")

    variants = {
        "original": img,
        "brightness": T.functional.adjust_brightness(img, 1.05),
        "rotate": T.functional.rotate(img, 2),
        "contrast": T.functional.adjust_contrast(img, 1.05),
    }

    cams = {}

    for name, variant in variants.items():
        input_tensor = transform(variant).unsqueeze(0).to(device)
        cam = gradcam.generate(input_tensor, CLASS_IDX)
        cams[name] = cam

    original_cam = cams["original"]

    for pert in ["brightness", "rotate", "contrast"]:

        cam = cams[pert]

        mad = np.mean(np.abs(original_cam - cam))
        cos = cosine_similarity(
            original_cam.flatten().reshape(1, -1),
            cam.flatten().reshape(1, -1)
        )[0][0]

        ssim_score = ssim(original_cam, cam, data_range=1.0)

        metrics[pert].append((mad, cos, ssim_score))

# ------------------------------
# Print results
# ------------------------------
print("\n===== FINAL STABILITY RESULTS (30 images) =====\n")

for pert in metrics:

    values = np.array(metrics[pert])
    mad_mean, cos_mean, ssim_mean = values.mean(axis=0)
    mad_std, cos_std, ssim_std = values.std(axis=0)

    print(f"{pert.upper()} Perturbation:")
    print(f"MAD:  {mad_mean:.6f} ± {mad_std:.6f}")
    print(f"Cosine: {cos_mean:.6f} ± {cos_std:.6f}")
    print(f"SSIM: {ssim_mean:.6f} ± {ssim_std:.6f}")
    print("-------------------------------------------")
