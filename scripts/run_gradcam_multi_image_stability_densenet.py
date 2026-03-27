import os
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import random
torch.autograd.set_detect_anomaly(False)
torch.backends.cudnn.enabled = True
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim

from src.models.densenet_baseline import get_densenet121
from scripts.gradcam import GradCAM

# ==========================
# CONFIG
# ==========================
IMAGE_DIR = "/home/azureuser/chexpert/data/images_30k_final"
CHECKPOINT_PATH = "/home/azureuser/chexpert/checkpoints/densenet_baseline_best.pt"
NUM_IMAGES = 30
CLASS_IDX = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# Load Model
# ==========================
model = get_densenet121(num_classes=5, pretrained=False)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.to(device)
model.eval()

# DenseNet target layer
target_layer = model.model.features[-1]
gradcam = GradCAM(model, target_layer)

# ==========================
# Transforms
# ==========================
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================
# Collect random images
# ==========================
all_images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]
selected_images = random.sample(all_images, NUM_IMAGES)

# ==========================
# Metrics storage
# ==========================
results = {
    "brightness": [],
    "rotate": [],
    "contrast": []
}

print("\nProcessing images...\n")

for img_name in tqdm(selected_images):

    img_path = os.path.join(IMAGE_DIR, img_name)
    img = Image.open(img_path).convert("RGB")

    variants = {
        "original": img,
        "brightness": T.functional.adjust_brightness(img, 1.05),
        "rotate": T.functional.rotate(img, 2),
        "contrast": T.functional.adjust_contrast(img, 1.05)
    }

    cams = {}

    for name, variant in variants.items():
        input_tensor = transform(variant).unsqueeze(0).to(device)
        cam = gradcam.generate(input_tensor, CLASS_IDX)
        cams[name] = cam

    original_cam = cams["original"].flatten()

    for perturb in ["brightness", "rotate", "contrast"]:

        perturbed_cam = cams[perturb].flatten()

        mad = np.mean(np.abs(original_cam - perturbed_cam))
        cos = cosine_similarity(
            original_cam.reshape(1, -1),
            perturbed_cam.reshape(1, -1)
        )[0][0]

        ssim_score = ssim(
            cams["original"],
            cams[perturb],
            data_range=1.0
        )

        results[perturb].append((mad, cos, ssim_score))

# ==========================
# Print Final Results
# ==========================
print("\n===== FINAL STABILITY RESULTS (DenseNet - 30 images) =====\n")

for perturb in results:

    mads = [r[0] for r in results[perturb]]
    coss = [r[1] for r in results[perturb]]
    ssims = [r[2] for r in results[perturb]]

    print(f"{perturb.upper()} Perturbation:")
    print(f"MAD:  {np.mean(mads):.6f} ± {np.std(mads):.6f}")
    print(f"Cosine: {np.mean(coss):.6f} ± {np.std(coss):.6f}")
    print(f"SSIM: {np.mean(ssims):.6f} ± {np.std(ssims):.6f}")
    print("-------------------------------------------")
