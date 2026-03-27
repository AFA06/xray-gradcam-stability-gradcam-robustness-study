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
IMAGE_FOLDER = "data/images_30k_final"
CHECKPOINT_PATH = "checkpoints/resnet18_epoch3.pt"
NUM_IMAGES = 50
CLASS_IDX = 0
THRESHOLD = 0.5

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
# Select random images
# ------------------------------
all_images = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(".jpg")]
selected_images = random.sample(all_images, NUM_IMAGES)


# ------------------------------
# Storage
# ------------------------------
results = {
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
    probs = {}

    for name, variant in variants.items():
        input_tensor = transform(variant).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output)[0, CLASS_IDX].item()

        probs[name] = prob

        cam = gradcam.generate(input_tensor, CLASS_IDX)
        cams[name] = cam

    original_cam = cams["original"]
    original_prob = probs["original"]
    original_pred = int(original_prob > THRESHOLD)

    for pert in ["brightness", "rotate", "contrast"]:

        cam = cams[pert]
        pert_prob = probs[pert]
        pert_pred = int(pert_prob > THRESHOLD)

        # Explanation stability
        mad = np.mean(np.abs(original_cam - cam))
        cos = cosine_similarity(
            original_cam.flatten().reshape(1, -1),
            cam.flatten().reshape(1, -1)
        )[0][0]
        ssim_score = ssim(original_cam, cam, data_range=1.0)

        # Prediction stability
        delta_prob = abs(original_prob - pert_prob)
        decision_changed = int(original_pred != pert_pred)

        results[pert].append((mad, cos, ssim_score, delta_prob, decision_changed))


# ------------------------------
# Print Results
# ------------------------------
print("\n===== Prediction vs Explanation Stability =====\n")

for pert in results:

    values = np.array(results[pert])

    mad_mean = values[:, 0].mean()
    ssim_mean = values[:, 2].mean()
    delta_prob_mean = values[:, 3].mean()
    decision_change_rate = values[:, 4].mean()

    print(f"\n{pert.upper()} Perturbation:")
    print(f"Heatmap MAD: {mad_mean:.6f}")
    print(f"Heatmap SSIM: {ssim_mean:.6f}")
    print(f"Mean Δ Probability: {delta_prob_mean:.6f}")
    print(f"Decision Change Rate: {decision_change_rate*100:.2f}%")
