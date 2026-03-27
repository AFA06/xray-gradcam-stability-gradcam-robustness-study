import os
import random
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim
from scipy.stats import ttest_rel

from src.models.resnet18 import CheXpertResNet18
from scripts.gradcam import GradCAM


# -------------------------------------------------
# CONFIG
# -------------------------------------------------
IMAGE_FOLDER = "data/images_30k_final"
CHECKPOINT_PATH = "checkpoints/resnet18_epoch3.pt"
NUM_IMAGES = 50
THRESHOLD = 0.5
NUM_CLASSES = 5
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------
# Load model
# -------------------------------------------------
model = CheXpertResNet18(num_classes=NUM_CLASSES, pretrained=False)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.to(device)
model.eval()

target_layer = model.model.layer4[-1]
gradcam = GradCAM(model, target_layer)


# -------------------------------------------------
# Transform
# -------------------------------------------------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# -------------------------------------------------
# Select random images
# -------------------------------------------------
all_images = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(".jpg")]
selected_images = random.sample(all_images, NUM_IMAGES)


# -------------------------------------------------
# Main Experiment
# -------------------------------------------------
print("\nProcessing images across all classes...\n")

for class_idx in range(NUM_CLASSES):

    print(f"\n========== CLASS {class_idx} ==========\n")

    results = {
        "brightness": [],
        "rotate": [],
        "contrast": []
    }

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
                prob = torch.sigmoid(output)[0, class_idx].item()

            probs[name] = prob
            cams[name] = gradcam.generate(input_tensor, class_idx)

        original_cam = cams["original"]
        original_prob = probs["original"]
        original_pred = int(original_prob > THRESHOLD)

        for pert in ["brightness", "rotate", "contrast"]:

            cam = cams[pert]
            pert_prob = probs[pert]
            pert_pred = int(pert_prob > THRESHOLD)

            # Explanation stability
            mad = np.mean(np.abs(original_cam - cam))
            ssim_score = ssim(original_cam, cam, data_range=1.0)

            # Prediction stability
            delta_prob = abs(original_prob - pert_prob)
            decision_changed = int(original_pred != pert_pred)

            results[pert].append((mad, ssim_score, delta_prob, decision_changed))

    # -------------------------------------------------
    # Summary + Statistical Testing
    # -------------------------------------------------
    brightness_ssim = np.array(results["brightness"])[:, 1]
    rotation_ssim = np.array(results["rotate"])[:, 1]

    for pert in results:

        values = np.array(results[pert])

        mad_mean = values[:, 0].mean()
        ssim_mean = values[:, 1].mean()
        delta_prob_mean = values[:, 2].mean()
        decision_rate = values[:, 3].mean()

        print(f"\n{pert.upper()} Perturbation:")
        print(f"Heatmap MAD: {mad_mean:.6f}")
        print(f"Heatmap SSIM: {ssim_mean:.6f}")
        print(f"Mean Δ Probability: {delta_prob_mean:.6f}")
        print(f"Decision Change Rate: {decision_rate*100:.2f}%")

    # Paired t-test (Brightness vs Rotation SSIM)
    t_stat, p_value = ttest_rel(brightness_ssim, rotation_ssim)

    print("\nStatistical Test (Brightness SSIM vs Rotation SSIM)")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.6f}")

    if p_value < 0.001:
        print("Result: Highly significant difference (p < 0.001)")
    elif p_value < 0.05:
        print("Result: Significant difference (p < 0.05)")
    else:
        print("Result: No significant difference")

    print("\n--------------------------------------------\n")
