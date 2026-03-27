import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

from src.models.densenet_baseline import get_densenet121
from scripts.gradcam import GradCAM

# ------------------------------
# 1. Load model
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_densenet121(pretrained=True)  # or False if you don't have pretrained weights
model.eval()
model.to(device)

# ------------------------------
# 2. Select target layer for Grad-CAM
# ------------------------------
# Note: DenseNetBaseline wraps the actual DenseNet in model.model
target_layer = model.model.features.denseblock4.denselayer16.conv2
gradcam = GradCAM(model, target_layer)

# ------------------------------
# 3. Load and preprocess image
# ------------------------------
img_path = "/home/azureuser/chexpert/data/images_30k_final/patient00001_study1_view1_frontal.jpg"  # <-- change this to your image
img = Image.open(img_path).convert("RGB")

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

input_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension

# ------------------------------
# 4. Forward pass and Grad-CAM
# ------------------------------
# Choose class index to visualize (e.g., 0)
class_idx = 0
cam = gradcam.generate(input_tensor, class_idx)

# ------------------------------
# 5. Overlay CAM on original image
# ------------------------------
# Convert PIL image to numpy
img_np = np.array(img.resize((224, 224)))
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
overlay = 0.5 * img_np + 0.5 * heatmap
overlay = overlay.astype(np.uint8)

# ------------------------------
# 6. Show results
# ------------------------------
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(img_np)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Grad-CAM Heatmap")
plt.imshow(cam, cmap="jet")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Overlay")
plt.imshow(overlay)
plt.axis("off")

plt.show()

# ------------------------------
# 7. Save results to disk
# ------------------------------
save_dir = "/home/azureuser/chexpert/gradcam_results"

orig_path = f"{save_dir}/original.jpg"
heatmap_path = f"{save_dir}/heatmap.jpg"
overlay_path = f"{save_dir}/overlay.jpg"

cv2.imwrite(orig_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
cv2.imwrite(heatmap_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

print("Grad-CAM images saved:")
print(orig_path)
print(heatmap_path)
print(overlay_path)
