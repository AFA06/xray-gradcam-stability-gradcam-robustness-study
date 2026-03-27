import matplotlib.pyplot as plt
from PIL import Image

paths = [
    "gradcam_results/phase4_instability/original_overlay.png",
    "gradcam_results/phase4_instability/bright_5pct_overlay.png",
    "gradcam_results/phase4_instability/bright_5pct_overlay.png",
    "gradcam_results/phase4_instability/rotate_2deg_overlay.png"
]

titles = [
    "Original",
    "Brightness +5%",
    "Contrast +5%",
    "Rotation +2°"
]

imgs = [Image.open(p) for p in paths]

plt.figure(figsize=(12,3))

for i, img in enumerate(imgs):
    plt.subplot(1,4,i+1)
    plt.imshow(img)
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()

plt.savefig("results/gradcam_perturbation_comparison.png", dpi=300)

print("Saved: results/gradcam_perturbation_comparison.png")