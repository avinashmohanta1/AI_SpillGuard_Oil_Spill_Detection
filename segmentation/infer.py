import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from modeling.segmentation.model import UNet
from feature_engineering.preprocess import SARPreprocess

# =====================
# CONFIG
# =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT = Path("modeling/segmentation/checkpoints/best_unet.pth")

IMAGE_DIR = Path(
    "03_data_preprocessing/segmentation/test/images"
)

CONF_THRESHOLD = 0.6
ALPHA = 0.7

# =====================
# LOAD MODEL
# =====================
model = UNet(in_channels=1, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

print(f"Using device: {DEVICE}")

# =====================
# LOAD IMAGE PATH
# =====================
img_path = list(IMAGE_DIR.iterdir())[0]
print(f"Running inference on: {img_path.name}")

# Load original for display ONLY
orig_img = Image.open(img_path).convert("L")
orig = np.array(orig_img).astype(np.float32)
orig = (orig - orig.min()) / (orig.max() - orig.min() + 1e-6)

# =====================
# PREPROCESS (PATH ONLY ✅)
# =====================
preprocess = SARPreprocess()
img_tensor = preprocess(img_path).unsqueeze(0).to(DEVICE)  # [1,1,H,W]

# =====================
# INFERENCE
# =====================
with torch.no_grad():
    logits = model(img_tensor)
    prob_mask = torch.sigmoid(logits)[0, 0].cpu().numpy()

print(f"Prob range: min={prob_mask.min():.3f}, max={prob_mask.max():.3f}")

# =====================
# CONFIDENCE-AWARE OVERLAY
# =====================
p = (prob_mask - prob_mask.min()) / (prob_mask.max() - prob_mask.min() + 1e-6)
p[p < CONF_THRESHOLD] = 0.0

orig_rgb = np.stack([orig, orig, orig], axis=-1)
overlay = orig_rgb.copy()

overlay[..., 0] = overlay[..., 0] * (1 - ALPHA * p) + ALPHA * p
overlay[..., 1] = overlay[..., 1] * (1 - ALPHA * p)
overlay[..., 2] = overlay[..., 2] * (1 - ALPHA * p)

overlay = np.clip(overlay, 0, 1)

# =====================
# VISUALIZATION
# =====================
plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.title("Original SAR")
plt.imshow(orig, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Predicted Mask (probability)")
plt.imshow(prob_mask, cmap="gray")
plt.colorbar(fraction=0.046, pad=0.04)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Overlay (Oil Spill in Red)")
plt.imshow(overlay)
plt.axis("off")

plt.tight_layout()
plt.show()
