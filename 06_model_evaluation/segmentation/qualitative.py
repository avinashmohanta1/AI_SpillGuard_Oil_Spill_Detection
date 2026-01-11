# ============================================================
# Qualitative Segmentation Evaluation (Option B)
# ============================================================

import sys
from pathlib import Path

# ------------------------------------------------------------
# FIX PYTHON PATH (CRITICAL)
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from modeling.segmentation.model import UNet
from modeling.segmentation.dataset import OilSpillSegmentationDataset

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = PROJECT_ROOT / "03_data_preprocessing" / "segmentation"
CHECKPOINT = PROJECT_ROOT / "modeling" / "segmentation" / "checkpoints" / "best_unet.pth"

SAVE_ROOT = PROJECT_ROOT / "06_model_evaluation" / "segmentation_qualitative"
BEST_DIR = SAVE_ROOT / "best"
AVG_DIR = SAVE_ROOT / "average"
WORST_DIR = SAVE_ROOT / "worst"

for d in [BEST_DIR, AVG_DIR, WORST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 10  # number of images per category

# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------
def dice_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2 * intersection + eps) / (pred.sum() + target.sum() + eps)


def create_overlay(image, mask):
    """
    image: (H, W)
    mask: (H, W) binary
    """
    image_rgb = np.stack([image, image, image], axis=-1)
    overlay = image_rgb.copy()
    overlay[..., 0] = np.maximum(overlay[..., 0], mask * 255)
    return overlay.astype(np.uint8)


def save_visual(img, pred, mask, save_path):
    img = img.squeeze()
    pred = pred.squeeze()
    mask = mask.squeeze()

    overlay = create_overlay(img * 255, (pred > 0.5).astype(np.uint8))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original SAR")
    axes[0].axis("off")

    axes[1].imshow(pred, cmap="gray")
    axes[1].set_title("Predicted Mask")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (Oil Spill in Red)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------
print(f"Using device: {DEVICE}")

model = UNet(in_channels=1, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
dataset = OilSpillSegmentationDataset(
    root_dir=DATA_ROOT,
    split="test"
)

# ------------------------------------------------------------
# Run qualitative evaluation
# ------------------------------------------------------------
results = []

print("Running qualitative segmentation evaluation...")

with torch.no_grad():
    for idx in tqdm(range(len(dataset))):
        img, mask = dataset[idx]

        img = img.to(DEVICE).unsqueeze(0)   # [1,1,H,W]
        mask = mask.to(DEVICE)

        pred = torch.sigmoid(model(img)).squeeze(0)

        score = dice_score(pred, mask)

        results.append((
            score.item(),
            img.cpu().numpy(),
            pred.cpu().numpy(),
            mask.cpu().numpy(),
            idx
        ))

# ------------------------------------------------------------
# Sort by Dice score
# ------------------------------------------------------------
results.sort(key=lambda x: x[0])

worst = results[:N_SAMPLES]
mid = results[len(results)//2 : len(results)//2 + N_SAMPLES]
best = results[-N_SAMPLES:]

# ------------------------------------------------------------
# Save images
# ------------------------------------------------------------
for rank, (score, img, pred, mask, idx) in enumerate(worst):
    save_visual(img[0], pred[0], mask[0],
                WORST_DIR / f"worst_{rank}_dice_{score:.3f}.png")

for rank, (score, img, pred, mask, idx) in enumerate(mid):
    save_visual(img[0], pred[0], mask[0],
                AVG_DIR / f"avg_{rank}_dice_{score:.3f}.png")

for rank, (score, img, pred, mask, idx) in enumerate(best):
    save_visual(img[0], pred[0], mask[0],
                BEST_DIR / f"best_{rank}_dice_{score:.3f}.png")

print("✅ Qualitative segmentation evaluation completed.")
