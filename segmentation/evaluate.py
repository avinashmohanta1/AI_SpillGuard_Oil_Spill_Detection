import torch
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from modeling.segmentation.dataset import OilSpillSegmentationDataset
from modeling.segmentation.model import UNet

# ===============================
# Config
# ===============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path("03_data_preprocessing/segmentation")
CHECKPOINT = Path("modeling/segmentation/checkpoints/best_unet.pth")
OUTPUT_DIR = Path("06_model_evaluation/segmentation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===============================
# Metrics
# ===============================
def dice_score(pred, target, eps=1e-7):
    intersection = (pred * target).sum()
    return (2 * intersection + eps) / (pred.sum() + target.sum() + eps)

def iou_score(pred, target, eps=1e-7):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + eps) / (union + eps)

# ===============================
# Load model
# ===============================
print(f"Using device: {DEVICE}")

model = UNet(in_channels=1, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

# ===============================
# Dataset
# ===============================
dataset = OilSpillSegmentationDataset(
    root_dir=ROOT,
    split="test",
    transform=None
)

# ===============================
# Evaluation
# ===============================
records = []

with torch.no_grad():
    for img, mask in tqdm(dataset, desc="Evaluating segmentation"):
        img = img.unsqueeze(0).to(DEVICE)   # [1,1,H,W]
        mask = mask.to(DEVICE)

        pred = torch.sigmoid(model(img))[0, 0]
        pred_bin = (pred > 0.5).float()
        target = mask[0]

        records.append({
            "dice": dice_score(pred_bin, target).item(),
            "iou": iou_score(pred_bin, target).item()
        })

# ===============================
# Save
# ===============================
df = pd.DataFrame(records)
df.to_csv(OUTPUT_DIR / "segmentation_metrics.csv", index=False)

print("\n✅ Segmentation Evaluation Completed")
print(df.describe())
print(f"\n📁 Results saved to {OUTPUT_DIR}")
