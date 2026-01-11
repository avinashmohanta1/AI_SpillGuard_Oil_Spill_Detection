# App configuration (paths, thresholds)
from pathlib import Path
import torch

# =========================================================
# PROJECT ROOT
# =========================================================
# This assumes deployment is inside oil_spill_detection
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# =========================================================
# DEVICE
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# MODEL PATHS
# =========================================================
# Classification model (binary oil / no-oil)
CLASSIFIER_CHECKPOINT = (
    PROJECT_ROOT
    / "modeling"
    / "classification"
    / "checkpoints"
    / "best_classifier.pth"
)

# Segmentation model (U-Net)
SEGMENTATION_CHECKPOINT = (
    PROJECT_ROOT
    / "modeling"
    / "segmentation"
    / "checkpoints"
    / "best_unet.pth"
)

# =========================================================
# THRESHOLDS
# =========================================================
CLASSIFICATION_THRESHOLD = 0.5
SEGMENTATION_THRESHOLD = 0.5

# =========================================================
# SAVE PATHS
# =========================================================
RESULTS_DIR = PROJECT_ROOT / "deployment" / "saved_results"
OVERLAY_DIR = RESULTS_DIR / "overlays"
MASK_DIR = RESULTS_DIR / "masks"
CSV_PATH = RESULTS_DIR / "predictions.csv"

OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# APP METADATA
# =========================================================
APP_TITLE = "🛢️ Oil Spill Detection System (SAR Images)"
APP_DESCRIPTION = (
    "Upload a SAR image to detect whether an oil spill is present. "
    "If detected, the system will localize the spill region using U-Net. "
    "All predictions are logged and can be downloaded as CSV."
)
