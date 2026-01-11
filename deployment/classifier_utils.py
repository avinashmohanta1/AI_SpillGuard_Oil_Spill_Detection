# deployment/classifier_utils.py

from pathlib import Path
import sys

# ----------------------------------------------------
# Ensure project root is in PYTHONPATH
# ----------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ----------------------------------------------------
# Imports
# ----------------------------------------------------
import torch
import torch.nn.functional as F
from PIL import Image

from feature_engineering.preprocess import SARPreprocess
from modeling.classification.model import get_model
from deployment.config import (
    DEVICE,
    CLASSIFIER_CHECKPOINT,
    CLASSIFICATION_THRESHOLD,
)

# ----------------------------------------------------
# Load model ONCE (important for Streamlit performance)
# ----------------------------------------------------
_preprocess = SARPreprocess()

_classifier = get_model(num_classes=2)
_classifier.load_state_dict(
    torch.load(CLASSIFIER_CHECKPOINT, map_location=DEVICE)
)
_classifier.to(DEVICE)
_classifier.eval()

# ----------------------------------------------------
# Public API
# ----------------------------------------------------
def predict_oil_spill(image: Image.Image) -> dict:
    """
    Predict whether a SAR image contains an oil spill.

    Args:
        image (PIL.Image): Input SAR image

    Returns:
        dict {
            "is_oil_spill": bool,
            "confidence": float,
            "probabilities": [no_oil_prob, oil_prob]
        }
    """

    # Ensure grayscale
    image = image.convert("L")

    with torch.no_grad():
        # Preprocess → [1, 1, H, W]
        tensor = _preprocess(image).unsqueeze(0).to(DEVICE)

        logits = _classifier(tensor)
        probs = F.softmax(logits, dim=1)[0]

        oil_prob = probs[1].item()
        is_oil = oil_prob >= CLASSIFICATION_THRESHOLD

    return {
        "is_oil_spill": bool(is_oil),
        "confidence": round(oil_prob, 4),
        "probabilities": [
            round(probs[0].item(), 4),
            round(probs[1].item(), 4),
        ],
    }
