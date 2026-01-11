import sys
from pathlib import Path

# --------------------------------------------------
# Fix imports
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import torch
import numpy as np
from PIL import Image
import pandas as pd

from modeling.segmentation.model import UNet

# --------------------------------------------------
# Config
# --------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "modeling/segmentation/checkpoints/best_unet.pth"
THRESHOLD = 0.6

# --------------------------------------------------
# Load model once
# --------------------------------------------------
@st.cache_resource
def load_model():
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()
    return model

model = load_model()

# --------------------------------------------------
# Simple preprocessing (NO SARPreprocess)
# --------------------------------------------------
def preprocess_pil(img: Image.Image):
    img = img.convert("L").resize((400, 400))
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return tensor.to(DEVICE)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
def predict(img: Image.Image):
    x = preprocess_pil(img)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    mask = (prob >= THRESHOLD).astype(np.uint8)

    oil_pixels = int(mask.sum())
    total_pixels = mask.size
    oil_pct = (oil_pixels / total_pixels) * 100
    confidence = float(prob.mean())

    return prob, mask, oil_pixels, total_pixels, oil_pct, confidence

# --------------------------------------------------
# Overlay
# --------------------------------------------------
def make_overlay(img, mask):
    img = np.array(img.convert("RGB").resize((400, 400)))
    overlay = img.copy()
    overlay[mask == 1] = [255, 0, 0]
    return overlay

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config("Oil Spill Detection", layout="wide")
st.title("🛢️ Oil Spill Detection using SAR Images")
st.caption("Simple segmentation-based detection (U-Net)")

uploaded = st.file_uploader(
    "Upload a SAR Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded:
    image = Image.open(uploaded)

    with st.spinner("Running inference..."):
        prob, mask, oil_px, total_px, oil_pct, conf = predict(image)
        overlay = make_overlay(image, mask)

    col1, col2, col3 = st.columns(3)

    col1.subheader("Original")
    col1.image(image, width=300)

    col2.subheader("Predicted Mask")
    col2.image(mask * 255, width=300)

    col3.subheader("Overlay")
    col3.image(overlay, width=300)

    st.divider()

    st.subheader("📊 Detection Summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Oil Pixels", oil_px)
    c2.metric("Total Pixels", total_px)
    c3.metric("Oil Coverage (%)", f"{oil_pct:.2f}")
    c4.metric("Confidence", f"{conf:.3f}")

    is_oil = oil_pct > 1.0

    if is_oil:
        st.error("🛢️ Oil Spill Detected")
    else:
        st.success("✅ No Oil Spill Detected")

    # --------------------------------------------------
    # CSV Download
    # --------------------------------------------------
    results = pd.DataFrame([{
        "oil_pixels": oil_px,
        "total_pixels": total_px,
        "oil_coverage_percent": oil_pct,
        "confidence": conf,
        "prediction": "Oil Spill" if is_oil else "No Oil Spill"
    }])

    st.download_button(
        "⬇️ Download Results as CSV",
        data=results.to_csv(index=False),
        file_name="oil_spill_result.csv",
        mime="text/csv"
    )
