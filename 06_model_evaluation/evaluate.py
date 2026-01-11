from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

from feature_engineering.preprocess import SARPreprocess
from modeling.dataset import OilSpillDataset
from modeling.model import SimpleCNN

# =====================
# Paths
# =====================
DATA_ROOT = Path("03_data_preprocessing/processed")
MODEL_PATH = Path("modeling/checkpoints/best_cnn.pth")
OUTPUT_DIR = Path("06_model_evaluation")
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# Load data
# =====================
preprocess = SARPreprocess()
test_ds = OilSpillDataset(DATA_ROOT, "test", preprocess)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

# =====================
# Load model
# =====================
model = SimpleCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# =====================
# Evaluation
# =====================
all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        outputs = model(x)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

# =====================
# Metrics
# =====================
report = classification_report(
    all_labels,
    all_preds,
    target_names=["No Spill", "Oil Spill"],
    output_dict=True
)

df = pd.DataFrame(report).transpose()
df.to_csv(OUTPUT_DIR / "metrics.csv")

print("📊 Classification Report")
print(df)

# =====================
# Confusion Matrix
# =====================
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Spill", "Oil Spill"]
)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
plt.close()

print("✅ Evaluation complete.")
print("📁 Results saved to 06_model_evaluation/")
