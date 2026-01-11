from pathlib import Path
import os
import random
import shutil
import csv

# =========================
# Paths
# =========================
BASE_DIR = Path("D:/Oil spill dataset/oil_spill_detection")

RAW_DATA_DIR = BASE_DIR / "01_data_collection" / "raw" / "CSIRO_DATA"

CLASS_0_SRC = RAW_DATA_DIR / "S1SAR_UnBalanced_400by400_Class_0" / "0"
CLASS_1_SRC = RAW_DATA_DIR / "S1SAR_UnBalanced_400by400_Class_1" / "1"

PROCESSED_DIR = BASE_DIR / "03_data_preprocessing" / "processed"

# Create destination folders
for split in ["train", "val", "test"]:
    for cls in ["0", "1"]:
        (PROCESSED_DIR / split / cls).mkdir(parents=True, exist_ok=True)

# =========================
# Split ratios
# =========================
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# =========================
# Helper function
# =========================
def split_and_copy(class_dir, class_label):
    images = os.listdir(class_dir)
    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    records = []

    for split, files in splits.items():
        for fname in files:
            src = class_dir / fname
            dst = PROCESSED_DIR / split / class_label / fname
            shutil.copy(src, dst)

            records.append([fname, class_label, split])

    return records

# =========================
# Perform split
# =========================
csv_records = []
csv_records += split_and_copy(CLASS_0_SRC, "0")
csv_records += split_and_copy(CLASS_1_SRC, "1")

# =========================
# Save CSV
# =========================
csv_path = BASE_DIR / "03_data_preprocessing" / "split_info.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "class", "split"])
    writer.writerows(csv_records)

print("✅ Data split completed successfully!")
print("📄 split_info.csv saved to 03_data_preprocessing/")
