from pathlib import Path
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Paths
# =========================
BASE_DIR = Path("D:/Oil spill dataset/oil_spill_detection")

DATA_DIR = BASE_DIR / "01_data_collection" / "raw" / "CSIRO_DATA"

CLASS_0_DIR = DATA_DIR / "S1SAR_UnBalanced_400by400_Class_0" / "0"
CLASS_1_DIR = DATA_DIR / "S1SAR_UnBalanced_400by400_Class_1" / "1"

OUTPUT_DIR = BASE_DIR / "02_data_exploration"
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================
# 1. Count images per class
# =========================
class_0_images = os.listdir(CLASS_0_DIR)
class_1_images = os.listdir(CLASS_1_DIR)

count_0 = len(class_0_images)
count_1 = len(class_1_images)

print(f"[INFO] Class 0 (No Spill): {count_0}")
print(f"[INFO] Class 1 (Oil Spill): {count_1}")
print(f"[INFO] Imbalance ratio (0/1): {count_0 / count_1:.2f}")

# =========================
# 2. Class distribution plot
# =========================
plt.figure(figsize=(6,4))
plt.bar(["No Spill", "Oil Spill"], [count_0, count_1], color=["gray", "red"])
plt.title("Class Distribution")
plt.ylabel("Number of Images")
plt.savefig(OUTPUT_DIR / "class_distribution.png")
plt.close()

# =========================
# 3. Image shape & SAR stats
# =========================
sample_img = cv2.imread(
    str(CLASS_0_DIR / class_0_images[0]),
    cv2.IMREAD_GRAYSCALE
)

print("[INFO] Image shape:", sample_img.shape)
print("[INFO] Pixel range:", sample_img.min(), "to", sample_img.max())

# =========================
# 4. Visualize sample images
# =========================
def save_sample_images(image_list, folder, title, filename):
    plt.figure(figsize=(10,4))
    for i in range(5):
        img = cv2.imread(str(folder / image_list[i]), cv2.IMREAD_GRAYSCALE)
        plt.subplot(1,5,i+1)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
    plt.suptitle(title)
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()

save_sample_images(class_0_images, CLASS_0_DIR,
                   "No Oil Spill (Class 0)", "sample_no_spill.png")

save_sample_images(class_1_images, CLASS_1_DIR,
                   "Oil Spill (Class 1)", "sample_oil_spill.png")

# =========================
# 5. Pixel intensity histogram
# =========================
def collect_pixels(image_list, folder, max_images=200):
    pixels = []
    for img_name in image_list[:max_images]:
        img = cv2.imread(str(folder / img_name), cv2.IMREAD_GRAYSCALE)
        pixels.extend(img.flatten())
    return pixels

pixels_0 = collect_pixels(class_0_images, CLASS_0_DIR)
pixels_1 = collect_pixels(class_1_images, CLASS_1_DIR)

plt.figure(figsize=(8,5))
plt.hist(pixels_0, bins=50, alpha=0.6, label="No Spill")
plt.hist(pixels_1, bins=50, alpha=0.6, label="Oil Spill")
plt.legend()
plt.title("Pixel Intensity Distribution (SAR)")
plt.savefig(OUTPUT_DIR / "pixel_intensity_histogram.png")
plt.close()

print("✅ EDA completed. Outputs saved to 02_data_exploration/")
