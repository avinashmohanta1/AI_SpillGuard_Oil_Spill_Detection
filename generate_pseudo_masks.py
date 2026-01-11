from pathlib import Path
import cv2
import numpy as np

# =========================
# Paths
# =========================
SEG_ROOT = Path("03_data_preprocessing/segmentation")
splits = ["train", "val", "test"]

MIN_REGION_AREA = 500  # remove tiny noise blobs

for split in splits:
    img_dir = SEG_ROOT / split / "images"
    mask_dir = SEG_ROOT / split / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    for img_path in img_dir.iterdir():
        if img_path.suffix.lower() not in [".png", ".jpg", ".jpeg", ".tif"]:
            continue

        # Load SAR image (grayscale)
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Normalize
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        # Adaptive threshold (oil spills appear darker)
        _, binary = cv2.threshold(
            img_norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Remove small regions
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        clean_mask = np.zeros_like(binary)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= MIN_REGION_AREA:
                clean_mask[labels == i] = 255

        # Save mask
        mask_path = mask_dir / img_path.name
        cv2.imwrite(str(mask_path), clean_mask)

    print(f"✅ Pseudo-masks generated for {split}")

print("🎯 All pseudo-masks generated successfully!")
