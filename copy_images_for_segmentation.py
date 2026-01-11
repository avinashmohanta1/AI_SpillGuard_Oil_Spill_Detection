from pathlib import Path
import shutil

SRC_ROOT = Path("03_data_preprocessing/processed")
DST_ROOT = Path("03_data_preprocessing/segmentation")

splits = ["train", "val", "test"]

for split in splits:
    src_split = SRC_ROOT / split
    dst_images = DST_ROOT / split / "images"

    dst_images.mkdir(parents=True, exist_ok=True)

    for class_dir in src_split.iterdir():
        if not class_dir.is_dir():
            continue

        for img in class_dir.iterdir():
            dst_path = dst_images / img.name
            shutil.copy(img, dst_path)

    print(f"✅ Copied images for {split}")

print("🎯 All images copied successfully!")
