from pathlib import Path

BASE_DIR = Path("03_data_preprocessing/segmentation")

splits = ["train", "val", "test"]
subfolders = ["images", "masks"]

for split in splits:
    for sub in subfolders:
        path = BASE_DIR / split / sub
        path.mkdir(parents=True, exist_ok=True)

print("✅ Segmentation folder structure created successfully!")
