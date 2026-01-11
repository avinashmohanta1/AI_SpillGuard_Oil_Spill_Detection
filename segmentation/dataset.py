from pathlib import Path
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class OilSpillSegmentationDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.image_dir = Path(root_dir) / split / "images"
        self.mask_dir = Path(root_dir) / split / "masks"
        self.images = sorted(list(self.image_dir.iterdir()))
        self.masks = sorted(list(self.mask_dir.iterdir()))
        self.transform = transform

        assert len(self.images) == len(self.masks), "Image-mask count mismatch"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.images[idx]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(self.masks[idx]), cv2.IMREAD_GRAYSCALE)

        img = img.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        img = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask
