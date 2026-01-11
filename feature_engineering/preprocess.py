import cv2
import numpy as np
import torch

class SARPreprocess:
    """
    Preprocessing pipeline for SAR images (Option A):
    - Read grayscale image
    - Min-Max normalization
    - Convert to PyTorch tensor
    """

    def __init__(self):
        pass

    def __call__(self, img_path):
        # Read image in grayscale
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"Could not read image: {img_path}")

        # Convert to float32
        img = img.astype(np.float32)

        # Min-Max normalization (SAR safe)
        min_val = img.min()
        max_val = img.max()

        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val)
        else:
            img = np.zeros_like(img)

        # Add channel dimension -> (1, H, W)
        img = np.expand_dims(img, axis=0)

        # Convert to PyTorch tensor
        img = torch.from_numpy(img)

        return img
