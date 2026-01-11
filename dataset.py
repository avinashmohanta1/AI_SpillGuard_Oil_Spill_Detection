from pathlib import Path
from torch.utils.data import Dataset
import torch

class OilSpillDataset(Dataset):
    def __init__(self, root_dir, split, preprocess):
        """
        root_dir: 03_data_preprocessing/processed
        split: train | val | test
        preprocess: SARPreprocess instance
        """
        self.preprocess = preprocess
        self.samples = []

        for cls in ["0", "1"]:
            cls_dir = Path(root_dir) / split / cls
            for img_path in cls_dir.iterdir():
                self.samples.append((img_path, int(cls)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        x = self.preprocess(img_path)          # (1, 400, 400)
        y = torch.tensor(label, dtype=torch.long)
        return x, y
