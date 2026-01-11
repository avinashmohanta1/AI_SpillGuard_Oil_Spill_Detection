from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from modeling.segmentation.dataset import OilSpillSegmentationDataset
from modeling.segmentation.model import UNet
from modeling.segmentation.loss import BCEDiceLoss

# ======================
# Config
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

EPOCHS = 50          # increased epochs
BATCH_SIZE = 4
LR = 1e-4

DATA_ROOT = "03_data_preprocessing/segmentation"
SAVE_DIR = Path("modeling/segmentation/checkpoints")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ======================
# Data
# ======================
train_ds = OilSpillSegmentationDataset(DATA_ROOT, "train")
val_ds = OilSpillSegmentationDataset(DATA_ROOT, "val")

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,      # 🔥 IMPORTANT FOR WINDOWS
    pin_memory=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,      # 🔥 IMPORTANT FOR WINDOWS
    pin_memory=True
)

# ======================
# Model
# ======================
model = UNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = BCEDiceLoss()

# ======================
# Training
# ======================
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for imgs, masks in loop:
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        loop.set_postfix(train_loss=loss.item())

    train_loss /= len(train_loader)

    # ======================
    # Validation
    # ======================
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            preds = model(imgs)
            loss = criterion(preds, masks)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f}"
    )

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), SAVE_DIR / "best_unet.pth")
        print("✅ Best model saved")

print("🎯 Training complete!")
