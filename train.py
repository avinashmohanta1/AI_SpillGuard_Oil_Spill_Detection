from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from feature_engineering.preprocess import SARPreprocess
from modeling.dataset import OilSpillDataset
from modeling.model import SimpleCNN
from modeling.utils import accuracy

# Paths
ROOT = Path("03_data_preprocessing/processed")
SAVE_DIR = Path("modeling/checkpoints")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Config
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data
preprocess = SARPreprocess()
train_ds = OilSpillDataset(ROOT, "train", preprocess)
val_ds   = OilSpillDataset(ROOT, "val", preprocess)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# Model
model = SimpleCNN().to(DEVICE)

# Handle class imbalance
counts = [0, 0]
for _, y in train_ds:
    counts[y.item()] += 1
weights = torch.tensor([1.0 / counts[0], 1.0 / counts[1]], device=DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = optim.Adam(model.parameters(), lr=LR)

best_val_acc = 0.0

# Training loop
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss, train_acc = 0.0, 0.0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc  += accuracy(out, y)

    train_loss /= len(train_loader)
    train_acc  /= len(train_loader)

    model.eval()
    val_acc = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            val_acc += accuracy(out, y)
    val_acc /= len(val_loader)

    print(f"Epoch {epoch:02d} | "
          f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | "
          f"Val Acc {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_DIR / "best_cnn.pth")
        print("✅ Saved best model")

print("🏁 Training complete.")
