import torch
from modeling.segmentation.model import UNet

model = UNet()
x = torch.randn(1, 1, 400, 400)

y = model(x)
print("Output shape:", y.shape)
print("Min:", y.min().item(), "Max:", y.max().item())
