import torch
from modeling.segmentation.loss import BCEDiceLoss

loss_fn = BCEDiceLoss()

pred = torch.rand(1, 1, 400, 400)
target = (torch.rand(1, 1, 400, 400) > 0.5).float()

loss = loss_fn(pred, target)
print("Loss:", loss.item())
