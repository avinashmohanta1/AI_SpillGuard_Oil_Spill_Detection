from modeling.segmentation.dataset import OilSpillSegmentationDataset

dataset = OilSpillSegmentationDataset(
    root_dir="03_data_preprocessing/segmentation",
    split="train"
)

img, mask = dataset[0]
print("Image:", img.shape, "Mask:", mask.shape)
