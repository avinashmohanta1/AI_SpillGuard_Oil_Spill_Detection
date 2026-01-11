from pathlib import Path
from feature_engineering.preprocess import SARPreprocess

# Path to processed training images (Class 0)
img_path = Path("03_data_preprocessing/processed/train/0")

# Pick one sample image
sample_img = list(img_path.iterdir())[0]

# Initialize preprocessing
preprocess = SARPreprocess()

# Apply preprocessing
tensor = preprocess(sample_img)

# Print results
print("Tensor shape:", tensor.shape)
print("Min:", tensor.min().item(), "Max:", tensor.max().item())
