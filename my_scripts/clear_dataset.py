import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class ClearDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.image_dir = os.path.join(data_path, "images")
        self.mask_dir = os.path.join(data_path, "masks")
        self.image_paths = sorted([
            os.path.join(self.image_dir, f)
            for f in os.listdir(self.image_dir) if f.endswith(".png")
        ])
        self.mask_paths = sorted([
            os.path.join(self.mask_dir, f)
            for f in os.listdir(self.mask_dir) if f.endswith(".png")
        ])
        assert len(self.image_paths) == len(self.mask_paths)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        mask = cv2.imread(self.mask_paths[idx], 0)
        if self.transform:
            img, mask = self.transform(img, mask, "train")
        return img, mask
