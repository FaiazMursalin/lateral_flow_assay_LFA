import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class LFADataset(Dataset):
    def __init__(self, root_path, transform=None):
        self.root_path = root_path
        self.transform = transform

        train_images = sorted(
            os.listdir(root_path + "/train_images/"),
            key=lambda content: int(''.join(d for d in content if d.isdigit()))
        )
        train_masks = sorted(
            os.listdir(root_path + "/train_masks/"),
            key=lambda content: int(''.join(d for d in content if d.isdigit()))
        )
        self.images = [root_path + "/train_images/" + img for img in train_images]
        self.masks = [root_path + "/train_masks/" + img for img in train_masks]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.array(Image.open(self.images[index]).convert("RGB"))
        mask = np.rot90(np.array(Image.open(self.masks[index]).convert("L"), dtype=np.float32))

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        return image, mask
