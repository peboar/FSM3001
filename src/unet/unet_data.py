from pathlib import Path
import numpy as np

from PIL import Image
import torch
from torch.utils.data import Dataset
from unet.unet_augment import ImageAugmenter

class SingleUnetDataset(Dataset):
    def __init__(self, packing_path, mode="train", image_size=512, noise_type="none"):
        if mode not in {"train", "validation", "test"}:
            raise ValueError(f"mode {mode} must be train, validation or test")

        self.packing_path = Path(packing_path)
        self.image_size = image_size
        self.mode = mode
        self.augmenter = ImageAugmenter(noise_type)

        self.path_slices = self.packing_path / "slices"
        self.path_masks = self.packing_path / "masks"

        self.slices = sorted([image for image in self.path_slices.glob("*.png") if image.is_file()])
        self.masks = sorted([image for image in self.path_masks.glob("*.png") if image.is_file()])

        if not self.slices:
            raise RuntimeError(f"No slices found in {self.path_slices}")

        if len(self.slices) != len(self.masks):
            raise RuntimeError(f"The number of slices: {len(self.slices)} " 
                               f"does not match the number of masks: {len(self.masks)}")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        image = Image.open(self.slices[index]).convert("L")
        mask = Image.open(self.masks[index]).convert("L")

        size = (self.image_size, self.image_size)
        image = image.resize(size, Image.BILINEAR)
        mask = mask.resize(size, Image.NEAREST)

        image = self.augmenter.augment(image)

        # Normalize gray scale colors to be between 0-1
        image = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0)
        # Adds another dimension corresponding to the batch size
        image = image.unsqueeze(0)

        mask_array = np.array(mask, dtype=np.uint8)
        # 0: Black voids, 1: Gray aggregates, 2: White aggregate boundaries
        mask_to_values = {0: 0, 128: 1, 255: 2}
        for mask_value, class_label in mask_to_values.items():
            mask_array[mask_array == mask_value] = class_label

        mask = Image.from_numpy(mask_array)

        return image, mask


path = r"/home/per/Desktop/Kth/Phd/Courses/FSM3001/Project/src/data/polyhedrons/packing_398_20260921_221507"
a = SingleUnetDataset(path)
