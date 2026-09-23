from pathlib import Path
import numpy as np

from PIL import Image
import torch
from torch.utils.data import Dataset

from unet.unet_augment import ImageAugmenter


def compute_class_weights(packing_paths):
    """Use normalized inverse pixel area as Weights."""

    pixel_counts = np.zeros(3, dtype=np.int64)

    for path in packing_paths:
        mask_paths = sorted((Path(path) / "masks").glob("*.png"))
        for mask_path in mask_paths:
            mask_array = np.array(Image.open(mask_path).convert("L"))
            # Compute all pixels
            pixel_counts[0] += (mask_array == 0).sum()
            pixel_counts[1] += (mask_array == 128).sum()
            pixel_counts[2] += (mask_array == 255).sum()

    weights = 1.0 / pixel_counts
    weights = weights / weights.sum()

    return torch.from_numpy(weights).float()


class SingleUnetDataset(Dataset):
    def __init__(
        self,
        packing_path,
        image_size=512,
        noise_type="none",
        phase_colors=None,
    ):
        self.packing_path = Path(packing_path)
        self.image_size = image_size

        self.augmenter = ImageAugmenter(
            noise_type=noise_type,
            phase_colors=phase_colors,
        )

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

        image_resolution = (self.image_size, self.image_size)

        image = image.resize(image_resolution, Image.BILINEAR)
        mask = mask.resize(image_resolution, Image.NEAREST)

        image = self.augmenter.augment(image)

        # Normalize gray scale colors to be between 0-1
        image = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0)
        # Adds channel dimension (H, W) -> (1, H, W)
        image = image.unsqueeze(0)

        mask_array = np.array(mask, dtype=np.uint8)

        # Clean template for the mask output. Uses int64 to ensure a PyTorch LongTensor
        class_mask = np.zeros_like(mask_array, dtype=np.int64)
        # 0: Black voids, 1: Gray aggregates, 2: White aggregate boundaries
        class_mask[mask_array == 128] = 1
        class_mask[mask_array == 255] = 2

        mask = torch.from_numpy(class_mask)

        return image, mask