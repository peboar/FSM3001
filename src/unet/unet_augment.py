"""
Image augmentation for the synthetic slices.
Augmentations are applied to make the synthetic slices look more similar to real CT-scans.
All augmentations are applied before the image is converted to a tensor.
The mask is untouched.
"""

import numpy as np
from PIL import Image, ImageFilter


class ImageAugmenter:
    def __init__(
        self,
        phase_colors,
        phase_standard_deviations,
        noise_type="none",
    ):
        if noise_type not in {"none", "artificial", "ct"}:
            raise ValueError(f"Unknown noise_type: {noise_type}")

        self.noise_type = noise_type
        self.phase_colors = phase_colors
        self.phase_standard_deviations = phase_standard_deviations

    def augment(self, image: Image.Image) -> Image.Image:
        if self.noise_type == "none":
            return image

        image_array = np.array(image, dtype=np.float32)

        masks = []
        for color in self.values():
            mask = image_array == color
            masks.append(mask)

        image = self._apply_noise(image, masks)
        image = self._apply_blur(image, 1)

        return image

    def _apply_blur(self, image: Image.Image, radius) -> Image.Image:
        """Apply Gaussian blur."""
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    def _apply_noise(self, image, masks):
        if self.noise_type == "ct":
            return self._apply_ct_noise(image, masks)

        if self.noise_type == "artificial":
            return self._apply_artificial_noise(image, masks)

        return image

    def _apply_artificial_noise(self, image, masks):
        image_array = np.array(image, dtype=np.float32)
        noisy_image = image_array.copy()

        for standard_deviation, mask in zip(self.phase_standard_deviations.values(), masks):
            if not np.any(mask):
                continue

            noise = np.random.normal(
                0,
                standard_deviation,
                image_array.shape,
            )

            noisy_image[mask] += noise[mask]

        noisy_image = np.clip(noisy_image, 0, 255)

        return Image.fromarray(noisy_image.astype(np.uint8))

    def _apply_ct_noise(self, image, masks):
        """TODO: implement real CT texture augmentation."""
        return image