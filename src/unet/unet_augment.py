"""
Image augmentation for the synthetic slices.
Augmentations are applied to make the synthetic slices look more similar to real CT-scans.
All augmentations are applied before the image is converted to a tensor.
The mask is untouched.
"""

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage


class ImageAugmenter:
    def __init__(
        self,
        noise_type="none",
        phase_colors=None,
        phase_standard_deviations=None,
    ):
        if noise_type not in {"none", "artificial", "ct"}:
            raise ValueError(f"Unknown noise_type: {noise_type}")

        self.noise_type = noise_type
        self.phase_colors = phase_colors or {}
        self.phase_standard_deviations = phase_standard_deviations or {}

    def augment(self, image):
        if self.noise_type == "none":
            return image

        # Masks from the original flat image
        # before anything touches the colors.
        image_array = np.array(image, dtype=np.float32)
        masks = [image_array == color for color in self.phase_colors.values()]

        image = self._apply_blur(image, radius=1.6)  # blur
        image = self._apply_noise(image, masks)  # noise added last, amplitude survives

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

            noise = np.random.normal(0, 1, image_array.shape)
            noise = ndimage.gaussian_filter(noise, sigma=0.6)  # small correlation length, its own blur
            noise = noise / noise.std() * standard_deviation  # rescale AFTER blurring, hits target exactly

            noisy_image[mask] += noise[mask]

        noisy_image = np.clip(noisy_image, 0, 255)
        return Image.fromarray(noisy_image.astype(np.uint8))

    def _apply_ct_noise(self, image, masks):
        """TODO: implement real CT texture augmentation."""
        return image
