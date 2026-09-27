"""
Image augmentation for the synthetic slices.

Augmentations are applied to make the synthetic slices look more similar
to real CT scans. All augmentations are applied before the image is
converted to a tensor. The mask is untouched.
"""

from pathlib import Path

import albumentations as A
import numpy as np
from PIL import Image


class ImageAugmenter:
    def __init__(
        self,
        noise_type="none",
        seed=None,
        phase_colors=None,
        phase_standard_deviations=None,
    ):
        if noise_type not in {"none", "artificial"}:
            raise ValueError(f"Unknown noise_type: {noise_type}")

        self.noise_type = noise_type
        self.seed = seed
        self.phase_colors = phase_colors or {}
        self.phase_standard_deviations = (
            phase_standard_deviations or {}
        )

        self.transforms = A.Compose(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.2, 0.2),
                    contrast_limit=(-0.2, 0.2),
                    p=0.5,
                ),
                A.Illumination(
                    mode="gaussian",
                    intensity_range=(0.05, 0.15),
                    effect_type="both",
                    center_range=(0.2, 0.8),
                    sigma_range=(0.4, 0.8),
                    p=0.5,
                ),
                A.GaussianBlur(
                    sigma_limit=(0.8, 2),
                    p=1.0,
                ),
                A.MultiplicativeNoise(
                    multiplier=(0.9, 1.1),
                    per_channel=False,
                    elementwise=True,
                    p=0.5,
                ),
                A.ShotNoise(
                    scale_range=(0.005, 0.01),
                    p=0.5,
                ),
            ],
            seed=seed,
        )

    def augment(self, image):
        if self.noise_type == "none":
            return image

        image_array = np.asarray(image, dtype=np.uint8)

        # Store phase masks before the image intensities are changed.
        masks = {
            phase: image_array == color
            for phase, color in self.phase_colors.items()
        }

        image_array = self.transforms(
            image=image_array
        )["image"]

        image_array = self._apply_gaussian_noise(
            image_array,
            masks,
        )

        return Image.fromarray(image_array)

    def _apply_gaussian_noise(self, image, masks):
        image_array = image.copy()

        for phase, mask in masks.items():
            if not np.any(mask):
                continue

            std = self.phase_standard_deviations[phase]

            phase_pixels = image_array[mask].reshape(-1, 1)

            transform = A.Compose(
                [
                    A.GaussNoise(
                        std_range=(
                            0.5 * std / 255,
                            1.5 * std / 255,
                        ),
                        p=1.0,
                    )
                ],
                seed=self.seed,
            )

            phase_pixels = transform(
                image=phase_pixels
            )["image"]

            image_array[mask] = phase_pixels.reshape(-1)

        return np.clip(
            image_array,
            0,
            255,
        ).astype(np.uint8)