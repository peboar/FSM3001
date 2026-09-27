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


class RandomBrightnessContrast:
    def __init__(
        self,
        brightness_limit=(-0.2, 0.2),
        contrast_limit=(-0.2, 0.2),
        p=0.5,
    ):
        self.transform = A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=p,
        )

    def __call__(self, image):
        return self.transform(image=image)["image"]


class Illumination:
    def __init__(self, p=0.5):
        self.transform = A.Illumination(
            mode="gaussian",
            intensity_range=(0.05, 0.15),
            effect_type="both",
            center_range=(0.2, 0.8),
            sigma_range=(0.4, 0.8),
            p=p,
        )

    def __call__(self, image):
        return self.transform(image=image)["image"]


class GaussianBlur:
    def __init__(self, sigma_limit=(0.8, 2), p=1):
        self.transform = A.GaussianBlur(
            sigma_limit=sigma_limit,
            p=p,
        )

    def __call__(self, image):
        return self.transform(image=image)["image"]


class MultiplicativeNoise:
    def __init__(
        self,
        multiplier_range=(0.9, 1.1),
        p=0.5,
    ):
        self.transform = A.MultiplicativeNoise(
            multiplier=multiplier_range,
            per_channel=False,
            elementwise=True,
            p=p,
        )

    def __call__(self, image):
        return self.transform(image=image)["image"]


class ShotNoise:
    def __init__(self, scale_range=(0.005, 0.01), p=0.5):
        self.transform = A.ShotNoise(
            scale_range=scale_range,
            p=p,
        )

    def __call__(self, image):
        return self.transform(image=image)["image"]


class GaussianNoise:
    def __init__(self, scale_range=(0.5, 1.5), p=0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, image, std):
        transform = A.GaussNoise(
            std_range=(
                self.scale_range[0] * std / 255,
                self.scale_range[1] * std / 255,
            ),
            p=self.p,
        )

        return transform(image=image)["image"]


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

        self.random_brightness_contrast = RandomBrightnessContrast()
        self.illumination = Illumination()
        self.gaussian_blur = GaussianBlur()

        self.multiplicative_noise = MultiplicativeNoise()
        self.shot_noise = ShotNoise()
        self.gaussian_noise = GaussianNoise()

    def augment(self, image):
        if self.noise_type == "none":
            return image

        image_array = np.asarray(image, dtype=np.uint8)

        # Store phase masks before the image intensities are changed.
        masks = {
            phase: image_array == color
            for phase, color in self.phase_colors.items()
        }
        image_array = self._apply_random_brightness_contrast(image_array)
        image_array = self._apply_illumination(image_array)

        image_array = self._apply_blur(image_array)

        image_array = self._apply_multiplicative_noise(image_array)
        image_array = self._apply_shot_noise(image_array)
        image_array = self._apply_gaussian_noise(image_array, masks)

        return Image.fromarray(image_array)

    def _apply_random_brightness_contrast(self, image):
        return self.random_brightness_contrast(image)

    def _apply_illumination(self, image):
        return self.illumination(image)

    def _apply_blur(self, image):
        return self.gaussian_blur(image)

    def _apply_gaussian_noise(self, image, masks):
        image_array = image.copy()

        for phase, mask in masks.items():
            if not np.any(mask):
                continue

            std = self.phase_standard_deviations[phase]

            phase_pixels = image_array[mask].reshape(-1, 1)

            phase_pixels = self.gaussian_noise(
                phase_pixels,
                std=std,
            )

            image_array[mask] = phase_pixels.reshape(-1)

        return np.clip(image_array, 0, 255).astype(np.uint8)

    def _apply_multiplicative_noise(self, image):
        return self.multiplicative_noise(image)

    def _apply_shot_noise(self, image):
        return self.shot_noise(image)
