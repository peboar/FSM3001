"""
Image augmentation for the synthetic slices.

Augmentations are applied to make the synthetic slices look more similar
to real CT scans. All augmentations are applied before the image is
converted to a tensor. The mask is untouched.
"""
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
        self.rng = np.random.default_rng(seed)

        self.phase_colors = phase_colors or {}
        self.phase_standard_deviations = (
            phase_standard_deviations or {}
        )

        self.transforms_before_illumination = A.Compose(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.2, 0.2),
                    contrast_limit=(-0.2, 0.2),
                    p=0.5,
                ),
            ],
            seed=self.seed,
        )

        self.transforms_after_illumination = A.Compose(
            [
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
                    scale_range=(0.005, 0.015),
                    p=0.5,
                ),
            ],
            seed=self.seed,
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

        image_array = self.transforms_before_illumination(
            image=image_array
        )["image"]

        image_array = self._apply_illumination(image_array)

        image_array = self.transforms_after_illumination(
            image=image_array
        )["image"]

        image_array = self._apply_gaussian_noise(
            image_array,
            masks,
        )

        return Image.fromarray(image_array)

    def _apply_gaussian_noise(self, image_array, masks, p=1):
        if self.rng.random() >= p:
            return image_array

        output_image_array = image_array.copy()

        for phase, mask in masks.items():
            if not np.any(mask):
                continue

            std = self.phase_standard_deviations[phase]

            phase_pixels = output_image_array[mask].reshape(-1, 1)

            transform = A.Compose(
                [
                    A.GaussNoise(
                        std_range=(
                            0.5 * std / 255,
                            1.5 * std / 255,
                        ),
                        p=p,
                    )
                ],
                seed=self.seed,
            )

            phase_pixels = transform(
                image=phase_pixels
            )["image"]

            output_image_array[mask] = phase_pixels.reshape(-1)

        return np.clip(
            output_image_array,
            0,
            255,
        ).astype(np.uint8)

    def _apply_illumination(self, image_array, p=0.5):
        if self.rng.random() >= p:
            return image_array

        image_height, image_width = image_array.shape

        num_circles = self.rng.integers(4, 16)
        max_radius = min(image_height, image_width)

        radii = (
                        np.linspace(0, 1, num_circles + 2)[1:-1] ** 2
                ) * max_radius

        center_x = np.clip(
            self.rng.normal(0.5, 0.15),
            0.2,
            0.8,
        )

        center_y = np.clip(
            self.rng.normal(0.5, 0.15),
            0.2,
            0.8,
        )

        center_x_pixel = center_x * image_width
        center_y_pixel = center_y * image_height

        y, x = np.ogrid[:image_height, :image_width]
        distance = np.sqrt(
            (x - center_x_pixel) ** 2
            + (y - center_y_pixel) ** 2
        )

        transform = A.Compose(
            [
                A.Illumination(
                    mode="gaussian",
                    intensity_range=(0.05, 0.2),
                    effect_type="both",
                    center_range=(0.2, 0.8),
                    sigma_range=(0.4, 0.8),
                    p=1.0,
                ),
            ],
            seed=self.seed,
        )

        output_image_array = image_array.copy()

        for radius in radii.astype(int):
            ring_mask = (
                    (distance >= radius - 5)
                    & (distance <= radius + 5)
            )

            output_image_array[ring_mask] = transform(
                image=output_image_array
            )["image"][ring_mask]

        return output_image_array