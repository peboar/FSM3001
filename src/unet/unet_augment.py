"""
Image augmentation for synthetic CT slices.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from medaugmentx import Compose, MedVolume, OneOf, Transform
from medaugmentx.transforms import (
    BeamHardening,
    GaussianBlur,
    GaussianNoise,
    WindowLevel,
)


class PhaseColors(Transform):
    def __init__(
        self,
        phase_colors=None,
        edge_factor=1.0,
        material_inclusions=None,
        seed=None,
    ):
        super().__init__(p=1.0, seed=seed)

        self.phase_colors = phase_colors
        self.number_of_phases = (
            None if not phase_colors else len(phase_colors)
        )
        self.edge_factor = edge_factor
        self.material_inclusions = material_inclusions

    def apply(self, volume):
        image_arr = volume.image.astype(
            np.float32,
            copy=False,
        )

        image_255_arr = image_arr * 255.0
        augmented_255_arr = image_255_arr.copy()

        if self.number_of_phases:
            min_distance = 30
            min_inclusion_distance = 30

            while True:
                colors = self.rng.choice(
                    np.arange(256),
                    size=self.number_of_phases,
                    replace=False,
                )

                if np.min(
                    np.diff(np.sort(colors))
                ) >= min_distance:
                    break

            for i, (phase, color) in enumerate(
                self.phase_colors.items()
            ):
                mask = image_255_arr == color
                augmented_255_arr[mask] = colors[i]

                edge_color = min(
                    255,
                    int(color * self.edge_factor),
                )

                mask_edges = image_255_arr == edge_color

                new_edge_color = min(
                    255,
                    int(colors[i] * self.edge_factor),
                )

                augmented_255_arr[mask_edges] = new_edge_color

                if (
                    self.material_inclusions
                    and phase in self.material_inclusions
                    and self.material_inclusions[phase]
                ):
                    contrast = self.material_inclusions[
                        phase
                    ]["contrast"]

                    inclusion_color = min(
                        255,
                        int(color * contrast),
                    )

                    mask_inclusion = (
                        image_255_arr == inclusion_color
                    )

                    max_inclusion_color = (
                        colors[i] - min_inclusion_distance
                    )

                    if max_inclusion_color > 0:
                        new_inclusion_color = (
                            self.rng.integers(
                                1,
                                max_inclusion_color + 1,
                            )
                        )

                        augmented_255_arr[
                            mask_inclusion
                        ] = new_inclusion_color

        augmented = augmented_255_arr / 255.0

        return volume.replace(
            image=augmented.astype(
                np.float32,
                copy=False,
            )
        )


class FFTCloudNoise(Transform):
    def __init__(
        self,
        cloud_scale=(15.0, 50.0),
        strength=(1.0, 10.0),
        p=0.5,
        seed=None,
    ):
        super().__init__(p=p, seed=seed)

        self.cloud_scale = cloud_scale
        self.strength = strength

    def apply(self, volume):
        image_arr = volume.image.astype(
            np.float32,
            copy=False,
        )

        image_255_arr = image_arr * 255.0

        noise = self.rng.normal(
            0.0,
            1.0,
            image_arr.shape,
        )

        height, width = image_arr.shape

        fy = np.fft.fftfreq(height)
        fx = np.fft.fftfreq(width)

        fx, fy = np.meshgrid(fx, fy)

        radius = np.sqrt(fx**2 + fy**2)

        cloud_scale = float(
            self.rng.uniform(*self.cloud_scale)
        )

        frequency_scale = 1.0 / cloud_scale

        frequency_filter = np.exp(
            -(radius / frequency_scale) ** 2
        )

        noise_fft = np.fft.fft2(noise)

        cloud = np.real(
            np.fft.ifft2(
                noise_fft * frequency_filter
            )
        )

        cloud -= cloud.mean()
        cloud /= cloud.std()

        strength = float(
            self.rng.uniform(*self.strength)
        )

        augmented_255 = (
            image_255_arr
            + strength * cloud
        )

        augmented_255 = np.clip(
            augmented_255,
            0.0,
            255.0,
        )

        augmented = augmented_255 / 255.0

        return volume.replace(
            image=augmented.astype(
                np.float32,
                copy=False,
            )
        )


class ImageAugmenter:
    def __init__(
        self,
        phase_colors=None,
        edge_factor=1.0,
        material_inclusions=None,
        seed=None,
    ):
        self.transform = Compose(
            [
                PhaseColors(
                    phase_colors=phase_colors,
                    edge_factor=edge_factor,
                    material_inclusions=material_inclusions,
                    seed=seed,
                ),
                GaussianBlur(
                    sigma=(1.0, 3.0),
                    p=0.5,
                ),
                WindowLevel(
                    center_shift_frac=0.05,
                    width_scale=(0.85, 1.15),
                    p=0.5,
                ),
                OneOf(
                    [
                        GaussianNoise(
                            std=(0.01, 0.05),
                        ),
                        GaussianNoise(
                            std=(0.05, 0.12),
                        ),
                    ],
                    p=1,
                ),
                OneOf(
                    [
                        BeamHardening(
                            alpha=(0.05, 0.10),
                            power=2.0,
                        ),
                        BeamHardening(
                            alpha=(0.10, 0.25),
                            power=2.0,
                        ),
                    ],
                    p=0.5,
                ),
            ],
            seed=seed,
        )

    def augment(self, image):
        image_array = np.asarray(
            image,
            dtype=np.float32,
        )

        image_array /= 255.0

        volume = MedVolume(
            image=image_array,
            spacing=(1.0, 1.0),
            metadata={"modality": "CT"},
        )

        augmented = self.transform(volume)

        output = np.asarray(
            augmented.image,
            dtype=np.float32,
        )

        output = np.clip(
            output,
            0.0,
            1.0,
        )

        output = (output * 255).astype(np.uint8)

        return Image.fromarray(output)


if __name__ == "__main__":
    image_path = Path(
        r"C:\KTH\Courses\FSM3001\Project\src\data\polyhedrons"
        r"\packing_398_20260921_221507\slices"
        r"\slice_z_003_89_aggregates_39.tif"
    )

    image = Image.open(image_path)

    phase_colors = {
        "granite": 93,
        "brick": 83,
        "void": 49,
    }

    granite_name = "granite"
    granite_inclusion = {
        "contrast": 1.5,
        "threshold": 0.025,
        "scale_x": 0.35,
        "scale_y": 0.025,
        "scale_z": 0.05,
        "octaves": 1,
    }

    brick_name = "brick"
    brick_inclusion = None

    material_inclusions = {
        granite_name: granite_inclusion,
        brick_name: brick_inclusion,
    }

    for seed in range(1, 100):
        augmenter = ImageAugmenter(
            phase_colors=phase_colors,
            edge_factor=1.0,
            material_inclusions=material_inclusions,
            seed=seed,
        )

        augmented = augmenter.augment(image)

        plt.figure()
        plt.imshow(augmented, cmap="gray")
        plt.axis("off")
        plt.show()