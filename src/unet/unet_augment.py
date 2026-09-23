"""
Image augmentation for the synthetic slices.
Augmentations are applied to make the synthetic slices look more similar to real CT-scans.
All augmentations are applied before the image is converted to a tensor.
The mask is untouched.

TODO: no augmentations are implemented and the images are returned without any augmentations.
"""
import numpy as np
from PIL import Image


class ImageAugmenter:
    def __init__(
            self,
            noise_type="none",
            phase_colors=None,
            ct_textures=None,
            ct_intensity=None
    ):

        if noise_type not in {"none", "ct"}:
            raise ValueError(f"Unknown noise_type: {noise_type}")

        self.noise_type = noise_type
        self.phase_colors = phase_colors
        self.ct_textures = ct_textures

    def augment(self, image: Image.Image) -> Image.Image:
        image = self._apply_blur(image)
        image = self._apply_noise(image)
        return image

    def _apply_blur(self, image: Image.Image) -> Image.Image:
        # TODO: Gaussian blur
        return image

    def _apply_noise(self, image: Image.Image) -> Image.Image:
        if self.noise_type == "ct":
            return self._apply_ct_noise(image)

        return image

    def _apply_gaussian_noise(self, image: Image.Image) -> Image.Image:
        # TODO: Random noise using a Gaussian distribution
        return image

    def _apply_ct_noise(self, image: Image.Image) -> Image.Image:
        image_array = np.array(image, dtype=np.float32)
        noisy_image = image_array.copy()

        for phase, color in self.phase_colors.items():
            mask = (image_array == color)

            if not np.any(mask):
                continue

            texture = self.ct_textures[phase]
            # Generate random texture infused with noise
            noise = self._generate_ct_texture(texture)

            # Add the texture to the phase
            noisy_image[mask] += noise[mask]

        # Remove values outside the valid 0-255 gray-scale range
        noisy_image = np.clip(noisy_image, 0, 255)

        return Image.fromarray(noisy_image.astype(np.uint8))

    def _generate_ct_texture(self, magnitude):
        """Generate a random image used to create a unique CT texture."""
        random_image = np.random.normal(
            0,
            1,
            magnitude.shape,
        )

        random_fft = np.fft.fftshift(
            np.fft.fft2(random_image)
        )

        phase = np.angle(random_fft)

        # Reconstruct new fft with noise
        spectrum = magnitude * np.exp(1j * phase)

        # Reconstruct texture with random noise
        texture = np.fft.ifft2(
            np.fft.ifftshift(spectrum)
        ).real

        # Noise should not shift the data, so remove the mean
        texture = texture - texture.mean()

        return texture


if __name__ == "__main__":
    from pathlib import Path

    from ct.ct_texture import ExtractCtTexture

    project_path = Path(__file__).resolve().parent.parent
    ct_path = project_path / "ct"

    image_path = r"/home/per/Desktop/Kth/Phd/Courses/FSM3001/Project/src/data/polyhedrons/packing_398_20260921_221507/slices/slice_z_047_21_aggregates_52.tif"

    image_size = 512
    phases = ["granite", "brick", "void"]

    textures = {}
    for phase in phases:
        texture = ExtractCtTexture(
            folder_path=ct_path / phase,
            image_size=image_size,
            image_extension="tif",
        )
        textures[phase] = texture

    phase_colors = {
        phase: texture.extract_mean_gray()
        for phase, texture in textures.items()
    }

    ct_textures = {
        phase: texture.extract_ct_texture()
        for phase, texture in textures.items()
    }

    image = Image.open(image_path).convert("L")

    augmenter = ImageAugmenter(
        noise_type="ct",
        phase_colors=phase_colors,
        ct_textures=ct_textures,
    )

    augmented_image = augmenter.augment(image)

    image.show(title="Original")
    augmented_image.show(title="CT texture")

