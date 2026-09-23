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

