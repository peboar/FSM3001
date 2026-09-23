"""
Image augmentation for the synthetic slices.
Augmentations are applied to make the synthetic slices look more similar to real CT-scans.
All augmentations are applied before the image is converted to a tensor.
The mask is untouched.

TODO: no augmentations are implemented and the images are returned without any augmentations.
"""
from PIL import Image


class ImageAugmenter:
    def __init__(self, noise_type="none", phase_colors=None):
        if noise_type not in {"none", "ct"}:
            raise ValueError(f"Unknown noise_type: {noise_type}")

        self.noise_type = noise_type
        self.phase_colors = phase_colors

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
        # TODO: FFT-amplitude noise from real ct-scans
        return image