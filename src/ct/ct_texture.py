from pathlib import Path
import numpy as np
from PIL import Image


class ExtractCtTexture:
    """Handles mean color extraction and noise extraction from patches of real CT-scans"""
    def __init__(self, folder_path, image_size=512, image_extension="tif"):
        self.folder_path = Path(folder_path)
        self.image_resolution = (image_size, image_size)
        self.image_paths = sorted(
            self.folder_path.glob(f"*.{image_extension}")
        )

        window_height = np.hanning(image_size)
        window_width = np.hanning(image_size)

        self.hanning_2d = np.outer(window_height, window_width)

    def _make_grayscale(self):
        if not self.image_paths:
            raise RuntimeError(f"No patches found in {self.folder_path}")

        gray_scale_images = []
        for image_path in self.image_paths:
            gray_scale_images.append(Image.open(image_path).convert("L"))

        return gray_scale_images

    def extract_gray_mean(self):
        images = self._make_grayscale()

        image_means = []
        image_weights = []

        for image in images:
            image_weights.append(image.width * image.height)
            image_means.append(np.average(np.array(image)))

        return int(np.average(image_means, weights=image_weights))

    def extract_gray_standard_deviation(self):
        images = self._make_grayscale()

        image_weights = []
        standard_deviations = []

        for image in images:
            image_weights.append(image.width * image.height)
            standard_deviations.append(np.array(image, dtype=np.float32).std())
        return np.average(standard_deviations, weights=image_weights)

    def _resize_image(self, image):

        resized_image = image.resize(
            self.image_resolution, Image.Resampling.BICUBIC
        )
        return resized_image

    def extract_ct_texture(self):
        """Extract the average fft amplitude of all patches"""
        images = self._make_grayscale()
        image_weights = []
        magnitudes = []

        for image in images:
            image_weights.append(image.width * image.height)
            image = self._resize_image(image)

            # Remove the average brightness
            image_array = np.array(image) - np.average(np.array(image))

            # Apply the Hanning window to get rid of boundary effects
            fourier_transform = np.fft.fft2(image_array*self.hanning_2d)
            fourier_transform_shift = np.fft.fftshift(fourier_transform)
            magnitude = np.abs(fourier_transform_shift)

            magnitudes.append(magnitude)

        return np.average(magnitudes, weights=image_weights, axis=0)