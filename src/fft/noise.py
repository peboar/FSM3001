from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

path = Path(__file__).resolve().parent
brick_path = path / "brick"
granite_path = path / "granite"
void_path = path / "void"


class ExtractCtNoise:
    """Handles mean color extraction and noise extraction from patches of real CT-scans"""
    def __init__(self, folder_path, image_size=512, image_extension="tif"):
        self.folder_path = folder_path
        self.image_resolution = (image_size, image_size)
        self.image_paths = sorted(
            self.folder_path.glob(f"*.{image_extension}")
        )

    def _make_grayscale(self):
        if not self.image_paths:
            raise RuntimeError(f"No patches found in {self.folder_path}")

        gray_scale_images = []
        for image_path in self.image_paths:
            gray_scale_images.append(Image.open(image_path).convert("L"))

        return gray_scale_images

    def extract_mean_gray(self):
        images = self._make_grayscale()

        image_means = []
        image_weights = []

        for image in images:
            image_weights.append(image.width * image.height)
            image_means.append(np.average(np.array(image)))

        return np.average(image_means, weights=image_weights)

    def _upscale_image(self):
        images = self._make_grayscale()
        upscaled_images = []

        for image in images:
            resized_image = image.resize(
                self.image_resolution, Image.Resampling.BICUBIC
            )
            upscaled_images.append(np.array(resized_image))

        return upscaled_images
