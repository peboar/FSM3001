from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes


def extract_container_coordinates(image_path):
    """Extract the cylinder container coordinates from a CT slice.

    The cylinder is assumed to be stationary across all CT slices, so the
    resulting coordinates can be reused for every slice in the scan.

    Pixels outside the estimated cylinder are set to white. Sensor noise
    below the intensity threshold is removed before filling the cylinder
    interior.

    Args:
        image_path: Path to a CT image used to determine the container.

    Returns:
        Boolean mask containing the pixels inside the container.
    """
    image = Image.open(image_path).convert("L")
    image_arr = np.array(image)

    height, width = image_arr.shape

    # Estimate the center of the cylindrical container.
    x_mid = height // 2
    y_mid = width // 2

    # Set pixels outside the estimated container to white.
    radius = 1.05 * min(height // 2, width // 2)

    for h in range(height):
        for w in range(width):
            if (h - y_mid) ** 2 + (w - x_mid) ** 2 > radius ** 2:
                image_arr[h, w] = 255

    # Identify pixels belonging to the container.
    non_black_pixels = image_arr > 50

    # Fill holes to obtain a solid container mask.
    container_mask = binary_fill_holes(non_black_pixels)

    # Create an image containing only the container region.
    cropped_image_arr = np.full_like(image_arr, 255)
    cropped_image_arr[container_mask] = image_arr[container_mask]

    cropped_image = Image.fromarray(cropped_image_arr)

    # Save the coordinates for reuse on all slices.
    container_coordinates = np.where(container_mask)

    script_directory = Path(__file__).resolve().parent
    coordinates_path = script_directory / "container_coordinates.npy"
    np.save(coordinates_path, container_coordinates)

    cropped_image.show()

    return container_mask


ct_image_path = (
    r"/home/per/Desktop/Kth/Phd/Courses/FSM3001/Project/"
    r"S01_1_Granite11_Brick11_SS/"
    r"S01.1 - Granite11 Brick11 SS/"
    r"Granite brick SS aft Y_0005.tif"
)

extract_container_coordinates(ct_image_path)