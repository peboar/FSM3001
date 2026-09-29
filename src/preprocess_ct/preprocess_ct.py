from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes


def extract_container_coordinates(image_path):
    """Extract the cylinder container coordinates from a CT slice.

    The cylinder is stationary across all CT slices, so the
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
    image_cropped_arr = np.full_like(image_arr, 255)
    image_cropped_arr[container_mask] = image_arr[container_mask]

    x_min = width
    x_max = 0
    y_min = height
    y_max = 0

    for h in range(height):
        for w in range(width):
            if image_cropped_arr[h][w] != 255:
                x_min = min(x_min, w)
                x_max = max(x_max, w)
                y_min = min(y_min, h)
                y_max = max(y_max, h)

    bounds = (x_min, y_min, x_max, y_max)
    image_cropped = Image.fromarray(image_cropped_arr)

    # Save the coordinates for reuse on all slices.
    container_coordinates = np.where(container_mask)

    script_directory = Path(__file__).resolve().parent
    coordinates_path = script_directory / "container_coordinates.npz"
    np.savez(coordinates_path, container_coordinates=container_coordinates, bounds=bounds)
    image_cropped.show()

def crop_ct_image(image_path, image_size=512):
    """Crop CT-images to make them easier to segment"""
    script_directory = Path(__file__).resolve().parent
    coordinates_path = script_directory / "container_coordinates.npz"

    image_resolution = (image_size, image_size)

    data = np.load(coordinates_path)
    container_coordinates = data["container_coordinates"]
    bounds = data["bounds"]
    mask_arr = tuple(container_coordinates)

    image = Image.open(image_path).convert("L")
    image_arr = np.array(image)

    image_cropped_arr = np.zeros_like(image_arr)
    image_cropped_arr[mask_arr] = image_arr[mask_arr]
    image_cropped = Image.fromarray(image_cropped_arr)
    image_cropped = image_cropped.crop(bounds)
    image_cropped.resize(image_resolution)

    return image_cropped


if __name__ == "__main__":
    # Use the first slice to detect the container boundaries
    ct_mask_image_path = (
        r"/home/per/Desktop/Kth/Phd/Courses/FSM3001/Project/"
        r"S01_1_Granite11_Brick11_SS/"
        r"S01.1 - Granite11 Brick11 SS/"
        r"Granite brick SS aft Y_0000.tif"
    )

    extract_container_coordinates(ct_mask_image_path)

    # Test if the crop works
    ct_image_path = (
        r"/home/per/Desktop/Kth/Phd/Courses/FSM3001/Project/"
        r"S01_1_Granite11_Brick11_SS/"
        r"S01.1 - Granite11 Brick11 SS/"
        r"Granite brick SS aft Y_0530.tif"
    )

    image = crop_ct_image(ct_image_path)
    image.save("cropped_granite_brick_ss_aft_y_0530.tif")