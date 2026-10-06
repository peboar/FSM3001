import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment


def aggregate_centroids(image_arr):
    rows, cols = image_arr.shape
    aggregate_centroids = []
    aggregate_pixels_dict = {}  # Stores aggregate_index -> list of (row, col) pixels
    aggregate_counter = 0

    directions = [
        (-1, 0),
        (0, -1),
        (0, 1),
        (1, 0),
    ]

    for row in range(rows):
        for column in range(cols):
            if image_arr[row][column] == 128:
                stack = [(row, column)]
                image_arr[row][column] = 0  # Mark as visited
                current_aggregate_pixels = []

                while stack:
                    r, c = stack.pop()
                    current_aggregate_pixels.append((r, c))

                    for d_r, d_c in directions:
                        next_r, next_c = r + d_r, c + d_c
                        if 0 <= next_r < rows and 0 <= next_c < cols:
                            if image_arr[next_r][next_c] == 128:
                                image_arr[next_r][next_c] = 0
                                stack.append((next_r, next_c))

                total_pixels = len(current_aggregate_pixels)
                centroid_x = sum(x for _, x in current_aggregate_pixels) / total_pixels
                centroid_y = sum(y for y, _ in current_aggregate_pixels) / total_pixels

                aggregate_centroids.append([centroid_x, centroid_y])
                # Save the pixel aggregate coordinates cleanly
                aggregate_pixels_dict[aggregate_counter] = current_aggregate_pixels
                aggregate_counter += 1
            else:
                image_arr[row][column] = 0

    return np.array(aggregate_centroids).reshape(-1, 2), aggregate_pixels_dict


def match_centroids(predicted_centroids, centroids, tolerance=15):
    squared_dist = (
        np.sum(predicted_centroids**2, axis=1, keepdims=True)
        + np.sum(centroids**2, axis=1)
        - 2 * np.dot(predicted_centroids, centroids.T)
    )
    distance_matrix = np.sqrt(squared_dist)
    row_indices, column_indices = linear_sum_assignment(distance_matrix)

    matches = []

    for row, column in zip(row_indices, column_indices):
        distance = distance_matrix[row, column]

        if distance <= tolerance:
            matches.append((row, column, distance))

    return np.array(matches)




if __name__ == "__main__":
    inference_path = Path(
            r"C:\KTH\Courses\FSM3001\Project\src\data\polyhedrons\packing_430_20260921_235616\inference\inference_z_007_90_aggregates_53.tif"
    )

    centroid_path = Path(
        r"C:\KTH\Courses\FSM3001\Project\src\data\polyhedrons\packing_430_20260921_235616\centroids\centroid_z_007_90_aggregates_53.npz"
    )

    inference = Image.open(inference_path)
    data = np.load(centroid_path, allow_pickle=True)
    mask_centroids = data["centroids"]
    # Pass a copy to your algorithm
    inference_arr = np.array(inference, copy=True)

    centroids, aggregates = aggregate_centroids(inference_arr)  # Returns a list of (row, col) tuples
    color_arr = np.array(inference.convert("RGB"))
    copy = color_arr.copy()

    if np.any(centroids):
        raw_x, raw_y = zip(*centroids)
        x = np.round(raw_x).astype(int)
        y = np.round(raw_y).astype(int)

        color_arr[y, x] = [255, 0, 0]

    if np.any(mask_centroids):
        raw_x, raw_y = zip(*mask_centroids)
        x = np.round(raw_x).astype(int)
        y = np.round(raw_y).astype(int)

        color_arr[y, x] = [0, 255, 0]

    final_image = Image.fromarray(color_arr)
    final_image.show()
    final_image = Image.fromarray(color_arr)
    matches = match_centroids(centroids, mask_centroids)
