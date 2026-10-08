from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment


class SegmentationEvaluator:

    def __init__(
        self,
        inference_path,
        mask_path,
        tolerance=15,
    ):
        self.inference_path = Path(inference_path)
        self.mask_path = Path(mask_path)

        self.inference_image = Image.open(
            self.inference_path
        )
        self.mask_image = Image.open(
            self.mask_path
        )

        self.tolerance = tolerance

    @staticmethod
    def aggregate_centroids(image):
        image_arr = np.array(image, copy=True)

        rows, cols = image_arr.shape

        aggregate_centroids = []
        aggregate_areas = []

        directions = [
            (-1, 0),
            (0, -1),
            (0, 1),
            (1, 0),
        ]

        for row in range(rows):
            for column in range(cols):

                if image_arr[row, column] != 128:
                    continue

                stack = [(row, column)]
                image_arr[row, column] = 0

                current_aggregate = []

                while stack:
                    r, c = stack.pop()

                    current_aggregate.append((r, c))

                    for d_r, d_c in directions:
                        next_r = r + d_r
                        next_c = c + d_c

                        if 0 <= next_r < rows and 0 <= next_c < cols:
                            if image_arr[next_r, next_c] == 128:
                                image_arr[next_r, next_c] = 0
                                stack.append(
                                    (next_r, next_c)
                                )

                total_pixels = len(current_aggregate)

                centroid_y = (
                    sum(r for r, _ in current_aggregate)
                    / total_pixels
                )

                centroid_x = (
                    sum(c for _, c in current_aggregate)
                    / total_pixels
                )

                aggregate_centroids.append(
                    [centroid_x, centroid_y]
                )

                aggregate_areas.append(total_pixels)

        return (
            np.array(aggregate_centroids),
            aggregate_areas,
        )

    def match_centroids(
        self,
        predicted_centroids,
        ground_truth_centroids,
    ):
        if (
            len(predicted_centroids) == 0
            or len(ground_truth_centroids) == 0
        ):
            return np.empty((0, 3))

        squared_distances = (
            np.sum(
                predicted_centroids**2,
                axis=1,
                keepdims=True,
            )
            + np.sum(
                ground_truth_centroids**2,
                axis=1,
            )
            - 2
            * np.dot(
                predicted_centroids,
                ground_truth_centroids.T,
            )
        )

        distances = np.sqrt(
            np.maximum(squared_distances, 0)
        )

        predicted_indices, ground_truth_indices = (
            linear_sum_assignment(distances)
        )

        matches = []

        for predicted_index, ground_truth_index in zip(
            predicted_indices,
            ground_truth_indices,
        ):
            distance = distances[
                predicted_index,
                ground_truth_index,
            ]

            if distance <= self.tolerance:
                matches.append(
                    (
                        predicted_index,
                        ground_truth_index,
                        distance,
                    )
                )

        return np.array(
            matches,
            dtype=np.float64,
        ).reshape(-1, 3)

    @staticmethod
    def aggregate_area_agreement(
        predicted_areas,
        ground_truth_areas,
        matches,
    ):
        if len(matches) == 0:
            return 0.0

        agreement = 0.0

        for predicted_index, ground_truth_index, _ in matches:

            predicted_index = int(predicted_index)
            ground_truth_index = int(ground_truth_index)

            predicted_area = predicted_areas[predicted_index]
            ground_truth_area = ground_truth_areas[
                ground_truth_index
            ]

            agreement += (
                1
                - abs(
                    predicted_area - ground_truth_area
                )
                / ground_truth_area
            )

        return agreement / len(matches)

    @staticmethod
    def dice_score(
        prediction,
        ground_truth,
        class_value,
    ):
        predicted_class = prediction == class_value
        ground_truth_class = ground_truth == class_value

        intersection = np.sum(
            predicted_class & ground_truth_class
        )

        denominator = (
            np.sum(predicted_class)
            + np.sum(ground_truth_class)
        )

        if denominator == 0:
            return 1.0

        return 2 * intersection / denominator

    def evaluate(self):
        inference = np.array(
            self.inference_image,
            copy=True,
        )

        mask = np.array(
            self.mask_image,
            copy=True,
        )

        predicted_centroids, predicted_areas = (
            self.aggregate_centroids(inference)
        )

        ground_truth_centroids, ground_truth_areas = (
            self.aggregate_centroids(mask)
        )

        matches = self.match_centroids(
            predicted_centroids,
            ground_truth_centroids,
        )

        aggregate_dice = self.dice_score(
            inference,
            mask,
            128,
        )

        boundary_dice = self.dice_score(
            inference,
            mask,
            255,
        )

        area_agreement = (
            self.aggregate_area_agreement(
                predicted_areas,
                ground_truth_areas,
                matches,
            )
        )

        return {
            "aggregate_dice": aggregate_dice,
            "boundary_dice": boundary_dice,
            "predicted_aggregates": len(
                predicted_centroids
            ),
            "ground_truth_aggregates": len(
                ground_truth_centroids
            ),
            "matched_aggregates": len(matches),
            "area_agreement": area_agreement,
        }




if __name__ == "__main__":

    inference_path = Path(
        r"C:\KTH\Courses\FSM3001\Project\src\data"
        r"\polyhedrons\packing_430_20260921_235616"
        r"\inference\inference_z_007_90_aggregates_53.tif"
    )

    mask_path = Path(
        r"C:\KTH\Courses\FSM3001\Project\src\data"
        r"\polyhedrons\packing_430_20260921_235616"
        r"\masks\mask_z_007_90_aggregates_53.tif"
    )

    evaluator = SegmentationEvaluator(
        inference_path,
        mask_path,
        tolerance=15,
    )

    results = evaluator.evaluate()

    print(results)




