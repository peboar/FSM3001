from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment


class SegmentationEvaluator:

    def __init__(
        self,
        packing_path,
        input_type="synthetic",
        tolerance=15,
    ):

        if input_type not in {"synthetic", "real"}:
            raise ValueError(
                "input_type should be 'synthetic' or 'real'"
            )

        self.packing_path = Path(packing_path)
        self.inference_path = self.packing_path / "inference"
        self.mask_path = self.packing_path / "masks"

        self.input_type = input_type
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
                                stack.append((next_r, next_c))

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

        inference_files = sorted(
            self.inference_path.glob("*.tif")
        )

        mask_files = sorted(
            self.mask_path.glob("*.tif")
        )

        if len(inference_files) != len(mask_files):
            raise ValueError(
                f"Number of inference files "
                f"({len(inference_files)}) does not match "
                f"number of mask files ({len(mask_files)})."
            )

        total_predicted = 0
        total_ground_truth = 0
        total_matched = 0

        aggregate_dice = []
        boundary_dice = []
        area_agreements = []

        for inference_file, mask_file in zip(
            inference_files,
            mask_files,
        ):

            inference = np.array(
                Image.open(inference_file),
                copy=True,
            )

            mask = np.array(
                Image.open(mask_file),
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

            total_predicted += len(
                predicted_centroids
            )

            total_ground_truth += len(
                ground_truth_centroids
            )

            total_matched += len(matches)

            if self.input_type == "synthetic":

                aggregate_dice.append(
                    self.dice_score(
                        inference,
                        mask,
                        128,
                    )
                )

                boundary_dice.append(
                    self.dice_score(
                        inference,
                        mask,
                        255,
                    )
                )

                area_agreements.append(
                    self.aggregate_area_agreement(
                        predicted_areas,
                        ground_truth_areas,
                        matches,
                    )
                )

        centroid_precision = (
            total_matched / total_predicted
            if total_predicted > 0
            else 0.0
        )

        centroid_recall = (
            total_matched / total_ground_truth
            if total_ground_truth > 0
            else 0.0
        )

        centroid_f1 = (
            2 * total_matched
            / (
                    total_predicted
                    + total_ground_truth
            )
            if total_predicted + total_ground_truth > 0
            else 0.0
        )

        results = {
            "predicted_aggregates": total_predicted,
            "ground_truth_aggregates": total_ground_truth,
            "matched_aggregates": total_matched,
            "centroid_precision": centroid_precision,
            "centroid_recall": centroid_recall,
            "centroid_f1": centroid_f1,
        }

        if self.input_type == "synthetic":
            results.update({
                "aggregate_dice": np.mean(
                    aggregate_dice
                ),
                "boundary_dice": np.mean(
                    boundary_dice
                ),
                "area_agreement": np.mean(
                    area_agreements
                ),
            })

        return results


if __name__ == "__main__":

    packing_path = Path(
        r"/unet_ouput_new_fft/edges/data/polyhedrons/packing_425_20260922_020837"
    )

    evaluator = SegmentationEvaluator(
        packing_path,
        input_type="synthetic",
        tolerance=15,
    )

    results = evaluator.evaluate()

    print(results)