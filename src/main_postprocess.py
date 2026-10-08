from pathlib import Path

import numpy as np

import config_data as cfg
from postprocess.evaluator import SegmentationEvaluator


project_path = Path(__file__).resolve().parent
data_path = project_path / "data" / (cfg.AGGREGATE_TYPE + "s")
results_path = project_path / "results"

results_path.mkdir(exist_ok=True)


if __name__ == "__main__":
    packing_dirs = sorted(
        file for file in data_path.iterdir() if file.is_dir()
    )

    for packing_dir in packing_dirs:
        inference_path = packing_dir / "inference"

        if not inference_path.is_dir():
            continue

        print(f"Processing {packing_dir.name}")

        evaluator = SegmentationEvaluator(
            packing_dir,
            input_type="synthetic",
            tolerance=15,
        )

        result = evaluator.evaluate()

        output_path = results_path / f"{packing_dir.name}.npz"

        np.savez(
            output_path,
            **result,
        )