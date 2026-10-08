from pathlib import Path

import numpy as np

import config_data as cfg
from postprocess.evaluator import SegmentationEvaluator


project_path = Path(__file__).resolve().parent
data_path = project_path / "data" / (cfg.AGGREGATE_TYPE + "s")

run_name = Path(__file__).resolve().parent.name
output_path = data_path / f"evaluation_{cfg.EVALUATION_INPUT_TYPE}_{run_name}.npz"

packing_names = []
results = []

packing_dirs = sorted(
    path for path in data_path.iterdir() if path.is_dir()
)

for packing_dir in packing_dirs:
    inference_path = packing_dir / "inference"

    if not inference_path.is_dir():
        continue

    print(f"Processing {packing_dir.name}")

    evaluator = SegmentationEvaluator(
        packing_dir,
        input_type=cfg.EVALUATION_INPUT_TYPE,
        tolerance=15,
    )

    packing_names.append(packing_dir.name)
    results.append(evaluator.evaluate())

np.savez(
    output_path,
    packing=np.array(packing_names),
    results=np.array(results, dtype=object),
)