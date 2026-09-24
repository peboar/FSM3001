from pathlib import Path
import sys

import config_data as cfg
from ct.ct_generator import CtDataGenerator

project_path = Path(__file__).resolve().parent
data_path = project_path / "data" / (cfg.AGGREGATE_TYPE + "s")

if __name__ == "__main__":
    i = 0
    for path in data_path.iterdir():
        packing_paths  = list(path.glob("*.npz"))

        if len(packing_paths) < 1:
            raise Exception("Number of packings must be greater than 1")

        if len(packing_paths) > 1:
            print("Warning the number of packings exceeds 1")
            print("Fallback to the first one")

        packing_path = packing_paths[0]
        if i == 0:
            ct_generator = CtDataGenerator(
                packing_path=packing_path,
                image_size=cfg.IMAGE_SIZE,
                image_extension=cfg.IMAGE_EXTENSION,
                dpi=cfg.DPI,
                clear_slices=cfg.CLEAR_SLICES
            )

            ct_generator.generate_ct_data(cfg.NUMBER_OF_SLICES)
        i+=1
