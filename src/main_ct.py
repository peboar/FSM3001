from pathlib import Path
import sys

import config_data as cfg
from ct.ct_generator import CtDataGenerator

project_path = script_path.parent
data_path = project_path / "data" / (cfg.AGGREGATE_TYPE + "s")

if __name__ == "__main__":
    for path in data_path.iterdir()[0]:
        ct_generator = CtDataGenerator(
            packing_path=path,
            image_size=cfg.IMAGE_SIZE,
            image_extension=cfg.IMAGE_EXTENSION,
            dpi=cfg.DPI,
            clear_slices=cfg.CLEAR_SLICES
        )

        print(ct_generator)
