from pathlib import Path
import sys

import config_data as cfg
from ct.ct_generator import CtDataGenerator


project_path = Path(__file__).resolve().parent
data_path = project_path / "data" / (cfg.AGGREGATE_TYPE + "s")

slice_bounds = ((-1.1*cfg.RADIUS, -1.1*cfg.RADIUS), (1.1*cfg.RADIUS, 1.1*cfg.RADIUS))

if __name__ == "__main__":
    packing_dirs = sorted(data_path.iterdir())
    number_of_packings = len(packing_dirs)

    for index, packing_dir in enumerate(packing_dirs, start=1):
        packing_paths = list(packing_dir.glob("*.npz"))

        if not packing_paths:
            raise RuntimeError(
                f"No packing file found in {packing_dir}"
            )

        if len(packing_paths) > 1:
            print(
                f"Warning: {len(packing_paths)} packing files found in "
                f"{packing_dir.name}. Using the first one."
            )

        packing_path = packing_paths[0]

        print(
            f"Generating CT slices for {packing_path.stem} "
            f"({index}/{number_of_packings})"
        )

        ct_generator = CtDataGenerator(
            packing_path=packing_path,
            image_size=cfg.IMAGE_SIZE,
            image_extension=cfg.IMAGE_EXTENSION,
            dpi=cfg.DPI,
            clear_slices=cfg.CLEAR_SLICES,
        )

        ct_generator.generate_ct_data(
            number_of_slices=cfg.NUMBER_OF_SLICES,
            bounds=slice_bounds,
            edge_color=cfg.EDGE_FACTOR
        )