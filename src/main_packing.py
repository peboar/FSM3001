from pathlib import Path
import sys

script_path = Path(__file__).resolve()

sys.path.append(str(script_path.parent))

import config as cfg
from blender.containers import CylindricalContainer
from blender.materials import Material
from blender.packing import Packing

def generate_packing():
    material_granite = Material(
        name=cfg.GRANITE_NAME,
        density=cfg.GRANITE_DENSITY,
        short_ratio=cfg.GRANITE_SHORT_RATIO,
        long_ratio=cfg.GRANITE_LONG_RATIO,
    )

    material_brick = Material(
        name=cfg.BRICK_NAME,
        density=cfg.BRICK_DENSITY,
        short_ratio=cfg.BRICK_SHORT_RATIO,
        long_ratio=cfg.BRICK_LONG_RATIO,
    )

    materials = [material_granite, material_brick]

    container = CylindricalContainer(
        radius=cfg.RADIUS,
        height=cfg.HEIGHT,
        friction=cfg.FRICTION_CONTAINER,
        damping=cfg.DAMPING_CONTAINER
    )

    packing = Packing(
        container=container,
        aggregate_type=cfg.AGGREGATE_TYPE,
        min_dimension=cfg.MIN_DIMENSION,
        max_dimension=cfg.MAX_DIMENSION,
        materials=materials,
        proportions=cfg.PROPORTIONS,
        target_aggregate_volume=cfg.TARGET_AGGREGATE_VOLUME,
    )

    packing.run_simulation(total_frames=cfg.TOTAL_FRAMES)
    packing.save_packing(save_blend=cfg.SAVE_BLEND)

if __name__ == "__main__":
    generate_packing()
