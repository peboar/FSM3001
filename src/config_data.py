# Number of packings
NUMBER_OF_PACKINGS = 2

# Container dimensions
RADIUS = 50
HEIGHT = 305
FRICTION_CONTAINER = 0.2
DAMPING_CONTAINER = 0.1

# Aggregate materials
GRANITE_NAME = "granite"
GRANITE_DENSITY = 2650
GRANITE_SHORT_RATIO = (0.7, 1.0)
GRANITE_LONG_RATIO = (1.0, 1.5)
GRANITE_INCLUSION = {
    "contrast": 1.25,
    "threshold": 0.05,   # raise to 0.07-0.08 for thicker lines
    "scale_x": 0.6,      # raise for more lines, lower for fewer
    "scale_y": 0.02,     # lower for longer streaks (try 0.015)
    "scale_z": 0.05,
    "octaves": 2,
}

BRICK_NAME = "brick"
BRICK_DENSITY = 2070
BRICK_SHORT_RATIO = (0.5, 0.9)
BRICK_LONG_RATIO = (1.2, 1.8)
BRICK_INCLUSION = None  # no inclusions

MATERIAL_INCLUSIONS = {
    GRANITE_NAME: GRANITE_INCLUSION,
    BRICK_NAME: BRICK_INCLUSION,
}

# Packing parameters
AGGREGATE_TYPE = "polyhedron"
MIN_DIMENSION = 11.2
MAX_DIMENSION = 16.0
PROPORTIONS = [0.5, 0.5]
TARGET_AGGREGATE_VOLUME = 566400

# Simulation frames
TOTAL_FRAMES = 1000

# Save blender output file
SAVE_BLEND = False

# CT-image generator parameters
IMAGE_SIZE = 512
IMAGE_EXTENSION = "tif" # No dot
DPI = 100
CLEAR_SLICES = True
NUMBER_OF_SLICES = 32
EDGE_FACTOR = 1.0



