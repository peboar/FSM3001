import config_data as cfg_data

# Only two materials for now. Phases are used to apply texture from the real scans
PHASES = [
    cfg_data.GRANITE_NAME,
    cfg_data.BRICK_NAME,
    "void",
]

AGGREGATE_TYPE = cfg_data.AGGREGATE_TYPE

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

RANDOM_SEED = 1

IMAGE_SIZE = cfg_data.IMAGE_SIZE
IMAGE_EXTENSION = cfg_data.IMAGE_EXTENSION

IN_CHANNELS = 1
NUM_CLASSES = 3

BATCH_SIZE = 8
LEARNING_RATE = 1e-3
MOMENTUM = 0.99
NUM_EPOCHS = 50

USE_CLASS_WEIGHTS = True

NOISE_TYPE = "none"