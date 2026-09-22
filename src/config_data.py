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

    BRICK_NAME = "brick"
    BRICK_DENSITY = 2070
    BRICK_SHORT_RATIO = (0.5, 0.9)
    BRICK_LONG_RATIO = (1.2, 1.8)

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
