import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def generate_granite_aggregate(sieve_min, sieve_max, num_points=None, seed=None):
    """
    Generates a single sharp, cube-based granite aggregate calibrated to pass
    through a square sieve size between sieve_min and sieve_max.
    """
    if seed is not None:
        np.random.seed(seed)

    sieve_target_size = np.random.uniform(sieve_min, sieve_max)

    length = 1.6
    depth = 1.0
    height = 1.0

    bounding_x = (-length / 2, length / 2)
    bounding_y = (-depth / 2, depth / 2)
    bounding_z = (-height / 2, height / 2)

    # Generate raw random points within the cubic bounding boxes
    x_pts = np.random.uniform(bounding_x[0], bounding_x[1], num_points)
    y_pts = np.random.uniform(bounding_y[0], bounding_y[1], num_points)
    z_pts = np.random.uniform(bounding_z[0], bounding_z[1], num_points)


    convex_hull_cloud = np.column_stack((x_pts, y_pts, z_pts))

    # Build the initial convex hull to measure current width
    initial_hull = ConvexHull(convex_hull_cloud)
    vertices = convex_hull_cloud[initial_hull.vertices]

    # Measure the current width along the Y-axis
    current_width = np.max(vertices[:, 1]) - np.min(vertices[:, 1])

    # Strict local scaling to match the sieve target size
    scale_factor = sieve_target_size / current_width

    # Center the particle around (0,0,0) and scale everything linearly
    center = np.mean(vertices, axis=0)
    final_points = (convex_hull_cloud - center) * scale_factor

    # Build the final perfectly scaled hull
    final_hull = ConvexHull(final_points)
    final_vertices = final_points[final_hull.vertices]

    return final_points, final_hull, final_vertices, sieve_target_size



# Call the function with local parameters
points, hull, vertices, target_size = generate_granite_aggregate(
    sieve_min=11.8,
    sieve_max=16.0,
    num_points=25,
    seed=None
)

# Measure final dimensions for verification
final_x_size = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
final_y_size = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
final_z_size = np.max(vertices[:, 2]) - np.min(vertices[:, 2])
true_3d_diagonal = np.max(np.linalg.norm(vertices[:, None, :] - vertices[None, :, :], axis=-1))


# Setup 3D Plot
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection='3d')

# Gather triangular faces
faces = [points[simplex] for simplex in hull.simplices]
poly3d = Poly3DCollection(faces, alpha=0.8, facecolor='darkgray', edgecolor='black', linewidths=0.8)
ax.add_collection3d(poly3d)

# Draw corner points
ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='red', s=20)

plt.show()
