import numpy as np
import trimesh
import matplotlib.pyplot as plt
import time

radius = 8e-3
pitch = 0.5 * radius

data = np.load("blender/packing.npz")

vertices = data["vertices"]
faces = data["faces"]

mesh = trimesh.Trimesh(
    vertices=vertices,
    faces=faces,
    process=True
)

voxels = mesh.voxelized(pitch=pitch)
voxel_matrix = voxels.matrix

mid_index = voxel_matrix.shape[2] // 2

# Plot vertices
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

ax.scatter(
    vertices[:, 0],
    vertices[:, 1],
    vertices[:, 2],
    s=5
)
ax.axis('equal')


# Plot original shape
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

ax.plot_trisurf(
    vertices[:, 0],
    vertices[:, 1],
    vertices[:, 2],
    triangles=faces
)

ax.axis('equal')


plt.show()

# Plot voxel
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

ax.voxels(voxel_matrix)

nx, ny, nz = voxel_matrix.shape
ax.set_box_aspect([nx, ny, nz])

plt.show()

fig2 = plt.figure()
slice_2d = voxel_matrix[:, :, mid_index]
plt.imshow(slice_2d, cmap="gray", vmin=0, vmax=1)
plt.show()