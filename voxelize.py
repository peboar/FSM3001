import numpy as np
import trimesh
import matplotlib.pyplot as plt

data = np.load("sphere_mesh.npz")

vertices = data["vertices"]
faces = data["faces"]

mesh = trimesh.Trimesh(
    vertices=vertices,
    faces=faces,
    process=True
)

voxels = mesh.voxelized(pitch=0.1).fill()
voxel_matrix = voxels.matrix


fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.axis('equal')

ax.voxels(voxel_matrix)

fig2 = plt.figure()
slice_2d = voxel_matrix[:, :, 0]
plt.imshow(slice_2d, cmap="gray", vmin=0, vmax=1)
plt.show()