import numpy as np
import trimesh
import matplotlib.pyplot as plt
import time

radius = 8e-3
pitch = 0.5 * radius

data = np.load("packing.npz")

vertices = data["vertices"]
faces = data["faces"]

mesh = trimesh.Trimesh(
    vertices=vertices,
    faces=faces,
    process=True
)

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
z_slice = radius

section = mesh.section(
    plane_origin=[0, 0, z_slice],
    plane_normal=[0, 0, 1]
)

section_2D, _ = section.to_planar()

fig, ax = plt.subplots(facecolor="black")
ax.set_facecolor("black")

for polygon in section_2D.polygons_full:
    x, y = polygon.exterior.xy

    ax.fill(x, y, color="gray")
    ax.plot(x, y, color="gray")

ax.set_aspect("equal", adjustable="box")
ax.axis("off")
fig.tight_layout(pad=0)
plt.show()