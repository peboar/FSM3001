import numpy as np
import trimesh
import matplotlib.pyplot as plt

data_path = np.load("../data/spheres/packing_test/packing_test.npz")


class CtDataGenerator:
    def __init__(self, data_path, image_size, min_dimension=1, pitch=1):
        self.data = np.load(data_path)
        self.image_size = image_size
        self.min_dimension = 1
        self.pitch = 1






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

# Generate raw cross sections
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

# Generate cross-section masks
fig, ax = plt.subplots(facecolor="black")
ax.set_facecolor("black")
for polygon in section_2D.polygons_full:
    x, y = polygon.exterior.xy

    ax.fill(x, y, color="gray")
    ax.plot(x, y, color="white")


ax.set_aspect("equal", adjustable="box")
ax.axis("off")
fig.tight_layout(pad=0)
plt.show()