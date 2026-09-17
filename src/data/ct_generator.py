import numpy as np
import trimesh
import matplotlib.pyplot as plt


class CtDataGenerator:
    def __init__(self, data_path, image_size, dpi=100, min_dimension=1, pitch=1):
        self.data = np.load(data_path)
        self.image_size = image_size
        self.dpi = dpi
        self.min_dimension = 1
        self.pitch = 1

        self.vertices = self.data["vertices"]
        self.faces = self.data["faces"]
        self.aggregate_ids = self.data["aggregate_ids"]
        self.mesh = trimesh.Trimesh(
            vertices=self.vertices,
            faces=self.faces,
            process=True
        )

    def plot_packing(self):
        fig, ax = plt.subplots(
           dpi=self.dpi,
           facecolor="black",
           subplot_kw={"projection": "3d"}
        )
        ax.plot_trisurf(
            self.vertices[:, 0],
            self.vertices[:, 1],
            self.vertices[:, 2],
            triangles=self.faces,
            color="gray"
        )
        ax.axis('equal')
        plt.show()

    def _generate_mask(self, z):
        section = self.mesh.section(
        plane_normal=[0, 0, 1],
        plane_origin=[0, 0, z]
        )
        if section is None:
            print(f"No valid section is found for z={z}")
            return

        section_2d, _ = section.to_planar()
        figure_size = self.image_size / self.dpi
        fig, ax = plt.subplots(figsize=(figure_size, figure_size),
                               dpi=self.dpi,
                               facecolor="black")

        for polygon in section_2d.polygons_full:
            x, y = polygon.exterior.xy
            ax.fill(x, y, color="gray", antialiased=False)
            ax.plot(x, y, color="white", antialiased=False)

        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        fig.tight_layout(pad=0)
        plt.show()

path = r"/home/per/Desktop/Kth/Phd/Courses/FSM3001/Project/src/data/spheres/packing_test/packing_test.npz"
ct_generator = CtDataGenerator(path, 512)
ct_generator.plot_packing()
dz = 1e-3
for i in range(1, 101):
    ct_generator._generate_mask(i*dz)
