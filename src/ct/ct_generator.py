import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import trimesh


class CtDataGenerator:
    def __init__(self, data_path, image_size, dpi=100, min_dimension=1, pitch=1):
        self.data = np.load(data_path, allow_pickle=True)

        self.image_size = image_size
        self.dpi = dpi
        self.min_dimension = min_dimension
        self.pitch = pitch
        self.line_width = 72 / self.dpi
        self.script_path = Path(__file__).resolve()

        self.aggregates = self.data["aggregates"].item()

        self.aggregate_colors = {
            aggregate_id: np.random.randint(150, 210)
            for aggregate_id in self.aggregates
        }

    @property
    def z_min(self):
        return min(
            aggregate["vertices"][:, 2].min()
            for aggregate in self.aggregates.values()
        )

    @property
    def z_max(self):
        return max(
            aggregate["vertices"][:, 2].max()
            for aggregate in self.aggregates.values()
        )

    def _get_sections(self, z):
        sections = {}

        global_matrix = trimesh.geometry.plane_transform(
            origin=[0, 0, z],
            normal=[0, 0, 1]
        )

        for aggregate_id, aggregate in self.aggregates.items():
            mesh = trimesh.Trimesh(
                vertices=aggregate["vertices"],
                faces=aggregate["faces"],
                process=True
            )

            section = mesh.section(
                plane_normal=[0, 0, 1],
                plane_origin=[0, 0, z]
            )

            if section is None:
                continue

            section_2d, _ = section.to_2D(
                to_2D=global_matrix,
                check=False
            )

            sections[aggregate_id] = section_2d

        return sections

    def _create_canvas(self, bounds):
        figure_size = self.image_size / self.dpi

        fig = plt.figure(
            figsize=(figure_size, figure_size),
            dpi=self.dpi,
            facecolor="black"
        )

        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor("black")

        (x_min, y_min), (x_max, y_max) = bounds

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.axis("off")

        return fig, ax

    def plot_packing(self):
        fig, ax = plt.subplots(
            dpi=self.dpi,
            facecolor="black",
            subplot_kw={"projection": "3d"}
        )

        for aggregate_id, aggregate in self.aggregates.items():
            vertices = aggregate["vertices"]
            faces = aggregate["faces"]

            color = self.aggregate_colors[aggregate_id]

            ax.plot_trisurf(
                vertices[:, 0],
                vertices[:, 1],
                vertices[:, 2],
                triangles=faces,
                color=(color / 255, color / 255, color / 255)
            )

        ax.set_aspect("equal")
        plt.show()

    def _generate_slice(
        self,
        z,
        filename="slice.png",
        bounds=((-55, -55), (55, 55))
    ):
        sections = self._get_sections(z)

        if not sections:
            return

        fig, ax = self._create_canvas(bounds)

        for aggregate_id, section_2d in sections.items():
            color = self.aggregate_colors[aggregate_id]

            for polygon in section_2d.polygons_full:
                x, y = polygon.exterior.xy

                ax.fill(
                    x,
                    y,
                    color=(color / 255, color / 255, color / 255),
                    antialiased=False
                )

        fig.canvas.draw()

        image = np.asarray(fig.canvas.buffer_rgba())
        image = image[:, :, :3].mean(axis=2).astype(np.uint8)

        Image.fromarray(image, mode="L").save(filename)

        plt.close(fig)

    def _generate_mask(
        self,
        z,
        filename="mask.png",
        bounds=((-55, -55), (55, 55))
    ):
        sections = self._get_sections(z)

        if not sections:
            return

        fig, ax = self._create_canvas(bounds)

        for section_2d in sections.values():
            for polygon in section_2d.polygons_full:
                x, y = polygon.exterior.xy

                ax.fill(
                    x,
                    y,
                    color="white",
                    antialiased=False
                )

        fig.canvas.draw()

        image = np.asarray(fig.canvas.buffer_rgba())
        image = image[:, :, :3].mean(axis=2).astype(np.uint8)

        Image.fromarray(image, mode="L").save(filename)

        plt.close(fig)

path = r"/home/per/Desktop/Kth/Phd/Courses/FSM3001/Project/src/data/polyhedrons/packing_500_20260918_112953/packing_500_20260918_112953.npz"
ct_generator = CtDataGenerator(path, 512)

ct_generator.plot_packing()
ct_generator._generate_slice(11, "slice1.png")
ct_generator._generate_slice(12, "slice2.png")
ct_generator._generate_slice(13, "slice3.png")

ct_generator._generate_mask(11, "mask1.png")
ct_generator._generate_mask(12, "mask2.png")
ct_generator._generate_mask(13, "mask3.png")
