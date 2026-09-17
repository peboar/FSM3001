import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import trimesh



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

    @property
    def z_min(self):
        return self.vertices[:, 2].min()

    @property
    def z_max(self):
        return self.vertices[:, 2].max()

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
        ax.axis("equal")
        plt.show()

    def _generate_mask(self, z, filename="test.png"):
        section = self.mesh.section(
            plane_normal=[0, 0, 1],
            plane_origin=[0, 0, z]
        )

        if section is None:
            print(f"No valid section is found for z={z}")
            return

        section_2d, _ = section.to_planar()

        figure_size = self.image_size / self.dpi

        fig = plt.figure(
            figsize=(figure_size, figure_size),
            dpi=self.dpi,
            facecolor="black"
        )

        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor("black")
        # NB!!!!!!!!! FIX BOUNDS USE CONFIGFILE to set the bounds based on the container size
        (x_min, y_min), (x_max, y_max) = section_2d.bounds
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.axis("off")

        for polygon in section_2d.polygons_full:
            x, y = polygon.exterior.xy

            ax.fill(x, y, color="gray", antialiased=False)
            ax.plot(x, y, color="white", antialiased=False, linewidth=0.2)

        fig.canvas.draw()

        image = np.asarray(fig.canvas.buffer_rgba())

        # Convert RGB to grayscale
        image = image[:, :, :3].mean(axis=2).astype(np.uint8)

        Image.fromarray(image, mode="L").save(filename)
        plt.close(fig)

    def animate_slicing(
        self,
        n_slices,
        filename="slicing.gif",
        duration=700,
        background="black",
        top_margin_diameters=2,
        line_extension=0.05
    ):

        z_min = self.vertices[:, 2].min()
        z_max = self.vertices[:, 2].max()

        max_aggregate_diameter = 0

        for aggregate_id in np.unique(self.aggregate_ids):

            vertex_indices = np.where(
                self.aggregate_ids == aggregate_id
            )[0]

            if len(vertex_indices) == 0:
                continue

            aggregate_vertices = self.vertices[
                vertex_indices
            ]

            dimensions = (
                aggregate_vertices.max(axis=0)
                - aggregate_vertices.min(axis=0)
            )

            diameter = dimensions.max()

            max_aggregate_diameter = max(
                max_aggregate_diameter,
                diameter
            )

        top_margin = (
            top_margin_diameters
            * max_aggregate_diameter
        )

        z_top = z_max - top_margin

        z_values = np.linspace(
            z_min,
            z_top,
            n_slices + 2
        )[1:-1]

        x_min = self.vertices[:, 0].min()
        x_max = self.vertices[:, 0].max()

        y_min = self.vertices[:, 1].min()
        y_max = self.vertices[:, 1].max()

        x_margin = line_extension * (x_max - x_min)

        line_x_min = x_min - x_margin
        line_x_max = x_max + x_margin

        frames = []

        for z in z_values:

            fig = plt.figure(
                figsize=(12, 6),
                dpi=self.dpi,
                facecolor=background
            )

            ax_3d = fig.add_subplot(
                121,
                projection="3d",
                facecolor=background
            )

            ax_2d = fig.add_subplot(
                122,
                facecolor=background
            )

            ax_3d.computed_zorder = False

            ax_3d.plot_trisurf(
                self.vertices[:, 0],
                self.vertices[:, 1],
                self.vertices[:, 2],
                triangles=self.faces,
                color="gray",
                edgecolor="white",
                linewidth=0.2,
                antialiased=True
            )

            ax_3d.set_xlim(x_min, x_max)
            ax_3d.set_ylim(y_min, y_max)
            ax_3d.set_zlim(z_min, z_max)

            ax_3d.set_box_aspect([
                x_max - x_min,
                y_max - y_min,
                z_max - z_min
            ])

            ax_3d.view_init(
                elev=0,
                azim=-90
            )

            ax_3d.set_axis_off()

            ax_3d.plot(
                [line_x_min, line_x_max],
                [y_max, y_max],
                [z, z],
                color="lightgreen",
                linewidth=5,
                alpha=0.45,
                solid_capstyle="butt",
                zorder=100
            )

            section = self.mesh.section(
                plane_normal=[0, 0, 1],
                plane_origin=[0, 0, z]
            )

            if section is not None:

                section_2d, _ = section.to_planar()

                for polygon in section_2d.polygons_full:

                    x, y = polygon.exterior.xy

                    ax_2d.fill(
                        x,
                        y,
                        color="gray",
                        edgecolor="lightgray",
                        linewidth=0.7,
                        antialiased=True
                    )

            ax_2d.set_xlim(x_min, x_max)
            ax_2d.set_ylim(y_min, y_max)
            ax_2d.set_aspect("equal", adjustable="box")
            ax_2d.set_axis_off()

            fig.tight_layout(pad=0)

            buffer = io.BytesIO()

            fig.savefig(
                buffer,
                format="png",
                facecolor=background,
                bbox_inches=None,
                pad_inches=0
            )

            buffer.seek(0)

            frames.append(
                Image.open(buffer).convert("RGB")
            )

            plt.close(fig)

        frames[0].save(
            filename,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0
        )

path = r"/home/per/Desktop/Kth/Phd/Courses/FSM3001/Project/src/data/polyhedrons/PRESENTATION/packing_500_20260917_101113.npz"
ct_generator = CtDataGenerator(path, 512)
ct_generator._generate_mask(100)