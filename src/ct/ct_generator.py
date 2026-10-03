from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from noise import pnoise3
import numpy as np
import shapely
import trimesh

from ct.ct_texture import ExtractCtTexture


class CtDataGenerator:
    def __init__(
        self,
        packing_path,
        image_size=512,
        image_extension="tif",
        dpi=100,
        clear_slices=True,
        inclusions = None,
    ):
        self.script_path = Path(__file__).resolve().parent
        self.packing_path = Path(packing_path)
        self.output_directory = self.packing_path.parent
        self.data = np.load(self.packing_path, allow_pickle=True)
        self.aggregates = self.data["aggregates"].item()

        self.image_width = image_size
        self.image_height = image_size
        self.image_extension = image_extension
        self.dpi = dpi
        self.clear_slices = clear_slices

        self.materials = {
            aggregate.get("material")
            for aggregate in self.aggregates.values()
            if aggregate.get("material") is not None
        }

        self.phases = list(self.materials) + ["void"]

        self.textures = {}
        for phase in self.phases:
            texture_path = self.script_path / phase
            texture = ExtractCtTexture(
                folder_path=texture_path,
                image_size=image_size,
                image_extension=self.image_extension,
            )
            self.textures[phase] = texture

        self.phase_colors = {
            phase: texture.extract_gray_mean()
            for phase, texture in self.textures.items()
        }

        self.aggregate_colors = {
            aggregate_id: self.phase_colors.get(
                aggregate.get("material"),
                128,
            )
            for aggregate_id, aggregate in self.aggregates.items()
        }
        inclusions = inclusions or {}

        self.inclusions = {
            aggregate_id: inclusions.get(aggregate.get("material"))
            for aggregate_id, aggregate in self.aggregates.items()
        }

        # Random seed for perlie noise
        seed = int(self.output_directory.stem.split('_')[-1])
        self.rng = np.random.default_rng(seed)


        self.aggregate_angles = {
            aggregate_id: self.rng.uniform(0.0, 360.0)
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

        for aggregate_id, aggregate in self.aggregates.items():
            mesh = trimesh.Trimesh(
                vertices=aggregate["vertices"],
                faces=aggregate["faces"],
                process=True,
            )

            section = mesh.section(
                plane_normal=[0, 0, 1],
                plane_origin=[0, 0, z],
            )

            if section is None:
                continue

            global_matrix = trimesh.geometry.plane_transform(
                origin=[0, 0, z],
                normal=[0, 0, 1],
            )

            section_2d, _ = section.to_2D(
                to_2D=global_matrix,
                check=False,
            )

            sections[aggregate_id] = section_2d

        return sections

    def _coordinates_to_pixels(self, bounds):
        (min_x, min_y), (max_x, max_y) = bounds

        range_x = max_x - min_x
        range_y = max_y - min_y

        y_indices, x_indices = np.indices(
            (self.image_height, self.image_width)
        )

        pixel_x = (
            (x_indices / (self.image_width - 1))
            * range_x
            + min_x
        )

        pixel_y = (
            (self.image_height - 1 - y_indices)
            / (self.image_height - 1)
            * range_y
            + min_y
        )

        return pixel_x, pixel_y

    def _generate_inclusions(self,
                             poly_mask,
                             pixel_x,
                             pixel_y,
                             z,
                             angle=0,
                             threshold=0.05,
                             scale_x=0.6,
                             scale_y=0.02,
                             scale_z=0.05,
                             octaves=2):
        """Boolean vein mask, evaluated only inside poly_mask, in global coordinates."""
        angle_radian = np.deg2rad(angle)

        dx = pixel_x[poly_mask]
        dy = pixel_y[poly_mask]
        rx = (np.cos(angle_radian) * dx - np.sin(angle_radian) * dy) * scale_x
        ry = (np.sin(angle_radian) * dx + np.cos(angle_radian) * dy) * scale_y
        rz = z * scale_z

        noise = np.array([
            pnoise3(px, py, rz, octaves=octaves)
            for px, py in zip(rx.tolist(), ry.tolist())
        ])

        veins = np.zeros_like(poly_mask, dtype=bool)
        veins[poly_mask] = noise > threshold
        return veins

    def plot_packing(self):
        fig, ax = plt.subplots(
            dpi=self.dpi,
            facecolor="black",
            subplot_kw={"projection": "3d"},
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
                color=(color / 255, color / 255, color / 255),
            )

        ax.set_aspect("equal")
        plt.show()

    def _generate_slice(
        self,
        z,
        bounds,
        edge_factor=1,
    ):
        sections = self._get_sections(z)

        if not sections:
            print(f"    No sections found at z: {z}")
            return


        canvas = (
            np.zeros(
                (self.image_height, self.image_width),
                dtype=np.uint8,
            )
        )

        # Add container void phase
        void_color = self.phase_colors["void"]

        radius = min(self.image_width, self.image_height) // 2
        center_width = self.image_width // 2
        center_height = self.image_height // 2
        y, x = np.ogrid[:self.image_height, :self.image_width]
        circle_mask = (x - center_width) ** 2 + (y - center_height) ** 2 <= radius ** 2

        canvas[circle_mask] = void_color

        kernel = np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype=np.uint8,
        )

        pixel_x, pixel_y = self._coordinates_to_pixels(bounds)

        for aggregate_id, section_2d in sections.items():
            inclusion_params = dict(self.inclusions[aggregate_id] or {})
            inclusion_color = inclusion_params.pop("color", 255)
            inclusion_params["angle"] = self.aggregate_angles[aggregate_id]

            for polygon in section_2d.polygons_full:
                if polygon.is_empty:
                    continue

                poly_mask = shapely.contains_xy(
                    polygon,
                    pixel_x,
                    pixel_y,
                )

                if not np.any(poly_mask):
                    continue

                color = self.aggregate_colors[aggregate_id]
                edge_color = int(color * edge_factor)

                poly_mask_uint8 = (
                    poly_mask.astype(np.uint8) * 255
                )

                eroded_mask = cv2.erode(
                    poly_mask_uint8,
                    kernel,
                    iterations=1,
                )

                edge_mask = cv2.subtract(
                    poly_mask_uint8,
                    eroded_mask,
                )

                canvas[poly_mask_uint8 == 255] = color

                if self.inclusions[aggregate_id]:
                    inclusion = self._generate_inclusions(
                        poly_mask,
                        pixel_x,
                        pixel_y,
                        z,
                        **inclusion_params,
                    )
                    canvas[inclusion] = inclusion_color

                canvas[edge_mask == 255] = edge_color

        return canvas, len(sections)

    def _generate_mask(
        self,
        z,
        bounds,
    ):
        sections = self._get_sections(z)

        if not sections:
            print(f"    No sections found at z: {z}")
            return

        canvas = np.zeros(
            (self.image_height, self.image_width),
            dtype=np.uint8,
        )

        kernel = np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype=np.uint8,
        )

        pixel_x, pixel_y = self._coordinates_to_pixels(bounds)

        for section_2d in sections.values():
            for polygon in section_2d.polygons_full:
                if polygon.is_empty:
                    continue

                poly_mask = shapely.contains_xy(
                    polygon,
                    pixel_x,
                    pixel_y,
                )

                if not np.any(poly_mask):
                    continue

                poly_mask_uint8 = (
                    poly_mask.astype(np.uint8) * 255
                )

                eroded_mask = cv2.erode(
                    poly_mask_uint8,
                    kernel,
                    iterations=1,
                )

                edge_mask = cv2.subtract(
                    poly_mask_uint8,
                    eroded_mask,
                )

                canvas[poly_mask_uint8 == 255] = 128
                canvas[edge_mask == 255] = 255

        return canvas, len(sections)

    def generate_ct_data(
        self,
        number_of_slices,
        bounds,
        edge_color=1,
        z_min=None,
        z_max=None,
    ):
        if z_min is None:
            z_min = self.z_min

        if z_max is None:
            z_max = self.z_max

        slicing_dir = self.output_directory / "slices"
        mask_dir = self.output_directory / "masks"

        slicing_dir.mkdir(exist_ok=True)
        mask_dir.mkdir(exist_ok=True)

        if self.clear_slices:
            [
                f.unlink()
                for f in slicing_dir.glob("*")
                if f.is_file()
            ]
            [
                f.unlink()
                for f in mask_dir.glob("*")
                if f.is_file()
            ]

        z_values = np.linspace(
            z_min,
            z_max,
            number_of_slices + 2,
        )[1:-1]

        for z in z_values:
            (
                canvas_slice,
                num_aggregates_slice,
            ) = self._generate_slice(
                z,
                bounds,
                edge_color,
            )

            (
                canvas_mask,
                num_aggregates_mask,
            ) = self._generate_mask(
                z,
                bounds,
            )

            z_string = f"{z:06.2f}".replace(".", "_")

            slicing_filename = (
                slicing_dir
                / (
                    f"slice_z_{z_string}_aggregates_"
                    f"{num_aggregates_slice}.{self.image_extension}"
                )
            )

            mask_filename = (
                mask_dir
                / (
                    f"mask_z_{z_string}_aggregates_"
                    f"{num_aggregates_mask}.{self.image_extension}"
                )
            )

            cv2.imwrite(
                str(slicing_filename),
                canvas_slice,
            )
            cv2.imwrite(
                str(mask_filename),
                canvas_mask,
            )

            print(f"    Generated slices at z={z:.2f}")

        print(f"    Generated {number_of_slices} slices in:")
        print(f"    {slicing_dir}")
        print(f"    {mask_dir}")

    def animate_slicing(
        self,
        num_frames,
        bounds,
        z_min=None,
        z_max=None,
        filename="slicing.gif",
        duration=300,
    ):
        if z_min is None:
            z_min = self.z_min

        if z_max is None:
            z_max = self.z_max

        z_values = np.linspace(
            z_min,
            z_max,
            num_frames,
        )

        figure = plt.figure(
            figsize=(16, 9),
            facecolor="black",
        )

        ax_3d = figure.add_subplot(
            121,
            projection="3d",
            facecolor="black",
        )

        ax_2d = figure.add_subplot(
            122,
            facecolor="black",
        )

        figure.tight_layout(
            pad=0.1,
            w_pad=0.0,
        )

        ax_3d.computed_zorder = False

        for aggregate_id, aggregate in self.aggregates.items():
            vertices = aggregate["vertices"]
            faces = aggregate["faces"]
            color = self.aggregate_colors[aggregate_id] / 255

            ax_3d.plot_trisurf(
                vertices[:, 0],
                vertices[:, 1],
                vertices[:, 2],
                triangles=faces,
                color=(color, color, color),
            )

        (min_x, min_y), (max_x, max_y) = bounds

        ax_3d.set_xlim(min_x, max_x)
        ax_3d.set_ylim(min_y, max_y)
        ax_3d.set_zlim(z_min, z_max)
        ax_3d.set_aspect("equal")

        ax_3d.view_init(
            elev=0,
            azim=-90,
        )

        ax_3d.set_facecolor("black")
        ax_3d.set_axis_off()

        ax_2d.set_xlim(min_x, max_x)
        ax_2d.set_ylim(min_y, max_y)
        ax_2d.set_aspect("equal")
        ax_2d.axis("off")

        pixel_x, pixel_y = self._coordinates_to_pixels(bounds)

        line_x = [min_x, max_x]
        line_y = [max_y, max_y]

        slicing_line, = ax_3d.plot(
            line_x,
            line_y,
            [z_values[0], z_values[0]],
            color="lightgreen",
            linewidth=2,
            alpha=0.65,
            zorder=100,
        )

        image = ax_2d.imshow(
            np.zeros(
                (self.image_height, self.image_width),
                dtype=np.uint8,
            ),
            cmap="gray",
            vmin=0,
            vmax=255,
            extent=[
                min_x,
                max_x,
                min_y,
                max_y,
            ],
            origin="upper",
        )

        def update(frame):
            z = z_values[frame]

            sections = self._get_sections(z)

            canvas = np.zeros(
                (self.image_height, self.image_width),
                dtype=np.uint8,
            )

            for aggregate_id, section_2d in sections.items():
                color = self.aggregate_colors[aggregate_id]

                for polygon in section_2d.polygons_full:
                    if polygon.is_empty:
                        continue

                    poly_mask = shapely.contains_xy(
                        polygon,
                        pixel_x,
                        pixel_y,
                    )

                    if not np.any(poly_mask):
                        continue

                    canvas[poly_mask] = color

            image.set_data(canvas)

            slicing_line.set_data(
                line_x,
                line_y,
            )

            slicing_line.set_3d_properties(
                [z, z],
            )

            return image, slicing_line

        animation = FuncAnimation(
            figure,
            update,
            frames=num_frames,
            interval=duration,
            blit=False,
        )

        animation.save(
            filename,
            writer=PillowWriter(
                fps=1000 / duration,
            ),
        )

        plt.close(figure)
