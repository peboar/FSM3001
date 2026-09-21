import random
import sys
from datetime import datetime
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


# Get the script path.
script_path = Path(__file__).resolve()

sys.path.append(str(script_path.parent))

from aggregates import PolyHedron, Sphere
from containers import CylindricalContainer
from materials import Material


class Packing:
    """Handles the generation of the aggregate packing inside the container."""

    def __init__(
        self,
        container,
        aggregate_type,
        min_dimension,
        max_dimension,
        materials,
        proportions,
        number_of_aggregates=None,
        target_aggregate_volume=None,
    ):
        """
        Initialize the aggregate generation parameters.

        Parameters
        ----------
        container : Container
            Container in which the aggregates are generated.
        aggregate_type : str
            Type of aggregate to generate.
        min_dimension : float
            Minimum dimension (sieve size) of the aggregate in [mm].
        max_dimension : float
            Maximum dimension (sieve size) of the aggregate in [mm].
        materials : list of Material
            Aggregate materials, each with density and grain aspect ratios.
        proportions : list of float
            Target share of each material by volume (normalised to sum to 1).
        number_of_aggregates : int, optional
            Number of aggregates to generate.
        target_aggregate_volume : float, optional
            Total volume of the aggregates, without voids, in [mm³].

        Give exactly one of number_of_aggregates and target_aggregate_volume.
        """
        if (number_of_aggregates is None) == (target_aggregate_volume is None):
            raise ValueError(
                "Give either number_of_aggregates or target_aggregate_volume."
            )

        if len(materials) != len(proportions):
            raise ValueError("Need one proportion per material.")

        self.container = container
        self.aggregate_type = aggregate_type
        self.number_of_aggregates = number_of_aggregates
        self.min_dimension = min_dimension
        self.max_dimension = max_dimension
        self.materials = materials
        self.proportions = np.array(proportions) / np.sum(proportions)
        self.target_aggregate_volume = target_aggregate_volume
        self.aggregates = []
        self.aggregate_materials = []  # Material of each aggregate

        self._configure_rigidbody_world()
        self._generate_aggregates()

    def _pick_material(self, material_volumes):
        """Return index of the material furthest below its target volume fraction."""
        total_volume = sum(material_volumes)

        # No grains yet: start with the first material.
        if total_volume == 0:
            return 0

        fractions = np.array(material_volumes) / total_volume
        surpluses = fractions - self.proportions

        return int(np.argmin(surpluses))

    def _is_finished(self, count, total_volume):
        """Check if the requested volume or number of aggregates is reached."""
        if self.target_aggregate_volume is not None:
            return total_volume >= self.target_aggregate_volume

        return count >= self.number_of_aggregates

    def _progress(self, count, total_volume):
        """Fraction of the requested aggregates generated so far, from 0 to 1."""
        if self.target_aggregate_volume is not None:
            return min(total_volume / self.target_aggregate_volume, 1.0)

        return min(count / self.number_of_aggregates, 1.0)

    def _generate_aggregates(self):
        """
        Generate aggregate objects.

        The generated aggregates are stored in self.aggregates.
        """
        if self.container.get_shape() == "CYLINDER":
            cylinder_radius, _ = self.container.get_cylinder_dimensions()

            # Start generating aggregates from the bottom of the container.
            prev_bounding_radius = 0
            total_volume = 0
            material_volumes = [0.0] * len(self.materials)

            while not self._is_finished(len(self.aggregates), total_volume):
                # Select the material furthest below its target volume fraction.
                material_index = self._pick_material(material_volumes)
                material = self.materials[material_index]

                dimension = random.uniform(self.min_dimension, self.max_dimension)

                # Create the selected aggregate type.
                if self.aggregate_type == "sphere":
                    aggregate = Sphere(dimension, 3, material.density)
                elif self.aggregate_type == "polyhedron":
                    # Dimensions for the convex hull using predefined aspect ratios
                    length, depth, height = material.sample_dimensions(dimension)

                    # Random point cloud for convex hull 15 to 35 looks decent.
                    number_of_points = random.randint(25, 50)

                    aggregate = PolyHedron(
                        length,
                        depth,
                        height,
                        number_of_points,
                        material.density,
                    )

                volume = aggregate.get_volume()
                total_volume += volume
                material_volumes[material_index] += volume

                bounding_radius = aggregate.get_bounding_radius()
                allowed_radius = cylinder_radius - bounding_radius
                std_dev = allowed_radius / 3.0

                x = random.gauss(0, std_dev)
                y = random.gauss(0, std_dev)

                # Checks that the entire aggregate remains inside the cylinder.
                while x**2 + y**2 >= allowed_radius**2:
                    x = random.gauss(0, std_dev)
                    y = random.gauss(0, std_dev)

                if not self.aggregates:
                    z = 1.01 * bounding_radius
                else:
                    z += 1.01 * (bounding_radius + prev_bounding_radius)

                prev_bounding_radius = bounding_radius
                aggregate.location((x, y, z))
                self.aggregates.append(aggregate)
                self.aggregate_materials.append(material)

                progress = self._progress(len(self.aggregates), total_volume)
                print(
                    f"Generated aggregate {len(self.aggregates)}, "
                    f"{100 * progress:.0f}% of target, "
                    f"aggregate volume {total_volume} mm³"
                )

    def _configure_rigidbody_world(self):
        if not bpy.context.scene.rigidbody_world:
            bpy.ops.rigidbody.world_add()

        rb_world = bpy.context.scene.rigidbody_world

        # Convex hull contact between the mesh container fails for small dimensions.
        # Must use millimeters.
        bpy.context.scene.unit_settings.system = "METRIC"
        bpy.context.scene.unit_settings.length_unit = "MILLIMETERS"
        bpy.context.scene.gravity = (0, 0, -9810)

        # Blender dimension for correct units. Dimensions and gravity is scaled down.
        bpy.context.scene.unit_settings.scale_length = 0.001

        rb_world.time_scale = 0.1
        rb_world.substeps_per_frame = 10
        rb_world.solver_iterations = 10

    def run_simulation(self, total_frames=1000):
        rb_world = bpy.context.scene.rigidbody_world
        rb_world.point_cache.frame_end = total_frames

        for frame in range(0, total_frames + 1):
            bpy.context.scene.frame_set(frame)
            bpy.context.view_layer.update()

            output_frequency = total_frames // 100

            if frame > 0 and frame % output_frequency == 0:
                print(
                    f"Simulation completion "
                    f"{100 * frame / total_frames:.0f}%"
                )

    def save_packing(self, save_blend=False):
        project_dir = script_path.parent.parent
        output_dir = project_dir / "data" / f"{self.aggregate_type}s"
        output_dir.mkdir(parents=True, exist_ok=True)

        aggregates = {}
        valid_aggregates = 0

        for aggregate_id, aggregate in enumerate(self.aggregates):
            obj = aggregate.obj

            transform = obj.matrix_world
            obj_vertices = obj.data.vertices
            centroid = Vector((0, 0, 0))

            vertices = []

            for vertex in obj_vertices:
                transformed_vertex = transform @ vertex.co
                vertices.append(transformed_vertex)
                centroid += transformed_vertex

            centroid /= len(obj_vertices)

            x, y, z = centroid

            # Check if an aggregate is outside the container.
            if self.container.get_shape() == "CYLINDER":
                if z <= 0 or x**2 + y**2 >= self.container.radius**2:
                    continue

            vertices = np.array(vertices)

            faces = np.array([
                polygon.vertices[:]
                for polygon in obj.data.polygons
            ])

            material = self.aggregate_materials[aggregate_id]

            aggregates[aggregate_id] = {
                "vertices": vertices,
                "faces": faces,
                "material": material.name,
            }

            valid_aggregates += 1

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"packing_{valid_aggregates}_{timestamp}"

        packing_dir = output_dir / filename
        packing_dir.mkdir(parents=True, exist_ok=True)

        np.savez(
            packing_dir / f"{filename}.npz",
            aggregates=np.array(aggregates, dtype=object)
        )

        if save_blend:
            bpy.ops.wm.save_as_mainfile(
                filepath=str(packing_dir / f"{filename}.blend")
            )


if __name__ == "__main__":
    container = CylindricalContainer(50, 305)

    granite = Material("granite", 2650, short_ratio=(0.9, 1.0), long_ratio=(1.0, 1.1))
    brick = Material("brick", 2070, short_ratio=(0.5, 0.9), long_ratio=(1.2, 1.8))

    packing = Packing(
        container,
        "polyhedron",
        11.2,
        16,
        [granite, brick],
        [0.5, 0.5],
        target_aggregate_volume=566400,  # [mm³], Blend CT
    )

    packing.run_simulation()
    packing.save_packing()