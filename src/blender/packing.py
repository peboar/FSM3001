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


class Packing:
    """Handles the generation of the aggregate packing inside the container."""

    def __init__(
        self,
        container,
        aggregate_type,
        number_of_aggregates,
        min_dimension,
        max_dimension,
        densities,
        proportions,
    ):
        """
        Initialize the aggregate generation parameters.

        Parameters
        ----------
        container : Container
            Container in which the aggregates are generated.
        aggregate_type : str
            Type of aggregate to generate.
        number_of_aggregates : int
            Number of aggregates to generate.
        min_dimension : float
            Minimum dimension of the aggregate.
        max_dimension : float
            Maximum dimension of the aggregate.
        densities : list of float
            Possible aggregate densities.
        proportions : list of float
            Relative proportions used to select the densities.
        """
        self.container = container
        self.aggregate_type = aggregate_type
        self.number_of_aggregates = number_of_aggregates
        self.min_dimension = min_dimension
        self.max_dimension = max_dimension
        self.densities = densities
        self.proportions = proportions
        self.aggregates = []
        self._configure_rigidbody_world()
        self._generate_aggregates()

    def _generate_aggregates(self):
        """
        Generate aggregate objects.

        The generated aggregates are stored in self.aggregates.
        """
        if self.container.get_shape() == "CYLINDER":
            locations = []
            cylinder_radius, cylinder_height = (
                self.container.get_cylinder_dimensions()
            )

            # Start generating aggregates from the bottom of the container.
            mean_radius = (self.min_dimension + self.max_dimension) / 2
            std_dev_radius = mean_radius / 3

            prev_bounding_radius = 0
            for i in range(self.number_of_aggregates):
                # Select the aggregate density according to the specified proportions.
                density = random.choices(
                    self.densities,
                    weights=self.proportions,
                    k=1,
                )[0]

                # Loop until a valid aggregate size is found.
                while True:
                    # random.gauss pulls from a true normal distribution.
                    dimension = random.gauss(mean_radius, std_dev_radius)
                    if self.min_dimension <= dimension <= self.max_dimension:
                        break

                # Create the selected aggregate type.
                if self.aggregate_type == "sphere":
                    aggregate = Sphere(dimension, 3, density)
                elif self.aggregate_type == "polyhedron":
                    # Random point cloud for convex hull 15 to 35 looks decent.
                    number_of_points = random.randint(15, 35)
                    aggregate = PolyHedron(
                        dimension,
                        dimension,
                        dimension,
                        number_of_points,
                        2650,
                    )

                bounding_radius = aggregate.get_bounding_radius()
                allowed_radius = cylinder_radius - bounding_radius
                std_dev = allowed_radius / 3.0

                x = random.gauss(0, std_dev)
                y = random.gauss(0, std_dev)

                # Checks that the entire aggregate remains inside the cylinder.
                while x**2 + y**2 >= allowed_radius**2:
                    x = random.gauss(0, std_dev)
                    y = random.gauss(0, std_dev)

                if i == 0:
                    z = 1.01 * bounding_radius
                else:
                    z += 1.01 * (bounding_radius + prev_bounding_radius)

                prev_bounding_radius = bounding_radius
                aggregate.location((x, y, z))
                self.aggregates.append(aggregate)

                print(
                    f"Generating aggregate "
                    f"{i + 1}/{self.number_of_aggregates}"
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
            centroid = obj.get_centroid()

            vertices = []

            for vertex in obj_vertices:
                transformed_vertex = transform @ vertex.co
                vertices.append(transformed_vertex)

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

            aggregates[aggregate_id] = {
                "vertices": vertices,
                "faces": faces
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
    container = CylindricalContainer(50, 205)

    packing = Packing(
        container,
        "polyhedron",
        500,
        11.6,
        16,
        [2650],
        [1],
    )

    packing.run_simulation()
    packing.save_packing()
