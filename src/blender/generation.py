import sys
import os
import random
import bpy
import numpy as np

sys.path.append(r"/home/per/Desktop/Kth/Phd/Courses/FSM3001/Project/src/blender")

from containers import CylindricalContainer
from aggregates import Sphere, PolyHedron


class Generation:
    """Handles the generation of aggregates inside a container."""

    def __init__(
            self,
            container,
            aggregate_type,
            number_of_aggregates,
            min_dimension,
            max_dimension,
            densities,
            proportions
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
        Generate aggregate objects

        The generated aggregates are stored in self.aggregates.
        """
        if self.container.get_shape() == "CYLINDER":
            locations = []
            cylinder_radius, cylinder_height = self.container.get_cylinder_dimensions()

            # Start generating aggregates from the bottom of the container.
            mean_radius = (self.min_dimension + self.max_dimension) / 2
            std_dev_radius = mean_radius / 3

            # Select the aggregate density according to the specified proportions.
            density = random.choices(
                self.densities,
                weights=self.proportions,
                k=1
            )[0]


            prev_bounding_radius = 0
            for i in range(self.number_of_aggregates):
                # Loop until a valid aggregate size is found
                while True:
                    # random.gauss pulls from a true normal distribution
                    dimension = random.gauss(mean_radius, std_dev_radius)
                    if self.min_dimension <= dimension <= self.max_dimension:
                        break # Valid normal sample found, break out of loop

                # Create the selected aggregate type.
                if self.aggregate_type == "sphere":
                    aggregate = Sphere(dimension, 3, density)
                elif self.aggregate_type == "polyhedron":
                    number_of_points = random.randint(15, 35)
                    aggregate = PolyHedron(dimension, dimension, dimension, number_of_points, 2650)

                bounding_radius = aggregate.get_bounding_radius()
                allowed_radius = cylinder_radius - bounding_radius
                std_dev = allowed_radius / 3.0

                x = random.gauss(0, std_dev)
                y = random.gauss(0, std_dev)

                # Checks that the entire aggregate remains inside the cylinder.
                while x ** 2 + y ** 2 >= allowed_radius ** 2:
                    x = random.gauss(0, std_dev)
                    y = random.gauss(0, std_dev)

                if i == 0:
                    z = 1.01 * bounding_radius
                else:
                    z += 1.01 * (bounding_radius + prev_bounding_radius)

                prev_bounding_radius = bounding_radius
                aggregate.location((x, y, z))
                self.aggregates.append(aggregate)

                print(f"Generating aggregate {i+1}/{self.number_of_aggregates}")

    def _configure_rigidbody_world(self):
        if not bpy.context.scene.rigidbody_world:
            bpy.ops.rigidbody.world_add()

        rb_world = bpy.context.scene.rigidbody_world

        # Convex hull contact between the mesh container fails for small dimensions. Must use millimeters
        bpy.context.scene.unit_settings.system = "METRIC"
        bpy.context.scene.unit_settings.length_unit = "MILLIMETERS"
        bpy.context.scene.unit_settings.scale_length = 0.001
        bpy.context.scene.gravity = (0, 0, -9810)

        rb_world.time_scale = 0.1  # Slows simulation and fixes failed contacts
        rb_world.substeps_per_frame = 10  # Default; increase if contacts fail
        rb_world.solver_iterations = 10  # Default; increase if contacts fail


    def run_simulation(self, total_frames=1000):
        rb_world = bpy.context.scene.rigidbody_world
        rb_world.point_cache.frame_end = total_frames
        for frame in range(0, total_frames + 1):
            bpy.context.scene.frame_set(frame)
            bpy.context.view_layer.update()  # Force Blender to update object positions
            output_frequency = total_frames // 100
            if frame > 0 and frame % output_frequency == 0:
                print(f"Simulation completion {100*frame/total_frames: .0f}%")

    def save_packing(self, filename="packing.npz"):
        vertices = []
        faces = []
        aggregate_ids = []

        vertex_offset = 0

        for aggregate_id, aggregate in enumerate(self.aggregates):

            obj = aggregate.obj

            transform = obj.matrix_world

            for vertex in obj.data.vertices:
                transformed_vertices = transform @ vertex.co
                vertices.append(transformed_vertices)

            # Offsetting the indices of the faces to ensure they map to the correct vertices
            for polygon in obj.data.polygons:
                faces.append([
                    vertex_index + vertex_offset
                    for vertex_index in polygon.vertices
                ])

            # Store aggregate indices
            aggregate_ids.append(aggregate_id)

            vertex_offset += len(obj.data.vertices)

        np.savez(
            filename,
            vertices=np.array(vertices),
            faces=np.array(faces),
            aggregate_ids=np.array(aggregate_ids)
        )

if __name__ == "__main__":
    container = CylindricalContainer(50, 205)

    generation = Generation(
        container,
        "polyhedron",
        10,
        11.6/2,
        16/2,
        [2500],
        [1]
    )
    generation.run_simulation()
    generation.save_packing()
