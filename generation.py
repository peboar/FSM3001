import sys
import os
import random
import bpy

sys.path.append(r"/home/per/Desktop/Kth/Phd/Courses/FSM3001/Project")

from containers import CylindricalContainer
from aggregates import Sphere


class Generation:
    """Handles the generation of aggregates inside a container."""

    def __init__(
            self,
            container,
            aggregate_type,
            number_of_aggregates,
            min_radius,
            max_radius,
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
        min_radius : float
            Minimum bounding radius of an aggregate.
        max_radius : float
            Maximum bounding radius of an aggregate.
        densities : list of float
            Possible aggregate densities.
        proportions : list of float
            Relative proportions used to select the densities.
        """
        self.container = container
        self.aggregate_type = aggregate_type
        self.number_of_aggregates = number_of_aggregates
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.densities = densities
        self.proportions = proportions
        self.aggregates = []
        self._configure_rigidbody_world()
        self._generate_aggregates()

    def _generate_aggregate_locations(self):
        """
        Generate bounding radii and locations for the aggregates.

        Returns
        -------
        tuple
            Bounding radii and locations of the generated aggregates.
        """
        if self.container.get_shape() == "CYLINDER":
            locations = []
            radii = []
            cylinder_radius, cylinder_height = self.container.get_cylinder_dimensions()

            # Start generating aggregates from the bottom of the container.
            z = cylinder_height
            mean_radius = (self.min_radius + self.max_radius) / 2
            std_dev_radius = mean_radius / 3

            while len(locations) < self.number_of_aggregates:
                while True:
                    # random.gauss pulls from a true normal distribution
                    sampled_radius = random.gauss(mean_radius, std_dev_radius)
                    if self.min_radius <= sampled_radius <= self.max_radius:
                        radius = sampled_radius
                        break # Valid normal sample found, break out of loop

                # Determine the region where the aggregate center can be placed.
                allowed_radius = cylinder_radius - radius
                std_dev = allowed_radius / 3.0

                # Generate a random position around the center of the cylinder.
                x = random.gauss(0, std_dev)
                y = random.gauss(0, std_dev)

                # Check that the entire aggregate remains inside the cylinder.
                if x ** 2 + y ** 2 < allowed_radius ** 2:
                    if not radii:
                        z += 1.01 * radius
                    else:
                        z += 1.01*(radius + radii[-1])
                    radii.append(radius)
                    locations.append((x, y, z))

            return radii, locations

    def _generate_aggregates(self):
        """
        Generate aggregate objects from the generated locations.

        The generated aggregates are stored in self.aggregates.
        """
        radii, locations = self._generate_aggregate_locations()

        for radius, location in zip(radii, locations):
            # Select the aggregate density according to the specified proportions.
            density = random.choices(
                self.densities,
                weights=self.proportions,
                k=1
            )[0]

            # Create the selected aggregate type.
            if self.aggregate_type == "sphere":
                sphere = Sphere(radius, density, location)
                self.aggregates.append(sphere)

    def _configure_rigidbody_world(self):
        if not bpy.context.scene.rigidbody_world:
            bpy.ops.rigidbody.world_add()

        rb_world = bpy.context.scene.rigidbody_world
        rb_world.time_scale = 0.1  # Slows simulation and fixes failed contacts
        rb_world.substeps_per_frame = 10  # Default; increase if contacts fail
        rb_world.solver_iterations = 10  # Default; increase if contacts fail

    def run_simulation(self, total_frames=1000):
        rb_world = bpy.context.scene.rigidbody_world
        rb_world.point_cache.frame_end = total_frames
        for frame in range(0, total_frames + 1):
            bpy.context.scene.frame_set(frame)
            bpy.context.view_layer.update()  # Force Blender to update object positions

            if frame % 100 == 0 and frame > 0:
                print(f"    Simulation frame {frame}/{total_frames} calculated...")


    def voxelize_aggregates(self, voxel_size):
        for aggregate in self.aggregates:
            aggregate.voxelize(voxel_size)

if __name__ == "__main__":
    container = CylindricalContainer(0.1, 0.5)

    generation = Generation(
        container,
        "sphere",
        100,
        11.6e-3/2,
        16e-3/2,
        [2500],
        [1]
    )
    generation.run_simulation()
    generation.voxelize_aggregates()