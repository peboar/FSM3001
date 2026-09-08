import bmesh
import bpy
from abc import ABC, abstractmethod
from mathutils import Vector


class Aggregate(ABC):
    """Parent class for handling varying rigid body physics particles."""

    def __init__(self, shape, density, location, friction, collision_shape="MESH"):
        self.shape = shape
        self.density = density  # Material density (kg/m³)
        self.location = Vector(location)
        self.friction = friction
        self.collision_shape = collision_shape
        self.obj = self._create_blender_primitive()
        if self.obj:
            self._apply_physics()
        else:
            raise RuntimeError(f"Failed to generate geometry")

    @abstractmethod
    def _create_blender_primitive(self):
        """Must be overridden by child classes to generate and return a Blender object."""
        pass

    def _calculate_mass_from_volume(self):
        """Calculates the true geometry volume using BMesh and returns Mass."""
        # Create a temporary bmesh directly from the final mesh geometry
        bm = bmesh.new()
        bm.from_mesh(self.obj.data)

        # Scale the geometry relative to world transforms to get accurate sizing
        bm.transform(self.obj.matrix_world)

        # Calculate volume (m³)
        volume = bm.calc_volume()
        bm.free()

        # Mass = Volume * Density
        return abs(volume) * self.density

    def _apply_physics(self):
        """Uniformly applies physics using dynamically calculated mass."""
        # Ensure the Rigid Body World exists FIRST before running any physics operators
        if not bpy.context.scene.rigidbody_world:
            bpy.context.view_layer.objects.active = self.obj
            self.obj.select_set(True)
            bpy.ops.rigidbody.world_add()
            self.obj.select_set(False)

        # Set the object as active and selected
        bpy.context.view_layer.objects.active = self.obj
        self.obj.select_set(True)

        # Safely add the rigid body component if it doesn't exist
        if not self.obj.rigid_body:
            bpy.ops.rigidbody.object_add()

        # Calculate mass and apply physics attributes
        dynamic_mass = self._calculate_mass_from_volume()

        self.obj.rigid_body.type = "ACTIVE"
        self.obj.rigid_body.mass = dynamic_mass
        self.obj.rigid_body.friction = (
            self.friction
        )  # Added self. to prevent NameError
        self.obj.rigid_body.collision_shape = self.collision_shape

        # Clean up selection state
        self.obj.select_set(False)


class Sphere(Aggregate):
    """Child class handling procedural spherical rigid body particles."""

    def __init__(self, radius, density, location, friction=0.7, subdivisions=5):
        self.radius = radius
        self.subdivisions = subdivisions

        # Initialize parent class with configuration details
        super().__init__(
            shape="SPHERE",
            density=density,
            location=location,
            friction=friction,
            collision_shape="SPHERE",  # Dynamic primitive shape optimization
        )

    def _create_blender_primitive(self):
        """Generates a procedural Ico Sphere mesh inside the scene."""
        # Create sphere object
        mesh_data = bpy.data.meshes.new("SphereMesh")
        sphere = bpy.data.objects.new("Sphere", mesh_data)

        bm = bmesh.new()

        bmesh.ops.create_icosphere(bm,
                                   subdivisions=self.subdivisions,
                                   radius=self.radius,
                                   )

        # Finalize and transfer geometry data to mesh data block
        bm.to_mesh(mesh_data)
        bm.free()

        sphere.location = self.location

        # Link to scene
        bpy.context.scene.collection.objects.link(sphere)
        return sphere


# Example execution to test the structural flow:
# This generates an interactive sphere with realistic calculated mass based on its volume.
test_sphere = Sphere(radius=1.5, density=2500.0, location=(0,0,5), friction=0.4, subdivisions=1)
