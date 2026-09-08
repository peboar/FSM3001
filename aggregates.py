import bmesh
import bpy
from abc import ABC, abstractmethod
from mathutils import Vector


class Aggregate(ABC):
    """Parent class for handling varying rigid body physics particles."""

    def __init__(self, shape, density, location, friction, damping, collision_shape="MESH"):
        self.shape = shape
        self.density = density
        self.location = Vector(location)
        self.friction = friction
        self.damping = damping
        self.collision_shape = collision_shape
        self.obj = self._create_blender_primitive()

        if self.obj:
            self._apply_physics()
        else:
            raise RuntimeError("Failed to generate geometry")

    @abstractmethod
    def _create_blender_primitive(self):
        """Must be overridden by child classes to generate and return a Blender object."""
        pass

    def _calculate_mass_from_volume(self):
        """Calculates the true geometry volume using BMesh and returns Mass."""
        bm = bmesh.new()
        bm.from_mesh(self.obj.data)

        bm.transform(self.obj.matrix_world)

        volume = bm.calc_volume()
        bm.free()

        return abs(volume) * self.density

    def _apply_physics(self):
        """Uniformly applies physics using dynamically calculated mass."""
        if not bpy.context.scene.rigidbody_world:
            bpy.context.view_layer.objects.active = self.obj
            self.obj.select_set(True)
            bpy.ops.rigidbody.world_add()
            self.obj.select_set(False)

        bpy.context.view_layer.objects.active = self.obj
        self.obj.select_set(True)

        if not self.obj.rigid_body:
            bpy.ops.rigidbody.object_add()

        dynamic_mass = self._calculate_mass_from_volume()

        self.obj.rigid_body.type = "ACTIVE"
        self.obj.rigid_body.mass = dynamic_mass
        self.obj.rigid_body.friction = self.friction
        self.obj.rigid_body.linear_damping = self.damping
        self.obj.rigid_body.angular_damping = self.damping
        self.obj.rigid_body.collision_shape = self.collision_shape

        self.obj.select_set(False)

    def voxelize(self, voxel_size):
        """Voxelize the aggregate.
            Parameters
            ----------
            voxel_size : Size of the voxel
        """
        if self.obj:

            remesh_mod = self.obj.modifiers.new(name="Remesh", type='REMESH')

            remesh_mod.mode = 'VOXEL'
            remesh_mod.voxel_size = voxel_size
            remesh_mod.use_smooth_shade = False

    def get_shape(self):
        return self.shape


class Sphere(Aggregate):
    """Child class handling procedural spherical rigid body particles."""

    def __init__(self, radius, density, location, subdivisions=3, friction=0.7, damping=0.1):
        self.radius = radius
        self.subdivisions = subdivisions

        super().__init__(
            shape="SPHERE",
            density=density,
            location=location,
            friction=friction,
            damping=damping,
            collision_shape="SPHERE",
        )

    def _create_blender_primitive(self):
        """Generates a procedural Ico Sphere mesh inside the scene."""
        mesh_data = bpy.data.meshes.new("SphereMesh")
        sphere = bpy.data.objects.new("Sphere", mesh_data)

        bm = bmesh.new()

        bmesh.ops.create_icosphere(
            bm,
            subdivisions=self.subdivisions,
            radius=self.radius,
        )

        bm.to_mesh(mesh_data)
        bm.free()

        sphere.location = self.location

        bpy.context.scene.collection.objects.link(sphere)

        return sphere



if __name__ == "__main__":
    aggregate = Sphere(8e-3, 2500,(0,0,0), 3, 0.1)
    aggregate.voxelize(0.01)
