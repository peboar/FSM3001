from abc import ABC, abstractmethod

import bmesh
import bpy


class Container(ABC):
    """Parent class for handling varying container shapes."""

    def __init__(self, shape, friction, damping):
        self.shape = shape
        self.friction = friction
        self.damping = damping
        self.collision_shape = "MESH"

        # Remove initial cube
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)

        self.obj = self._create_blender_primitive()
        if self.obj:
            self._apply_physics()
        else:
            raise RuntimeError("Failed to generate geometry")

    @abstractmethod
    def _create_blender_primitive(self):
        """Must be overridden by child classes to generate and return a Blender object."""
        pass

    def _apply_physics(self):
        """Uniformly applies physics to the container."""
        if not bpy.context.scene.rigidbody_world:
            bpy.context.view_layer.objects.active = self.obj
            self.obj.select_set(True)
            bpy.ops.rigidbody.world_add()
            self.obj.select_set(False)

        bpy.context.view_layer.objects.active = self.obj
        self.obj.select_set(True)

        if not self.obj.rigid_body:
            bpy.ops.rigidbody.object_add()

        self.obj.rigid_body.type = "PASSIVE"
        self.obj.rigid_body.friction = self.friction
        self.obj.rigid_body.linear_damping = self.damping
        self.obj.rigid_body.angular_damping = self.damping
        self.obj.rigid_body.collision_shape = self.collision_shape

        self.obj.select_set(False)

    def get_shape(self):
        return self.shape

class CylindricalContainer(Container):
    """Child class handling a procedural cylindrical container."""

    def __init__(self, radius, height, friction=0.2, damping=0.1):
        self.radius = radius
        self.height = height

        super().__init__(
            shape="CYLINDER",
            friction=friction,
            damping=damping,
        )

    def _create_blender_primitive(self):
        """Generates a procedural cylindrical container."""
        if any("Container" in obj.name for obj in bpy.data.objects):
            raise RuntimeError("Container already exists in the tree")

        mesh_data = bpy.data.meshes.new("ContainerMesh")
        cylinder_obj = bpy.data.objects.new("Container", mesh_data)
        cylinder_obj.display_type = 'WIRE' # Wire for visibility
        bm = bmesh.new()

        bmesh.ops.create_cone(
            bm,
            cap_ends=True,
            cap_tris=False,
            segments=32,
            radius1=self.radius,
            radius2=self.radius,
            depth=self.height
        )

        top_face = None

        for face in bm.faces:
            if face.normal.dot((0, 0, 1)) > 0.99:
                top_face = face
                break

        if top_face:
            bmesh.ops.delete(
                bm,
                geom=[top_face],
                context="FACES"
            )

        bm.to_mesh(mesh_data)
        bm.free()

        cylinder_obj.location.z = self.height / 2

        bpy.context.scene.collection.objects.link(cylinder_obj)

        return cylinder_obj

    def get_cylinder_dimensions(self):
        return (self.radius, self.height)