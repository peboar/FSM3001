from abc import ABC, abstractmethod
import random

import bmesh
import bpy
from mathutils import Vector


class Aggregate(ABC):
    """Parent class for handling varying rigid body physics particles."""

    def __init__(
        self,
        shape,
        density,
        friction,
        damping,
        collision_shape="MESH",
    ):
        self.shape = shape
        self.density = density
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
        """Must be overridden by child classes to generate a Blender object."""
        pass

    def location(self, location):
        self.obj.location = Vector(location)

    def _calculate_mass_from_volume(self):
        """Calculates the true geometry volume using BMesh and returns Mass."""
        bm = bmesh.new()
        bm.from_mesh(self.obj.data)

        bm.transform(self.obj.matrix_world)

        volume = bm.calc_volume()
        bm.free()

        return abs(volume) * self.density

    @abstractmethod
    def get_bounding_radius(self):
        """Must be overridden by child classes to return the maximum bounding radius."""
        pass

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

    def get_shape(self):
        return self.shape


class Sphere(Aggregate):
    """Child class handling procedural spherical rigid body particles."""

    def __init__(
        self,
        diameter,
        subdivisions=3,
        density=2650,
        friction=0.7,
        damping=0.1,
    ):
        self.diameter = diameter
        self.radius = diameter / 2
        self.subdivisions = subdivisions

        super().__init__(
            shape="SPHERE",
            density=density,
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

        bpy.context.scene.collection.objects.link(sphere)

        return sphere

    def get_bounding_radius(self):
        return self.radius


class PolyHedron(Aggregate):
    """Child class handling procedural convex polyhedron particles."""

    def __init__(
        self,
        length,
        depth,
        height,
        number_of_points=15,
        density=2650,
        friction=0.45,
        damping=0.1,
    ):
        self.length = length
        self.depth = depth
        self.height = height
        self.number_of_points = number_of_points

        super().__init__(
            shape="POLYHEDRON",
            density=density,
            friction=friction,
            damping=damping,
            collision_shape="CONVEX_HULL",
        )

    def _create_blender_primitive(self):
        """Generates a procedural convex polyhedron."""
        mesh_data = bpy.data.meshes.new("ConvexHullMesh")
        poly_hedra = bpy.data.objects.new("PolyHedra", mesh_data)

        bm = bmesh.new()

        for _ in range(self.number_of_points):
            x = random.uniform(-self.length / 2, self.length / 2)
            y = random.uniform(-self.depth / 2, self.depth / 2)
            z = random.uniform(-self.height / 2, self.height / 2)

            bm.verts.new([x, y, z])

        bmesh.ops.convex_hull(bm, input=bm.verts)

        bm.to_mesh(mesh_data)
        bm.free()

        bpy.context.scene.collection.objects.link(poly_hedra)

        return poly_hedra

    def get_bounding_radius(self):
        return max(vertex.co.length for vertex in self.obj.data.vertices)