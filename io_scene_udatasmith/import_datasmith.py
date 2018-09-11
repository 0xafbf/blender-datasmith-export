import sys
import struct
import bpy
import bmesh
from mathutils import Matrix

from .data_types import UDMesh, UDScene, UDActor

b_major, b_minor, b_patch = bpy.app.version

def load_materials(scene:UDScene):
    ' just ensure that there are materials with the same name for now '
    for name, mat in scene.materials.items():
        blender_mat = bpy.data.materials.get(name)
        if not blender_mat:
            blender_mat = bpy.data.materials.new(name)
        
        # here maybe something like mat.populate(blender_mat) ?


def load_meshes(uscene: UDScene):
    for name, umesh in uscene.meshes.items():
        blender_mesh = bpy.data.meshes.get(umesh.name)
        if not blender_mesh:
            blender_mesh = bpy.data.meshes.new(umesh.name)
            # this from_pydata is a convenience, misses things like normals and smoothing groups
            blender_mesh.from_pydata(umesh.vertices, [], list(zip(*(iter(umesh.triangles),)*3)))
            for idx, poly in enumerate(blender_mesh.polygons):
                poly.material_index = umesh.tris_material_slot[idx]
            
            blender_mesh.use_auto_smooth = True
            blender_mesh.create_normals_split()
            blender_mesh.normals_split_custom_set(umesh.vertex_normals)
            for slot, mat in umesh.materials.items():
                blender_mesh.materials.append(bpy.data.materials.get(mat))

        # maybe we can do here some updates even if already existed?

def load_actor(context, name: str, actor:UDActor, parent:UDActor = None):
    b_object = bpy.data.objects.get(name)
    if not b_object:
        mesh_name = getattr(actor, 'mesh', None) # is not valid in plain actors
        b_mesh = None
        if mesh_name:
            b_mesh = bpy.data.meshes.get(mesh_name)
        
        b_object = bpy.data.objects.new(name, b_mesh)
        b_object.location = actor.transform.loc
        b_object.rotation_quaternion = actor.transform.rot
        b_object.scale = actor.transform.scale
        if b_minor >= 80:
            context.scene.collection.objects.link(b_object)
            # maybe in the future it would be good to use collections instead of empties?
        else:
            context.scene.objects.link(b_object)

        #load children
        for child_name, child in actor.objects.items():
            load_actor(context, child_name, child, b_object)

        if parent:
            b_parent = bpy.data.objects.get(parent.name)
            if b_parent:
                # pass
                #b_object.parent = b_parent
                #b_object.matrix_parent_inverse = b_parent.matrix_world.inverted()


def load_objects(uscene: UDScene, context):
    for name, uobject in uscene.objects.items():
        load_actor(context, name, uobject)
        # maybe update transform conditionally (for example if I move things in other app)


def load(operator, context, filepath, *, use_smooth_groups = False):
    scene = UDScene(path=filepath)
    load_materials(scene)
    load_meshes(scene)
    load_objects(scene, context=context)
    return {'FINISHED'}


