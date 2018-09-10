import sys
import struct
import bpy
import bmesh
from .data_types import UDMesh, UDScene


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


def load_objects(uscene: UDScene, context):
    for name, uobject in uscene.actors.items():
        b_object = bpy.data.objects.get(name)
        if not b_object:
            b_mesh = bpy.data.meshes.get(uobject.mesh)
            b_object = bpy.data.objects.new(name, b_mesh)
            b_object.location = uobject.loc
            b_object.rotation_quaternion = uobject.rot
            b_object.scale = uobject.scale
            context.scene.collection.objects.link(b_object)
        # maybe update transform conditionally (for example if I move things in other app)


def load(operator, context, filepath, *, use_smooth_groups = False):
    scene = UDScene(path=filepath)
    load_materials(scene)
    load_meshes(scene)
    load_objects(scene, context=context)
    return {'FINISHED'}

#if __name__ == '__main__':
    # this won't work without blender anymore
    # load(None, "C:/Users/boterock/Desktop/ud_test.udatasmith")


    #load("C:/Users/boterock/Desktop/cube.udatasmith")



# '''
# for testing inside blender console:

#     from io_scene_udatasmith.data_types import UDScene
#     scene = UDScene(path="C:\\Users\\boterock\\Desktop\\cube.udatasmith")

# '''

