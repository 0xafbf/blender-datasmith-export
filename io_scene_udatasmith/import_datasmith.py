import sys
import struct
import bpy
import bmesh
from mathutils import Matrix, Vector

from .data_types import UDMesh, UDScene, UDActor, UDTexture

b_major, b_minor, b_patch = bpy.app.version

matrix_datasmith = Matrix.Scale(1/100, 4)
matrix_datasmith[1][1] *= -1.0   

matrix_normals = Matrix()
matrix_normals[1][1] = -1.0


def mat_compose(a, b, *args):
    a_b = None
    if b_minor >= 80:
        a_b = a @ b
    else:
        a_b = a * b
    if args:
        return mat_compose(a_b, *args)
    return a_b

def load_materials(scene:UDScene):
    ' just ensure that there are materials with the same name for now '
    for name, umat in scene.materials.items():
        blender_mat = bpy.data.materials.get(name)
        if not blender_mat:
            blender_mat = bpy.data.materials.new(name)
            if "Color" in umat.properties:
                blender_mat.diffuse_color = umat.properties['Color'].value[:3]
            if "Texture" in umat.properties:
                img_name = umat.properties['Texture'].value
                blender_mat.use_nodes = True
                image_node = blender_mat.node_tree.nodes.new('ShaderNodeTexImage')
                utex = UDTexture.find(name=img_name, parent=scene)
                im = bpy.data.images.load(utex.abs_path())
                image_node.image = im

                shader_node = blender_mat.node_tree.nodes[1]
                blender_mat.node_tree.links.new(image_node.outputs['Color'], shader_node.inputs['Base Color'])


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
            
            blender_mesh.transform(matrix_datasmith)

            normals = [mat_compose(matrix_normals, Vector(norm)) for norm in umesh.vertex_normals]

            blender_mesh.use_auto_smooth = True
            blender_mesh.create_normals_split()
            blender_mesh.normals_split_custom_set(normals)
            for slot, mat in umesh.materials.items():
                blender_mesh.materials.append(bpy.data.materials.get(mat))

            if b_minor >= 80:
                blender_mesh.uv_layers.new()
            else:
                blender_mesh.uv_textures.new()
            uv0 = blender_mesh.uv_layers[0]
            for idx in range(len(uv0.data)):
                uv0.data[idx].uv = umesh.uvs[idx]

            


        # maybe we can do here some updates even if already existed?\



def load_actor(context, name: str, actor:UDActor, parent:UDActor = None):
    b_object = bpy.data.objects.get(name)
    print("loading: ", name)
    if not b_object:
        mesh_name = getattr(actor, 'mesh', None) # is not valid in plain actors
        b_mesh = None
        if mesh_name:
            b_mesh = bpy.data.meshes.get(mesh_name)
        
        b_object = bpy.data.objects.new(name, b_mesh)

        if b_minor >= 80:
            b_object.empty_display_type = 'SPHERE'
            b_object.empty_display_size = 0.1
        else:
            b_object.empty_draw_type = 'SPHERE'
            b_object.empty_draw_size = 0.1
        
        b_object.location = actor.transform.loc
        b_object.rotation_mode='QUATERNION'
        b_object.rotation_quaternion = actor.transform.rot
        b_object.scale = actor.transform.scale

        b_object.matrix_basis = mat_compose(matrix_datasmith, b_object.matrix_basis, matrix_datasmith.inverted())

        collection = bpy.data.collections.get(actor.layer)
        if collection is None:
            collection = bpy.data.collections.new(actor.layer)
            context.scene.collection.children.link(collection)
        collection.objects.link(b_object)

        if parent:
            b_parent = bpy.data.objects.get(parent.name)
            if b_parent:
                b_object.parent = b_parent

                # we can do any of these three, I don't know what would be preferred.
                
                # this is just keeping every coordinate world space
                b_object.matrix_parent_inverse = b_object.parent.matrix_world.inverted()
                
                # this is projecting every coordinate to local space
                #b_object.matrix_basis = mat_compose( b_object.parent.matrix_world.inverted(), b_object.matrix_basis)
                
                # this is the other, I like it more because resetting to default is just reset transform
                # but this would only work for objects with parent, so this is a turn down.
                # b_object.matrix_parent_inverse = (b_object.parent.matrix_world.inverted() @ b_object.matrix_basis)
                # b_object.matrix_basis = Matrix()

        if b_object.parent:
            b_object.matrix_world = mat_compose(parent.matrix_world, b_object.matrix_parent_inverse, b_object.matrix_basis)
        else:
            b_object.matrix_world = b_object.matrix_basis

        #load children
        for child_name, child in actor.objects.items():
            load_actor(context, child_name, child, b_object)




def load_objects(uscene: UDScene, context):
    for name, uobject in uscene.objects.items():
        load_actor(context, name, uobject)
        # maybe update transform conditionally (for example if I move things in other app)


def load(operator, context, filepath, *, use_smooth_groups = False, **kwargs):
    scene = UDScene(source=filepath)
    load_materials(scene)
    load_meshes(scene)
    load_objects(scene, context=context)
    return {'FINISHED'}


