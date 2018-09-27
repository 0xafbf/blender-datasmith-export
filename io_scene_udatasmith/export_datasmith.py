
import bpy
import bmesh
import math
from io_scene_udatasmith.data_types import (
	UDScene, UDActor, UDActorMesh, UDMaterial, UDMasterMaterial,
	UDActorLight, UDActorCamera, UDMesh) 
from mathutils import Matrix


matrix_datasmith = Matrix.Scale(100, 4)
matrix_datasmith[1][1] *= -1.0   

matrix_normals = Matrix.Scale(-1, 4)
#matrix_normals[1][1] = -1

# used for lights and cameras, whose forward is (0, 0, -1) and its right is (1, 0, 0)
matrix_forward = Matrix((
	(0, 1, 0, 0),
	(0, 0, -1, 0),
	(-1, 0, 0, 0),
	(0, 0, 0, 1)
))

def collect_materials(materials, uscene):
	for mat in materials:
		mat_name = getattr(mat, 'name', 'default_material')
		if mat_name in uscene.materials:
			continue

		umat = UDMasterMaterial.new(parent=uscene, name=mat_name)
		if mat:
			umat.properties['Color'] = UDMasterMaterial.PropColor(*mat.diffuse_color)


def collect_mesh(bl_mesh, uscene):
	umesh = UDMesh.new(name=bl_mesh.name, parent=uscene)
	
	# create copy to triangulate
	m = bl_mesh.copy()
	m.transform(matrix_datasmith)
	bm = bmesh.new()
	bm.from_mesh(m)
	bmesh.ops.triangulate(bm, faces=bm.faces[:])
	# this is just to make sure a UV layer exists
	bm.loops.layers.uv.verify()
	bm.to_mesh(m)
	bm.free()
	# not sure if this is the best way to read normals
	m.calc_normals_split()

	
	umesh.materials = [mat.name if mat else 'None' for mat in bl_mesh.materials]
	collect_materials(bl_mesh.materials, uscene)

	#for idx, mat in enumerate(bl_mesh.materials):
	#    umesh.materials[idx] = getattr(mat, 'name', 'DefaultMaterial')
	
	umesh.tris_material_slot = [p.material_index for p in m.polygons]
	umesh.tris_smoothing_group = [0 for p in m.polygons] # no smoothing groups for now
	
	umesh.vertices = [v.co.copy() for v in m.vertices]
	
	umesh.triangles = [l.vertex_index for l in m.loops]
	umesh.vertex_normals = [matrix_normals * l.normal for l in m.loops] # don't know why, but copy is needed for this only
	umesh.uvs = [m.uv_layers[0].data[l.index].uv.copy() for l in m.loops]
	
	bpy.data.meshes.remove(m) 

def collect_object(bl_obj, uscene, parent = None):
	if parent is None:
		parent = uscene
	uobj = None
	
	kwargs = {}
	kwargs['parent'] = parent
	kwargs['name'] = bl_obj.name
	
	if bl_obj.type == 'MESH':
		uobj = UDActorMesh.new(**kwargs)
		collect_mesh(bl_obj.data, uscene)
		uobj.mesh = bl_obj.data.name
	elif bl_obj.type == 'CAMERA':
		uobj = UDActorCamera.new(**kwargs)
		bl_cam = bl_obj.data
		uobj.focal_length = bl_cam.lens
		uobj.focus_distance = bl_cam.dof_distance
		uobj.sensor_width = bl_cam.sensor_width

	elif bl_obj.type == 'LAMP' or bl_obj.type == 'LIGHT':
		uobj = UDActorLight.new(**kwargs)
		bl_light = bl_obj.data

		node = bl_light.node_tree.nodes['Emission']
		uobj.color = node.inputs['Color'].default_value
		uobj.intensity = node.inputs['Strength'].default_value * 16 # rough translation to candelas

		if bl_light.type == 'POINT':
			uobj.type = UDActorLight.LIGHT_POINT
		elif bl_light.type == 'SPOT':
			uobj.type = UDActorLight.LIGHT_SPOT
			angle = bl_light.spot_size * 180 / (2*math.pi)
			uobj.outer_cone_angle = angle
			uobj.inner_cone_angle = angle - angle * bl_light.spot_blend

	else: # maybe empties
		uobj = UDActor.new(**kwargs)

	obj_mat = matrix_datasmith * bl_obj.matrix_world * matrix_datasmith.inverted()

	if bl_obj.type == 'CAMERA' or bl_obj.type == 'LAMP':
		obj_mat = obj_mat * matrix_forward

	uobj.transform.loc = obj_mat.to_translation()
	uobj.transform.rot = obj_mat.to_quaternion()
	uobj.transform.scale = obj_mat.to_scale()
	
	for child in bl_obj.children:
		collect_object(child, uscene, uobj)

def collect_to_uscene(context):
	all_objects = context.scene.objects
	root_objects = [obj for obj in all_objects if obj.parent is None]
	
	uscene = UDScene()
	for obj in root_objects:
		uobj = collect_object(obj, uscene)
	
	return uscene

def save(operator, context,*, filepath, **kwargs):
	
	from os import path
	basepath, ext = path.splitext(filepath)
	basedir, basename = path.split(basepath)
	scene = collect_to_uscene(bpy.context)
	scene.save(basedir, basename) 

	return {'FINISHED'}
	
if __name__ == '__main__':
	save(operator=None, context=bpy.context, filepath='C:\\Users\\boterock\\Desktop\\export.datasmith') 