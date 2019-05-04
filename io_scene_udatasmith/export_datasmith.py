
import bpy
import bmesh
import math
from io_scene_udatasmith.data_types import (
	UDScene, UDActor, UDActorMesh, UDMaterial, UDMasterMaterial,
	UDActorLight, UDActorCamera, UDMesh, Node, UDTexture, sanitize_name)
from mathutils import Matrix

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


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




def TexNode(type, texture):
	return Node(type, {
		'tex': texture,
		'coordinate': 0,
		'sx': 1.0,
		'sy': 1.0,
		'ox': 0.0,
		'oy': 0.0,
		'mx': 0,
		'my': 0,
		'rot': 0.0,
		'mul': 1.0,
		'channel': 0,
		'inv': 0,
		'cropped':0,

	})

def make_scalar_node(name, value, **kwargs):
	return Node(name, attrs={'value': value, **kwargs})

def make_rgba_node(name, color):
	return Node(name, attrs={
		'R': color[0],
		'G': color[1],
		'B': color[2],
		'A': color[3],
		})

def make_basic_ushader(base_name, material):
	shader = Node('Shader')
	shader.children = [
		make_rgba_node('Diffusecolor', material.diffuse_color),
		make_scalar_node('Metalval', material.metallic),
		make_scalar_node('Specularval', material.specular_intensity),
		make_scalar_node('Roughnessval', material.roughness, desc='Roughness'),
	]
	return shader


def make_default_node(field, name):
	node = Node(name)
	default = field.default_value
	if field.type == 'VALUE':
		node['value'] = "%.6f" % default
	elif field.type == 'RGBA':
		node['R'] = '%.6f' % default[0]
		node['G'] = '%.6f' % default[1]
		node['B'] = '%.6f' % default[2]
		node['A'] = '%.6f' % default[3]

	return node


def make_field_node(field, name, default_name):
	if field.links:
		input_node = field.links[0].from_node
		if input_node.type == 'TEX_IMAGE':
			# if it is a texture
			image = input_node.image
			image_name = sanitize_name(image.name)
			node = TexNode(name, image_name)
			# also make sure that the scene has a reference to this texture
			texture = UDScene.current_scene.get_field(UDTexture, image_name)
			texture.image = image

			return node
		elif input_node.type == 'HUE_SAT':
			return make_field_node(input_node.inputs['Color'], name, default_name)
		elif input_node.type == 'MIX_RGB':
			factor = input_node.inputs['Fac'].default_value
			selected_input = 'Color1' if factor < 0.5 else 'Color2'
			return make_field_node(input_node.inputs[selected_input], name, default_name)
		elif input_node.type == 'RGB':
			value = input_node.outputs[0]
			return make_default_node(value, default_name)
		else:
			log.error("unhandled node")

	return make_default_node(field, default_name)

def make_ushader(base_name, mat_node=None):

	shader = Node('Shader')
	shader['name'] = base_name + '_0'
	if not mat_node:
		return shader

	# TODO handle anything other than principled bsdf
	if mat_node.type == 'MIX_SHADER':
		pass
	elif mat_node.type == 'BSDF_PRINCIPLED':
		inputs = mat_node.inputs

		params = []
		base_color = inputs['Base Color']
		log.info("material name" + str(base_name))
		log.info("base_color links:" + str(base_color.links))
		params.append(make_field_node(base_color, 'Diffuse', 'Diffusecolor'))
		params.append(make_field_node(inputs['IOR'], 'IOR', 'IOR'))
		params.append(make_field_node(inputs['Metallic'], 'Metallic', 'Metalval'))
		params.append(make_field_node(inputs['Roughness'], 'Roughness', 'Roughnessval'))
		params.append(make_field_node(inputs['Specular'], 'Specular', 'Specularval'))


		shader.children = params

	elif mat_node.type == 'BSDF_TRANSPARENT':
		pass

	return shader



def collect_materials(scene, materials):
	umats = []
	for mat in materials:
		if not mat: # material may be none
			continue

		mat_name = sanitize_name(mat.name)
		umat = scene.get_field(UDMaterial, mat_name)
		umats.append(umat)

		if not mat.node_tree:
			umat.children = [make_basic_ushader(mat_name, mat)]
			continue

		log.debug(mat.name)
		log.debug(umat.name)
		output = mat.node_tree.get_output_node('EEVEE') # this could be 'CYCLES' or 'ALL'

		# this might need a big refactoring, to handle all the possible cases
		output_links = output.inputs['Surface'].links

		if output_links:
			surface_input = output_links[0].from_node
			umat.children = [make_ushader(mat_name, surface_input)]
		else:
			umat.children = [make_ushader(mat_name)]

	return umats




def collect_mesh(bl_mesh, uscene):

	# when using linked libraries, some meshes can have the same name
	mesh_name = sanitize_name(bl_mesh.name)
	if bl_mesh.library:
		#import pdb; pdb.set_trace()
		lib_path = bpy.path.clean_name(bl_mesh.library.filepath)
		prefix = lib_path.strip('_')
		if prefix.endswith('_blend'):
			prefix = prefix[:-6] + '_'
		mesh_name = prefix + mesh_name

	umesh = uscene.get_field(UDMesh, name=mesh_name)
	if len(umesh.vertices) > 0:
		# mesh already processed
		return umesh
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


	umats = collect_materials(uscene, bl_mesh.materials)
	umesh.materials = [umat.name for umat in umats]

	#for idx, mat in enumerate(bl_mesh.materials):
	#    umesh.materials[idx] = getattr(mat, 'name', 'DefaultMaterial')

	umesh.tris_material_slot = [p.material_index for p in m.polygons]
	umesh.tris_smoothing_group = [0 for p in m.polygons] # no smoothing groups for now

	umesh.vertices = [v.co.copy() for v in m.vertices]

	umesh.triangles = [l.vertex_index for l in m.loops]
	umesh.vertex_normals = [matrix_normals @ l.normal for l in m.loops] # don't know why, but copy is needed for this only
	umesh.uvs = [fix_uv(m.uv_layers[0].data[l.index].uv.copy()) for l in m.loops]

	bpy.data.meshes.remove(m)
	return umesh

def fix_uv(data):
	return (data[0], 1-data[1])

def collect_object(bl_obj, uscene, context, dupli_matrix=None, name_override=None):

	uobj = None

	kwargs = {}
	kwargs['name'] = bl_obj.name
	if name_override:
		kwargs['name'] = name_override

	if bl_obj.type == 'MESH':
		bl_mesh = bl_obj.data
		if len(bl_mesh.polygons) > 0:
			uobj = UDActorMesh(**kwargs)
			umesh = collect_mesh(bl_obj.data, uscene)
			uobj.mesh = umesh.name
			for idx, slot in enumerate(bl_obj.material_slots):
				if slot.link == 'OBJECT':
					collect_materials([slot.material], uscene)
					uobj.materials[idx] = slot.material.name

		else: # if is a mesh with no polys, treat as empty
			uobj = UDActor(**kwargs)

	elif bl_obj.type == 'CAMERA':
		uobj = UDActorCamera(**kwargs)
		bl_cam = bl_obj.data
		uobj.focal_length = bl_cam.lens
		uobj.focus_distance = bl_cam.dof_distance
		uobj.sensor_width = bl_cam.sensor_width

	elif bl_obj.type == 'LIGHT':
		uobj = UDActorLight(**kwargs)
		bl_light = bl_obj.data

		if bl_light.node_tree:

			node = bl_light.node_tree.nodes['Emission']
			uobj.color = node.inputs['Color'].default_value
			uobj.intensity = node.inputs['Strength'].default_value # have to check how to relate to candelas
		else:
			uobj.color = bl_light.color
			uobj.intensity = bl_light.energy

		if bl_light.type == 'POINT':
			uobj.type = UDActorLight.LIGHT_POINT
		elif bl_light.type == 'SPOT':
			uobj.type = UDActorLight.LIGHT_SPOT
			angle = bl_light.spot_size * 180 / (2*math.pi)
			uobj.outer_cone_angle = angle
			uobj.inner_cone_angle = angle - angle * bl_light.spot_blend

	else: # maybe empties
		uobj = UDActor(**kwargs)

	mat_basis = bl_obj.matrix_world
	if dupli_matrix:
		mat_basis = dupli_matrix @ mat_basis

	obj_mat = matrix_datasmith @ mat_basis @ matrix_datasmith.inverted()

	if bl_obj.type == 'CAMERA' or bl_obj.type == 'LIGHT':
		# use this correction because lights/cameras in blender point -Z
		obj_mat = obj_mat @ matrix_forward

	loc, rot, scale = obj_mat.decompose()
	uobj.transform.loc = loc
	uobj.transform.rot = rot
	uobj.transform.scale = scale

	if bl_obj.instance_type == 'COLLECTION':
		duplis = bl_obj.instance_collection.objects
		for idx, dup in enumerate(duplis):
			dupli_name = '{parent}_{dup_idx}'.format(parent=dup.name, dup_idx=idx)
			new_obj = collect_object(dup, uscene, context=context, dupli_matrix=bl_obj.matrix_world, name_override=dupli_name)
			uobj.objects[new_obj.name] = new_obj

	if dupli_matrix is None: # this was for blender 2.7, maybe 2.8 works without this
		for child in bl_obj.children:
			new_obj = collect_object(child, uscene, context=context)
			uobj.objects[new_obj.name] = new_obj

	return uobj

def collect_to_uscene(context):
	all_objects = context.scene.objects
	root_objects = [obj for obj in all_objects if obj.parent is None]

	uscene = UDScene()
	UDScene.current_scene = uscene # FIXME
	for obj in root_objects:
		uobj = collect_object(obj, uscene, context=context)
		uscene.objects[uobj.name] = uobj
	return uscene

def save(operator, context,*, filepath, **kwargs):

	from os import path
	basepath, ext = path.splitext(filepath)
	basedir, basename = path.split(basepath)
	scene = collect_to_uscene(bpy.context)
	scene.save(basedir, basename)
	UDScene.current_scene = None # FIXME

	return {'FINISHED'}

if __name__ == '__main__':
	save(operator=None, context=bpy.context, filepath='C:\\Users\\boterock\\Desktop\\export.datasmith')
