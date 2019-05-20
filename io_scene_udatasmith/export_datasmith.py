# Copyright Andrés Botero 2019

import bpy
import bmesh
import math
from io_scene_udatasmith.data_types import (
	UDScene, UDActor, UDActorMesh,
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


def exp_color(value):
	return Node("Color", {
		"Name": "",
		"constant": "(R=%.6f,G=%.6f,B=%.6f,A=%.6f)"%tuple(value)
		})
def exp_scalar(value):
	return Node("Scalar", {
		"Name": "",
		"constant": "%f"%value
		})




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

def make_ushader(mat_node):

	shader = Node('Shader')
	if not mat_node:
		return shader

	# TODO handle anything other than principled bsdf
	if mat_node.type == 'BSDF_PRINCIPLED':
		inputs = mat_node.inputs

		params = []
		base_color = inputs['Base Color']
		log.info("base_color links:" + str(base_color.links))
		params.append(make_field_node(base_color, 'Diffuse', 'Diffusecolor'))
		params.append(make_field_node(inputs['IOR'], 'IOR', 'IOR'))
		params.append(make_field_node(inputs['Metallic'], 'Metallic', 'Metalval'))
		params.append(make_field_node(inputs['Roughness'], 'Roughness', 'Roughnessval'))
		params.append(make_field_node(inputs['Specular'], 'Specular', 'Specularval'))


		shader.children = params


	return shader

def exp_texcoord(index=0, u_tiling=1.0, v_tiling=1.0):
	n = Node("TextureCoordinate")
	n["Index"] = "0"
	n["UTiling"] = u_tiling
	n["VTiling"] = v_tiling
	return n

def exp_texture(path, tex_coord_exp):
	n = Node("Texture")
	n["Name"] = ""
	n["PathName"] = path
	n.push(Node("Coordinates", {
		"expression": tex_coord_exp,
		"OutputIndex": 0,
		}))
	return n

# the operation options are:
# 'ADD', 'SUBTRACT', 'MULTIPLY', 'DIVIDE', 'POWER', 'LOGARITHM',
# 'SQRT', 'ABSOLUTE', 'MINIMUM', 'MAXIMUM', 'LESS_THAN',
# 'GREATER_THAN', 'ROUND', 'FLOOR', 'CEIL', 'FRACT', 'MODULO', 'SINE',
# 'COSINE', 'TANGENT', 'ARCSINE', 'ARCCOSINE', 'ARCTANGENT', 'ARCTAN2'
op_map = {
	'ADD': "Add",
	'SUBTRACT': "Subtract",
	'MULTIPLY': "Multiply",
	'DIVIDE': "Divide",
	'POWER': "Power",
	'SQRT': "Sqrt",
	'ABSOLUTE': "Abs",
	'MINIMUM': "Min",
	'MAXIMUM': "Max",
}

def exp_math(node, exp_list):
	op = node.operation
	n = Node(op_map[op])
	if node.inputs[0].links:
		exp_1 = get_expression(node.inputs[0], exp_list)
		n.push(Node("0", {"expression": exp_1}))
	else:
		n.push(Node("0", {
			"name": "constA",
			"type": "Float",
			"val": node.inputs[0].default_value
		}))
	if node.inputs[1].links:
		exp_1 = get_expression(node.inputs[1], exp_list)
		n.push(Node("1", {"expression": exp_1}))
	else:
		n.push(Node("1", {
			"name": "constB",
			"type": "Float",
			"val": node.inputs[1].default_value
		}))
	# TODO: test in unreal if I have an expression with two imputs which
	# one takes place, if it crashes or what
	return exp_list.push(n)

op_map2 = {
	'OVERLAY': "/Engine/Functions/Engine_MaterialFunctions03/Blends/Blend_Overlay",
}

def exp_mixrgb(node, exp_list):
	op = node.blend_type
	n = Node("FunctionCall", { "Function": op_map2[op]})
	exp_1 = get_expression(node.inputs['Color1'], exp_list)
	n.push(Node("0", {"expression": exp_1}))
	exp_2 = get_expression(node.inputs['Color2'], exp_list)
	n.push(Node("1", {"expression": exp_2}))

	exp_blend = exp_list.push(n)

	lerp = Node("LinearInterpolate")
	lerp.push(Node("0", {"expression": exp_1}))
	lerp.push(Node("1", {"expression": exp_blend}))
	exp_fac = get_expression(node.inputs['Fac'], exp_list)
	lerp.push(Node("2", {"expression": exp_fac}))

	return exp_list.push(lerp)

# TODO: this depends on having the material functions in UE4
def exp_hsv(node, exp_list):
	n = Node("FunctionCall", { "Function": "/BlenderDatasmithAdditions/BlenderAdditions/AdjustHSV"})
	exp_hue = get_expression(node.inputs['Hue'], exp_list)
	n.push(Node("0", {"expression": exp_hue}))
	exp_sat = get_expression(node.inputs['Saturation'], exp_list)
	n.push(Node("1", {"expression": exp_sat}))
	exp_value = get_expression(node.inputs['Value'], exp_list)
	n.push(Node("2", {"expression": exp_value}))
	exp_fac = get_expression(node.inputs['Fac'], exp_list)
	n.push(Node("3", {"expression": exp_fac}))
	exp_color = get_expression(node.inputs['Color'], exp_list)
	n.push(Node("4", {"expression": exp_color}))
	# TODO: test in unreal if I have an expression with two imputs which
	# one takes place, if it crashes or what
	return exp_list.push(n)



def get_expression(field, exp_list):
	if not field.links:
		if field.type == 'VALUE':
			return exp_list.push(exp_scalar(field.default_value))
		elif field.type == 'RGBA':
			return exp_list.push(exp_color(field.default_value))

	from_node = field.links[0].from_node
	if from_node.type == 'TEX_IMAGE':
		tex_coord = exp_list.push(exp_texcoord()) # TODO: maybe not even needed?
		image = from_node.image
		name = sanitize_name(image.name) # name_full?

		texture = UDScene.current_scene.get_field(UDTexture, name)
		texture.image = image

		texture_exp = exp_texture(name, tex_coord)
		return exp_list.push(texture_exp)
	elif from_node.type == 'MATH':
		return exp_math(from_node, exp_list)
	elif from_node.type == 'MIX_RGB':
		return exp_mixrgb(from_node, exp_list)
	elif from_node.type == 'HUE_SAT':
		return exp_hsv(from_node, exp_list)
	elif from_node.type == 'ATTRIBUTE':
		log.warn("unimplemented node ATTRIBUTE")
		return exp_list.push(Node("VertexColor"))
	elif from_node.type == 'RGB':
		return exp_list.push(exp_color(from_node.outputs[0].default_value))

	log.warn("node not handled" + from_node.type)
	return exp_list.push(exp_scalar(0))


def get_bsdf_expression(node, exp_list):
	expressions = {}
	if node.type == 'BSDF_PRINCIPLED':
		expressions["BaseColor"] = get_expression(node.inputs['Base Color'], exp_list)
		expressions["Metallic"] = get_expression(node.inputs['Metallic'], exp_list)
		expressions["Roughness"] = get_expression(node.inputs['Roughness'], exp_list)
	elif node.type == 'BSDF_DIFFUSE':
		expressions["BaseColor"] = get_expression(node.inputs['Color'], exp_list)
		expressions["Roughness"] = exp_list.push(exp_scalar(1.0))
	elif node.type == 'BSDF_GLOSSY':
		expressions["BaseColor"] = get_expression(node.inputs['Color'], exp_list)
		expressions["Roughness"] = get_expression(node.inputs['Roughness'], exp_list)
		expressions["Metallic"] = exp_list.push(exp_scalar(1.0))
	elif node.type == 'BSDF_VELVET':
		expressions["BaseColor"] = get_expression(node.inputs['Color'], exp_list)
		expressions["Roughness"] = exp_list.push(exp_scalar(1.0))
		log.warn("BSDF_VELVET incomplete implementation")

	elif node.type == 'ADD_SHADER':
		expressions = get_bsdf_expression(node.inputs[0].links[0].from_node, exp_list)
		expressions1 = get_bsdf_expression(node.inputs[1].links[0].from_node, exp_list)
		for name, exp in expressions1.items():
			if name in expressions:
				n = Node("Add")
				n.push(Node("0", {"expression": expressions[name]}))
				n.push(Node("1", {"expression": exp}))
				expressions[name] = exp_list.push(n)
			else:
				expressions[name] = exp
	else:
		log.warn("bsdf not handled" + node.type)

	return expressions



def pbr_nodetree_material(material):
	n = Node("UEPbrMaterial")
	n['name'] = sanitize_name(material.name)
	exp = Node("Expressions")
	n.push(exp)

	output_node = material.node_tree.get_output_node('EEVEE') # this could be 'CYCLES' or 'ALL'

	# this might need a big refactoring, to handle all the possible cases
	output_links = output_node.inputs['Surface'].links

	if not output_links:
		log.warn("material %s with use_nodes does not have nodes" % material.name)
		return n

	surface_node = output_links[0].from_node

	expressions = get_bsdf_expression(surface_node, exp)
	for key, value in expressions.items():
		n.push(Node(key, {
		"expression": value,
		"OutputIndex": "0"
		}))

	return n


def pbr_default_material():
	n = Node("UEPbrMaterial")
	n["name"] = "DefaultMaterial"
	exp = Node("Expressions")

	basecolor_idx = exp.push(exp_color((0.8, 0.8, 0.8, 1.0)))
	roughness_idx = exp.push(exp_scalar(0.5))
	n.push(Node("BaseColor", {
		"expression": basecolor_idx,
		"OutputIndex": "0"
		}))
	n.push(Node("Roughness", {
		"expression": roughness_idx,
		"OutputIndex": "0"
		}))
	return n

def pbr_basic_material(material):
	n = Node("UEPbrMaterial")
	n['name'] = sanitize_name(material.name)
	exp = Node("Expressions")
	n.push(exp)

	basecolor_idx = exp.push(exp_color(material.diffuse_color))
	roughness_idx = exp.push(exp_scalar(material.roughness))
	metallic_idx = exp.push(exp_scalar(material.metallic))
	specular_idx = exp.push(exp_scalar(material.specular_intensity))

	n.push(Node("BaseColor", {
		"expression": basecolor_idx,
		"OutputIndex": "0"
		}))
	n.push(Node("Roughness", {
		"expression": roughness_idx,
		"OutputIndex": "0"
		}))
	n.push(Node("Metallic", {
		"expression": metallic_idx,
		"OutputIndex": "0"
		}))
	n.push(Node("Specular", {
		"expression": specular_idx,
		"OutputIndex": "0"
		}))

	return n



def collect_pbr_material(material):
	if material is None:
		log.debug("creating default material")
		return pbr_default_material()
	if not material.use_nodes:
		log.debug("creating material %s without nodes" % material.name)
		return pbr_basic_material(material)
	log.debug("creating material %s with node_tree " % material.name)
	return pbr_nodetree_material(material)

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



	for idx, mat in enumerate(bl_mesh.materials):
	    umesh.materials[idx] = getattr(mat, 'name', 'DefaultMaterial')

	umesh.tris_material_slot = [p.material_index for p in m.polygons]
	umesh.tris_smoothing_group = [0 for p in m.polygons] # no smoothing groups for now

	umesh.vertices = [v.co.copy() for v in m.vertices]

	umesh.triangles = [l.vertex_index for l in m.loops]
	umesh.vertex_normals = [matrix_normals @ l.normal for l in m.loops] # don't know why, but copy is needed for this only
	umesh.uvs = [fix_uv(m.uv_layers[0].data[l.index].uv.copy()) for l in m.loops]
	if (m.vertex_colors):
		umesh.vertex_colors = [color_uchar(m.vertex_colors[0].data[l.index].color) for l in m.loops]
	bpy.data.meshes.remove(m)
	return umesh

def fix_uv(data):
	return (data[0], 1-data[1])

def color_uchar(data):
	return (
		int(data[0]*255),
		int(data[1]*255),
		int(data[2]*255),
		int(data[3]*255),
	)


def collect_object(bl_obj, uscene, context, dupli_matrix=None, name_override=None):

	uobj = None

	kwargs = {}
	kwargs['name'] = bl_obj.name
	if name_override:
		kwargs['name'] = name_override

	if bl_obj.type == 'MESH':

		uscene.materials |= {slot.material for slot in bl_obj.material_slots}

		bl_mesh = bl_obj.data
		if len(bl_mesh.polygons) > 0:
			uobj = UDActorMesh(**kwargs)
			umesh = collect_mesh(bl_obj.data, uscene)
			uobj.mesh = umesh.name
			for idx, slot in enumerate(bl_obj.material_slots):
				if slot.link == 'OBJECT':
					#collect_materials([slot.material], uscene)
					uobj.materials[idx] = sanitize_name(slot.material.name)

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
	log.info("collecting objects")
	for obj in root_objects:
		uobj = collect_object(obj, uscene, context=context)
		uscene.objects[uobj.name] = uobj
	log.info("collecting materials")
	uscene.material_nodes = [collect_pbr_material(mat) for mat in uscene.materials]


	return uscene

def save(operator, context,*, filepath, **kwargs):

	from os import path
	basepath, ext = path.splitext(filepath)
	basedir, basename = path.split(basepath)

	log.info("Starting collection of scene")
	scene = collect_to_uscene(bpy.context)
	log.info("finished collecting, now saving")
	scene.save(basedir, basename)

	UDScene.current_scene = None # FIXME

	return {'FINISHED'}

if __name__ == '__main__':
	save(operator=None, context=bpy.context, filepath='C:\\Users\\boterock\\Desktop\\export.datasmith')
