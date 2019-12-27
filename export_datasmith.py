# Copyright Andrés Botero 2019

import bpy
import bmesh
import math
import os
import time
from os import path
from .data_types import (
	UDActor, UDActorMesh,
	UDActorLight, UDActorCamera, UDMesh, Node, UDTexture, sanitize_name)
from mathutils import Matrix, Vector, Euler

import logging
log = logging.getLogger("bl_datasmith")

matrix_datasmith = Matrix.Scale(100, 4)
matrix_datasmith[1][1] *= -1.0

matrix_normals2 = Matrix.Scale(-1, 4)
#matrix_normals[1][1] = -1
matrix_normals = [
	[-1, 0, 0],
	[0, -1, 0],
	[0, 0, -1],
]

# used for lights and cameras, whose forward is (0, 0, -1) and its right is (1, 0, 0)
matrix_forward = Matrix((
	(0, 1, 0, 0),
	(0, 0, -1, 0),
	(-1, 0, 0, 0),
	(0, 0, 0, 1)
))

def exp_vector(value, exp_list):
	# n = Node("Color", {
	# nocheckin: may not work
	n = Node("Color", {
		# "Name": name,
		"constant": "(R=%.6f,G=%.6f,B=%.6f,A=1.0)"%tuple(value)
		})
	return exp_list.push(n)

def exp_color(value, exp_list, name=""):
	n = Node("Color", {
		"Name": name,
		"constant": "(R=%.6f,G=%.6f,B=%.6f,A=%.6f)"%tuple(value)
		})
	return exp_list.push(n)

def exp_scalar(value, exp_list):
	n = Node("Scalar", {
		"Name": "",
		"constant": "%f"%value
		})
	return exp_list.push(n)

def exp_texcoord(exp_list, index=0, u_tiling=1.0, v_tiling=1.0):
	n = Node("TextureCoordinate")
	n["Index"] = "0"
	n["UTiling"] = u_tiling
	n["VTiling"] = v_tiling
	return {"expression": exp_list.push(n)}

# instead of setting coordinates here, use coordinates when creating
# the texture expression instead
def exp_texture(path): # , tex_coord_exp):
	n = Node("Texture")
	n["Name"] = ""
	n["PathName"] = path
	#n.push(Node("Coordinates", tex_coord_exp))
	return n

# these map 1:1 with UE4 nodes:
op_map = {
	'ADD': "Add",
	'SUBTRACT': "Subtract",
	'MULTIPLY': "Multiply",
	'DIVIDE': "Divide",
	'POWER': "Power",
	'MINIMUM': "Min",
	'MAXIMUM': "Max",
	'MODULO': "Fmod",
	'ARCTAN2': "Arctangent2",
}

# these use only one input in UE4
op_map_one_input = {
	'SQRT': "SquareRoot",
	'ABSOLUTE': "Abs",
	'ROUND': "Round",
	'FLOOR': "Floor",
	'CEIL': "Ceil",
	'FRACT': "Frac",
	'SINE': "Sine",
	'COSINE': "Cosine",
	'TANGENT': "Tangent",
	'ARCSINE': "Arcsine",
	'ARCCOSINE': "Arccosine",
	'ARCTANGENT': "Arctangent",
}

# these require specific implementations:
op_map_custom = {
	'LOGARITHM', # ue4 only has log2 and log10
	'LESS_THAN', # use UE4 If node
	'GREATER_THAN', # use UE4 If node
}

def exp_math(node, exp_list):
	op = node.operation
	n = None
	if op in op_map:
		n = Node(op_map[op])
		exp_0 = get_expression(node.inputs[0], exp_list)
		n.push(Node("0", exp_0))
		exp_1 = get_expression(node.inputs[1], exp_list)
		n.push(Node("1", exp_1))
	elif op in op_map_one_input:
		n = Node(op_map_one_input[op])
		exp = get_expression(node.inputs[0], exp_list)
		n.push(Node("0", exp))
	elif op in op_map_custom:
		# all of these use two inputs
		in_0 = get_expression(node.inputs[0], exp_list)
		in_1 = get_expression(node.inputs[1], exp_list)
		if op == 'LOGARITHM': # take two logarithms and divide
			log0 = Node("Logarithm2")
			log0.push(Node("0", in_0))
			exp_0 = exp_list.push(log0)
			log1 = Node("Logarithm2")
			log1.push(Node("0", in_1))
			exp_1 = exp_list.push(log1)
			n = Node("Divide")
			n.push(Node("0", {"expression": exp_0}))
			n.push(Node("1", {"expression": exp_1}))
		elif op == 'LESS_THAN':
			n = Node("If")
			one = {"expression": exp_scalar(1.0, exp_list)}
			zero = {"expression": exp_scalar(0.0, exp_list)}
			n.push(Node("0", in_0)) # A
			n.push(Node("1", in_1)) # B
			n.push(Node("2", zero)) # A > B
			n.push(Node("3", one)) # A == B
			n.push(Node("4", one)) # A < B
		elif op == 'GREATER_THAN':
			n = Node("If")
			one = {"expression": exp_scalar(1.0, exp_list)}
			zero = {"expression": exp_scalar(0.0, exp_list)}
			n.push(Node("0", in_0)) # A
			n.push(Node("1", in_1)) # B
			n.push(Node("2", one)) # A > B
			n.push(Node("3", zero)) # A == B
			n.push(Node("4", zero)) # A < B

	assert n, "unrecognized math operation: %s" % op
	exp = exp_list.push(n)
	return {"expression": exp, "OutputIndex": 0}

op_map_color = {
# MIX is handled manually
	'DARKEN': "/Engine/Functions/Engine_MaterialFunctions03/Blends/Blend_Darken",
# MULTIPLY is handled in op_map
	'BURN': "/Engine/Functions/Engine_MaterialFunctions03/Blends/Blend_ColorBurn",
	# TODO: check for blender implementation of burn, it could mean this:
	#'BURN': "/Engine/Functions/Engine_MaterialFunctions03/Blends/Blend_LinearBurn",
	'LIGHTEN': "/Engine/Functions/Engine_MaterialFunctions03/Blends/Blend_Lighten",
	'SCREEN': "/Engine/Functions/Engine_MaterialFunctions03/Blends/Blend_Screen",
	'DODGE': "/Engine/Functions/Engine_MaterialFunctions03/Blends/Blend_ColorDodge",
	'OVERLAY': "/Engine/Functions/Engine_MaterialFunctions03/Blends/Blend_Overlay",
# ADD is handled in op_map
	'SOFT_LIGHT': "/Engine/Functions/Engine_MaterialFunctions03/Blends/Blend_SoftLight",
	'LINEAR_LIGHT': "/Engine/Functions/Engine_MaterialFunctions03/Blends/Blend_LinearLight",
	'DIFFERENCE': "/Engine/Functions/Engine_MaterialFunctions03/Blends/Blend_Difference",
# SUBTRACT is handled in op_map
# DIVIDE is handled in op_map
# HUE is unhandled TODO
# SATURATION is unhandled
# COLOR is unhandled
# VALUE is unhandled
}

def exp_blend(exp_0, exp_1, blend_type, exp_list):
	if blend_type == 'MIX':
		return exp_1
	n = None
	if blend_type in {'ADD', 'SUBTRACT', 'MULTIPLY', 'DIVIDE'}:
		n = Node(op_map[blend_type])
	else:
		n = Node("FunctionCall", { "Function": op_map_color[blend_type]})
	assert n
	n.push(Node("0", exp_0))
	n.push(Node("1", exp_1))
	return {"expression": exp_list.push(n)}



def exp_mixrgb(node, exp_list):
	exp_1 = get_expression(node.inputs['Color1'], exp_list)
	exp_2 = get_expression(node.inputs['Color2'], exp_list)

	exp_result = exp_blend(exp_1, exp_2, node.blend_type, exp_list)

	lerp = Node("LinearInterpolate")
	lerp.push(Node("0", exp_1))
	lerp.push(Node("1", exp_result))
	exp_fac = get_expression(node.inputs['Fac'], exp_list)
	lerp.push(Node("2", exp_fac))

	return exp_list.push(lerp)

op_custom_functions = {
	"HUE_SAT":   "/BlenderDatasmithAdditions/MaterialFunctions/AdjustHSV",
	"MAPPING_TEX2D": "/BlenderDatasmithAdditions/MaterialFunctions/MappingTexture2D",
	"MAPPING_POINT2D": "/BlenderDatasmithAdditions/MaterialFunctions/MappingPoint2D",
	"LAYER_WEIGHT": "/BlenderDatasmithAdditions/MaterialFunctions/LayerWeight",
	"CURVE_RGB": "/BlenderDatasmithAdditions/MaterialFunctions/RGBCurveLookup",
	"NORMAL_FROM_HEIGHT": "/Engine/Functions/Engine_MaterialFunctions03/Procedurals/NormalFromHeightmap",
	"FRESNEL": "/BlenderDatasmithAdditions/MaterialFunctions/BlenderFresnel",
}


# TODO: this depends on having the material functions in UE4
def exp_hsv(node, exp_list):
	n = Node("FunctionCall", { "Function": op_custom_functions["HUE_SAT"]})
	exp_hue = get_expression(node.inputs['Hue'], exp_list)
	n.push(Node("0", exp_hue))
	exp_sat = get_expression(node.inputs['Saturation'], exp_list)
	n.push(Node("1", exp_sat))
	exp_value = get_expression(node.inputs['Value'], exp_list)
	n.push(Node("2", exp_value))
	exp_fac = get_expression(node.inputs['Fac'], exp_list)
	n.push(Node("3", exp_fac))
	exp_color = get_expression(node.inputs['Color'], exp_list)
	n.push(Node("4", exp_color))
	# TODO: test in unreal if I have an expression with two inputs which
	# one takes place, if it crashes or what
	return exp_list.push(n)

def exp_invert(node, exp_list):
	n = Node("OneMinus")
	exp_color = get_expression(node.inputs['Color'], exp_list)
	n.push(Node("0", exp_color))
	invert_exp = exp_list.push(n)

	blend = Node("LinearInterpolate")
	exp_fac = get_expression(node.inputs['Fac'], exp_list)
	blend.push(Node("0", exp_color))
	blend.push(Node("1", {"expression": invert_exp}))
	blend.push(Node("2", exp_fac))

	return exp_list.push(blend)

def exp_mapping(node, exp_list):
	# TODO: cases for 'TEXTURE', 'POINT', 'VECTOR', 'NORMAL'
	n = None
	mapping_type = 'MAPPING_POINT2D'
	if node.vector_type == 'TEXTURE':
		mapping_type = 'MAPPING_TEX2D'
	n = Node("FunctionCall", { "Function": op_custom_functions[mapping_type]})

	input_vector = get_expression(node.inputs['Vector'], exp_list)
	input_location = get_expression(node.inputs['Location'], exp_list)
	input_rotation = get_expression(node.inputs['Rotation'], exp_list)
	input_scale = get_expression(node.inputs['Scale'], exp_list)
	n.push(Node("0", input_vector))
	n.push(Node("1", input_location))
	n.push(Node("2", input_rotation))
	n.push(Node("3", input_scale))

	return {"expression": exp_list.push(n)}

def exp_layer_weight(socket, exp_list):
	expr = None
	if socket.node in reverse_expressions:
		expr = reverse_expressions[socket.node]
	else:
		exp_blend = get_expression(socket.node.inputs['Blend'], exp_list)
		n = Node("FunctionCall", { "Function": op_custom_functions['LAYER_WEIGHT']})
		n.push(Node("0", exp_blend))
		expr = exp_list.push(n)
		reverse_expressions[socket.node] = expr

	log.warn("layer weight missing normal input")

	if socket.name == "Fresnel":
		return {"expression": expr, "OutputIndex": 0}
	elif socket.name == "Facing":
		return {"expression": expr, "OutputIndex": 1}
	log.error("LAYER_WEIGHT node from unknown socket")
	return {"expression": expr, "OutputIndex": 0}


def add_material_curve(curve):
	curves = datasmith_context["material_curves"]
	idx = len(curves)
	curves.append(curve)
	return idx

def exp_color_ramp(from_node, exp_list):
	ramp = from_node.color_ramp
	values = [ramp.evaluate(idx/255) for idx in range(256)]

	idx = add_material_curve(values)

	level = get_expression(from_node.inputs['Fac'], exp_list)

	curve_idx = exp_scalar(idx, exp_list)
	pixel_offset = exp_scalar(0.5, exp_list)
	vertical_res = exp_scalar(1/256, exp_list) # curves texture size
	n = Node("Add")
	n.push(Node("0", {"expression": curve_idx}))
	n.push(Node("1", {"expression": pixel_offset}))
	curve_y = exp_list.push(n)
	n2 = Node("Multiply")
	n2.push(Node("0", {"expression": curve_y}))
	n2.push(Node("1", {"expression": vertical_res}))
	curve_v = exp_list.push(n2)

	n3 = Node("AppendVector")
	n3.push(Node("0", level))
	n3.push(Node("1", {"expression": curve_v}))
	tex_coord = exp_list.push(n3)

	texture_exp = exp_texture("datasmith_curves")
	texture_exp.push(Node("Coordinates", {"expression":tex_coord}))

	return exp_list.push(texture_exp)

def exp_curvergb(from_node, exp_list):
	mapping = from_node.mapping
	mapping.initialize()
	curves = mapping.curves
	ev = lambda x : (
		curves[0].evaluate(x),
		curves[1].evaluate(x),
		curves[2].evaluate(x),
		curves[3].evaluate(x),
	) # the curves[3] is a global multiplier, not an alpha
	values = [ev(idx/255) for idx in range(256)]

	idx = add_material_curve(values)

	factor = get_expression(from_node.inputs['Fac'], exp_list)
	color = get_expression(from_node.inputs['Color'], exp_list)

	curve_idx = exp_scalar(idx, exp_list)
	pixel_offset = exp_scalar(0.5, exp_list)
	vertical_res = exp_scalar(1/256, exp_list) # curves texture size
	n = Node("Add")
	n.push(Node("0", {"expression": curve_idx}))
	n.push(Node("1", {"expression": pixel_offset}))
	curve_y = exp_list.push(n)
	n2 = Node("Multiply")
	n2.push(Node("0", {"expression": curve_y}))
	n2.push(Node("1", {"expression": vertical_res}))
	curve_v = exp_list.push(n2)

	texture = exp_texture_object("datasmith_curves", exp_list)

	lookup = Node("FunctionCall", { "Function": op_custom_functions["CURVE_RGB"]})
	lookup.push(Node("0", color))
	lookup.push(Node("1", {"expression": curve_v}))
	lookup.push(Node("2", {"expression": texture}))
	blend_exp = exp_list.push(lookup)


	blend = Node("LinearInterpolate")
	blend.push(Node("0", color))
	blend.push(Node("1", {"expression": blend_exp}))
	blend.push(Node("2", factor))
	result = exp_list.push(blend)

	return result

def exp_texture_object(name, exp_list):
	n = Node("TextureObject")
	n.push(Node("0", {
		"name": "Texture",
		"type": "Texture",
		"val": name,
	}))
	return exp_list.push(n)


def exp_bump(node, exp_list):
	height_input = node.inputs['Height']
	if height_input.links:
		from_node = height_input.links[0].from_node
		if from_node.type == 'TEX_IMAGE':
			image = from_node.image
			name = sanitize_name(image.name)

			# ensure that texture is exported
			texture = get_or_create_texture(name)
			texture.image = image

			image_object = exp_texture_object(name, exp_list)
			bump_node = Node("FunctionCall", { "Function": op_custom_functions["NORMAL_FROM_HEIGHT"]})
			bump_node.push(Node("0", {"expression": image_object}))
			bump_node.push(Node("1", get_expression(node.inputs['Strength'], exp_list)))
			bump_node.push(Node("2", get_expression(node.inputs['Distance'], exp_list)))
			bump_node.push(Node("3", get_expression(from_node.inputs['Vector'], exp_list)))
			exp = exp_list.push(bump_node)
			return {"expression": exp}

		else:
			log.warn("trying to export bump node, but input is not an image")
	else:
		log.warn("trying to export bump node without connections")


group_context = {}
def exp_group(socket, exp_list):
	node = socket.node
	global group_context
	global reverse_expressions
	previous_reverse = reverse_expressions
	reverse_expressions = {}
	previous_context = group_context
	group_context = {}
	for input in node.inputs:
		group_context[input.name] = get_expression(input, exp_list)

	# now traverse the inner graph
	output_name = socket.name

	node_tree_outputs = node.node_tree.nodes['Group Output'] # Should we rely on output nodes having the default name?
	inner_socket = node_tree_outputs.inputs[output_name]
	inner_exp = get_expression(inner_socket, exp_list)

	group_context_dict = previous_context
	reverse_expressions = previous_reverse
	return inner_exp

def exp_group_input(socket, exp_list):
	outer_expression = group_context[socket.name]
	return outer_expression

def exp_fresnel(node, exp_list):
	n = Node("FunctionCall", { "Function": op_custom_functions["FRESNEL"]})
	exp_ior = get_expression(node.inputs['IOR'], exp_list)
	n.push(Node("0", exp_ior))
	return exp_list.push(n)


context_stack = []
def push_context(context):
	context_stack.append(context)

def pop_context():
	context_stack.pop()

def get_context():
	if context_stack:
		return context_stack[-1]


expression_log_prefix = ""
def get_expression(field, exp_list):
	# this may return none for fields without default value
	# most of the time blender doesn't have default value for vector
	# node inputs, but it does for scalars and colors
	# TODO: check which cases we should be careful
	global expression_log_prefix
	log.debug(expression_log_prefix + "found field:"+field.node.name+"/"+field.name+":"+field.type)

	if not field.links:
		if field.type == 'VALUE':
			exp = exp_scalar(field.default_value, exp_list)
			return {"expression": exp, "OutputIndex": 0}
		elif field.type == 'RGBA':
			exp = exp_color(field.default_value, exp_list)
			return {"expression": exp, "OutputIndex": 0}
		elif field.type == 'VECTOR':
			if type(field.default_value) is Vector:
				exp = exp_vector(field.default_value, exp_list)
				return {"expression": exp, "OutputIndex": 0}
			if type(field.default_value) is Euler:
				exp = exp_vector(field.default_value, exp_list)
				return {"expression": exp, "OutputIndex": 0}
		elif field.type == 'SHADER':
			# same as holdout shader
			bsdf = {
				"BaseColor": {"expression": exp_scalar(0.0, exp_list)},
				"Roughness": {"expression": exp_scalar(1.0, exp_list)},
			}
			return bsdf
		log.debug("field has no links, and no default value " + str(field))
		return None

	prev_prefix = expression_log_prefix
	expression_log_prefix += "|   "
	return_exp = get_expression_inner(field, exp_list)
	expression_log_prefix = prev_prefix

	# if a color output is connected to a scalar input, average by using dot product
	if field.type == 'VALUE':
		other_output = field.links[0].from_socket
		if other_output.type == 'RGBA' or other_output.type == 'VECTOR':
			#TODO: check if we should do some colorimetric aware convertion to grayscale
			n = Node("DotProduct")
			exp_0 = return_exp
			n.push(Node("0", exp_0))
			exp_1 = exp_vector((0.333333, 0.333333, 0.333333), exp_list)
			n.push(Node("1", {"expression": exp_1}))
			dot_exp = exp_list.push(n)
			return_exp = {"expression": dot_exp}

	socket = field.links[0].from_socket
	reverse_expressions[socket] = return_exp
	return return_exp

def get_expression_inner(field, exp_list):

	node = field.links[0].from_node
	socket = field.links[0].from_socket
	log.debug("getting node:"+node.name+" output:"+socket.name)
	# if this node is already exported, connect to that instead
	# I am considering in
	if socket in reverse_expressions:
		return reverse_expressions[socket]

	# The cases are ordered like in blender Add menu, but shaders are first
	# TODO: all the shaders are missing normal maps

	# Shader nodes return a dictionary
	bsdf = None
	if node.type == 'BSDF_PRINCIPLED':
		bsdf = {
			"BaseColor": get_expression(node.inputs['Base Color'], exp_list),
			"Metallic": get_expression(node.inputs['Metallic'], exp_list),
			"Roughness": get_expression(node.inputs['Roughness'], exp_list),
		}

		# only add opacity if transmission != 0
		transmission_field = node.inputs['Transmission']
		add_transmission = False
		if len(transmission_field.links) != 0:
			add_transmission = True
		elif transmission_field.default_value != 0:
			add_transmission = True
		if add_transmission:
			n = Node("OneMinus")
			exp_transmission = get_expression(node.inputs['Transmission'], exp_list)
			n.push(Node("0", exp_transmission))
			exp_opacity = {"expression": exp_list.push(n)}
			bsdf['Opacity'] = exp_opacity


	elif node.type == 'BSDF_DIFFUSE':
		bsdf = {
			"BaseColor": get_expression(node.inputs['Color'], exp_list),
			"Roughness": {"expression": exp_scalar(1.0, exp_list)},
			"Metallic": {"expression": exp_scalar(0.0, exp_list)},
		}
	elif node.type == 'BSDF_TOON':
		log.warn("BSDF_TOON incomplete implementation")
		bsdf = {
			"BaseColor": get_expression(node.inputs['Color'], exp_list),
			"Roughness": {"expression": exp_scalar(1.0, exp_list)},
			"Metallic": {"expression": exp_scalar(0.0, exp_list)},
		}
	elif node.type == 'BSDF_GLOSSY':
		bsdf = {
			"BaseColor": get_expression(node.inputs['Color'], exp_list),
			"Roughness": get_expression(node.inputs['Roughness'], exp_list),
			"Metallic": {"expression": exp_scalar(1.0, exp_list)},
		}
	elif node.type == 'BSDF_VELVET':
		log.warn("BSDF_VELVET incomplete implementation")
		bsdf = {
			"BaseColor": get_expression(node.inputs['Color'], exp_list),
			"Roughness": {"expression": exp_scalar(1.0, exp_list)},
		}
	elif node.type == 'BSDF_TRANSPARENT':
		log.warn("BSDF_TRANSPARENT incomplete implementation")
		bsdf = {
			"BaseColor": get_expression(node.inputs['Color'], exp_list),
			"Refraction": {"expression": exp_scalar(1.0, exp_list)},
			"Opacity": {"expression": exp_scalar(0.0, exp_list)},
		}
	elif node.type == 'BSDF_TRANSLUCENT':
		log.warn("BSDF_TRANSLUCENT incomplete implementation")
		bsdf = {
			"BaseColor": get_expression(node.inputs['Color'], exp_list),
		}
	elif node.type == 'BSDF_GLASS':
		log.warn("BSDF_GLASS incomplete implementation")
		bsdf = {
			"BaseColor": get_expression(node.inputs['Color'], exp_list),
			"Roughness": get_expression(node.inputs['Roughness'], exp_list),
			"Refraction": get_expression(node.inputs['IOR'], exp_list),
			"Opacity": {"expression": exp_scalar(0.5, exp_list)},
		}
	elif node.type == 'BSDF_HAIR':
		log.warn("BSDF_HAIR incomplete implementation")
		bsdf = {
			"BaseColor": get_expression(node.inputs['Color'], exp_list),
			"Roughness": {"expression": exp_scalar(0.5, exp_list)},
		}
	elif node.type == 'SUBSURFACE_SCATTERING':
		log.warn("node SUBSURFACE_SCATTERING incomplete implementation")
		bsdf = {
			"BaseColor": get_expression(node.inputs['Color'], exp_list)
		}

	if bsdf:
		if "Normal" in node.inputs:
			normal_expression = get_expression(node.inputs['Normal'], exp_list)
			if normal_expression:
				bsdf["Normal"] = normal_expression
		return bsdf

	if node.type == 'EMISSION':
		mult = Node("Multiply")
		mult.push(Node("0", get_expression(node.inputs['Color'], exp_list)))
		mult.push(Node("1", get_expression(node.inputs['Strength'], exp_list)))
		mult_exp = exp_list.push(mult)
		return {
			"EmissiveColor": {"expression": mult_exp}
		}

	if node.type == 'HOLDOUT':
		return {
			"BaseColor": {"expression": exp_scalar(0.0, exp_list)},
			"Roughness": {"expression": exp_scalar(1.0, exp_list)},
		}

	if node.type == 'ADD_SHADER':
		expressions = get_expression(node.inputs[0], exp_list)
		expressions1 = get_expression(node.inputs[1], exp_list)
		for name, exp in expressions1.items():
			if name in expressions:
				n = Node("Add")
				n.push(Node("0", expressions[name]))
				n.push(Node("1", exp))
				expressions[name] = {"expression":exp_list.push(n)}
			else:
				expressions[name] = exp
		return expressions
	if node.type == 'MIX_SHADER':
		expressions = get_expression(node.inputs[1], exp_list)
		expressions1 = get_expression(node.inputs[2], exp_list)
		if ("Opacity" in expressions) or ("Opacity" in expressions1):
			# if there is opacity in any, both should have opacity
			if "Opacity" not in expressions:
				expressions["Opacity"] = {"expression": exp_scalar(1, exp_list)}
			if "Opacity" not in expressions1:
				expressions1["Opacity"] = {"expression": exp_scalar(1, exp_list)}
		fac_expression = get_expression(node.inputs['Fac'], exp_list)
		for name, exp in expressions1.items():
			if name in expressions:
				n = Node("LinearInterpolate")
				n.push(Node("0", expressions[name]))
				n.push(Node("1", exp))
				n.push(Node("2", fac_expression))
				expressions[name] = {"expression":exp_list.push(n)}
			else:
				expressions[name] = exp
		return expressions


	# from here the return type should be {expression:node_idx, OutputIndex: socket_idx}
	# Add > Input

	# if node.type == 'AMBIENT_OCCLUSION':
	if node.type == 'ATTRIBUTE':
		log.warn("incomplete node ATTRIBUTE")
		exp = exp_list.push(Node("VertexColor"))
		return {"expression": exp, "OutputIndex": 0}
	# if node.type == 'BEVEL':
	# if node.type == 'CAMERA':
	if node.type == 'FRESNEL':
		exp = exp_fresnel(node, exp_list)
		return {"expression": exp}
	# if node.type == 'NEW_GEOMETRY':
	# if node.type == 'HAIR_INFO':
	if node.type == 'LAYER_WEIGHT': # fresnel and facing, with "blend" (power?) and normal param
		return exp_layer_weight(socket, exp_list)
	# if node.type == 'LIGHT_PATH':
	# if node.type == 'OBJECT_INFO':
	# if node.type == 'PARTICLE_INFO':

	if node.type == 'RGB':
		exp = exp_color(node.outputs[0].default_value, exp_list)
		return {"expression": exp, "OutputIndex": 0}

	# if node.type == 'TANGENT':
	if node.type == 'TEX_COORD':
		if socket.name == 'UV':
			return exp_texcoord(exp_list)
		else:
			log.warn("found node texcoord with name:"+socket.name)

	# if node.type == 'UVMAP':
	# if node.type == 'VALUE':
	if node.type == 'VALUE':
		exp = exp_scalar(node.outputs[0].default_value, exp_list)
		return {"expression": exp}
	# if node.type == 'WIREFRAME':


	# Add > Texture
	# if node.type == 'TEX_BRICK':
	# if node.type == 'TEX_CHECKER':
	# if node.type == 'TEX_ENVIRONMENT':
	# if node.type == 'TEX_GRADIENT':
	# if node.type == 'TEX_IES':
	if node.type == 'TEX_IMAGE':
		cached_node = None
		if node in reverse_expressions:
			cached_node = reverse_expressions[node]

		if not cached_node:

			tex_coord = get_expression(node.inputs['Vector'], exp_list)
			image = node.image
			name = sanitize_name(image.name) # name_full?

			# ensure that texture is exported
			texture = get_or_create_texture(name)
			if (get_context() == 'NORMAL'):
				texture.normal_map_flag = True

			texture.image = image

			texture_exp = exp_texture(name)
			if tex_coord:
				texture_exp.push(Node("Coordinates", tex_coord))
			cached_node = exp_list.push(texture_exp)
			reverse_expressions[node] = cached_node

		output_index = 0 # RGB
		# indices 1, 2, 3 are separate RGB channels
		if socket.name == 'Alpha':
			output_index = 4 #

		return { "expression": cached_node, "OutputIndex": output_index }


	# Add > Color
	if node.type == 'BRIGHTCONTRAST':
		log.warn("unimplemented node BRIGHTCONTRAST")
		return get_expression(node.inputs['Color'], exp_list)
	# if node.type == 'GAMMA':
	if node.type == 'HUE_SAT':
		exp = exp_hsv(node, exp_list)
		return {"expression": exp, "OutputIndex": 0}

	if node.type == 'INVERT':
		exp = exp_invert(node, exp_list)
		return {"expression": exp}
	# if node.type == 'LIGHT_FALLOFF':
	# if node.type == 'TEX_CHECKER':
	# if node.type == 'TEX_CHECKER':

	if node.type == 'MIX_RGB':
		exp = exp_mixrgb(node, exp_list)
		return {"expression": exp, "OutputIndex": 0}

	if node.type == 'CURVE_RGB':
		exp = exp_curvergb(node, exp_list)
		return {"expression": exp, "OutputIndex": 0}

	# Add > Vector

	if node.type == 'BUMP':
		return exp_bump(node, exp_list)
	# if node.type == 'DISPLACEMENT':
	if node.type == 'MAPPING':
		return exp_mapping(node, exp_list)
	# if node.type == 'NORMAL':
	if node.type == 'NORMAL_MAP':
		node_input = node.inputs['Color']
		# hack: is it safe to assume that everything under here is normal?
		# maybe not, because it could be masks to mix normals
		push_context("NORMAL")
		return_exp = get_expression(node_input, exp_list)
		pop_context()
		return return_exp
	# if node.type == 'CURVE_VEC':
	# if node.type == 'VECTOR_DISPLACEMENT':
	# if node.type == 'VECT_TRANSFORM':

	# Add > Converter

	# if node.type == 'BLACKBODY':
	if node.type == 'VALTORGB':
		exp = exp_color_ramp(node, exp_list)
		return {"expression": exp, "OutputIndex": 0}

	# if node.type == 'COMBHSV':
	# if node.type == 'COMBRGB':
	# if node.type == 'COMBXYZ':
	if node.type == 'MATH':
		exp = exp_math(node, exp_list)
		return exp

	# if node.type == 'RGBTOBW':
	# if node.type == 'SEPHSV':
	# if node.type == 'SEPRGB':
	# if node.type == 'SEPXYZ':
	# if node.type == 'SHADERTORGB':
	# if node.type == 'VECT_MATH':
	# if node.type == 'WAVELENGTH':

	# Others:

	# if node.type == 'SCRIPT':
	if node.type == 'GROUP':
		# exp = exp_group(node, exp_list)
		# as exp_group can output shaders (dicts with basecolor/roughness)
		# or other types of values (dicts with expression:)
		# it may be better to return as is and handle internally
		return exp_group(socket, exp_list)# TODO node trees can have multiple outputs

	if node.type == 'GROUP_INPUT':
		return exp_group_input(socket, exp_list)

	if node.type == 'REROUTE':
		return get_expression(node.inputs['Input'], exp_list)

	log.error("node not handled" + node.type)
	exp = exp_scalar(0, exp_list)
	return {"expression": exp}


def pbr_nodetree_material(material):
	log.debug("collecting material:"+material.name)
	n = Node("UEPbrMaterial")
	n['name'] = sanitize_name(material.name)
	exp_list = Node("Expressions")
	n.push(exp_list)

	output_node = (
		material.node_tree.get_output_node('EEVEE')
		or material.node_tree.get_output_node('ALL')
		or material.node_tree.get_output_node('CYCLES')
	)

	global reverse_expressions
	reverse_expressions = dict()

	surface_field = output_node.inputs['Surface']
	if not surface_field.links:
		log.warn("material %s with use_nodes does not have nodes" % material.name)
		return n

	expressions = get_expression(surface_field, exp_list)
	for key, value in expressions.items():
		n.push(Node(key, value))

	# apparently this happens automatically, we may want to
	# choose if we export with masked blend mode
	#if "Opacity" in expressions:
	#	n.push(Node("Blendmode", {"value": "2.0"}))

	return n


def pbr_default_material():
	n = Node("UEPbrMaterial")
	n["name"] = "DefaultMaterial"
	exp_list = Node("Expressions")

	basecolor_idx = exp_color((0.8, 0.8, 0.8, 1.0), exp_list)
	roughness_idx = exp_scalar(0.5, exp_list)
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
	exp_list = Node("Expressions")
	n.push(exp_list)

	basecolor_idx = exp_color(material.diffuse_color, exp_list)
	roughness_idx = exp_scalar(material.roughness, exp_list)
	metallic_idx = exp_scalar(material.metallic, exp_list)
	specular_idx = exp_scalar(material.specular_intensity, exp_list)

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

import numpy as np

def mesh_copy_triangulated(bl_mesh):
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
	return m



def collect_mesh(bl_mesh):

	# when using linked libraries, some meshes can have the same name
	mesh_name = sanitize_name(bl_mesh.name)
	if bl_mesh.library:
		#import pdb; pdb.set_trace()
		lib_path = bpy.path.clean_name(bl_mesh.library.filepath)
		prefix = lib_path.strip('_')
		if prefix.endswith('_blend'):
			prefix = prefix[:-6] + '_'
		mesh_name = prefix + mesh_name

	umesh = get_or_create_mesh(mesh_name)
	if len(umesh.vertices) > 0:
		# mesh already processed
		return umesh
	# create copy to triangulate

	m = mesh_copy_triangulated(bl_mesh)

	for idx, mat in enumerate(bl_mesh.materials):
	    umesh.materials[idx] = sanitize_name(getattr(mat, 'name', 'DefaultMaterial'))

	polygons = m.polygons
	num_polygons = len(polygons)
	material_slots = np.empty(num_polygons, np.uint32)

	polygons.foreach_get("material_index", material_slots)
	umesh.tris_material_slot = material_slots # [p.material_index for p in m.polygons]

	umesh.tris_smoothing_group = np.zeros(num_polygons, np.uint32)

	vertices = m.vertices
	num_vertices = len(vertices)

	vertices_array = np.empty(num_vertices* 3, np.float32)
	vertices.foreach_get("co", vertices_array)

	umesh.vertices = vertices_array.reshape(-1, 3)

	loops = m.loops
	num_loops = len(loops)

	triangles = np.empty(num_loops, np.uint32)
	loops.foreach_get("vertex_index", triangles)

	umesh.triangles = triangles

	normals = np.empty(num_loops* 3, np.float32)
	loops.foreach_get("normal", normals)
	normals = normals.reshape((num_loops, 3))
	normals = normals @ matrix_normals

	umesh.vertex_normals = np.ascontiguousarray(normals, "<f4")

	uvs = np.empty(num_loops* 2, np.float32)
	uv_data = m.uv_layers[0].data

	uv_data.foreach_get("uv", uvs)
	uvs[1::2] = 1 - uvs[1::2]
	umesh.uvs = uvs.reshape((-1, 2))

	if (m.vertex_colors):
		vertex_colors = np.empty(num_loops * 4)
		m.vertex_colors[0].data.foreach_get("color", vertex_colors)
		vertex_colors *= 255
		vertex_colors = vertex_colors.reshape((-1, 4))
		vertex_colors[:, [0, 2]] = vertex_colors[:, [2, 0]]
		umesh.vertex_colors = vertex_colors.astype(np.uint8)

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


def collect_object(bl_obj, dupli_matrix=None, name_override=None):

	uobj = None

	kwargs = {}
	kwargs['name'] = bl_obj.name
	if name_override:
		kwargs['name'] = name_override

	if bl_obj.type == 'MESH':
		material_list = datasmith_context["materials"]
		for slot in bl_obj.material_slots:
			material_list.append(slot.material)

		bl_mesh = bl_obj.data
		if len(bl_mesh.polygons) > 0:
			uobj = UDActorMesh(**kwargs)
			umesh = collect_mesh(bl_obj.data)
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
		uobj.sensor_width = bl_cam.sensor_width
		# blender does not have aspect ratio for cameras
		# uobj.sensor_aspect_ratio = 1.777778
		uobj.enable_dof = bl_cam.dof.use_dof
		if uobj.enable_dof:
			uobj.focus_distance = bl_cam.dof.focus_distance * 100 # to centimeters
			if bl_cam.dof.focus_object:
				# TODO test this, I don't know if this look_at_actor
				# is for focus or for rotating the camera.
				uobj.look_at_actor = sanitize_name(bl_cam.dof.focus_object.name)
			uobj.f_stop = bl_cam.dof.aperture_fstop

		uobj.focal_length = bl_cam.lens

	elif bl_obj.type == 'LIGHT':
		uobj = UDActorLight(**kwargs)
		bl_light = bl_obj.data

		if bl_light.use_nodes and bl_light.node_tree:

			node = bl_light.node_tree.nodes['Emission']
			uobj.color = node.inputs['Color'].default_value
			uobj.intensity = node.inputs['Strength'].default_value # have to check how to relate to candelas
			log.warn("unsupported: using nodetree for light " + bl_obj.name)
		else:
			uobj.color = bl_light.color
			uobj.intensity_units = UDActorLight.LIGHT_UNIT_LUMENS
			uobj.intensity = bl_light.energy
			if bl_light.type == 'SPOT':
				uobj.intensity_units = UDActorLight.LIGHT_UNIT_CANDELAS

				uobj.intensity = bl_light.energy * 0.08 # came up with this constant by brute force
				# blender watts unit match ue4 lumens unit, but in spot lights the brightness
				# changes with the spot angle when using lumens while candelas do not.
				# TODO: put this behind a flag, and maybe docs


		# this seems to give a good result
		uobj.attenuation_radius = 20 * math.pow(bl_light.energy, 0.666666)

		if bl_light.type == 'SUN':
			uobj.intensity = bl_light.energy # suns are in lux
			uobj.type = UDActorLight.LIGHT_SUN
		elif bl_light.type == 'AREA':
			uobj.type = UDActorLight.LIGHT_AREA


			size_w = size_h = bl_light.size
			if (bl_light.shape == 'RECTANGLE'
				or bl_light.shape == 'ELLIPSE'):
				size_h = bl_light.size_y



			# light_shape fills the light with geometry, so better set none instead
			light_shape = 'None'
			# light_shape = 'Rectangle'
			# if (bl_light.shape == 'DISK'
			# 	or bl_light.shape == 'ELLIPSE'):
			# 	light_shape = 'Disc'



			uobj.shape = Node('Shape', {
				"type": light_shape, # can be Rectangle, Disc, Sphere, Cylinder, None
				"width": size_w * 100, # convert to cm
				"length": size_h * 100,
				"LightType": "Rect", # can be "Point", "Spot", "Rect"
				})

		uobj.node_props.append(
			Node('SourceSize', {
				"value": bl_light.shadow_soft_size * 100,
				}
			)
		)

		if bl_light.type == 'POINT':
			uobj.type = UDActorLight.LIGHT_POINT
		elif bl_light.type == 'SPOT':
			uobj.type = UDActorLight.LIGHT_SPOT
			angle = bl_light.spot_size * 180 / (2*math.pi)
			uobj.outer_cone_angle = angle
			inner_angle = angle * (1 - bl_light.spot_blend)
			uobj.inner_cone_angle = inner_angle if inner_angle > 0.0001 else 0.0001

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
			new_obj = collect_object(dup, dupli_matrix=bl_obj.matrix_world, name_override=dupli_name)
			uobj.objects[new_obj.name] = new_obj

	if dupli_matrix is None: # this was for blender 2.7, maybe 2.8 works without this
		for child in bl_obj.children:
			new_obj = collect_object(child)
			uobj.objects[new_obj.name] = new_obj

	return uobj

def collect_environment(world):

	log.info("Collecting environment")
	if not world.use_nodes:
		return
	log.info("Collecting environment")
	nodes = world.node_tree
	output = nodes.get_output_node('EEVEE') or nodes.get_output_node('ALL') or nodes.get_output_node('CYCLES')
	background_node = output.inputs['Surface'].links[0].from_node
	if not background_node.inputs['Color'].links:
		return
	source_node = background_node.inputs['Color'].links[0].from_node
	if source_node.type != 'TEX_ENVIRONMENT':
		log.info("Background texture is "+ source_node.type)
		return
	log.info("Collecting environment")
	image = source_node.image

	tex_name = sanitize_name(image.name)
	texture = get_or_create_texture(tex_name)
	texture.image = image

	tex_node = Node("Texture", {
		"tex": tex_name,
		})

	n2 = Node("Environment", {
		"name": "world_environment_lighting",
		"label": "world_environment_lighting",
		})
	n2.push(tex_node)
	n2.push(Node("Illuminate", {
		"enabled": "1"
		}))
	n = Node("Environment", {
		"name": "world_environment_background",
		"label": "world_environment_background",
		})
	n.push(tex_node)
	n.push(Node("Illuminate", {
		"enabled": "0"
		}))

	return [n, n2]



def get_file_header():

	n = Node('DatasmithUnrealScene')
	n.push(Node('Version', children=['0.22']))
	n.push(Node('SDKVersion', children=['4.22E0']))
	n.push(Node('Host', children=['Blender']))
	n.push(Node('Application', {
		'Vendor': 'Blender',
		'ProductName': 'Blender',
		'ProductVersion': '2.80',
		}))

	n.push(Node('User', {
		'ID': '00000000000000000000000000000000',
		'OS': 'Windows 8.1',
		}))
	return n


def get_or_create_texture(name):
	textures = datasmith_context["textures"]
	for tex in textures:
		if name == tex.name:
			return tex
	new_tex = UDTexture(name)
	textures.append(new_tex)
	return new_tex

def get_or_create_mesh(name):
	meshes = datasmith_context["meshes"]
	for mesh in meshes:
		if name == mesh.name:
			return mesh
	new_mesh = UDMesh(name)
	meshes.append(new_mesh)
	return new_mesh

datasmith_context = None
def collect_and_save(context, args, save_path):

	start_time = time.monotonic()

	global datasmith_context
	datasmith_context = {
		"objects": [],
		"textures": [],
		"meshes": [],
		"materials": [],
		"material_curves": [],
	}

	all_objects = context.scene.objects
	root_objects = [obj for obj in all_objects if obj.parent is None]

	objects = []
	log.info("collecting objects")
	for obj in root_objects:
		uobj = collect_object(obj)
		objects.append(uobj)

	log.info("collecting materials")
	materials = datasmith_context["materials"]
	unique_materials = []
	for material in materials:
		if material in unique_materials:
			continue
		unique_materials.append(material)
	material_nodes = [collect_pbr_material(mat) for mat in unique_materials]

	log.info("baking curves")
	curves_image = None
	if "datasmith_curves" in bpy.data.images:
		curves_image = bpy.data.images["datasmith_curves"]
	else:
		curves_image = bpy.data.images.new("datasmith_curves", 256, 256, alpha=True, float_buffer=True)
		curves_image.colorspace_settings.is_data = True

	pixels = curves_image.pixels
	curve_list = datasmith_context["material_curves"]
	log.info("curves: "+str(len(curve_list)))
	for idx, curve in enumerate(curve_list):
		row_idx = (255-idx) * 256
		for i in range(256):
			pixel_idx = (row_idx + i) * 4
			pixels[pixel_idx] = curve[i][0]
			pixels[pixel_idx+1] = curve[i][1]
			pixels[pixel_idx+2] = curve[i][2]
			pixels[pixel_idx+3] = curve[i][3]

	texture = get_or_create_texture("datasmith_curves")
	texture.image = curves_image

	log.info("finished collecting, now saving")

	basedir, file_name = path.split(save_path)
	folder_name = file_name + '_Assets'
	# make sure basepath_Assets directory exists
	try:
		os.makedirs(path.join(basedir, folder_name))
	except FileExistsError as e:
		pass

	log.info("writing meshes")
	for mesh in datasmith_context["meshes"]:
		mesh.save(basedir, folder_name)

	log.info("writing textures")
	for tex in datasmith_context["textures"]:
		tex.save(basedir, folder_name)

	log.info("building XML tree")

	n = get_file_header()
	for obj in objects:
		n.push(obj.node())

	environment = collect_environment(context.scene.world)
	if environment:
		for env in environment:
			n.push(env)

	for mesh in datasmith_context["meshes"]:
		n.push(mesh.node())
	for mat in material_nodes:
		n.push(mat)

	use_experimental_tex_mode = args["experimental_tex_mode"]
	print("Using experimental tex mode:%s", use_experimental_tex_mode)
	for tex in datasmith_context["textures"]:
		n.push(tex.node(folder_name, use_experimental_tex_mode))

	end_time = time.monotonic()
	total_time = end_time - start_time

	log.info("preparing data took:%f"%total_time)

	result = n.string_rep(first=True)

	log.info("writing to file")
	filename = path.join(basedir, file_name + '.udatasmith')
	with open(filename, 'w') as f:
		f.write(result)
	log.info("export successful")



def save(context,*, filepath, **kwargs):

	use_logging = False
	handler = None
	if "use_logging" in kwargs:
		use_logging = bool(kwargs["use_logging"])

	if use_logging:
		handler = logging.FileHandler(filepath + ".log", mode='w')
		log.addHandler(handler)

	try:
		from os import path
		basepath, ext = path.splitext(filepath)

		log.info("Starting Datasmith Export")
		collect_and_save(context, kwargs, basepath)
		log.info("Finished Datasmith Export")

	except Exception as error:
		log.error("Datasmith export error:")
		log.error(error)
		raise

	finally:
		if use_logging:
			handler.close()
			log.removeHandler(handler)

	return {'FINISHED'}

