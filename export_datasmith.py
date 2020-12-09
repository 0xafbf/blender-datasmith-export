# Copyright Andrés Botero 2019

import bpy
import idprop
import bmesh
import math
import os
import time
import hashlib
import shutil
from os import path
from .data_types import UDMesh, Node, sanitize_name
from mathutils import Matrix, Vector, Euler

import logging
log = logging.getLogger("bl_datasmith")

matrix_datasmith = Matrix.Scale(100, 4)
matrix_datasmith[1][1] *= -1.0

matrix_normals = [
	[1, 0, 0],
	[0, -1, 0],
	[0, 0, 1],
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

def exp_color(value, exp_list, name=None):
	n = Node("Color", {
		"constant": "(R=%.6f,G=%.6f,B=%.6f,A=%.6f)"%tuple(value)
		})
	if name:
		n["Name"] = name
	return exp_list.push(n)

def exp_scalar(value, exp_list):
	n = Node("Scalar", {
		# "Name": "",
		"constant": "%f"%value
		})
	return exp_list.push(n)

def exp_texcoord(exp_list, index=0, u_tiling=1.0, v_tiling=1.0):
	n = Node("TextureCoordinate")
	n["Index"] = index
	n["UTiling"] = u_tiling
	n["VTiling"] = v_tiling

	pad = Node("AppendVector")
	pad.push(Node("0", {"expression": exp_list.push(n) }))
	zero = exp_scalar(0, exp_list)
	pad.push(Node("1", {"expression": zero }))
	return {"expression": exp_list.push(pad) }

def exp_texcoord_node(socket, exp_list):
	socket_name = socket.name
	if socket_name == "Generated":
		n = Node("FunctionCall", { "Function": "/Engine/Functions/Engine_MaterialFunctions02/UVs/BoundingBoxBased_0-1_UVW"})
		return { "expression": exp_list.push(n) }
	# if socket_name == "Normal":
	if socket_name == "UV":
		return exp_texcoord(exp_list)
	if socket_name == "Object":
		n = Node("FunctionCall", { "Function": op_custom_functions["LOCAL_POSITION"]})
		return { "expression": exp_list.push(n) }
	# if socket_name == "Camera":
	#	`position from camera in camera space (blue is camera forward, green is camera up`
	# if socket_name == "Window":
	#	seems to be viewport coordinates
	# if socket_name == "Reflection":
	#	direction of reflection in world coordinates
	log.warn("Texcoord node doesn't implement %s yet" % socket_name)

def exp_tex_noise(socket, exp_list):

	# default if socket.name == "Fac"
	function_path = "/DatasmithBlenderContent/MaterialFunctions/TexNoise"
	out_socket = 0
	if socket.name == "Color":
		out_socket = 1
	n = Node("FunctionCall", { "Function": function_path})
	exp_1 = get_expression(socket.node.inputs['Vector'], exp_list)
	exp_2 = get_expression(socket.node.inputs['Scale'], exp_list)
	if exp_1:
		n.push(Node("0", exp_1))
	n.push(Node("1", exp_2))
	return { "expression": exp_list.push(n), "OutputIndex":out_socket }


def exp_tex_checker(socket, exp_list):
	if socket.node in cached_nodes:
		exp = { "expression": cached_nodes[socket.node] }
	else:
		exp = exp_function_call(
			"/DatasmithBlenderContent/MaterialFunctions/TexChecker",
			exp_list=exp_list,
			inputs=socket.node.inputs,
			force_default=True,
		)
		cached_nodes[socket.node] = exp["expression"]

	# could be faster by comparing to constants instead?
	exp["OutputIndex"] = socket.node.outputs.find(socket.name)

	return exp


def exp_uvmap(node, exp_list):
	channel_name = node.uv_map
	owner = datasmith_context["material_owner"]
	uv_index = 0
	m = owner.data
	if type(m) is bpy.types.Mesh:
		for idx, uv in enumerate(m.uv_layers):
			if uv.name == id:
				uv_index = idx
	return exp_texcoord(exp_list, uv_index)

# instead of setting coordinates here, use coordinates when creating
# the texture expression instead
def exp_texture(path, name=None): # , tex_coord_exp):
	n = Node("Texture")
	if name:
		n["Name"] = name
	n["PathName"] = path
	#n.push(Node("Coordinates", tex_coord_exp))
	return n

def exp_rgb_to_bw(socket, exp_list):
	input_exp = get_expression(socket.node.inputs[0], exp_list)
	n = Node("DotProduct")
	n.push(Node("0", input_exp))
	exp_1 = exp_vector( (0.2126, 0.7152, 0.0722), exp_list )
	n.push( Node( "1", { "expression": exp_1 } ) )
	dot_exp = exp_list.push(n)
	return { "expression": dot_exp }

def exp_make_vec3(socket, exp_list):
	node = socket.node
	output = Node("FunctionCall", { "Function": "/Engine/Functions/Engine_MaterialFunctions02/Utility/MakeFloat3" })
	output.push(Node("0", get_expression(node.inputs[0], exp_list)))
	output.push(Node("1", get_expression(node.inputs[1], exp_list)))
	output.push(Node("2", get_expression(node.inputs[2], exp_list)))
	return { "expression": exp_list.push(output) }

def exp_make_hsv(socket, exp_list):
	vec3_input = exp_make_vec3(socket, exp_list)
	output = Node("FunctionCall",  { "Function": "/DatasmithBlenderContent/MaterialFunctions/HSV_To_RGB" })
	output.push(Node("0", vec3_input))
	return { "expression": exp_list.push(output) }

def exp_break_vec3(socket, exp_list):
	expression_idx = -1
	if socket.node in cached_nodes:
		expression_idx = cached_nodes[socket.node]
	else:
		output = Node("FunctionCall",  { "Function": "/Engine/Functions/Engine_MaterialFunctions02/Utility/BreakOutFloat3Components" })
		output.push(Node("0", get_expression(socket.node.inputs[0], exp_list)))
		expression_idx = exp_list.push(output)
		cached_nodes[socket.node] = expression_idx

	output_index = socket.node.outputs.find(socket.name) # could be faster by comparing to constants instead?
	return { "expression": expression_idx, "OutputIndex": output_index }

def exp_break_hsv(socket, exp_list):

	expression_idx = -1
	if socket.node in cached_nodes:
		expression_idx = cached_nodes[socket.node]
	else:
		input = Node("FunctionCall",  { "Function": "/DatasmithBlenderContent/MaterialFunctions/RGB_To_HSV" })
		hsv_expression_idx = input.push(Node("0", get_expression(socket.node.inputs[0], exp_list)))

		output = Node("FunctionCall",  { "Function": "/Engine/Functions/Engine_MaterialFunctions02/Utility/BreakOutFloat3Components" })
		output.push(Node("0", { "expression": hsv_expression_idx }))
		expression_idx = exp_list.push(output)
		cached_nodes[socket.node] = expression_idx

	output_index = socket.node.outputs.find(socket.name) # could be faster by comparing to constants instead?
	return { "expression": expression_idx, "OutputIndex": output_index }


MATH_CUSTOM_FUNCTIONS = {
	'INVERSE_SQRT': (1, "/DatasmithBlenderContent/MaterialFunctions/MathInvSqrt"),
	'EXPONENT':     (1, "/DatasmithBlenderContent/MaterialFunctions/MathExp"),
	'SINH':         (1, "/DatasmithBlenderContent/MaterialFunctions/MathSinH"),
	'COSH':         (1, "/DatasmithBlenderContent/MaterialFunctions/MathCosH"),
	'TANH':         (1, "/DatasmithBlenderContent/MaterialFunctions/MathTanH"),
	'MULTIPLY_ADD': (3, "/DatasmithBlenderContent/MaterialFunctions/MathMultiplyAdd"),
	'COMPARE':      (3, "/DatasmithBlenderContent/MaterialFunctions/MathCompare"),
	'SMOOTH_MIN':   (3, "/DatasmithBlenderContent/MaterialFunctions/MathSmoothMin"),
	'SMOOTH_MAX':   (3, "/DatasmithBlenderContent/MaterialFunctions/MathSmoothMax"),
	'WRAP':         (3, "/DatasmithBlenderContent/MaterialFunctions/MathWrap"),
	'SNAP':         (2, "/DatasmithBlenderContent/MaterialFunctions/MathSnap"),
	'PINGPONG':    (2, "/DatasmithBlenderContent/MaterialFunctions/MathPingPong"),
}

# these map 1:1 with UE4 nodes:
MATH_TWO_INPUTS = {
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
MATH_ONE_INPUT = {
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
	'SIGN': "Sign",
	'TRUNC': "Truncate",
}

# these require specific implementations:
MATH_CUSTOM_IMPL = {
	'LOGARITHM', # ue4 only has log2 and log10
	'LESS_THAN', # use UE4 If node
	'GREATER_THAN', # use UE4 If node
	'RADIANS',
	'DEGREES',
}

def exp_generic(name, inputs, exp_list, force_default=False):
	n = Node(name)
	for idx, input in enumerate(inputs):
		input_exp = get_expression(input, exp_list, force_default)
		n.push(Node(str(idx), input_exp))
	return { "expression": exp_list.push(n) }

def exp_function_call(path, inputs, exp_list, force_default=False):
	n = Node("FunctionCall", {"Function": path})
	for idx, input in enumerate(inputs):
		input_exp = get_expression(input, exp_list, force_default)
		n.push(Node(str(idx), input_exp))
	return { "expression": exp_list.push(n) }

def exp_math(node, exp_list):
	op = node.operation
	exp = None
	if op in MATH_TWO_INPUTS:
		exp = exp_generic(
			name= MATH_TWO_INPUTS[op],
			inputs= node.inputs[:2],
			exp_list=exp_list,
			force_default=True,
		)
	elif op in MATH_ONE_INPUT:
		exp = exp_generic(
			name= MATH_ONE_INPUT[op],
			inputs= node.inputs[:1],
			exp_list=exp_list,
			force_default=True,
		)
	elif op in MATH_CUSTOM_FUNCTIONS:
		size, path = MATH_CUSTOM_FUNCTIONS[op]
		exp = exp_function_call(
			path,
			inputs= node.inputs[:size],
			exp_list=exp_list,
		)
	elif op in MATH_CUSTOM_IMPL:
		in_0 = get_expression(node.inputs[0], exp_list)
		n = None
		if op == 'RADIANS':
			n = Node("Multiply")
			n.push(Node("0", in_0))
			n.push(Node("1", { "expression": exp_scalar(math.tau / 360, exp_list)}))
		elif op == 'DEGREES':
			n = Node("Multiply")
			n.push(Node("0", in_0))
			n.push(Node("1", { "expression": exp_scalar(360 / math.tau, exp_list)}))
		else:
			# these use two inputs
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
		assert n
		exp = { "expression": exp_list.push(n) }


	assert exp, "unrecognized math operation: %s" % op

	if getattr(node, "use_clamp", False):
		clamp = Node("Saturate")
		clamp.push(Node("0", exp))
		exp = { "expression": exp_list.push(clamp) }
	return exp

# these nodes should only be built-ins (green nodes)
VECT_MATH_SAME_AS_SCALAR = {
	'ADD',
	'SUBTRACT',
	'MULTIPLY',
	'DIVIDE',

	'ABSOLUTE',
	'MINIMUM',
	'MAXIMUM',
	'FLOOR',
	'CEIL',
	'MODULO',
	'SINE',
	'COSINE',
	'TANGENT',
}


VECT_MATH_NODES = {
	'CROSS_PRODUCT': (2, "CrossProduct"),
	'DOT_PRODUCT':   (2, "DotProduct"),
	'DISTANCE':      (2, "Distance"),
	'NORMALIZE':     (1, "Normalize"),
	'FRACTION':      (1, "Frac"),
}
VECT_MATH_FUNCTIONS = { # tuples are (input_count, path)

	'WRAP': (3, "/DatasmithBlenderContent/MaterialFunctions/VectWrap"),
	'SNAP': (2, "/DatasmithBlenderContent/MaterialFunctions/VectSnap"),
	'PROJECT': (2, "/DatasmithBlenderContent/MaterialFunctions/VectProject"),
	'REFLECT': (2, "/DatasmithBlenderContent/MaterialFunctions/VectReflect"),
}

def exp_vect_math(node, exp_list):
	node_op = node.operation
	if node_op in VECT_MATH_SAME_AS_SCALAR:
		return exp_math(node, exp_list)
	elif node_op in VECT_MATH_NODES:
		size, name = VECT_MATH_NODES[node_op]
		return exp_generic(
			name=name,
			inputs=node.inputs[:size],
			exp_list=exp_list,
			force_default=True,
		)
	elif node_op in VECT_MATH_FUNCTIONS:
		size, path = VECT_MATH_FUNCTIONS[node_op]
		return exp_function_call(
			path,
			inputs= node.inputs[:size],
			exp_list=exp_list,
			force_default=True,
		)
	elif node_op == 'SCALE':
		return exp_generic(
			name= "Multiply",
			inputs= (node.inputs[0], node.inputs[3]),
			exp_list=exp_list,
			force_default=True,
		)
	elif node_op == 'LENGTH':
		n = Node("Distance")
		n.push(Node("0", get_expression(node.inputs[0], exp_list) ))
		n.push(Node("1", { "expression": exp_vector((0,0,0), exp_list) } ))
		return { "expression": exp_list.push(n) }

	log.error("VECT_MATH node operation:%s not found" % node_op)

# TODO: make test cases for all math nodes

def exp_gamma(node, exp_list):
	n = Node(MATH_TWO_INPUTS['POWER'])
	exp_0 = get_expression(node.inputs["Color"], exp_list)
	n.push(Node("0", exp_0))
	exp_1 = get_expression(node.inputs["Gamma"], exp_list)
	n.push(Node("1", exp_1))
	return {"expression": exp_list.push(n)}

op_map_color = {
# MIX is handled manually
	'DARKEN': "/Engine/Functions/Engine_MaterialFunctions03/Blends/Blend_Darken",
# MULTIPLY is handled in MATH_TWO_INPUTS
	'BURN': "/Engine/Functions/Engine_MaterialFunctions03/Blends/Blend_ColorBurn",
	# TODO: check for blender implementation of burn, it could mean this:
	#'BURN': "/Engine/Functions/Engine_MaterialFunctions03/Blends/Blend_LinearBurn",
	'LIGHTEN': "/Engine/Functions/Engine_MaterialFunctions03/Blends/Blend_Lighten",
	'SCREEN': "/Engine/Functions/Engine_MaterialFunctions03/Blends/Blend_Screen",
	'DODGE': "/Engine/Functions/Engine_MaterialFunctions03/Blends/Blend_ColorDodge",
	'OVERLAY': "/Engine/Functions/Engine_MaterialFunctions03/Blends/Blend_Overlay",
# ADD is handled in MATH_TWO_INPUTS
	'SOFT_LIGHT': "/Engine/Functions/Engine_MaterialFunctions03/Blends/Blend_SoftLight",
	'LINEAR_LIGHT': "/Engine/Functions/Engine_MaterialFunctions03/Blends/Blend_LinearLight",
	'DIFFERENCE': "/Engine/Functions/Engine_MaterialFunctions03/Blends/Blend_Difference",
# SUBTRACT is handled in MATH_TWO_INPUTS
# DIVIDE is handled in MATH_TWO_INPUTS
	'HUE': "/DatasmithBlenderContent/MaterialFunctions/Blend_Hue",
	'SATURATION': "/DatasmithBlenderContent/MaterialFunctions/Blend_Saturation",
	'COLOR': "/DatasmithBlenderContent/MaterialFunctions/Blend_Color",
	'VALUE': "/DatasmithBlenderContent/MaterialFunctions/Blend_Value",
}

def exp_blend(exp_0, exp_1, blend_type, exp_list):
	if blend_type == 'MIX':
		return exp_1
	n = None
	if blend_type in {'ADD', 'SUBTRACT', 'MULTIPLY', 'DIVIDE'}:
		n = Node(MATH_TWO_INPUTS[blend_type])
	else:
		n = Node("FunctionCall", { "Function": op_map_color[blend_type]})
	assert n
	n.push(Node("0", exp_0))
	n.push(Node("1", exp_1))
	return {"expression": exp_list.push(n)}

def exp_mixrgb(node, exp_list):
	exp_1 = get_expression(node.inputs['Color1'], exp_list)
	exp_2 = get_expression(node.inputs['Color2'], exp_list)
	# TODO: optimize case fac is disconnected and equals one (or zero)
	# TODO: add logic for clamp

	exp_result = exp_blend(exp_1, exp_2, node.blend_type, exp_list)

	lerp = Node("LinearInterpolate")
	lerp.push(Node("0", exp_1))
	lerp.push(Node("1", exp_result))
	exp_fac = get_expression(node.inputs['Fac'], exp_list)
	lerp.push(Node("2", exp_fac))

	return exp_list.push(lerp)

op_custom_functions = {
	"BRIGHTCONTRAST":     "/DatasmithBlenderContent/MaterialFunctions/BrightContrast",
	"COLOR_RAMP":         "/DatasmithBlenderContent/MaterialFunctions/ColorRamp",
	"CURVE_RGB":          "/DatasmithBlenderContent/MaterialFunctions/RGBCurveLookup2",
	"FRESNEL":            "/DatasmithBlenderContent/MaterialFunctions/BlenderFresnel",
	"HUE_SAT":            "/DatasmithBlenderContent/MaterialFunctions/AdjustHSV",
	"LAYER_WEIGHT":       "/DatasmithBlenderContent/MaterialFunctions/LayerWeight",
	"LOCAL_POSITION":     "/DatasmithBlenderContent/MaterialFunctions/BlenderLocalPosition",
	"MAPPING_POINT2D":    "/DatasmithBlenderContent/MaterialFunctions/MappingPoint2D_2",
	"MAPPING_POINT3D":    "/DatasmithBlenderContent/MaterialFunctions/MappingPoint3D",
	"MAPPING_TEX2D":      "/DatasmithBlenderContent/MaterialFunctions/MappingTexture2D_2",
	"MAPPING_TEX3D":      "/DatasmithBlenderContent/MaterialFunctions/MappingTexture3D",
	"MAPPING_NORMAL":      "/DatasmithBlenderContent/MaterialFunctions/MappingNormal",
	"NORMAL_FROM_HEIGHT": "/Engine/Functions/Engine_MaterialFunctions03/Procedurals/NormalFromHeightmap",
	"WORLD_POSITION":     "/DatasmithBlenderContent/MaterialFunctions/BlenderWorldPosition",
}



def exp_generic_function(node, exp_list, node_type, socket_names):
	n = Node("FunctionCall", { "Function": op_custom_functions[node_type]})
	for idx, socket_name in enumerate(socket_names):
		input_expression = get_expression(node.inputs[socket_name], exp_list)
		n.push(Node(str(idx), input_expression))
	return {"expression": exp_list.push(n) }

def exp_bright_contrast(node, exp_list):
	return exp_generic_function(node, exp_list, 'BRIGHTCONTRAST', ('Color', 'Bright', 'Contrast'))

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
	if node.vector_type == 'NORMAL':
		mapping_type = 'MAPPING_NORMAL'
	else:
		node_input_rot = node.inputs["Rotation"]
		default_rot = node_input_rot.default_value
		uses_3d_rot = default_rot.x != 0 or default_rot.y != 0
		use_2d_node = not node_input_rot.links and not uses_3d_rot
		if node.vector_type == 'POINT' or node.vector_type == 'VECTOR':
			if use_2d_node:
				mapping_type = 'MAPPING_POINT2D'
			else:
				mapping_type = 'MAPPING_POINT3D'
		elif node.vector_type == 'TEXTURE':
			if use_2d_node:
				mapping_type = 'MAPPING_TEX2D'
			else:
				mapping_type = 'MAPPING_TEX3D'

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
def exp_normal_map(socket, exp_list):
	node_input = socket.node.inputs['Color']
	# hack: is it safe to assume that everything under here is normal?
	# maybe not, because it could be masks to mix normals
	# most certainly, these wouldn't be colors (so should be non-srgb)
	push_context("NORMAL")
	return_exp = get_expression(node_input, exp_list)
	pop_context()

	strength_input = socket.node.inputs["Strength"]
	if strength_input.links or strength_input.default_value != 1.0:
		node_strength = Node("FunctionCall", {"Function": "/DatasmithBlenderContent/MaterialFunctions/NormalStrength"})
		node_strength.push(Node("0", return_exp))
		node_strength.push(Node("1", get_expression(strength_input, exp_list)))
		return_exp = { "expression": exp_list.push(node_strength) }
	return return_exp


def exp_new_geometry(socket, exp_list):
	socket_name = socket.name
	if socket_name == "Position":
		blend = Node("FunctionCall", { "Function": op_custom_functions["WORLD_POSITION"]})
		n = exp_list.push(blend)
		return { "expression": n }
	if socket_name == "Normal":
		blend = Node("PixelNormalWS")
		n = exp_list.push(blend)
		return { "expression": n }
	if socket_name == "Tangent":
		blend = Node("VertexTangentWS")
		n = exp_list.push(blend)
		return { "expression": n }
	if socket_name == "True Normal":
		blend = Node("VertexNormalWS")
		n = exp_list.push(blend)
		return { "expression": n }
	# if socket_name == "Incoming":
	# 	this would be cameraposition - worldposition
	# if socket_name == "Parametric":
	#	this appears to be per-triangle barycentric coordinates
	# if socket_name == "Backfacing":
	#	exactly what it says, I thought UE4 had this
	if socket_name == "Pointiness":
		exp = exp_scalar(0, exp_list)
		return {"expression": exp}
	# if socket_name == "Random Per Island":
	log.error("Node NEW_GEOMETRY has unhanded socket:%s" % socket_name)


def exp_texture_coordinates(socket, exp_list):
	socket_name = socket.name
	if socket_name == "Position":
		blend = Node("FunctionCall", { "Function": op_custom_functions["WORLD_POSITION"]})
		n = exp_list.push(blend)
		return { "expression": n }
	if socket_name == "Normal":
		blend = Node("PixelNormalWS")
		n = exp_list.push(blend)
		return { "expression": n }
	if socket_name == "Tangent":
		blend = Node("VertexTangentWS")
		n = exp_list.push(blend)
		return { "expression": n }
	if socket_name == "True Normal":
		blend = Node("VertexNormalWS")
		n = exp_list.push(blend)
		return { "expression": n }
	# if socket_name == "Incoming":
	# 	this would be cameraposition - worldposition
	# if socket_name == "Parametric":
	#	this appears to be per-triangle barycentric coordinates
	# if socket_name == "Backfacing":
	#	exactly what it says, I thought UE4 had this
	if socket_name == "Pointiness":
		exp = exp_scalar(0, exp_list)
		return {"expression": exp}
	# if socket_name == "Random Per Island":
	log.error("Node NEW_GEOMETRY has unhanded socket:%s" % socket_name)


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

def exp_light_path(socket, exp_list):
	log.warn("incomplete node implementation: LIGHT_PATH")
	n = exp_scalar(1, exp_list)
	return {"expression": n}


def exp_object_info(socket, exp_list):
	field = socket.name
	if field == "Location":
		# TODO: check if we need to transform these to blender space
		exp = exp_list.push(Node("ObjectPositionWS"))
	elif field == "Random":
		exp = exp_list.push(Node("PerInstanceRandom"))
	elif field == "Object Index":
		log.warning("Node Object Info>Object Index translated to random as it is used to randomize too")
		exp = exp_list.push(Node("PerInstanceRandom"))
	else:
		log.error("Can't write Material node 'Object Info' field:%s" % field)
		exp = exp_scalar(0, exp_list)

	return {"expression": exp, "OutputIndex": 0}


DATASMITH_TEXTURE_SIZE = 1024

def add_material_curve2(curve):

	# do some material curves initialization
	material_curves = datasmith_context["material_curves"]
	if material_curves is None:
		material_curves = np.zeros((DATASMITH_TEXTURE_SIZE, DATASMITH_TEXTURE_SIZE, 4))
		datasmith_context["material_curves"] = material_curves
		datasmith_context["material_curves_count"] = 0

	mat_curve_idx = datasmith_context["material_curves_count"]
	datasmith_context["material_curves_count"] = mat_curve_idx + 1
	log.info("writing curve:%s" % mat_curve_idx)

	# write texture from top
	row_idx = DATASMITH_TEXTURE_SIZE - mat_curve_idx - 1
	values = material_curves[row_idx]
	factor = DATASMITH_TEXTURE_SIZE - 1

	# check for curve type, do sampling
	curve_type = type(curve)
	if curve_type == bpy.types.ColorRamp:
		for idx in range(DATASMITH_TEXTURE_SIZE):
			values[idx] = curve.evaluate(idx/factor)

	elif curve_type == bpy.types.CurveMapping:
		curves = curve.curves

		position = 0
		for idx in range(DATASMITH_TEXTURE_SIZE):
			position = idx/factor
			values[idx, 0] = curve.evaluate(curves[0], position)
			values[idx, 1] = curve.evaluate(curves[1], position)
			values[idx, 2] = curve.evaluate(curves[2], position)
			values[idx, 3] = curve.evaluate(curves[3], position)

	return mat_curve_idx

def exp_blackbody(from_node, exp_list):
	n = Node("BlackBody")
	exp_0 = get_expression(from_node.inputs[0], exp_list)
	n.push(Node("0", exp_0))
	exp = exp_list.push(n)
	return {"expression": exp}

def exp_color_ramp(from_node, exp_list):
	ramp = from_node.color_ramp

	idx = add_material_curve2(ramp)

	level = get_expression(from_node.inputs['Fac'], exp_list)

	curve_idx = exp_scalar(idx, exp_list)
	compatibility_mode = datasmith_context["compatibility_mode"]
	if compatibility_mode:
		pixel_offset = exp_scalar(0.5, exp_list)
		vertical_res = exp_scalar(1/DATASMITH_TEXTURE_SIZE, exp_list) # curves texture size
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

		texture_exp = exp_texture("datasmith_curves", "datasmith_curves")
		texture_exp.push(Node("Coordinates", {"expression":tex_coord}))

		return exp_list.push(texture_exp)

	else:
		vertical_res = exp_scalar(DATASMITH_TEXTURE_SIZE, exp_list) # curves texture size
		texture = exp_texture_object("datasmith_curves", exp_list)

		lookup = Node("FunctionCall", { "Function": op_custom_functions["COLOR_RAMP"]})
		lookup.push(Node("0", level))
		lookup.push(Node("1", {"expression": curve_idx } ))
		lookup.push(Node("2", {"expression": vertical_res } ))
		lookup.push(Node("3", {"expression": texture } ))
		result = exp_list.push(lookup)

		return result

def exp_curvergb(from_node, exp_list):
	mapping = from_node.mapping
	mapping.initialize()

	idx = add_material_curve2(mapping)

	factor = get_expression(from_node.inputs['Fac'], exp_list)
	color = get_expression(from_node.inputs['Color'], exp_list)

	curve_idx = exp_scalar(idx, exp_list)
	vertical_res = exp_scalar(DATASMITH_TEXTURE_SIZE, exp_list) # curves texture size

	texture = exp_texture_object("datasmith_curves", exp_list)

	lookup = Node("FunctionCall", { "Function": op_custom_functions["CURVE_RGB"]})
	lookup.push(Node("0", color))
	lookup.push(Node("1", {"expression": curve_idx}))
	lookup.push(Node("2", {"expression": vertical_res}))
	lookup.push(Node("3", {"expression": texture}))
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
			get_or_create_texture(name, image)

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
	global cached_nodes
	new_context = {}
	new_cached_nodes = {}
	for input in node.inputs:
		new_context[input.name] = get_expression(input, exp_list)

	previous_reverse = reverse_expressions
	reverse_expressions = {}
	previous_context = group_context
	previous_cached_nodes = cached_nodes
	group_context = new_context
	cached_nodes = new_cached_nodes

	# now traverse the inner graph
	output_name = socket.name

	node_tree = node.node_tree

	# search for active output node:
	output_node = None
	for node in node_tree.nodes:
		if type(node) == bpy.types.NodeGroupOutput:
			if node.is_active_output or output_node is None:
				output_node = node

	# TODO: handle case when output_node is None

	inner_socket = output_node.inputs[output_name]
	inner_exp = get_expression(inner_socket, exp_list)

	group_context = previous_context
	cached_nodes = previous_cached_nodes
	reverse_expressions = previous_reverse
	return inner_exp

def exp_group_input(socket, exp_list):
	outer_expression = group_context[socket.name]
	return outer_expression
def exp_attribute(socket, exp_list):
	exp = exp_list.push(Node("VertexColor"))
	ret = {"expression": exp, "OutputIndex": 0}
	# average channels if socket is Fac
	if socket.name == "Fac":
		#TODO: check if we should do some colorimetric aware convertion to grayscale
		n = Node("DotProduct")
		n.push(Node("0", ret))
		exp_1 = exp_vector((0.333333, 0.333333, 0.333333), exp_list)
		n.push(Node("1", {"expression": exp_1}))
		dot_exp = exp_list.push(n)
		ret = {"expression": dot_exp}
	return ret

def exp_vertex_color(socket, exp_list):
	exp = exp_list.push(Node("VertexColor"))
	if socket.name == "Color":
		return {"expression": exp, "OutputIndex": 0}
	elif socket.name == "Alpha":
		return {"expression": exp, "OutputIndex": 4}

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
def get_expression(field, exp_list, force_default=False):
	# this may return none for fields without default value
	# most of the time blender doesn't have default value for vector
	# node inputs, but it does for scalars and colors
	# TODO: check which cases we should be careful
	global expression_log_prefix
	field_path = f"{field.node.name}/{field.name}:{field.type}"
	log.debug(expression_log_prefix + field_path)

	if not field.links:
		if field.type == 'VALUE':
			exp = exp_scalar(field.default_value, exp_list)
			return {"expression": exp, "OutputIndex": 0}
		elif field.type == 'RGBA':
			exp = exp_color(field.default_value, exp_list)
			return {"expression": exp, "OutputIndex": 0}
		elif field.type == 'VECTOR':
			use_vector_default = force_default or type(field.default_value) in {Vector, Euler}
			if use_vector_default:
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

	log.debug("%send field:%s = %s" % (expression_log_prefix, field_path, return_exp))

	return return_exp

def get_expression_inner(field, exp_list):

	node = field.links[0].from_node
	socket = field.links[0].from_socket
	log.debug(f"{expression_log_prefix} get_expression_inner {node.name} {socket.name}")
	# if this node is already exported, connect to that instead
	# I am considering in
	if socket in reverse_expressions:
		return reverse_expressions[socket]

	# The cases are ordered like in blender Add menu, others first, shaders second, then the rest

	# these are handled first as these can refer bsdfs
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

	# Shader nodes return a dictionary
	bsdf = None
	if node.type == 'BSDF_PRINCIPLED':
		bsdf = {
			"BaseColor": get_expression(node.inputs['Base Color'], exp_list),
			"Metallic": get_expression(node.inputs['Metallic'], exp_list),
			"Roughness": get_expression(node.inputs['Roughness'], exp_list),
			"Specular": get_expression(node.inputs['Specular'], exp_list),
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
	if node.type == 'EEVEE_SPECULAR':
		log.warn("EEVEE_SPECULAR incomplete implementation")
		bsdf = {
			"BaseColor": get_expression(node.inputs['Base Color'], exp_list),
			"Roughness": get_expression(node.inputs['Roughness'], exp_list),
		}

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
			"Metallic": { "expression": exp_scalar(1, exp_list) },
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
	elif node.type == 'BSDF_REFRACTION':
		log.warn("BSDF_REFRACTION incomplete implementation")
		bsdf = {
			"BaseColor": get_expression(node.inputs['Color'], exp_list),
			"Roughness": get_expression(node.inputs['Roughness'], exp_list),
			"Refraction": get_expression(node.inputs['IOR'], exp_list),
			"Opacity": {"expression": exp_scalar(0.5, exp_list)},
		}
	elif node.type == 'BSDF_ANISOTROPIC':
		log.warn("BSDF_ANISOTROPIC incomplete implementation")
		bsdf = {
			"BaseColor": get_expression(node.inputs['Color'], exp_list),
			"Roughness": get_expression(node.inputs['Roughness'], exp_list),
			# TODO: read inputs 'Anisotropy' and 'Rotation' and 'Tangent'
		}



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
		assert expressions

		expressions1 = get_expression(node.inputs[1], exp_list)
		assert expressions1
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
		assert expressions

		expressions1 = get_expression(node.inputs[2], exp_list)
		assert expressions1

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


	if field.type == 'SHADER':

		if bsdf:
			if "Normal" in node.inputs:
				normal_expression = get_expression(node.inputs['Normal'], exp_list)
				if normal_expression:
					bsdf["Normal"] = normal_expression
		else:
			log.error(f"couldn't find bsdf for field {field.name}")
		return bsdf
	# from here the return type should be {expression:node_idx, OutputIndex: socket_idx}
	# Add > Input

	# if node.type == 'AMBIENT_OCCLUSION':
	if node.type == 'ATTRIBUTE':
		return exp_attribute(socket, exp_list)
	if node.type == 'VERTEX_COLOR':
		return exp_vertex_color(socket, exp_list)

	# if node.type == 'BEVEL':
	# if node.type == 'CAMERA':
	if node.type == 'FRESNEL':
		exp = exp_fresnel(node, exp_list)
		return {"expression": exp}
	if node.type == 'NEW_GEOMETRY':
		result = exp_new_geometry(socket, exp_list)
		if result:
			return result
	# if node.type == 'HAIR_INFO':
	if node.type == 'LAYER_WEIGHT': # fresnel and facing, with "blend" (power?) and normal param
		return exp_layer_weight(socket, exp_list)
	if node.type == 'LIGHT_PATH':
		return exp_light_path(socket, exp_list)
	if node.type == 'OBJECT_INFO':
		return exp_object_info(socket, exp_list)
	# if node.type == 'PARTICLE_INFO':

	if node.type == 'RGB':
		exp = exp_color(node.outputs[0].default_value, exp_list)
		return {"expression": exp, "OutputIndex": 0}

	# if node.type == 'TANGENT':
	if node.type == 'TEX_COORD':
		exp = exp_texcoord_node(socket, exp_list)
		if exp:
			return exp

	if node.type == 'UVMAP':
		return exp_uvmap(node, exp_list)
	if node.type == 'VALUE':
		exp = exp_scalar(node.outputs[0].default_value, exp_list)
		return {"expression": exp}
	# if node.type == 'WIREFRAME':


	# Add > Texture
	# if node.type == 'TEX_BRICK':
	if node.type == 'TEX_CHECKER':
		return exp_tex_checker(socket, exp_list)
	# if node.type == 'TEX_ENVIRONMENT':
	# if node.type == 'TEX_GRADIENT':
	# if node.type == 'TEX_IES':
	if node.type == 'TEX_NOISE':
		return exp_tex_noise(socket, exp_list)
	if node.type == 'TEX_IMAGE':
		cached_node = None
		if node in reverse_expressions:
			cached_node = reverse_expressions[node]

		if not cached_node:
			image = node.image
			if not image:
				return { "expression": exp_scalar(0, exp_list) }


			tex_coord = get_expression(node.inputs['Vector'], exp_list)


			name = ""
			if image:
				name = sanitize_name(image.name) # name_full?

				# ensure that texture is exported
				texture_type = get_context() or 'SRGB'
				get_or_create_texture(name, image, texture_type)

			texture_exp = exp_texture(name)
			if tex_coord:
				if node.projection == 'BOX':
					proj = Node("FunctionCall", { "Function": "/DatasmithBlenderContent/MaterialFunctions/TexCoord_Box"})
					proj.push(Node("0", tex_coord))
					mask_expression = { "expression": exp_list.push(proj) }
					texture_exp.push(Node("Coordinates", mask_expression))
				else:
					if node.projection != 'FLAT':
						log.error("node TEXTURE_COORDINATE has unhandled projection: %s" % node.projection)
					mask = Node("ComponentMask")
					mask.push(Node("0", tex_coord))
					mask_expression = { "expression": exp_list.push(mask) }
					texture_exp.push(Node("Coordinates", mask_expression))

			cached_node = exp_list.push(texture_exp)
			reverse_expressions[node] = cached_node

		output_index = 0 # RGB
		# indices 1, 2, 3 are separate RGB channels
		if socket.name == 'Alpha':
			output_index = 4 #

		return { "expression": cached_node, "OutputIndex": output_index }

	# Add > Color
	if node.type == 'BRIGHTCONTRAST':
		return exp_bright_contrast(node, exp_list)
	if node.type == 'GAMMA':
		return exp_gamma(node, exp_list)
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
		return exp_normal_map(socket, exp_list)
	# if node.type == 'CURVE_VEC':
	# if node.type == 'VECTOR_DISPLACEMENT':
	# if node.type == 'VECT_TRANSFORM':

	# Add > Converter

	# if node.type == 'WAVELENGTH':
	if node.type == 'BLACKBODY':
		return exp_blackbody(node, exp_list)
	if node.type == 'VALTORGB':
		exp = exp_color_ramp(node, exp_list)
		return {"expression": exp, "OutputIndex": 0}

	if node.type == 'COMBRGB':
		return exp_make_vec3(socket, exp_list)
	if node.type == 'COMBXYZ':
		return exp_make_vec3(socket, exp_list)
	if node.type == 'COMBHSV':
		return exp_make_hsv(socket, exp_list)

	if node.type == 'SEPRGB':
		return exp_break_vec3(socket, exp_list)
	if node.type == 'SEPXYZ':
		return exp_break_vec3(socket, exp_list)
	if node.type == 'SEPHSV':
		return exp_break_hsv(socket, exp_list)

	if node.type == 'RGBTOBW':
		return exp_rgb_to_bw(socket, exp_list)
	if node.type == 'MATH':
		return exp_math(node, exp_list)
	if node.type == 'VECT_MATH':
		return exp_vect_math(node, exp_list)

	# if node.type == 'SHADERTORGB':

	# Others:

	# if node.type == 'SCRIPT':


	log.error("node not handled" + node.type)
	exp = exp_scalar(0, exp_list)
	return {"expression": exp}


def pbr_nodetree_material(material):
	log.info("Collecting material: "+material.name)
	n = Node("UEPbrMaterial")
	n['name'] = sanitize_name(material.name)
	exp_list = Node("Expressions")
	n.push(exp_list)

	output_node = (
		material.node_tree.get_output_node('EEVEE')
		or material.node_tree.get_output_node('ALL')
		or material.node_tree.get_output_node('CYCLES')
	)

	if not output_node:
		log.warn("material %s with use_nodes does not have nodes" % material.name)
		return n

	surface_field = output_node.inputs['Surface']
	if not surface_field.links:
		log.warn("material %s with use_nodes does not have nodes" % material.name)
		return n

	global reverse_expressions
	reverse_expressions = dict()

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
	grey = 0.906332
	basecolor_idx = exp_color((grey, grey, grey, 1.0), exp_list)
	roughness_idx = exp_scalar(0.4, exp_list)
	n.push(exp_list)
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


cached_nodes = {}
def collect_pbr_material(mat_with_owner):
	datasmith_context["material_owner"] = mat_with_owner[1]
	global cached_nodes
	cached_nodes = {}
	material = mat_with_owner[0]
	if material is None:
		log.debug("creating default material")
		return pbr_default_material()
	if not material.use_nodes:
		log.debug("creating material %s without nodes" % material.name)
		return pbr_basic_material(material)
	log.debug("creating material %s with node_tree " % material.name)
	return pbr_nodetree_material(material)

import numpy as np




def fill_umesh(umesh, bl_mesh):
	# create copy to triangulate
	m = bl_mesh.copy()
	bm = bmesh.new()
	bm.from_mesh(m)
	bmesh.ops.triangulate(bm, faces=bm.faces[:])
	# this is just to make sure a UV layer exists
	bm.loops.layers.uv.verify()
	bm.to_mesh(m)
	bm.free()
	# not sure if this is the best way to read normals
	m.calc_normals_split()

	loops = m.loops
	num_loops = len(loops)

	normals = np.empty(num_loops* 3, np.float32)
	loops.foreach_get("normal", normals)
	normals = normals.reshape((num_loops, 3))
	normals = normals @ matrix_normals

	m.transform(matrix_datasmith)

	#finish inline mesh_copy_triangulate
	if len(bl_mesh.materials) == 0:
		umesh.materials[0] = 'DefaultMaterial'
	else:
		for idx, mat in enumerate(bl_mesh.materials):
			umesh.materials[idx] = sanitize_name(getattr(mat, 'name', 'DefaultMaterial'))

	polygons = m.polygons
	num_polygons = len(polygons)
	material_slots = np.empty(num_polygons, np.uint32)

	polygons.foreach_get("material_index", material_slots)
	umesh.tris_material_slot = material_slots # [p.material_index for p in m.polygons]

	smoothing_groups = m.calc_smooth_groups()[0];
	umesh.tris_smoothing_group = np.array(smoothing_groups, np.uint32)

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

	umesh.vertex_normals = np.ascontiguousarray(normals, "<f4")


	uvs = []
	num_uvs = min(8, len(m.uv_layers))
	active_uv = 0
	for idx in range(num_uvs):
		if m.uv_layers[idx].active_render:
			active_uv = idx
	for idx in range(num_uvs):
		uv_idx = idx # swap active_render UV with channel 0
		if uv_idx == 0:
			uv_idx = active_uv
		elif uv_idx == active_uv:
			uv_idx = 0

		uv_channel = np.empty(num_loops * 2, np.float32)
		uv_data = m.uv_layers[uv_idx].data
		uv_data.foreach_get("uv", uv_channel)
		uv_channel = uv_channel.reshape((num_loops, 2))
		uv_channel[:,1] = 1 - uv_channel[:,1]
		uvs.append(uv_channel)
	umesh.uvs = uvs

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

def node_transform(mat):
	loc, rot, scale = mat.decompose()
	n = Node('Transform')
	n['tx'] = f(loc.x)
	n['ty'] = f(loc.y)
	n['tz'] = f(loc.z)
	n['qw'] = f(rot.w)
	n['qx'] = f(rot.x)
	n['qy'] = f(rot.y)
	n['qz'] = f(rot.z)
	n['sx'] = f(scale.x)
	n['sy'] = f(scale.y)
	n['sz'] = f(scale.z)
	return n

def collect_object(
	bl_obj,
	name_override=None,
	instance_matrix=None,
	selected_only=False,
	apply_modifiers=False,
	export_animations=False,
	export_metadata=False,
):

	n = Node('Actor')

	n['name'] = sanitize_name(bl_obj.name)
	if name_override:
		n['name'] = name_override
	log.debug("reading object:%s" % bl_obj.name)

	n['layer'] = bl_obj.users_collection[0].name_full


	child_nodes = []

	for child in bl_obj.children:
		new_obj = collect_object(child,
			selected_only=selected_only,
			apply_modifiers=apply_modifiers,
			export_animations=export_animations,
			export_metadata = export_metadata,
		)
		if new_obj:
			child_nodes.append(new_obj)

	# if we are exporting only selected items, we should only continue
	# if this is selected, or if there is any child that needs this
	# object to be placed in hierarchy
	# TODO: collections don't work this way, investigate (export chair from classroom)
	export_empty_because_unselected = False
	if selected_only:
		is_selected = bl_obj in bpy.context.selected_objects
		if selected_only and not is_selected:
			if len(child_nodes) == 0:
				# We skip this object as it is not selected, and has no children selected
				return None
			else:
				# we aren't selected, but we have selected children, so create minimal object
				export_empty_because_unselected = True

	# from here, we're absolutely sure that this object should be exported

	obj_mat = collect_object_transform(bl_obj, instance_matrix)
	transform = node_transform(obj_mat)

	# if an object is not selected but is in hierarchy, we don't write data for it
	if not export_empty_because_unselected:
		# TODO: use instanced static meshes
		depsgraph = datasmith_context["depsgraph"]

		if bl_obj.is_instancer:
			dups = []
			dup_idx = 0
			for dup in depsgraph.object_instances:
				if dup.parent and dup.parent.original == bl_obj:
					dup_name = '%s_%s' % (dup.instance_object.original.name, dup_idx)
					dup_name = sanitize_name(dup_name)
					new_obj = collect_object(
						dup.instance_object.original,
						instance_matrix=dup.matrix_world.copy(),
						name_override=dup_name,
						selected_only=False, # if is instancer, maybe all child want to be instanced
						apply_modifiers=False, # if is instancer, applying modifiers may end in a lot of meshes
						export_animations=False, # TODO: test how would animation work mixed with instancing
						export_metadata=False,
					)
					child_nodes.append(new_obj)
					#dups.append((dup.instance_object.original, dup.matrix_world.copy()))
					dup_idx += 1

		collect_object_custom_data(bl_obj, n, apply_modifiers, obj_mat, depsgraph, export_metadata)

	# todo: maybe make some assumptions? like if obj is probe or reflection, don't add to animated objects list

	if export_animations:
		datasmith_context["anim_objects"].append((bl_obj, n["name"], obj_mat))

	if export_metadata:
		collect_object_metadata(n["name"], "Actor", bl_obj)

	# just to make children appear last
	n.push(transform)

	if len(child_nodes) > 0:
		children_node = Node("children");
		# strange, this visibility flag is read from the "children" node. . . 
		children_node["visible"] = not bl_obj.hide_render
		for child in child_nodes:
			if child:
				children_node.push(child)
		n.push(children_node)


	return n


def collect_object_custom_data(bl_obj, n, apply_modifiers, obj_mat, depsgraph, export_metadata=False):
		# I think that these should be ordered by how common they are
		if bl_obj.type == 'EMPTY':
			pass
		elif bl_obj.type == 'MESH':
			bl_mesh = bl_obj.data
			bl_mesh_name = bl_mesh.name

			if bl_obj.modifiers and apply_modifiers:
				bl_mesh = bl_obj.evaluated_get(depsgraph).to_mesh()
				bl_mesh_name = "%s__%s" % (bl_obj.name, bl_mesh.name)

			if bl_mesh.library:
				libraries_dict = datasmith_context["libraries"]
				prefix = libraries_dict.get(bl_mesh.library)

				if prefix is None:
					lib_filename = bpy.path.basename(bl_mesh.library.filepath)
					lib_clean_name = bpy.path.clean_name(lib_filename)
					prefix = lib_clean_name.strip("_")
					if prefix.endswith("_blend"):
						prefix = prefix[:-5] # leave the underscore
					next_prefix = prefix
					try_count = 1
					libraries_prefixes = libraries_dict.values()
					# just to reaaally make sure there are no collisions
					while next_prefix in libraries_prefixes:
						next_prefix = "%s%d_" % (prefix, try_count)
						try_count += 1
					libraries_dict[bl_mesh.library] = next_prefix
					prefix = next_prefix
				bl_mesh_name = prefix + bl_mesh_name


			bl_mesh_name = sanitize_name(bl_mesh_name)
			meshes = datasmith_context["meshes"]
			umesh = None
			for mesh in meshes:
				if bl_mesh_name == mesh.name:
					umesh = mesh

			if umesh == None:
				if len(bl_mesh.polygons) > 0:
					umesh = UDMesh(bl_mesh_name)
					meshes.append(umesh)
					fill_umesh(umesh, bl_mesh)

					if export_metadata:
						collect_object_metadata(n["name"], "StaticMesh", bl_mesh)

					material_list = datasmith_context["materials"]
					if len(bl_obj.material_slots) == 0:
						material_list.append((None, bl_obj))
					else:
						for slot in bl_obj.material_slots:
							material_list.append((slot.material, bl_obj))

			if umesh:
				n.name = 'ActorMesh'
				n.push(Node('mesh', {'name': umesh.name}))

				for idx, slot in enumerate(bl_obj.material_slots):
					if slot.link == 'OBJECT':
						#collect_materials([slot.material], uscene)
						safe_name = sanitize_name(slot.material.name)
						n.push(Node('material', {'id':idx, 'name':safe_name}))

		elif bl_obj.type == 'CURVE':

			# as we cannot get geometry before evaluating depsgraph,
			# we better evaluate first, and check if it has polygons.
			# this might end with repeated geometry, gotta find solution.
			# maybe cache "evaluated curve without modifiers"?

			bl_mesh = bl_obj.evaluated_get(depsgraph).to_mesh()
			if bl_mesh and len(bl_mesh.polygons) > 0:
				bl_curve = bl_obj.data
				bl_curve_name = "%s_%s" % (bl_curve.name, bl_obj.name)
				bl_curve_name = sanitize_name(bl_curve_name)

				umesh = UDMesh(bl_curve_name)
				meshes = datasmith_context["meshes"]
				meshes.append(umesh)

				fill_umesh(umesh, bl_mesh)
				material_list = datasmith_context["materials"]

				n.name = 'ActorMesh'
				n.push(Node('mesh', {'name': umesh.name}))

				if len(bl_obj.material_slots) == 0:
					material_list.append((None, bl_obj))
				else:
					for idx, slot in enumerate(bl_obj.material_slots):
						material_list.append((slot.material, bl_obj))
						if slot.link == 'OBJECT':
							#collect_materials([slot.material], uscene)
							safe_name = sanitize_name(slot.material.name)
							n.push(Node('material', {'id':idx, 'name':safe_name}))

		elif bl_obj.type == 'CAMERA':

			bl_cam = bl_obj.data
			n.name = 'Camera'

			# TODO
			# look_at_actor = sanitize_name(bl_cam.dof.focus_object.name)

			use_dof = "1" if bl_cam.dof.use_dof else "0"
			n.push(Node("DepthOfField", {"enabled": use_dof}))
			n.push(node_value('SensorWidth', bl_cam.sensor_width))
			# blender doesn't have per-camera aspect ratio
			sensor_aspect_ratio = 1.777778
			n.push(node_value('SensorAspectRatio', sensor_aspect_ratio))
			n.push(node_value('FocusDistance', bl_cam.dof.focus_distance * 100)) # to centimeters
			n.push(node_value('FStop', bl_cam.dof.aperture_fstop))
			n.push(node_value('FocalLength', bl_cam.lens))
			n.push(Node('Post'))
		# maybe move up as lights are more common?
		elif bl_obj.type == 'LIGHT':

			bl_light = bl_obj.data
			n.name = 'Light'

			n['type'] = 'PointLight'
			n['enabled'] = '1'
			n.push(node_value('SourceSize', bl_light.shadow_soft_size * 100))
			light_intensity = bl_light.energy
			light_attenuation_radius = 100 * math.sqrt(bl_light.energy)
			light_color = bl_light.color
			light_intensity_units = 'Lumens' # can also be 'Candelas' or 'Unitless'
			light_use_custom_distance = bl_light.use_custom_distance

			if bl_light.type == 'SUN':
				n['type'] = 'DirectionalLight'
				light_use_custom_distance = False
				# light_intensity = bl_light.energy # suns are in lux

			elif bl_light.type == 'SPOT':
				n['type'] = 'SpotLight'
				outer_cone_angle = bl_light.spot_size * 180 / (2*math.pi)
				inner_cone_angle = outer_cone_angle * (1 - bl_light.spot_blend)
				if inner_cone_angle < 0.0001:
					inner_cone_angle = 0.0001
				n.push(node_value('InnerConeAngle', inner_cone_angle))
				n.push(node_value('OuterConeAngle', outer_cone_angle))

				spot_use_candelas = False # TODO: test this thoroughly
				if spot_use_candelas:
					light_intensity_units = 'Candelas'
					light_intensity = bl_light.energy * 0.08 # came up with this constant by brute force
					# blender watts unit match ue4 lumens unit, but in spot lights the brightness
					# changes with the spot angle when using lumens while candelas do not.

			elif bl_light.type == 'AREA':
				n['type'] = 'AreaLight'

				size_w = size_h = bl_light.size
				if bl_light.shape == 'RECTANGLE' or bl_light.shape == 'ELLIPSE':
					size_h = bl_light.size_y

				n.push(Node('Shape', {
					"type": 'None', # can be Rectangle, Disc, Sphere, Cylinder, None
					"width": size_w * 100, # convert to cm
					"length": size_h * 100,
					"LightType": "Rect", # can be "Point", "Spot", "Rect"
				}))
			if light_use_custom_distance:
				light_attenuation_radius = 100 * bl_light.cutoff_distance
			# TODO: check how lights work when using a node tree
			# if bl_light.use_nodes and bl_light.node_tree:

			# 	node = bl_light.node_tree.nodes['Emission']
			# 	light_color = node.inputs['Color'].default_value
			# 	light_intensity = node.inputs['Strength'].default_value # have to check how to relate to candelas
			# 	log.error("unsupported: using nodetree for light " + bl_obj.name)

			n.push(node_value('Intensity', light_intensity))
			n.push(node_value('AttenuationRadius', light_attenuation_radius))
			n.push(Node('IntensityUnits', {'value': light_intensity_units}))
			n.push(Node('Color', {
				'usetemp': '0',
				'temperature': '6500.0',
				'R': f(light_color[0]),
				'G': f(light_color[1]),
				'B': f(light_color[2]),
				}))
		elif bl_obj.type == 'LIGHT_PROBE':
			# TODO: LIGHT PROBE
			n.name = 'CustomActor'
			bl_probe = bl_obj.data
			if bl_probe.type == 'PLANAR':
				n["PathName"] = "/DatasmithBlenderContent/Blueprints/BP_BlenderPlanarReflection"

			elif bl_probe.type == 'CUBEMAP':
				## we could also try using min/max if it makes a difference
				_, _, obj_scale = obj_mat.decompose()
				avg_scale = (obj_scale.x + obj_scale.y + obj_scale.z) * 0.333333

				if bl_probe.influence_type == 'BOX':
					n["PathName"] = "/DatasmithBlenderContent/Blueprints/BP_BlenderBoxReflection"


					falloff = bl_probe.falloff # this value is 0..1
					transition_distance = falloff * avg_scale
					prop = Node("KeyValueProperty", {"name": "TransitionDistance", "type":"Float", "val": "%.6f"%transition_distance})
					n.push(prop)
				else: # if bl_probe.influence_type == 'ELIPSOID'
					n["PathName"] = "/DatasmithBlenderContent/Blueprints/BP_BlenderSphereReflection"
					probe_radius = bl_probe.influence_distance * 100 * avg_scale
					radius = Node("KeyValueProperty", {"name": "Radius", "type":"Float", "val": "%.6f"%probe_radius})
					n.push(radius)
			elif bl_probe.type == 'GRID':
				# for now we just export to custom object, but it doesn't affect the render on
				# the unreal side. would be cool if it made a difference by setting volumetric importance volume
				n["PathName"] = "/DatasmithBlenderContent/Blueprints/BP_BlenderGridProbe"

				# blender influence_distance is outwards, maybe we should grow the object to match?
				# outward_influence would be 1.0 + influence_distance / size maybe?
				# obj_mat = obj_mat @ Matrix.Scale(outward_influence, 4)

			else:
				log.error("unhandled light probe")
		elif bl_obj.type == 'ARMATURE':
			pass
		else:
			log.error("unrecognized object type: %s" % bl_obj.type)



def collect_object_transform(bl_obj, instance_matrix=None):
	mat_basis = instance_matrix or bl_obj.matrix_world
	obj_mat = matrix_datasmith @ mat_basis @ matrix_datasmith.inverted()

	if bl_obj.type in 'CAMERA' or bl_obj.type == 'LIGHT':
		obj_mat = obj_mat @ matrix_forward
	elif bl_obj.type == 'LIGHT_PROBE':
		bl_probe = bl_obj.data
		if bl_probe.type == 'PLANAR':
			obj_mat = obj_mat @ Matrix.Scale(0.05, 4)
		elif bl_probe.type == 'CUBEMAP':
			if bl_probe.influence_type == 'BOX':
				size = bl_probe.influence_distance * 100
				obj_mat = obj_mat @ Matrix.Scale(size, 4)

	obj_mat.freeze() # TODO: check if this is needed
	return obj_mat


def collect_object_metadata(obj_name, obj_type, obj):
	metadata = None
	found_metadata = False
	obj_props = obj.keys()
	for prop_name in obj_props:
		if prop_name in {"_RNA_UI", "cycles", "cycles_visibility"}:
			continue
		if prop_name.startswith("archipack_"):
			continue
		if metadata is None:
			names = (obj_type, obj_name)
			metadata = Node("MetaData", {"name": "%s_%s"%names, "reference":"%s.%s"%names } )

		out_value = prop_value = obj[prop_name]
		prop_type = type(prop_value)
		out_type = None
		if prop_type is str:
			out_type = "String"
		elif prop_type in {float, int}:
			out_type = "Float"
			out_value = f(prop_value)
		elif prop_type is idprop.types.IDPropertyArray:
			out_type = "Vector"
			out_value = ",".join(f(v) for v in prop_value)
		elif prop_type is idprop.types.IDPropertyGroup:
			if len(out_value) == 0:
				continue
			out_type = "String"
			out_value = str(prop_value.to_dict())
		# elif prop_type is list:
			# archipack uses some list props, I don't think these are useful
			# but we should check if there's something specific we should do.
		else:
			log.error("%s: %s has unsupported metadata with type:%s" % (obj_type, obj_name, prop_type))
			# write as string, and sanitize output
			out_type = "String"
			out_value = str(out_value)

		if out_type == "String":
			out_value = out_value.replace("<", "&lt;")
			out_value = out_value.replace(">", "&gt;")
			out_value = out_value.replace('"', "&quot;")

		kvp = Node("KeyValueProperty", {"name": prop_name, "val": out_value, "type": out_type } )
		metadata.push(kvp)
		found_metadata = True
	if found_metadata:
		datasmith_context["metadata"].append(metadata)

def node_value(name, value):
	return Node(name, {'value': '{:6f}'.format(value)})
def f(value):
	return '{:6f}'.format(value)

def collect_environment(world):

	if not world.use_nodes:
		return

	log.info("Collecting environment")
	nodes = world.node_tree
	output = nodes.get_output_node('EEVEE') or nodes.get_output_node('ALL') or nodes.get_output_node('CYCLES')
	background_node = output.inputs['Surface'].links[0].from_node
	if not 'Color' in background_node.inputs:
		return
	if not background_node.inputs['Color'].links:
		return
	source_node = background_node.inputs['Color'].links[0].from_node
	if source_node.type != 'TEX_ENVIRONMENT':
		log.info("Background texture is "+ source_node.type)
		return

	log.info("found environment, collecting...")
	image = source_node.image

	tex_name = sanitize_name(image.name)
	get_or_create_texture(tex_name, image)

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

	from . import bl_info
	plugin_version = bl_info['version']
	plugin_version_string = "%s.%s.%s" % plugin_version
	n.push(Node('Version', children=[plugin_version_string]))
	n.push(Node('SDKVersion', children=['4.24E0']))
	n.push(Node('Host', children=['Blender']))

	blender_version = bpy.app.version_string
	n.push(Node('Application', {
		'Vendor': 'Blender Foundation',
		'ProductName': 'Blender',
		'ProductVersion': blender_version,
		}))

	import os, platform
	os_name = "%s %s" % (platform.system(), platform.release())
	user_name = os.getlogin()

	n.push(Node('User', {
		'ID': user_name,
		'OS': os_name,
		}))
	return n


# in_type can be SRGB, LINEAR or NORMAL
def get_or_create_texture(in_name, in_image, in_type='SRGB'):
	textures = datasmith_context["textures"]
	for name, tex, _ in textures:
		if name == in_name:
			return tex
	log.debug("collecting texture:%s" % in_name)

	new_tex = (in_name, in_image, in_type)
	textures.append(new_tex)
	return new_tex

def get_datasmith_curves_image():
	log.info("baking curves")

	curve_list = datasmith_context["material_curves"]
	if curve_list is None:
		return None

	curves_image = None
	if "datasmith_curves" in bpy.data.images:
		curves_image = bpy.data.images["datasmith_curves"]
	else:
		curves_image = bpy.data.images.new(
			"datasmith_curves",
			DATASMITH_TEXTURE_SIZE,
			DATASMITH_TEXTURE_SIZE,
			alpha=True,
			float_buffer=True
		)
		curves_image.colorspace_settings.is_data = True
		curves_image.file_format = 'OPEN_EXR'

	curves_image.pixels[:] = curve_list.reshape((-1,))
	return curves_image


TEXTURE_MODE_DIFFUSE = "0"
TEXTURE_MODE_SPECULAR = "1"
TEXTURE_MODE_NORMAL = "2"
TEXTURE_MODE_NORMAL_GREEN_INV = "3"
TEXTURE_MODE_DISPLACE = "4"
TEXTURE_MODE_OTHER = "5"
TEXTURE_MODE_BUMP = "6" # this converts textures to normal maps automatically

# saves image, and generates node with image description to add to export
def save_texture(texture, basedir, folder_name, minimal_export = False, use_gamma_hack=False):
	name, image, img_type = texture

	log.info("writing texture:"+name)

	ext = ".png"
	if image.file_format == 'JPEG':
		ext = ".jpg"
	elif image.file_format == 'HDR':
		ext = ".hdr"
	elif image.file_format == 'OPEN_EXR':
		ext = ".exr"
	elif image.file_format == 'TARGA' or image.file_format == 'TARGA_RAW':
		ext = ".tga"

	safe_name = sanitize_name(name) + ext
	image_path = path.join(basedir, folder_name, safe_name)
	skip_image = minimal_export and not path.exists(image_path)

	# fix for invalid images, like one in mr_elephant sample.
	valid_image = (image.channels != 0)
	if valid_image and not skip_image:
		source_path = image.filepath_from_user()

		if image.packed_file:
			with open(image_path, "wb") as f:
				f.write(image.packed_file.data)
		elif source_path and source_path != image_path:
			shutil.copyfile(source_path, image_path)
		else:
			image.filepath_raw = image_path
			image.save()
			if source_path:
				image.filepath_raw = source_path

	n = Node('Texture')
	n['name'] = name
	n['file'] = path.join(folder_name, safe_name)
	n['rgbcurve'] = 0.0
	n['srgb'] = "1" # this parameter is only read on 4.25 onwards

	n['texturemode'] = TEXTURE_MODE_DIFFUSE
	if image.file_format == 'HDR':
		n['texturemode'] = TEXTURE_MODE_OTHER
		n['rgbcurve'] = "1.000000"
	elif img_type == 'NORMAL':
		n['texturemode'] = TEXTURE_MODE_NORMAL_GREEN_INV
		n['srgb'] = "2" # only read on 4.25 onwards, but we can still write it
	elif image.colorspace_settings.is_data:
		n['texturemode'] = TEXTURE_MODE_SPECULAR
		n['srgb'] = "2" # only read on 4.25 onwards, but we can still write it
		if use_gamma_hack:
			n['rgbcurve'] = "0.454545"

	n['texturefilter'] = "3"
	if valid_image:
		img_hash = calc_hash(image_path)
		n.push(Node('Hash', {'value': img_hash}))
	return n


def calc_hash(image_path):
	hash_md5 = hashlib.md5()
	with open(image_path, "rb") as f:
		for chunk in iter(lambda: f.read(4096), b""):
			hash_md5.update(chunk)
	return hash_md5.hexdigest()


datasmith_context = None
def collect_and_save(context, args, save_path):

	start_time = time.monotonic()

	global datasmith_context
	datasmith_context = {
		"objects": [],
		"anim_objects": [],
		"textures": [],
		"meshes": [],
		"materials": [],
		"material_curves": None,
		"metadata": [],
		"compatibility_mode": args["compatibility_mode"],
		"libraries": {},
	}

	log.info("collecting objects")
	datasmith_context['depsgraph'] = context.evaluated_depsgraph_get()
	all_objects = context.scene.objects
	root_objects = [obj for obj in all_objects if obj.parent is None]

	objects = []

	selected_only = args["export_selected"]
	apply_modifiers = args["apply_modifiers"]
	minimal_export = args["minimal_export"]
	export_animations = args["export_animations"]

	if export_animations:
		frame_at_export_time = context.scene.frame_current
		frame_start = context.scene.frame_start
		frame_end = context.scene.frame_end

	write_metadata = args["write_metadata"]

	for obj in root_objects:
		uobj = collect_object(obj,
			selected_only=selected_only,
			apply_modifiers=apply_modifiers,
			export_animations=export_animations,
			export_metadata=write_metadata,
		)
		if uobj:
			objects.append(uobj)

	log.info("collecting animations")
	anims = []
	if export_animations:

		# TODO: found a bit late about this: we need to test and profile
		# https://docs.blender.org/api/current/bpy_extras.anim_utils.html

		anim_objs = datasmith_context["anim_objects"]

		num_frames = frame_end - frame_start + 1
		num_objects = len(anim_objs)
		object_timelines = [[Matrix() for frame in range(num_frames)] for obj in range(num_objects)]
		object_animates = [False for num in range(num_objects)]
		# collect phase?

		for arr_idx, frame_idx in enumerate(range(frame_start, frame_end+1)):

			context.scene.frame_set(frame_idx)

			for obj_idx, obj in enumerate(anim_objs):

				obj_mat = collect_object_transform(obj[0])
				object_timelines[obj_idx][arr_idx] = obj_mat

				if arr_idx == 0:
					continue

				if not object_animates[obj_idx]:
					changed = obj_mat != object_timelines[obj_idx][arr_idx -1]
					if changed:
						object_animates[obj_idx] = True

		anims_strings = []
		# write phase:
		to_deg = 360 / math.tau
		rot_fix = np.array((to_deg, -to_deg, to_deg))
		for idx, timeline in enumerate(object_timelines):
			if not object_animates[idx]:
				continue
			log.error(f"writing obj:{idx}")

			timeline_repr = ['''{
				"actor": "''', anim_objs[idx][1], '",'
			]

			translations = np.empty((num_frames, 4), dtype=np.float32)
			rotations = np.empty((num_frames, 4), dtype=np.float32)
			scales = np.empty((num_frames, 4), dtype=np.float32)
			translations[:, 0] = np.arange(frame_start, frame_end+1)
			rotations[:, 0] = np.arange(frame_start, frame_end+1)
			scales[:, 0] = np.arange(frame_start, frame_end+1)

			for frame_idx, frame_mat in enumerate(timeline):
				loc, rot, scale = frame_mat.decompose()
				tx_slice = (frame_idx, slice(1,4))
				translations[frame_idx, 1:4] = loc
				rotations[frame_idx, 1:4] = rot_fix * rot.to_euler('XYZ')
				scales[frame_idx, 1:4] = scale

			trans_expression = ",".join(
				'{"id":%d,"x":%f,"y":%f,"z":%f}'% tuple(v)
				for v in translations
			)
			timeline_repr.extend(('"trans":[', trans_expression, '],'))

			rot_expression = ",".join(
				'{"id":%d,"x":%f,"y":%f,"z":%f}'% tuple(v)
				for v in rotations
			)
			timeline_repr.extend(('"rot":[', rot_expression, '],'))

			scale_expression = ",".join(
				'{"id":%d,"x":%f,"y":%f,"z":%f}'% tuple(v)
				for v in scales
			)
			timeline_repr.extend(('"scl":[', scale_expression, '],'))

			timeline_repr.append('"type":"transform"}')
			result = "".join(timeline_repr)
			anims_strings.append(result)

		if anims_strings:
			output = ["""
			{
		"version": "0.1",
		"fps": """,
			str(context.scene.render.fps),
		""",
		"animations": [""",
				",".join(anims_strings),
				"]}"
			]

			output_text = "".join(output)
			anims.append(output_text)

		# cleanup
		context.scene.frame_set(frame_at_export_time)


	environment = collect_environment(context.scene.world)

	log.info("Collecting materials")
	materials = datasmith_context["materials"]
	unique_materials = []
	for material in materials:
		found = False
		for mat in unique_materials:
			if material[0] is mat[0]:
				found = True
				break
		if not found:
			unique_materials.append(material)
	material_nodes = [collect_pbr_material(mat) for mat in unique_materials]

	curves_image = get_datasmith_curves_image()
	if curves_image:
		get_or_create_texture("datasmith_curves", curves_image)

	log.info("finished collecting, now saving")

	basedir, file_name = path.split(save_path)
	folder_name = file_name + '_Assets'
	# make sure basepath_Assets directory exists
	try:
		os.makedirs(path.join(basedir, folder_name))
	except FileExistsError as e:
		pass

	log.info("writing anims")
	anim_nodes = []
	for anim in anims:

		filename = path.join(basedir, folder_name, "anim_new.json")
		log.info("writing to file:%s" % filename)
		with open(filename, 'w') as f:
			f.write(output_text)

		anim = Node("LevelSequence", {"name": "anim_new"})
		anim.push(Node("File", {"path": f"{folder_name}/anim_new.json"}))
		anim_nodes.append(anim)




	log.info("writing meshes")
	for mesh in datasmith_context["meshes"]:
		mesh.save(basedir, folder_name)



	log.info("writing textures")

	tex_nodes = []
	use_gamma_hack = args["use_gamma_hack"]
	for tex in datasmith_context["textures"]:
		tex_node = save_texture(tex, basedir, folder_name, minimal_export, use_gamma_hack)
		tex_nodes.append(tex_node)

	log.info("building XML tree")

	n = get_file_header()

	for anim in anim_nodes:
		n.push(anim)

	for obj in objects:
		n.push(obj)

	if environment:
		for env in environment:
			n.push(env)

	for mesh in datasmith_context["meshes"]:
		n.push(mesh.node())
	for mat in material_nodes:
		n.push(mat)

	for tex in tex_nodes:
		n.push(tex)

	for metadata in datasmith_context["metadata"]:
		n.push(metadata)

	end_time = time.monotonic()
	total_time = end_time - start_time

	log.info("generating datasmith data took:%f"%total_time)
	n.push(
		Node("Export", {"Duration":total_time})
	)

	log.info("generating xml")
	result = n.string_rep(first=True)

	filename = path.join(basedir, file_name + '.udatasmith')
	log.info("writing to file: %s" % filename)

	with open(filename, 'w') as f:
		f.write(result)
	log.info("export finished")



def save(context, *, filepath, **kwargs):

	handler = None
	use_logging = bool(kwargs["use_logging"])

	if use_logging:
		log_path = filepath + ".log"
		handler = logging.FileHandler(log_path, mode='w')

		formatter = logging.Formatter(
			fmt='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
			datefmt='%Y-%m-%d %H:%M:%S'
		)
		handler.setFormatter(formatter)
		log.addHandler(handler)
		log.setLevel(logging.DEBUG)
		handler.setLevel(logging.DEBUG)
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
			log.info("Finished logging to path:" + log_path)
			handler.close()
			log.removeHandler(handler)

	return {'FINISHED'}

