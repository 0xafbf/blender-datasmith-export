# Copyright Andrés Botero 2019

import struct
from xml.etree import ElementTree
import os
from os import path
import itertools
import bpy
from functools import reduce

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

def read_array_data(io, data_struct):
	struct_size = struct.calcsize(data_struct)
	data_struct = "<" + data_struct # force little endianness

	count = struct.unpack("<I", io.read(4))[0]
	data = io.read(count * struct_size)
	unpacked_data = list(struct.iter_unpack(data_struct, data))
	return [tup[0] if len(tup) == 1 else tup for tup in unpacked_data ]


def flatten(it):
	data = []
	for d in it:
		if isinstance(d, float) or isinstance(d, int):
			data.append(d)
		else:
			data += [*d]
	return data

def write_array_data(io, data_struct, data):
	# first get data length
	length = len(data)
	data_struct = '<I' + (data_struct) * length
	flat_data = flatten(data)
	output = struct.pack(data_struct, length, *flat_data)
	io.write(output)

def read_data(io, data_struct):
	struct_size = struct.calcsize(data_struct)
	data_struct = "<" + data_struct	# force little endianness
	data = io.read(struct_size)
	unpacked_data = struct.unpack(data_struct, data)
	return unpacked_data

def write_data(io, data_struct, *args):
	data_struct = '<' + data_struct
	packed = struct.pack(data_struct, *args)
	io.write(packed)

def read_string(io):
	count = struct.unpack("<I", io.read(4))[0]
	data = io.read(count)
	return data.decode('utf-8').strip('\0')


def write_null(io, num_bytes):
	io.write(b'\0' * num_bytes)

def write_string(io, string):
	string_bytes = string.encode('utf-8') + b'\0'
	length = len(string_bytes)
	io.write(struct.pack('<I', length))
	io.write(string_bytes)

def sanitize_name(name):
	return name.replace('.', '_dot')


def f(x):
	return '{:6f}'.format(x)

# i am introducing this as I want to change some of how the API works


class Node:
	prefix = ""
	def __init__(self, name, attrs=None, children=None):
		self.name = name
		self.children = children or []
		if attrs:
			assert type(attrs) is dict
		self.attrs = attrs or {}

	def __getitem__(self, key):
		return self.attrs[key]

	def __setitem__(self, key, value):
		self.attrs[key] = value

	def string_rep(self, first=False):
		previous_prefix = Node.prefix
		if first:
			Node.prefix = ""
		else:
			Node.prefix += "\t"
		output = Node.prefix + '<{}'.format(self.name)
		if first:
			Node.prefix = "\n"
		for attr in self.attrs:
			output += ' {key}="{value}"'.format(key=attr, value=self.attrs[attr])

		if self.children:
			output += '>'
			for child in self.children:
				output += str(child)
			if len(self.children) == 1 and type(self.children[0]) == str:
				output += '</{}>'.format(self.name)
			else:
				output += Node.prefix + '</{}>'.format(self.name)
		else:
			output += '/>'
		Node.prefix = previous_prefix
		return output

	def __str__(self):
		return self.string_rep()
	def push(self, value):
		size = len(self.children)
		self.children.append(value)
		return size


def node_value(name, value):
	return Node(name, {'value': f(value)})


class UDMesh():
	node_type = 'StaticMesh'
	node_group = 'meshes'

	def __init__(self, path=None, node:ElementTree.Element = None, name=None):
		self.name = name
		if path:
			self.init_with_path(path)

		else:
			self.init_fields()

		self.check_fields() # to test if it is possible for these fields to have different values

	def init_fields(self):
		self.source_models = 'SourceModels'
		self.struct_property = 'StructProperty'
		self.datasmith_mesh_source_model = 'DatasmithMeshSourceModel'

		self.materials = {}

		self.tris_material_slot = []
		self.tris_smoothing_group = []
		self.vertices = []
		self.triangles = []
		self.vertex_normals = []
		self.uvs = []
		self.vertex_colors = [] # In 0-255 range
		self.relative_path = None
		self.hash = ''


	def check_fields(self):
		assert self.name != None
		assert self.source_models == 'SourceModels'
		assert self.struct_property == 'StructProperty'
		assert self.datasmith_mesh_source_model == 'DatasmithMeshSourceModel'


			# this may need some work, found some documentation:
			# Engine/Source/Developer/Rawmesh
	def write_to_path(self, path):
		with open(path, 'wb') as file:
			#write_null(file, 8)
			file.write(b'\x01\x00\x00\x00\xfd\x04\x00\x00')

			file_start = file.tell()
			write_string(file, self.name)
			#write_null(file, 5)
			file.write(b'\x00\x01\x00\x00\x00')
			write_string(file, self.source_models)
			write_string(file, self.struct_property)
			write_null(file, 8)

			write_string(file, self.datasmith_mesh_source_model)

			write_null(file, 25)

			size_loc = file.tell() # here we have to write the rawmesh size two times
			write_data(file, 'II', 0, 0) # just some placeholder data, to rewrite at the end

			file.write(b'\x7d\x00\x00\x00\x00\x00\x00\x00') #125 and zero

			#here starts rawmesh
			mesh_start = file.tell()
			file.write(b'\x01\x00\x00\x00') # raw mesh version
			file.write(b'\x00\x00\x00\x00') # raw mesh lic  version

			# further analysis revealed:
			# this loops are per triangle
			write_array_data(file, 'I', self.tris_material_slot)
			write_array_data(file, 'I', self.tris_smoothing_group)
			# per vertex
			write_array_data(file, 'fff', self.vertices) # VertexPositions
			# per vertexloop
			write_array_data(file, 'I', self.triangles) # WedgeIndices
			write_null(file, 4) # WedgeTangentX
			write_null(file, 4) # WedgeTangentY
			write_array_data(file, 'fff', self.vertex_normals) # WedgeTangentZ
			write_array_data(file, 'ff', self.uvs) # WedgeTexCoords[0]
			write_null(file, 4 * 7) # WedgeTexCoords[1..7]
			write_array_data(file, 'BBBB', self.vertex_colors) # WedgeColors
			write_null(file, 4) # MaterialIndexToImportIndex

			#here ends rawmesh
			mesh_end = file.tell()

			write_null(file, 16)
			write_null(file, 4)
			file_end = file.tell()

			mesh_size = mesh_end-mesh_start
			file.seek(size_loc)
			write_data(file, 'II', mesh_size, mesh_size)

			file.seek(0)
			write_data(file, 'II', 1, file_end - file_start)

	def node(self):
		n = Node('StaticMesh')
		n['label'] = self.name
		n['name'] = self.name

		for idx, m in self.materials.items():
			n.push(Node('Material', {'id':idx, 'name':m}))
		if self.relative_path:
			path = self.relative_path.replace('\\', '/')
			n.push(Node('file', {'path':path }))
		n.push(Node('LightmapUV', {'value': '-1'}))
		n.push(Node('Hash', {'value': self.hash}))
		return n

	def save(self, basedir, folder_name):
		self.relative_path = path.join(folder_name, self.name + '.udsmesh')
		abs_path = path.join(basedir, self.relative_path)
		self.write_to_path(abs_path)

		import hashlib
		hash_md5 = hashlib.md5()
		with open(abs_path, "rb") as f:
			for chunk in iter(lambda: f.read(4096), b""):
				hash_md5.update(chunk)
		self.hash = hash_md5.hexdigest()


class UDTexture():
	node_type = 'Texture'
	node_group = 'textures'

	TEXTURE_MODE_DIFFUSE = "0"
	TEXTURE_MODE_SPECULAR = "1"
	TEXTURE_MODE_NORMAL = "2"
	TEXTURE_MODE_NORMAL_GREEN_INV = "3"
	TEXTURE_MODE_DISPLACE = "4"
	TEXTURE_MODE_OTHER = "5"
	TEXTURE_MODE_BUMP = "6" # this converts textures to normal maps automatically

	def __init__(self, *, name=None):
		self.name = name
		self.image = None
		self.texture_mode = UDTexture.TEXTURE_MODE_OTHER

	def abs_path(self):
		safe_name = sanitize_name(self.name)
		return "{}/{}.png".format(UDScene.current_scene.export_path, safe_name)

	def node(self):
		n = Node('Texture')
		n['name'] = self.name
		n['file'] = self.abs_path()
		n['rgbcurve'] = 0.0

		# FIXME: Find a way to tell unreal to read as non-srgb
		if self.image.colorspace_settings.is_data:
			self.texture_mode = UDTexture.TEXTURE_MODE_SPECULAR
			n['rgbcurve'] = "0.454545"
		else:
			self.texture_mode = UDTexture.TEXTURE_MODE_DIFFUSE


		n['texturemode'] = self.texture_mode
		n.push(Node('Hash', {'value': self.hash}))
		return n

	def save(self):
		image_path = path.join(UDScene.current_scene.basedir, self.abs_path())
		old_path = self.image.filepath_raw
		self.image.file_format = 'PNG'
		self.image.filepath_raw = image_path
		self.image.save()
		if old_path:
			self.image.filepath_raw = old_path
		import hashlib
		hash_md5 = hashlib.md5()
		with open(image_path, "rb") as f:
			for chunk in iter(lambda: f.read(4096), b""):
				hash_md5.update(chunk)
		self.hash = hash_md5.hexdigest()


class UDActor():

	node_type = 'Actor'
	node_group = 'objects'

	def get_field(self, cls, name, **kwargs):
		group = getattr(self, cls.node_group)
		if not group:
			log.error("trying to get invalid group")

		if name in group:
			return group[name]

		new_object = cls(name=name, **kwargs)
		group[name] = new_object
		return new_object

	class Transform():
		def __init__(self, tx=0, ty=0, tz=0,
					 qw=0, qx=0, qy=0, qz=0,
					 sx=0, sy=0, sz=0, qhex = None):
			self.loc = (float(tx), float(ty), float(tz))
			self.rot = (float(qw), float(qx), float(qy), float(qz))
			self.scale = (float(sx), float(sy), float(sz))
			# don't know what qhex is
		def node(self):
			n = Node('Transform')
			n['tx'] = f(self.loc.x)
			n['ty'] = f(self.loc.y)
			n['tz'] = f(self.loc.z)
			n['qw'] = f(self.rot.w)
			n['qx'] = f(self.rot.x)
			n['qy'] = f(self.rot.y)
			n['qz'] = f(self.rot.z)
			n['sx'] = f(self.scale.x)
			n['sy'] = f(self.scale.y)
			n['sz'] = f(self.scale.z)
			return n


	def __init__(self, *, node=None, name=None, layer='Default'):
		self.transform = UDActor.Transform()
		self.objects = {}
		self.materials = {}
		self.name = name
		self.layer = layer
		if node: # for import
			self.name = node.attrib['name']
			self.layer = node.attrib['layer']
			node_transform = node.find('Transform')
			if node_transform is not None:
				self.transform = UDActor.Transform(**node_transform.attrib)
			else:
				import pdb; pdb.set_trace()
			node_children = node.find('children')
			if node_children is not None:
				for child in node_children:
					name = child.attrib["name"]
					if child.tag == "Actor":
						UDActor(name=name, node=child)
					if child.tag == "ActorMesh":
						UDActorMesh(name=name, node=child)

	def node(self):
		n = Node('Actor')
		n['name'] = self.name
		n['layer'] = self.layer
		n.push(self.transform.node())

		if len(self.objects) > 0:
			children_node = Node("children");
			for name, child in self.objects.items():
				children_node.push(child.node())
			n.push(children_node)
		return n


class UDActorMesh(UDActor):

	node_type = 'ActorMesh'

	def __init__(self, *, node=None, name=None):
		super().__init__(node=node, name=name)
		if node:
			self.mesh = node.find('mesh').attrib['name']
			self.materials = {n.attrib['id']: n.attrib['name'] for n in node.findall('material')}

	def node(self):
		n = super().node()
		n.name = 'ActorMesh'
		n.push(Node('mesh', {'name': self.mesh}))
		return n


class UDActorLight(UDActor):

	node_type = 'Light'

	LIGHT_POINT = 'PointLight'
	LIGHT_SPOT = 'SpotLight'

	LIGHT_UNIT_CANDELAS = 'Candelas'

	def __init__(self, *, node=None, name=None, light_type = LIGHT_POINT, color = (1.0,1.0,1.0)):
		super().__init__(node=node, name=name)
		self.type = light_type
		self.intensity = 1500
		self.intensity_units = UDActorLight.LIGHT_UNIT_CANDELAS
		self.color = color
		self.inner_cone_angle = 22.5
		self.outer_cone_angle = 25
		self.post = []
		if node:
			self.parse(node)
	def parse(self, node):
		self.type = node.attrib['type']

		# self.intensity =       	node.find('Intensity').attrib['value']
		# self.intensity_units = 	node.find('IntensityUnits').attrib['value']
		# self.color =           	node.find('Color').attrib['value']
		# self.inner_cone_angle =	node.find('InnerConeAngle').attrib['value']
		# self.outer_cone_angle =	node.find('OuterConeAngle').attrib['value']

	def node(self):
		n = super().node()
		n.name = 'Light'
		n['type'] = self.type
		n['enabled'] = '1'
		val = node_value
		n.push(val('Intensity', self.intensity))
		n.push(Node('IntensityUnits', {'value': self.intensity_units}))
		n.push(Node('Color', {
			'usetemp': '0',
			'temperature': '6500.0',
			'R': f(self.color[0]),
			'G': f(self.color[1]),
			'B': f(self.color[2]),
			}))
		if self.type == UDActorLight.LIGHT_SPOT:
			n.push(val('InnerConeAngle', self.inner_cone_angle))
			n.push(val('OuterConeAngle', self.outer_cone_angle))
		return n

class UDActorCamera(UDActor):

	node_type = 'Camera'

	def __init__(self, *, node=None, name=None):
		super().__init__(node=node, name=name)

		self.sensor_width = 36.0
		self.sensor_aspect_ratio = 1.777778
		self.enable_dof = False
		self.focus_distance = 1000.0
		self.f_stop = 2.8
		self.focal_length = 50.0
		self.look_at_actor = None
		self.post = []
		if node:
			self.parse(node)

	def node(self):
		n = super().node()
		n.name = 'Camera'
		val = node_value
		n.push(val('SensorWidth', self.sensor_width))
		n.push(val('SensorAspectRatio', self.sensor_aspect_ratio))
		n.push(val('FocusDistance', self.focus_distance))
		n.push(val('FStop', self.f_stop))
		n.push(val('FocalLength', self.focal_length))
		n.push(Node('Post'))
		return n


class UDScene():

	node_type = 'DatasmithUnrealScene'
	current_scene = None

	def __init__(self, source=None):
		self.name = 'udscene_name'

		self.materials = set()
		self.material_nodes = []
		self.meshes = {}
		self.objects = {}
		self.textures = {}


	def get_field(self, cls, name, **kwargs):
		group = getattr(self, cls.node_group)

		if name in group:
			return group[name]

		new_object = cls(name=name, **kwargs)
		group[name] = new_object
		return new_object

	def node(self):
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

		for name, obj in self.objects.items():
			n.push(obj.node())
		for name, mesh in self.meshes.items():
			n.push(mesh.node())
		for mat in self.material_nodes:
			n.push(mat)
		for name, tex in self.textures.items():
			n.push(tex.node())

		return n

	def save(self, basedir, name):
		self.name = name
		self.basedir = basedir

		folder_name = name + '_Assets'
		self.export_path = folder_name
		# make sure basepath_Assets directory exists
		try:
			os.makedirs(path.join(basedir, folder_name))
		except FileExistsError as e:
			pass

		for _name, mesh in self.meshes.items():
			mesh.save(basedir, folder_name)
		for _name, tex in self.textures.items():
			tex.save()

		log.info("building XML tree")

		result = self.node().string_rep(first=True)

		pretty_print = False
		if pretty_print:
			from xml.dom import minidom
			result = minidom.parseString(tree).toprettyxml()
		#pretty_xml = tree

		filename = path.join(basedir, self.name + '.udatasmith')
		log.info("writing to file")
		with open(filename, 'w') as f:
			f.write(result)
		log.info("export successful")

