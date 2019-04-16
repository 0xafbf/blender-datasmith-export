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
	return name.replace('.', '_')

# i am introducing this as I want to change some of how the API works


class Node:
	def __init__(self, name, attrs=None, children=None):
		self.name = name
		self.children = children or []
		self.attrs = attrs or {}

	def __getitem__(self, key):
		return self.attrs[key]
	
	def __setitem__(self, key, value):
		self.attrs[key] = value        
		
	def __str__(self):
		output = '<{}'.format(self.name)
		for attr in self.attrs:
			output += ' {key}="{value}"'.format(key=attr, value=self.attrs[attr])
			
		if not self.children:
			output += '/>'
			return output
		output += '>'
		
		for child in self.children:
			
			output += str(child)
			
		output += '</{}>'.format(self.name)
		
		return output
	





class UDElement:
	"""convenience for all elements in the udatasmith file"""
	node_type = 'Element'
	node_group = None

	class UDElementException(Exception):
		pass

	@classmethod
	def new(cls, name, parent=None, **kwargs): # I want to deprecate all this stuff
		if parent is None:
			raise UDElementException('Tried to create an element without a parent.')
		if cls.node_group is None:
			raise UDElementException("%s doesn't override `node_group`, without it, parent registration won't work.")
		
		group = getattr(parent, cls.node_group, {})

		name = sanitize_name(name)

		elem = group.get(name)
		if elem:
			return elem
			
		new_object = cls(parent=parent, name=name, **kwargs)
		
		if not new_object.name:
			raise UDElementException("object created without name")

		group = getattr(parent, cls.node_group, {})
		group[new_object.name] = new_object
		setattr(parent, cls.node_group, group)

		return new_object


	def render(self, parent): # maybe the only thing, but still not convinced
		elem = ElementTree.SubElement(parent, self.node_type)
		elem.attrib['name'] = self.name
		return elem

	def __repr__(self):
		return '{}: {}'.format(type(self).__name__, self.name)


class UDMesh(UDElement):
	node_type = 'StaticMesh'
	node_group = 'meshes'

	def __init__(self, path=None, node:ElementTree.Element = None, parent = None, name=None):
		self.parent = parent
		self.name = name
		if node:
			self.init_with_xmlnode(node)
		elif path:
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
		self.relative_path = None
		self.hash = ''


	def check_fields(self):
		assert self.name != None
		assert self.source_models == 'SourceModels'
		assert self.struct_property == 'StructProperty'
		assert self.datasmith_mesh_source_model == 'DatasmithMeshSourceModel'

	def init_with_xmlnode(self, node:ElementTree.Element):
		self.name = node.attrib['name']
		self.label = node.attrib['label']
		self.relative_path = node.find('file').attrib['path']
		
		parent_path = path.dirname(os.path.abspath(self.parent.path))
		self.init_with_path(path.join(parent_path, self.relative_path))
		# self.materials = {n.attrib['id']: n.attrib['name'] for n in node.iter('Material')}

		# flatten material lists
		material_map = {int(n.attrib['id']): idx for idx, n in enumerate(node.iter('Material'))}
		self.materials = {idx: n.attrib['name'] for idx, n in enumerate(node.iter('Material'))}
		if 0 not in material_map:
			last_index = len(material_map)
			material_map[0] = last_index
			self.materials[last_index] = 'default_material'

		self.tris_material_slot = list(map(lambda x: material_map.get(x, 0), self.tris_material_slot))


	def init_with_path(self, path):
		with open(path, 'rb') as file:


			# this may need some work, found some documentation:
			# Engine/Source/Developer/Rawmesh

			self.a01 = read_data(file, 'II') # a 1 and the whole bytes size
			self.name = read_string(file)

			self.a02 = file.read(5)
			
			self.source_models = read_string(file)
			self.struct_property = read_string(file)
			self.a03 = file.read(8)

			self.datasmith_mesh_source_model = read_string(file)
			
			self.a04 = file.read(25) # just zeros

			self.payload_length = read_data(file, 'II') # this is the size of the rawmesh

			self.a04_b = file.read(8) # this is a 125 and a zero, no idea what it is

			self.raw_mesh_version = read_data(file, 'i') # always 1 for what I can see
			self.raw_mesh_lic_version = read_data(file, 'i') # always 0 for what I can see

			self.tris_material_slot = read_array_data(file, "I") #FaceMaterialIndices
			self.tris_smoothing_group = read_array_data(file, "I") #FaceSmoothingMasks
			
			self.vertices = read_array_data(file, "fff") #VertexPositions
			self.triangles = read_array_data(file, "I") #WedgeIndices
			
			self.a05 = read_array_data(file, "I") # WedgeTangentX (maybe)
			self.a06 = read_array_data(file, "I") # WedgeTangentY (maybe)

			self.vertex_normals = read_array_data(file, "fff") #WedgeTangentZ
			self.uvs = read_array_data(file, "ff") #WedgeTexCoords

			self.a07 = file.read(28) # these may be WedgeTexCoods[1,2...7]
			# ue4 defines 8 texcoord layers, here we read the other seven zeros
			self.a07_b = file.read(4) # these seem to be WedgeColors count
			self.a07_c = file.read(4) # MaterialIndexToImportIndex?

			self.checksum = file.read(16) # I guess this is Guid?
			
			self.a08 = file.read() # And maybe this is bGuidIsHash
			
			# small check here to crash if something is suspicious
			assert len(self.triangles) == len(self.uvs)
			assert len(self.vertex_normals) == len(self.uvs)
			assert self.a08 == b'\x00\x00\x00\x00' # just to be sure its the end
		
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
			write_array_data(file, 'fff', self.vertices)
			# per vertexloop
			write_array_data(file, 'I', self.triangles)
			write_null(file, 8)
			write_array_data(file, 'fff', self.vertex_normals)
			write_array_data(file, 'ff', self.uvs)
			write_null(file, 36)
			
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

	def render(self, parent):
		elem = super().render(parent=parent)
		elem.attrib['label'] = self.name
		for idx, m in enumerate(self.materials):
			ElementTree.SubElement(elem, 'Material', id=str(idx), name=sanitize_name(m))
		if self.relative_path:
			path = self.relative_path.replace('\\', '/')
			ElementTree.SubElement(elem, 'file', path=path)
		lm_uv = ElementTree.SubElement(elem, 'LightmapUV', value='-1')
		ElementTree.SubElement(elem, 'Hash', value=self.hash)
		return elem

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



class UDMaterial(Node):
	node_type = 'Material'
	node_group = 'materials'

	def __init__(self, name: str, node=None, parent=None, **kwargs):
		super().__init__(self.node_type)
		self.attrs['name'] = name

class UDShader(UDElement):
	node_type = 'Shader'
	node_group = 'shaders'
	shader_count = 0
	def __init__(self):
		self.name = "Shader_%d" % (UDShader.shader_count)
		UDShader.shader_count += 1

class UDMasterMaterial(UDMaterial):
	def prop_color(value):
		data = {
			'prop_type': 'Color'
		}
		data['value'] = tuple(map(float, re.match(r"\(R=(-?[\d.]*),G=(-?[\d.]*),B=(-?[\d.]*),A=(-?[\d.]*)\)", src).groups()))
		return data

	def prop_bool(Prop):
		data = {
			'prop_type': 'Bool'
		}
		data['value'] = True if value == 'true' else False
		return data

	def prop_texture(value):
		return {
			'prop_type': 'Texture'
		}

	def prop_float(value):
		return {
			'prop_type': 'Float',
			'value': float(value),
		}

	types = {
		"Color": prop_color,
		"Bool": prop_bool,
		"Texture": prop_texture,
		"Float": prop_float,
	}

	'''sketchup datasmith outputs Master material, it may be different'''
	''' has params Type and Quality'''
	node_type = 'MasterMaterial'
	def __init__(self, *args, node=None, **kwargs):
		super().__init__(*args, node=node, **kwargs)
		self.properties = {}
		if node is not None:
			for prop in node.findall('KeyValueProperty'):
				prop_name = prop.attrib['name']
				prop_type = prop.attrib['type']

				self.properties[name] = UDMasterMaterial.types[prop.attrib['type']](prop.attrib['val'])

	@staticmethod
	def new(name, parent, node):
		ob = UDScene.current_scene.get_field(UDMasterMaterial, name)
		if ob:
			return ob
		return UDMasterMaterial(node=node, name=name)

class UDTexture(UDElement):
	node_type = 'Texture'
	node_group = 'textures'

	@classmethod
	def new(cls, name, node=None,**kwargs):
	# Need to override as it is possible to have textures with the same name
	# but different path
		
		if node:
			folder, file = path.split(node.attrib['file'])
			name = file

		new_object = UDScene.current_scene.get_field(UDTexture, name)

		# TODO: check when loaded textures have same name but different path
		return new_object

	def __init__(self, *, node=None, name=None):
		self.name = name
		self.image = None
		if node:
			self.folder, self.file = path.split(node.attrib['file'])
			self.texturemode = node.attrib['texturemode']

	def abs_path(self):
		ext = 'png'
		if self.image:
			if self.image.file_format == 'PNG':
				pass
				#ext = 'png'
		return "{}/{}.{}".format(UDScene.current_scene.export_path, self.name, ext)

	def __str__(self):
		r = Node('Texture')
		r['name'] = self.name
		r['file'] = self.abs_path()
		r['rgbcurve'] = 1.0
		r['texturemode'] = '5'
		r.children = [
			Node('Hash', {'value': self.hash})
		]
		return str(r)

	def save(self):
		image_path = path.join(UDScene.current_scene.basedir, self.abs_path())
		old_path = self.image.filepath
		self.image.filepath = image_path
		self.image.save()
		self.image.filepath = old_path
		import hashlib
		hash_md5 = hashlib.md5()
		with open(image_path, "rb") as f:
			for chunk in iter(lambda: f.read(4096), b""):
				hash_md5.update(chunk)
		self.hash = hash_md5.hexdigest()


class UDActor(UDElement):

	node_type = 'Actor'
	node_group = 'objects'

	class Transform:
		def __init__(self, tx=0, ty=0, tz=0, 
					 qw=0, qx=0, qy=0, qz=0,
					 sx=0, sy=0, sz=0, qhex = None):
			self.loc = (float(tx), float(ty), float(tz))
			self.rot = (float(qw), float(qx), float(qy), float(qz))
			self.scale = (float(sx), float(sy), float(sz))
			# don't know what qhex is
		def render(self, parent):
			f = lambda n: "{:.6f}".format(n)
			tx, ty, tz = self.loc
			qw, qx, qy, qz = self.rot
			sx, sy, sz = self.scale
			return ElementTree.SubElement(parent, 'Transform',
					tx=f(tx), ty=f(ty), tz=f(tz), 
					qw=f(qw), qx=f(qx), qy=f(qy), qz=f(qz),
					sx=f(sx), sy=f(sy), sz=f(sz),
				)



	def __init__(self, *, parent, node=None, name=None, layer='Default'):
		self.transform = UDActor.Transform()
		self.objects = {}
		self.materials = {}
		self.name = name
		self.layer = layer
		if node:
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
						UDActor.new(name=name, node=child, parent=self)
					if child.tag == "ActorMesh":
						UDActorMesh.new(name=name, node=child, parent=self)

	def render(self, parent):
		elem = super().render(parent)
		elem.attrib['layer'] = self.layer
		self.transform.render(elem)

		if len(self.objects) > 0:
			children = ElementTree.SubElement(elem, 'children')
			for name, child in self.objects.items():
				child.render(children)

		return elem


class UDActorMesh(UDActor):

	node_type = 'ActorMesh'

	def __init__(self, *, parent, node=None, name=None):
		super().__init__(parent=parent, node=node, name=name)
		if node:
			self.mesh = node.find('mesh').attrib['name']
			self.materials = {n.attrib['id']: n.attrib['name'] for n in node.findall('material')}

	def render(self, parent):
		elem = super().render(parent)
		mesh = ElementTree.SubElement(elem, 'mesh')
		mesh.attrib['name'] = sanitize_name(self.mesh)
		

class UDActorLight(UDActor):

	node_type = 'Light'

	LIGHT_POINT = 'PointLight'
	LIGHT_SPOT = 'SpotLight'

	LIGHT_UNIT_CANDELAS = 'Candelas'

	def __init__(self, *, parent, node=None, name=None, light_type = LIGHT_POINT, color = (1.0,1.0,1.0)):
		super().__init__(parent=parent, node=node, name=name)
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

	def render(self, parent):
		elem = super().render(parent)
		elem.attrib['type'] = self.type
		elem.attrib['enabled'] = '1'
		ElementTree.SubElement(elem, 'Intensity',     	value='{:6f}'.format(self.intensity))
		ElementTree.SubElement(elem, 'IntensityUnits',	value=self.intensity_units)
		f= '{:6f}'
		ElementTree.SubElement(	elem, 'Color', usetemp='0', temperature='6500.0',
								R=f.format(self.color[0]),
								G=f.format(self.color[1]),
								B=f.format(self.color[2]),
		)
		if self.type == UDActorLight.LIGHT_SPOT:
			ElementTree.SubElement(elem, 'InnerConeAngle',	value='{:6f}'.format(self.inner_cone_angle))
			ElementTree.SubElement(elem, 'OuterConeAngle',	value='{:6f}'.format(self.outer_cone_angle))
		return elem


class UDActorCamera(UDActor):

	node_type = 'Camera'

	def __init__(self, *, parent, node=None, name=None):
		super().__init__(parent=parent, node=node, name=name)
		self.sensor_width = 36.0
		self.sensor_aspect_ratio = 1.777778
		self.focus_distance = 1000.0
		self.f_stop = 5.6
		self.focal_length = 32.0
		self.post = []
		if node:
			self.parse(node)

	def parse(self, node):
		self.sensor_width =       	node.find('SensorWidth').attrib['value']
		self.sensor_aspect_ratio =	node.find('SensorAspectRatio').attrib['value']
		self.focus_distance =     	node.find('FocusDistance').attrib['value']
		self.f_stop =             	node.find('FStop').attrib['value']
		self.focal_length =       	node.find('FocalLength').attrib['value']

	def render(self, parent):
		elem = super().render(parent)
		ElementTree.SubElement(elem, 'SensorWidth',      	value='{:6f}'.format(self.sensor_width))
		ElementTree.SubElement(elem, 'SensorAspectRatio',	value='{:6f}'.format(self.sensor_aspect_ratio))
		ElementTree.SubElement(elem, 'FocusDistance',    	value='{:6f}'.format(self.focus_distance))
		ElementTree.SubElement(elem, 'FStop',            	value='{:6f}'.format(self.f_stop))
		ElementTree.SubElement(elem, 'FocalLength',      	value='{:6f}'.format(self.focal_length))
		ElementTree.SubElement(elem, 'Post')
		return elem





class UDScene(UDElement):

	node_type = 'DatasmithUnrealScene'

	current_scene = None

	def __init__(self, source=None):
		self.init_fields()
		if type(source) is str:
			self.path = source
			self.init_with_path(self.path)

		self.check_fields() # to test if it is possible for these fields to have different values

	def get_field(self, cls, name):
		group = getattr(self, cls.node_group)
		if not group:
			log.error("trying to get invalid group")

		if name in group:
			return group[name]

		new_object = cls(name=name)
		group[name] = new_object
		return new_object


	def init_fields(self):
		self.name = 'udscene_name'

		self.materials = {}
		self.meshes = {}
		self.objects = {}
		self.textures = {}

	def check_fields(self):
		pass

	def init_with_path(self, path):

		tree = ElementTree.parse(path)
		root = tree.getroot()

		self.version = root.find('Version').text
		self.sdk_version = root.find('SDKVersion').text
		self.host = root.find('Host').text
		# there are other bunch of data that i'll skip for now

		classes = [
			UDTexture,
			UDMaterial,
			UDMasterMaterial,
			UDMesh,
			UDActor,
			UDActorMesh,
			UDActorCamera,
			UDActorLight,
		]

		mappings = {cls.node_type:cls for cls in classes} 

		UDScene.current_scene = self
		for node in root:
			name = node.get('name') # most relevant nodes have a name as identifier
			cls = mappings.get(node.tag)
			if cls:
				cls.new(parent=self, name=name, node=node)

		print("loaded")

	def render(self):
		tree = ElementTree.Element('DatasmithUnrealScene')

		version = ElementTree.SubElement(tree, 'Version')
		version.text = '0.20' # to get from context?

		sdk = ElementTree.SubElement(tree, 'SDKVersion')
		sdk.text = '4.20E1' # to get from context?

		host = ElementTree.SubElement(tree, 'Host')
		host.text = 'Blender'

		application = ElementTree.SubElement(tree, 'Application', Vendor='Blender', ProductName='Blender', ProductVersion='2.80')
		user = ElementTree.SubElement(tree, 'User', ID='00000000000000000000000000000000', OS='Windows 8.1')

		for name, obj in self.objects.items():
			obj.render(tree)
		for name, mesh in self.meshes.items():
			mesh.render(tree)
		for name, mat in self.materials.items():
			tree.append(ElementTree.XML(str(mat)))
		for name, tex in self.textures.items():
			tree.append(ElementTree.XML(str(tex)))


		return tree

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
		
		tree = self.render()
		txt = ElementTree.tostring(tree)
		from xml.dom import minidom
		pretty_xml = minidom.parseString(txt).toprettyxml()

		filename = path.join(basedir, self.name + '.udatasmith')

		with open(filename, 'w') as f:
			f.write(pretty_xml)
	
