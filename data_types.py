# Copyright Andrés Botero 2019

import struct
import os
from os import path
import itertools
import bpy
import numpy as np
import logging
import hashlib
log = logging.getLogger("bl_datasmith")

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
	flat_data = None
	output = b''
	if isinstance(data, np.ndarray):

		output += struct.pack('<I', length)
		output += data.tobytes()
	else:
		flat_data = flatten(data)
		output = struct.pack(data_struct, length, *flat_data)
	if io:
		io.write(output)
	return output

def write_data(io, data_struct, *args):
	data_struct = '<' + data_struct
	packed = struct.pack(data_struct, *args)
	io.write(packed)

def write_null(io, num_bytes):
	io.write(b'\0' * num_bytes)

def write_string(io, string):
	string_bytes = string.encode('utf-8') + b'\0'
	length = len(string_bytes)
	io.write(struct.pack('<I', length))
	io.write(string_bytes)


def is_char_valid(in_char):
	char_num = ord(in_char)
	if char_num > 47 and char_num < 58:
		return True
	elif char_num > 64 and char_num < 99:
		return True
	elif char_num > 96 and char_num < 123:
		return True
	return False

def sanitize_name(name):
	first_char = ""
	if not is_char_valid(name[0]):
		first_char = "DS"
	sanitized = "".join(letter if is_char_valid(letter) else '_' for letter in name)
	return first_char + sanitized

def f(x):
	return '{:6f}'.format(x)

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

	def __init__(self, name):
		self.name = name

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

		self.test = []

		self.relative_path = None
		self.hash = ''

	# this may need some work, found some documentation:
	# Engine/Source/Developer/Rawmesh
	def write_to_path(self, path):
		with open(path, 'wb') as file:
			log.debug("writing mesh:"+self.name)
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

			# b2 = write_array_data(file, 'fff', self.test)
			# print(self.vertices)
			# print(self.test)
			# print(b1[0:10])
			# print(b2[0:10])


			# per vertexloop
			write_array_data(file, 'I', self.triangles) # WedgeIndices


			write_null(file, 4) # WedgeTangentX
			write_null(file, 4) # WedgeTangentY
			write_array_data(file, 'fff', self.vertex_normals) # WedgeTangentZ

			num_uvs = len(self.uvs)
			for idx in range(num_uvs):
				write_array_data(file, 'ff', self.uvs[idx]) # WedgeTexCoords[0]

			num_empty_uvs = 8 - num_uvs
			write_null(file, 4 * num_empty_uvs) # WedgeTexCoords[n..7]
			write_array_data(file, 'BBBB', self.vertex_colors) # WedgeColors
			# b2 = write_array_data(file, 'BBBB', self.test) # WedgeTexCoords[0]

			# print("old and new are same? {}".format(b1 == b2))
			# print(b2[4:24])
			# print(self.vertex_colors.tobytes()[:20])
			# print(self.vertex_colors[:20])
			# print(self.test[:20])

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
		log.debug("writing mesh:"+self.name)
		self.relative_path = path.join(folder_name, self.name + '.udsmesh')
		abs_path = path.join(basedir, self.relative_path)
		self.write_to_path(abs_path)
		self.hash = calc_hash(abs_path)


def calc_hash(image_path):
	hash_md5 = hashlib.md5()
	with open(image_path, "rb") as f:
		for chunk in iter(lambda: f.read(4096), b""):
			hash_md5.update(chunk)
	return hash_md5.hexdigest()

