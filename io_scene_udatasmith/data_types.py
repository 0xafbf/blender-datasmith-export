import struct
from xml.etree import ElementTree
import os


def read_array_data(io, data_struct):
    struct_size = struct.calcsize(data_struct)
    data_struct = "<" + data_struct # force little endianness

    count = struct.unpack("<I", io.read(4))[0]
    data = io.read(count * struct_size)
    unpacked_data = list(struct.iter_unpack(data_struct, data))
    return [tup[0] if len(tup) == 1 else tup for tup in unpacked_data ]

def read_data(io, data_struct):
    struct_size = struct.calcsize(data_struct)
    data_struct = "<" + data_struct # force little endianness

    data = io.read(struct_size)
    unpacked_data = struct.unpack(data_struct, data)
    return unpacked_data

def read_string(io):
    count = struct.unpack("<I", io.read(4))[0]
    data = io.read(count)
    return data.decode('utf-8').strip('\0')


class UDMesh:

    def __init__(self, path=None, xmlnode:ElementTree.Element = None, parent = None):
        self.parent = parent
        if xmlnode:
            self.init_with_xmlnode(xmlnode)
        elif path:
            self.init_with_path(path)
        
        else:
            self.init_fields()

        self.check_fields() # to test if it is possible for these fields to have different values

        self.parent.meshes[self.name] = self

    def init_fields(self):
        self.name = 'udmesh_name'
        self.source_models = 'SourceModels'
        self.struct_property = 'StructProperty'
        self.datasmith_mesh_source_model = 'DatasmithMeshSourceModel'

        self.materials = {}

    def check_fields(self):
        assert self.source_models == 'SourceModels'
        assert self.struct_property == 'StructProperty'
        assert self.datasmith_mesh_source_model == 'DatasmithMeshSourceModel'

    def init_with_xmlnode(self, node:ElementTree.Element):
        self.name = node.attrib['name']
        self.label = node.attrib['label']
        self.relative_path = node.find('file').attrib['path']
        
        parent_path = os.path.dirname(os.path.abspath(self.parent.path))
        self.init_with_path(os.path.join(parent_path, self.relative_path))
        self.materials = {n.attrib['id']: n.attrib['name'] for n in node.iter('Material')}


    def init_with_path(self, path):
        with open(path, 'rb') as file:


            self.a01 = read_data(file, 'II')
            self.name = read_string(file)

            self.a02 = file.read(5)
            
            self.source_models = read_string(file)
            self.struct_property = read_string(file)
            self.a03 = file.read(8)

            self.datasmith_mesh_source_model = read_string(file)
            
            self.a04 = file.read(49)

            self.tris_material_slot = read_array_data(file, "I")
            self.tris_smoothing_group = read_array_data(file, "I")
            
            self.vertices = read_array_data(file, "fff")
            self.triangles = read_array_data(file, "I")
            
            self.a05 = read_array_data(file, "I") # not sure about this structure
            self.a06 = read_array_data(file, "I") # not sure about this structure

            self.vertex_normals = read_array_data(file, "fff")
            self.uvs = read_array_data(file, "ff")
            
            self.a07 = file.read(36) # hmmm
            
            self.checksum = file.read(16) # I guess... seems to be non deterministic
            
            self.a08 = file.read()
            
            # small check here to crash if something is suspicious
            assert len(self.triangles) == len(self.uvs)
            assert len(self.vertex_normals) == len(self.uvs)
            assert self.a08 == b'\x00\x00\x00\x00' # just to be sure its the end
            

    
    def report(self):
        print(f'UDMESH {self.name}')
        print(f'    Vertices: {len(self.vertices)}')
        print(f'    Triangles: {len(self.triangles)}/3')
        print(f'    Normals: {len(self.vertex_normals)}')
        print(f'    UVs: {len(self.uvs)}')
        
        

class UDMaterial:
    def __new__(cls, name: str, parent=None, **kwargs):
        try:
            return parent.materials[name]
        except Exception:
            return super().__new__(cls)

    def __init__(self, name: str, node=None, parent=None, **kwargs):
        self.name = name # this is like the primary identifier
        if node:
            self.label = node.attrib['label']
        # datasmith file has a subnode 'shader' with some for now irrelevant info
        if parent:
            parent.materials[self.name] = self

class UDMasterMaterial(UDMaterial):
    '''sketchup datasmith outputs Master material, it may be different'''
    ''' has params Type and Quality'''
    pass

class UDActor(UDMaterial):

    def __init__(self, name:str, parent, node=None, layer=None):
        self.name = name
        if node:
            self.mesh = node.find('mesh').attrib['name']
            self.materials = {n.attrib['id']: n.attrib['name'] for n in node.iter('material')}
            self.loc, self.rot, self.scale = self.transform_from_params(**node.find('Transform').attrib)
            self.layer = node.attrib['layer']

        parent.actors[name] = self        

    def transform_from_params(self, **kwargs):
        p = lambda name: float(kwargs[name])
        location = (p('tx'), p('ty'), p('tz'))
        rotation = (p('qw'), p('qx'), p('qy'), p('qz'))
        scale = (p('sx'), p('sy'), p('sz'))
        return (location, rotation, scale)

class UDScene:
    def __init__(self, path=None, parent=None):
        self.init_fields()
        if path:
            self.path = path
            self.init_with_path(path)

        self.check_fields() # to test if it is possible for these fields to have different values

    def init_fields(self):
        self.name = 'udscene_name'

        self.materials = {}
        self.meshes = {}
        self.actors = {}

    def check_fields(self):
        pass

    def init_with_path(self, path):
        tree = ElementTree.parse(path)
        root = tree.getroot()

        self.version = root.find('Version').text
        self.sdk_version = root.find('SDKVersion').text
        self.host = root.find('Host').text
        # there are other bunch of data that i'll skip for now

        for mat in root.iter('Material'):
            UDMaterial(name=mat.attrib['name'], node=mat, parent=self)
        
        for mat in root.iter('MasterMaterial'):
            UDMasterMaterial(name=mat.attrib['name'], node=mat, parent=self)
        
        for mesh in root.iter('StaticMesh'):
            UDMesh(xmlnode=mesh, parent=self)

        for actor in root.iter('ActorMesh'):
            UDActor(name=actor.attrib['name'], node=actor, parent=self)


    
    def report(self):
        print(f'UDSCENE {self.name}')
        
 