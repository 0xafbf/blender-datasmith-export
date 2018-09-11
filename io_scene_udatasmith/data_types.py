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

class UDActor:
    class Transform:
        def __init__(self, tx=0, ty=0, tz=0, 
                     qw=0, qx=0, qy=0, qz=0,
                     sx=0, sy=0, sz=0, qhex = None):
            self.loc = (float(tx), float(ty), float(tz))
            self.rot = (float(qw), float(qx), float(qy), float(qz))
            self.scale = (float(sx), float(sy), float(sz))
            # don't know what qhex is

    def __init__(self, *, parent, node=None):
        if node:
            self.name = node.attrib['name']
            self.objects = {}
            node_transform = node.find('Transform')
            if node_transform is not None:
                self.transform = UDActor.Transform(**node_transform.attrib)
            else:
                import pdb; pdb.set_trace()
            node_children = node.find('children')
            if node_children:
                for child in node_children:
                    if child.tag == "Actor":
                        UDActor(node=child, parent=self)
                    if child.tag == "ActorMesh":
                        UDActorMesh(node=child, parent=self)
            parent.objects[self.name] = self


class UDActorMesh(UDActor):

    def __init__(self, *, parent, node=None):
        super().__init__(parent=parent, node=node)
        if node:
            self.mesh = node.find('mesh').attrib['name']
            self.materials = {n.attrib['id']: n.attrib['name'] for n in node.findall('material')}


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
        self.objects = {}

    def check_fields(self):
        pass

    def init_with_path(self, path):
        tree = ElementTree.parse(path)
        root = tree.getroot()

        self.version = root.find('Version').text
        self.sdk_version = root.find('SDKVersion').text
        self.host = root.find('Host').text
        # there are other bunch of data that i'll skip for now

        for node in root:
            name = node.get('name') # most relevant nodes have a name as identifier
            if node.tag == 'Material':
                UDMaterial(name=name, node=node, parent=self)
            
            if node.tag == 'MasterMaterial':
                UDMasterMaterial(name=name, node=node, parent=self)
            
            if node.tag == 'StaticMesh':
                UDMesh(xmlnode=node, parent=self)

            if node.tag == 'Actor':
                UDActor(node=node, parent=self)
            if node.tag == "ActorMesh":
                UDActorMesh(node=node, parent=self)

        print(self.objects)
    
        
 