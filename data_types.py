import struct

def read_array_data(io, data_struct):
    struct_size = struct.calcsize(data_struct)
    data_struct = "<" + data_struct # force little endianness

    count = struct.unpack("<I", io.read(4))[0]
    data = io.read(count * struct_size)
    unpacked_data = struct.iter_unpack(data_struct, data)
    if len(unpacked_data[0]) == 1:
        return [tup[0] for tup in unpacked_data]
    return list(unpacked_data)

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


    def __init__(self, path=None):
        if path:
            self.init_with_path(path)
        
        else:
            self.init_fields()

        self.check_fields() # to test if it is possible for these fields to have different values

    def init_fields(self):
        self.name = 'udmesh_name'
        self.source_models = 'SourceModels'
        self.struct_property = 'StructProperty'
        self.datasmith_mesh_source_model = 'DatasmithMeshSourceModel'

    def check_fields(self):
        assert self.source_models == 'SourceModels'
        assert self.struct_property == 'StructProperty'
        assert self.datasmith_mesh_source_model == 'DatasmithMeshSourceModel'

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

            self.material_slots = read_array_data(file, "I")
            self.smoothing_groups = read_array_data(file, "I")
            
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
        
        


class UDScene:


    def __init__(self, path=None):
        if path:
            self.init_with_path(path)
        
        else:
            self.init_fields()

        self.check_fields() # to test if it is possible for these fields to have different values

    def init_fields(self):
        self.name = 'udscene_name'

    def check_fields(self):
        pass

    def init_with_path(self, path):
        with open(path, 'r') as file:
            print(file.read())
    
    def report(self):
        print(f'UDSCENE {self.name}')
        
 