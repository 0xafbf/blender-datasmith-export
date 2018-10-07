# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

# <pep8 compliant>

bl_info = {
    "name": "Unreal Datasmith format",
    "author": "Andres Botero",
    "blender": (2, 80, 0),
    "location": "File > Export > Datasmith (.udatasmith)",
    "description": "Export scene as Datasmith asset",
    "warning": "",
    "wiki_url": "http://wiki.blender.org/index.php/Extensions:2.6/Py/"
                "Scripts/Import-Export/DSM",
    "category": "Import-Export",
}

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# To support reload properly, try to access a package var,
# if it's there, reload everything
if "bpy" in locals():
    import importlib
    if "export_datasmith" in locals():
        importlib.reload(export_datasmith)
    if "import_datasmith" in locals():
        importlib.reload(import_datasmith)



import bpy
from bpy.props import (
        StringProperty,
        BoolProperty,
        FloatProperty,
        EnumProperty,
        )
from bpy_extras.io_utils import (
        ImportHelper,
        ExportHelper,
        path_reference_mode,
        axis_conversion,
        )


b_major, b_minor, b_patch = bpy.app.version

class ImportDatasmith(bpy.types.Operator, ImportHelper):
    """Load a Datasmith file"""
    bl_idname = "import_scene.udatasmith"
    bl_label = "Import Datasmith"
    bl_options = {'PRESET', 'UNDO'}

    filename_ext = ".udatasmith"
    filter_glob = StringProperty(
        default="*.udatasmith",
        options={'HIDDEN'}
    )

    use_smooth_groups = BoolProperty(
        name="Smooth Groups",
        description="Surround smooth groups by sharp edges",
        default=True,
    )

    def execute(self, context):
        keywords = self.as_keywords(ignore=("filter_glob",))
        from . import import_datasmith
        return import_datasmith.load(self, context, **keywords)

    def draw(self, context):
        layout = self.layout

        layout.prop(self, 'use_smooth_groups')
        

class ExportDatasmith(bpy.types.Operator, ExportHelper):
    """Write a Datasmith file"""
    bl_idname = "export_scene.datasmith"
    bl_label = "Export Datasmith"
    bl_options = {'UNDO', 'PRESET'}

    filename_ext = ".udatasmith"
    filter_glob = StringProperty(default="*.udatasmith", options={'HIDDEN'})

    # List of operator properties, the attributes will be assigned
    # to the class instance from the operator settings before calling.

    version = EnumProperty(
            items=(('UD_420', "Datasmith for UE 4.20", "Datasmith for Unreal Engine 4.20"),
                   ('UD_421', "Datasmith for UE 4.21", "Datasmith for Unreal Engine 4.21"),
                   ),
            name="Version",
            description="Choose which version of the exporter to use",
            )

    use_selection = BoolProperty(
            name="Selected Objects",
            description="Export selected objects on visible layers",
            default=False,
            )
    global_scale = FloatProperty(
            name="Scale",
            description="Scale all data (Some importers do not support scaled armatures!)",
            min=0.001, max=1000.0,
            soft_min=0.01, soft_max=1000.0,
            default=1.0,
            )
    
    # TODO remove, left as doc for a multiple choice enum
    '''object_types = EnumProperty(
            name="Object Types",
            options={'ENUM_FLAG'},
            items=(('EMPTY', "Empty", ""),
                   ('CAMERA', "Camera", ""),
                   ('LIGHT', "Lamp", ""),
                   ('ARMATURE', "Armature", "WARNING: not supported in dupli/group instances"),
                   ('MESH', "Mesh", ""),
                   ('OTHER', "Other", "Other geometry types, like curve, metaball, etc. (converted to meshes)"),
                   ),
            description="Which kind of object to export",
            default={'EMPTY', 'CAMERA', 'LIGHT', 'ARMATURE', 'MESH', 'OTHER'},
            )'''

    # TODO remove: some of these are for fbx, maybe are useful for datasmith
    '''
    use_mesh_modifiers = BoolProperty(
            name="Apply Modifiers",
            description="Apply modifiers to mesh objects (except Armature ones) - "
                        "WARNING: prevents exporting shape keys",
            default=True,
            )
    use_mesh_modifiers_render = BoolProperty(
            name="Use Modifiers Render Setting",
            description="Use render settings when applying modifiers to mesh objects",
            default=True,
            )
    mesh_smooth_type = EnumProperty(
            name="Smoothing",
            items=(('OFF', "Normals Only", "Export only normals instead of writing edge or face smoothing data"),
                   ('FACE', "Face", "Write face smoothing"),
                   ('EDGE', "Edge", "Write edge smoothing"),
                   ),
            description="Export smoothing information "
                        "(prefer 'Normals Only' option if your target importer understand split normals)",
            default='OFF',
            )
    use_mesh_edges = BoolProperty(
            name="Loose Edges",
            description="Export loose edges (as two-vertices polygons)",
            default=False,
            )
    # 7.4 only
    use_tspace = BoolProperty(
            name="Tangent Space",
            description="Add binormal and tangent vectors, together with normal they form the tangent space "
                        "(will only work correctly with tris/quads only meshes!)",
            default=False,
            )
    # 7.4 only'''


    include_metadata = BoolProperty(
            name="Include Metadata",
            description="Include meshes metadata in file",
            default=False,
            )
    
    
    #I don't know what this does
    path_mode = path_reference_mode
    # 7.4 only
    embed_textures = BoolProperty(
            name="Embed Textures",
            description="Embed textures in Datasmith binary file (only for \"Copy\" path mode!)",
            default=False,
            )
    batch_mode = EnumProperty(
            name="Batch Mode",
            items=(('OFF', "Off", "Active scene to file"),
                   ('SCENE', "Scene", "Each scene as a file"),
                   ('GROUP', "Group", "Each group as a file"),
                   ),
            )
    

    def draw(self, context):
        layout = self.layout

        layout.prop(self, "version")
        layout.prop(self, "use_selection")
        layout.prop(self, "global_scale")
        layout.prop(self, "include_metadata")
        layout.prop(self, "embed_textures")
        layout.prop(self, "batch_mode")



    @property
    def check_extension(self):
        return self.batch_mode == 'OFF'

    def execute(self, context):
        keywords = self.as_keywords(ignore=("filter_glob",))
        from . import export_datasmith
        return export_datasmith.save(self, context, **keywords)

def menu_func_export(self, context):
    self.layout.operator(ExportDatasmith.bl_idname, text="Datasmith (.udatasmith)")

def menu_func_import(self, context):
    self.layout.operator(ImportDatasmith.bl_idname, text="Datasmith (.udatasmith)")

classes = (
    ExportDatasmith,
    ImportDatasmith,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    if b_minor >= 80:
        bpy.types.TOPBAR_MT_file_export.append(menu_func_export)
        bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    else:
        bpy.types.INFO_MT_file_export.append(menu_func_export)
        bpy.types.INFO_MT_file_import.append(menu_func_import)


def unregister():
    if b_minor >= 80:
        bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
        bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    else:
        bpy.types.INFO_MT_file_export.remove(menu_func_export)
        bpy.types.INFO_MT_file_import.remove(menu_func_import)


    for cls in classes:
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
