# Copyright Andrés Botero 2019

bl_info = {
	"name": "Unreal Datasmith format",
	"author": "Andrés Botero",
	"version": (0, 1, 0),
	"blender": (2, 80, 0),
	"location": "File > Export > Datasmith (.udatasmith)",
	"description": "Export scene as Datasmith asset",
	"warning": "",
	"category": "Import-Export",
	"support": 'COMMUNITY',
	"wiki_url": "https://github.com/0xafbf/blender-datasmith-export",
}


if "bpy" in locals():
	import importlib
	if "export_datasmith" in locals():
		importlib.reload(export_datasmith)

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

class ExportDatasmith(bpy.types.Operator, ExportHelper):
	"""Write a Datasmith file"""
	bl_idname = "export_scene.datasmith"
	bl_label = "Export Datasmith"
	bl_options = {'PRESET'}

	filename_ext = ".udatasmith"
	filter_glob: StringProperty(default="*.udatasmith", options={'HIDDEN'})

	use_logging: BoolProperty(
            name="Enable logging",
            description="Enable logging to Window > System console",
            default=False,
            )

	def execute(self, context):
		keywords = self.as_keywords(ignore=("filter_glob",))
		from . import export_datasmith
		profile = False
		if not profile:
			return export_datasmith.save(context, **keywords)
		else:
			import cProfile
			pr = cProfile.Profile()
			pr.enable()
			result = export_datasmith.save(context, **keywords)
			pr.disable()
			path = "datasmith.prof"
			pr.dump_stats(path)
			return result

def menu_func_export(self, context):
	self.layout.operator(ExportDatasmith.bl_idname, text="Datasmith (.udatasmith)")

def register():
	bpy.utils.register_class(ExportDatasmith)
	bpy.types.TOPBAR_MT_file_export.append(menu_func_export)

def unregister():
	bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
	bpy.utils.unregister_class(ExportDatasmith)


if __name__ == "__main__":
	register()
