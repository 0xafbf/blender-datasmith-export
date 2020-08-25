#
#
# datasmith export test suite
# run this file with the following command:
# blender -b sample_file.blend -P test_datasmith_export.py

import bpy
import bpy.ops
import logging
import os
import shutil
import sys
import time

is_benchmark = "-benchmark" in sys.argv

logging_level = logging.DEBUG # INFO

if is_benchmark:
	logging_level = logging.WARNING

logging.basicConfig(
	level=logging_level,
	# format='%(asctime)s.%(msecs)03d %(name)-12s %(levelname)-8s %(message)s',
	format='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
	datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger()

clean_path = os.path.normpath(bpy.data.filepath)

base_dir, file_name = os.path.split(clean_path)
name, ext = os.path.splitext(file_name)
target_path = os.path.join(base_dir, name + ".udatasmith")


log.info("basedir %s", base_dir)
use_diff = True
backup_path = None
if use_diff and os.path.isfile(target_path):
	log.info("backing up previous test")
	last_modification_time = os.path.getmtime(target_path)
	time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(last_modification_time))
	backup_path = os.path.join(base_dir, "%s_%s.udatasmith" % (name, time_str))
	log.debug("last modification was:%s", backup_path)
	shutil.copy(target_path, backup_path)

log.info("Starting automated export")

custom_args = {}
custom_args["use_gamma_hack"] = False
custom_args["apply_modifiers"] = True
custom_args["export_animations"] = True
custom_args["compatibility_mode"] = False
custom_args["minimal_export"] = False
custom_args["use_logging"] = True
custom_args["use_profiling"] = False
custom_args["write_metadata"] = False

if "-benchmark" in sys.argv:
	custom_args["use_logging"] = False


bpy.ops.export_scene.datasmith(filepath=target_path, **custom_args)
log.info("Ended automated export")

# right now this is not so useful as the export is non deterministic
# i guess it is because the usage of dictionaries
do_file_diff = True

if "-benchmark" in sys.argv:
	do_file_diff = False

# todo: if size is less than 2MB
if backup_path and do_file_diff:
	log.info("writing diff file")
	import difflib

	with open(backup_path) as ff:
		from_lines = ff.readlines()
	with open(target_path) as tf:
		to_lines = tf.readlines()

	diff = difflib.unified_diff(from_lines, to_lines, backup_path, target_path)

	new_modification_time = os.path.getmtime(target_path)
	new_time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(new_modification_time))
	diff_path = os.path.join(base_dir, "export_diff_%s.diff" % new_time_str)
	with open(diff_path, 'w') as diff_file:
		diff_file.writelines(diff)
	static_diff_path = os.path.join(base_dir, "export_diff.diff")
	shutil.copy(diff_path, static_diff_path)


