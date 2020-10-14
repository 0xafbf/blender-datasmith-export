# Changelog
All notable changes to this project will be documented in this file.
Features marked with `*` require the UE4 plugin to work.

## [Unreleased]
### Fixed
* Export metadata is now a flag, off by default, to help in case other addons store info there.
* Libraries with long file paths are now referenced by the blend file name only.

## [1.0.3] 2020-08-19
### Added
+ Texture node now supports box projection.
+ Smoothing groups are now written in geometry data.
+ TGA texture support. _Thanks KeyToon9_

### Fixed
* Vertex normals now export correctly (you may need to use the Triangulate modifier). _Thanks KeyToon9_
* Fixed image files names when first char is unvalid. _Thanks KeyToon9_
* Improved light attenuation radius calculation.
* Animation is now correctly exported for non-root objects.

## [1.0.2] 2020-07-24
### Fixed
* Fixed an issue when writing some custom props, that would break the XML structure.

## [1.0.0] 2020-07-01
### Added
+ Added support for object and meshes metadata (custom properties in Blender).
+ Added an option to prefer custom nodes to generate simpler graphs if you have the UE4 plugin.*
+ Added new material nodes:
  - Texture Coordinates (partial support)*
  - Noise Texture (simulated, not entirely accurate)*
  - UV Map*
  - Geometry (partial support)*
  - Attribute
  - Combine/Separate RGB/XYZ
  - Combine/Separate HSV*
  - Math (some new nodes like SINH, COSH, etc...)*
  - Vector Math
  - Vector Math (Round, Wrap, Project, Reflect)*
  - Mapping (can now do 3d rotations, uses simpler variants when there is no XY rotation)*
  - Checker Texture*

+ **This is a big one:** Object animations are now exported in a Level Sequence.


### Fixed
* Fixed nodegroups with output node named different than "Group Output".
* Names are now sanitized to be only alphanumeric.
* Increased size for datasmith_curves image to allow for more baked curves (up to 1024)
* All UV nodes are upcasted to Vector3 to better reflect Blender operations, and masked down to Vector2 when reading from them.
* Improved images recollection flow. Unpacked files are directly copied and packed files are written from packed data instead of resaving them (this fixes some strange formats like RG8, which Blender can read and pack, but cannot save).
* Improved behavior of **ColorRamp** and **RGBCurveLookup** nodes*
* Improved behavior of RGB sockets connected to VALUE sockets
* Math nodes with `Clamp` option are now wrapped with a `Saturate` operation.
* Normal maps with strength different than 1 are now wrapped with a `FlattenNormal` operation.
* Principled BSDF nodes now write the Specular values.
* UV maps set as `Render Active` are now set at UV0 to be read as default UVs in UE4. \
* Improved handling of some meshes with empty materials or no material slots.
* Tested on **Blender 2.83.1**


## [0.4.0] 2020-03-26
### Added
+ Added support for exporting curves as geometry
+ Added minimal export option, which skips textures for faster export
+ Added new material nodes:
  - Blackbody
  - Bright/Contrast*
  - Object Info
  - Gamma (Power)

### Fixed
* Improved materials rgb_curves node support
* Improved logging output
* We now support **Blender 2.82**

## [0.3.0] 2020-03-09

### Added
+ Added support for multiple uv maps (datasmith supports up to 8)
+ Added support for sphere reflection probes
+ Added support for box reflection probes
+ Added support for planar reflection probes

### Fixed
* Fixed export of material nodetrees inside nodetrees
* Improved export speed of material curves
* Fixed normal map flags for ue4.25
* Fixed profiling flag as export option
* Fixed export for multiple scenes, these scenes export, although some are not
  tested in UE4 yet, tested scenes include:
  + archiviz
  + blender_splash_fishy_cat
  + classroom
  + forest
  + mr_elephant
  + pabellon_barcelona
  + pokedstudio
  + race_spaceship
  + temple
  + the_junk_shop
  + tree_creature
  + wanderer
  + wasp_bot

## [0.2.0]

This was the first release, changelog wasn't used before this.
