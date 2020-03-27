# Changelog
All notable changes to this project will be documented in this file.
Features marked with `*` require the UE4 plugin to work.

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
