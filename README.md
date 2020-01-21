# Blender Datasmith Export

Export your Blender scene to UE4 using the Datasmith format.

It aims to export all the Datasmith format supports. For now it exports:

* __Meshes__ with normals, vertex colors and UV coordinates.
* __Hierarchy__ is exported keeping meshes references, transforms, parents and
per-instance material overrides from blender.
* __Textures and materials__ are exported using data from the shader graphs.
Materials are closely approximated and a good amount of nodes are supported
(math, mix, fresnel, vertex color and others)
* __Cameras__ are exported trying to match Blender data, keeping focus
distance, focal length, and aperture
* __Lights__ are exported, keeping their type, power, color and size data.

Check out an overview of a previous version here:
https://youtu.be/bUUDqerdqAc

## Sample:
__Blender Eevee:__
![Blender Eevee render](docs/blender.jpg)
__UE4 using Datasmith:__
![UE4 render](docs/unreal.jpg)

This result relies on the **DatasmithBlenderContent**, which is a UE4 Plugin
that improves material import compatibility. Consider supporting the project by
purchasing it from [here][gumroad] (Epic Games store support will be added later)

[gumroad]: https://gum.co/DQvTL

This result is in a custom UE4 build, which fixes some issues of the UE4
importer. If you are technical to compile the engine, you can check the fork
[here][ue4 fork]. Hopefully we can get to integrate our fixes into the main
branch.

[ue4 fork]: https://github.com/0xafbf/UnrealEngine/tree/master

Now [__Download the latest development version__][download_link] and install
from the Blender addons preferences pane.

[download_link]: https://github.com/0xafbf/blender-datasmith-export/archive/master.zip

If you want to support the project, consider supporting via [Patreon].

[patreon]: https://www.patreon.com/0xafbf

Please please, [join the project Discord][join_discord] and share your results!
I want to see what you make and I am open to any feedback you have.

[join_discord]: https://discord.gg/h2GHqMq

