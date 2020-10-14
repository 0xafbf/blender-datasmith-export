# Blender Datasmith Export

Export your Blender scene to UE4 using the Datasmith format.

It aims to export all the Datasmith format supports. For now it exports:

* __Meshes__ with normals, vertex colors and up to 8 UV channels.
* __Hierarchy__ is exported keeping meshes references, transforms, parents and
per-instance material overrides from blender.
* __Textures and materials__ are exported using data from the shader graphs.
Materials are closely approximated and a good amount of nodes are supported
(math, mix, fresnel, vertex color and others)
* __Cameras__ are exported trying to match Blender data, keeping focus
distance, focal length, and aperture
* __Lights__ are exported, keeping their type, power, color and size data.
* __Reflection probes__ including Planar, Sphere and Box captures.

Check out an overview of a previous version here:
https://youtu.be/bUUDqerdqAc

## Sample:
You can click the images to open a large preview.

__Blender Eevee:__

<img alt="Blender Eevee render" src="docs/blender.jpg" width="300">

__UE4 using Datasmith:__

<img alt="UE4 render" src="docs/unreal.jpg" width="300">

This result relies on the **DatasmithBlenderContent**, which is a UE4 Plugin
that improves material import compatibility. Consider supporting the project by
purchasing it from [here][gumroad] (Epic Games store support will be added later)

[gumroad]: https://gum.co/DQvTL

This result is in a custom UE4 build, which fixes some issues of the UE4
importer. If you are technical to compile the engine, you can check the fork
[here][ue4 fork]. Hopefully we can get to integrate our fixes into the main
branch.

[ue4 fork]: https://github.com/0xafbf/UnrealEngine/tree/master

## Installation:

Now [__Download the latest development version__][download_link] and install
from the Blender addons preferences pane.

[download_link]: https://github.com/0xafbf/blender-datasmith-export/archive/master.zip

## Frequently Asked Questions:

__Q: Does this support weighted normals/smoothing groups?__

A: Yes, but the plugin is unable to triangulate correctly. For the time
being, you can add a `Triangulate` modifier with the `Keep Normals` option to
work around this.

__Q: Why are some material nodes not exported?__

A: Most of the nodes are exported, but not all of them are imported from the
UE4 side. The Datasmith Blender Additions (mentioned above) improves this by
adding implementations for some of these nodes. There is a
[list of nodes in the wiki] with more information on which nodes are
supported, and which require the UE4 plugin to work.

[list of nodes in the wiki]: https://github.com/0xafbf/blender-datasmith-export/wiki/Supported-Material-Nodes

__Q: What is this "custom build" you talked about earlier?__
A: I modified some of the UE4 build to fix a couple of errors when importing
the scenes generated from Blender. These are related to normal maps import
and very specific import issues with lights. If you're interested you can
check this [custom build discussion].

[custom build discussion]: https://github.com/0xafbf/blender-datasmith-export/issues/25

If you want to support the project, consider supporting via [Patreon].

[patreon]: https://www.patreon.com/0xafbf

Please please, [join the project Discord][join_discord] and share your results!
I want to see what you make and I am open to any feedback you have.

[join_discord]: https://discord.gg/NJt5ADJ

