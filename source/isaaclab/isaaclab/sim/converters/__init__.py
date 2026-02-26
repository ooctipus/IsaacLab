# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing converters for converting various file types to USD.

In order to support direct loading of various file types into Omniverse, we provide a set of
converters that can convert the file into a USD file. The converters are implemented as
sub-classes of the :class:`AssetConverterBase` class.

The following converters are currently supported:

* :class:`UrdfConverter`: Converts a URDF file into a USD file.
* :class:`MeshConverter`: Converts a mesh file into a USD file. This supports OBJ, STL and FBX files.

"""

from isaaclab.utils.lazy_imports import lazy_export

lazy_export(
    ("asset_converter_base", "AssetConverterBase"),
    ("asset_converter_base_cfg", "AssetConverterBaseCfg"),
    ("mesh_converter", "MeshConverter"),
    ("mesh_converter_cfg", "MeshConverterCfg"),
    ("mjcf_converter", "MjcfConverter"),
    ("mjcf_converter_cfg", "MjcfConverterCfg"),
    ("urdf_converter", "UrdfConverter"),
    ("urdf_converter_cfg", "UrdfConverterCfg"),
)
