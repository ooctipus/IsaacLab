# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing utilities for creating prims in Omniverse.

Spawners are used to create prims into Omniverse simulator. At their core, they are calling the
USD Python API or Omniverse Kit Commands to create prims. However, they also provide a convenient
interface for creating prims from their respective config classes.

There are two main ways of using the spawners:

1. Using the function from the module

   .. code-block:: python

    import isaaclab.sim as sim_utils
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

    # spawn from USD file
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd")
    prim_path = "/World/myAsset"

    # spawn using the function from the module
    sim_utils.spawn_from_usd(prim_path, cfg)

2. Using the `func` reference in the config class

   .. code-block:: python

    import isaaclab.sim as sim_utils
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

    # spawn from USD file
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd")
    prim_path = "/World/myAsset"

    # use the `func` reference in the config class
    cfg.func(prim_path, cfg)

For convenience, we recommend using the second approach, as it allows to easily change the config
class and the function call in a single line of code.

Depending on the type of prim, the spawning-functions can also deal with the creation of prims
over multiple prim path. These need to be provided as a regex prim path expressions, which are
resolved based on the parent prim paths using the :meth:`isaaclab.sim.utils.clone` function decorator.
For example:

* ``/World/Table_[1,2]/Robot`` will create the prims ``/World/Table_1/Robot`` and ``/World/Table_2/Robot``
  only if the parent prim ``/World/Table_1`` and ``/World/Table_2`` exist.
* ``/World/Robot_[1,2]`` will **NOT** create the prims ``/World/Robot_1`` and
  ``/World/Robot_2`` as the prim path expression can be resolved to multiple prims.

"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        # spawner base cfg classes
        "spawner_cfg": ["SpawnerCfg", "RigidObjectSpawnerCfg", "DeformableObjectSpawnerCfg"],
        # from_files
        "from_files": [
            "spawn_from_mjcf",
            "spawn_from_urdf",
            "spawn_from_usd",
            "spawn_from_usd_with_compliant_contact_material",
            "spawn_ground_plane",
            "GroundPlaneCfg",
            "MjcfFileCfg",
            "UrdfFileCfg",
            "UsdFileCfg",
            "UsdFileWithCompliantContactCfg",
        ],
        # lights
        "lights": [
            "spawn_light",
            "CylinderLightCfg",
            "DiskLightCfg",
            "DistantLightCfg",
            "DomeLightCfg",
            "LightCfg",
            "SphereLightCfg",
        ],
        # materials
        "materials": [
            "spawn_deformable_body_material",
            "spawn_rigid_body_material",
            "DeformableBodyMaterialCfg",
            "PhysicsMaterialCfg",
            "RigidBodyMaterialCfg",
            "spawn_from_mdl_file",
            "spawn_preview_surface",
            "GlassMdlCfg",
            "MdlFileCfg",
            "PreviewSurfaceCfg",
            "VisualMaterialCfg",
        ],
        # meshes
        "meshes": [
            "spawn_mesh_capsule",
            "spawn_mesh_cone",
            "spawn_mesh_cuboid",
            "spawn_mesh_cylinder",
            "spawn_mesh_sphere",
            "MeshCapsuleCfg",
            "MeshCfg",
            "MeshConeCfg",
            "MeshCuboidCfg",
            "MeshCylinderCfg",
            "MeshSphereCfg",
        ],
        # sensors
        "sensors": ["spawn_camera", "FisheyeCameraCfg", "PinholeCameraCfg"],
        # shapes
        "shapes": [
            "spawn_capsule",
            "spawn_cone",
            "spawn_cuboid",
            "spawn_cylinder",
            "spawn_sphere",
            "CapsuleCfg",
            "ConeCfg",
            "CuboidCfg",
            "CylinderCfg",
            "ShapeCfg",
            "SphereCfg",
        ],
        # wrappers
        "wrappers": ["spawn_multi_asset", "spawn_multi_usd_file", "MultiAssetSpawnerCfg", "MultiUsdFileCfg"],
    },
)
