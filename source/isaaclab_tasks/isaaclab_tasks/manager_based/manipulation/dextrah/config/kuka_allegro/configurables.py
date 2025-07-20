# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim import CuboidCfg, SphereCfg, CapsuleCfg, ConeCfg, RigidBodyMaterialCfg
from isaaclab.utils import configclass

# from isaaclab.utils.assets import LOCAL_ASSET_PATH_DIR
# ISAACLAB_NUCLEUS_DIR = "source/isaaclab_assets/data"
# objects_dir = f"{ISAACLAB_NUCLEUS_DIR}/Props/Dextrah/Objects"
# sub_dirs = sorted(os.listdir(objects_dir))
# dirs = [object_name for object_name in sub_dirs if os.path.isdir(os.path.join(objects_dir, object_name))]

@configclass
class EnvConfigurables:
    env: dict[str, any] = {
        "scene.object": {
            # "visdex": RigidObjectCfg(
            #     prim_path="{ENV_REGEX_NS}/Object",
            #     spawn=sim_utils.MultiAssetSpawnerCfg(
            #         assets_cfg=[
            #             sim_utils.UsdFileCfg(usd_path=os.path.join(objects_dir, name, f"{name}.usd")) for name in dirs],
            #         random_choice=True,
            #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
            #             rigid_body_enabled=True,
            #             solver_position_iteration_count=16,
            #             solver_velocity_iteration_count=0,
            #             kinematic_enabled=False,
            #             disable_gravity=False,
            #             sleep_threshold=0.005,
            #             stabilization_threshold=0.0025,
            #             max_linear_velocity=1000.0,
            #             max_angular_velocity=1000.0,
            #             max_depenetration_velocity=1000.0,
            #         ),
            #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            #         mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            #     ),
            #     init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.55, 0.1, 0.35)),
            # ),
            "cube": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Object",  
                spawn=sim_utils.MultiAssetSpawnerCfg(
                    assets_cfg=[
                        CuboidCfg(size=(0.1, 0.1, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                        CuboidCfg(size=(0.05, 0.1, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                        CuboidCfg(size=(0.05, 0.05, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                        CuboidCfg(size=(0.025, 0.1, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                        CuboidCfg(size=(0.025, 0.05, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                        CuboidCfg(size=(0.025, 0.025, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                    ],
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=0,
                        disable_gravity=False,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.55, 0.1, 0.35)),
            ),
            "geometry": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Object",
                spawn=sim_utils.MultiAssetSpawnerCfg(
                    assets_cfg=[
                        CuboidCfg(size=(0.05, 0.1, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                        CuboidCfg(size=(0.05, 0.05, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                        CuboidCfg(size=(0.025, 0.1, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                        CuboidCfg(size=(0.025, 0.05, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                        CuboidCfg(size=(0.025, 0.025, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                        CuboidCfg(size=(0.01, 0.1, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                        SphereCfg(radius=0.05, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                        SphereCfg(radius=0.025, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                        CapsuleCfg(radius=0.04, height=0.025, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                        CapsuleCfg(radius=0.04, height=0.01, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                        CapsuleCfg(radius=0.04, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                        CapsuleCfg(radius=0.025, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                        CapsuleCfg(radius=0.025, height=0.2, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                        CapsuleCfg(radius=0.01, height=0.2, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                        ConeCfg(radius=0.05, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                        ConeCfg(radius=0.025, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                    ],
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=0,
                        disable_gravity=False,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.55, 0.1, 0.35)),
            ),
        },
    }
