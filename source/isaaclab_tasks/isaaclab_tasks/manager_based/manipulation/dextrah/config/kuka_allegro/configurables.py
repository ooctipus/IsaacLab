# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim import CapsuleCfg, ConeCfg, CuboidCfg, RigidBodyMaterialCfg, SphereCfg
from isaaclab.utils import configclass

from ... import mdp


@configclass
class KukaAllegroPCAActionCfg:
    actions = mdp.PCAHandActionCfg(asset_name="robot")


@configclass
class KukaAllegroFabricActionCfg:
    actions = mdp.FabricActionCfg(asset_name="robot")


@configclass
class EnvConfigurables:
    env: dict[str, any] = {
        "actions": {"pca": KukaAllegroPCAActionCfg(), "geometry_fabric": KukaAllegroFabricActionCfg()},
        "scene.object": {
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
                        CapsuleCfg(
                            radius=0.04, height=0.025, physics_material=RigidBodyMaterialCfg(static_friction=0.5)
                        ),
                        CapsuleCfg(
                            radius=0.04, height=0.01, physics_material=RigidBodyMaterialCfg(static_friction=0.5)
                        ),
                        CapsuleCfg(radius=0.04, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                        CapsuleCfg(
                            radius=0.025, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)
                        ),
                        CapsuleCfg(
                            radius=0.025, height=0.2, physics_material=RigidBodyMaterialCfg(static_friction=0.5)
                        ),
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
