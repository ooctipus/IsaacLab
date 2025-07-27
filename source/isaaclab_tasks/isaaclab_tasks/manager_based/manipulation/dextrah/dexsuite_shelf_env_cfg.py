# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from .dexsuite_env_cfg import DexSuiteReorientEnvCfg, EventCfg
from .rigid_object_generator.rigid_object_generator_usd import build_shelf_usd
from . import mdp

def generate_shelf_usds(num_shelfs) -> list[sim_utils.SpawnerCfg]:
    for i in range(num_shelfs):
        build_shelf_usd(
            f"/tmp/generated_shelves/shelf{i}.usd",
            length_range=(0.8, 1.6),
            depth_range=(0.3, 0.4),
            height_range=(0.5, 1.2),
            thickness_range=(0.02, 0.05),
            row_range=(3, 5),
            col_range=(2, 5)
        )
    return [sim_utils.UsdFileCfg(usd_path=f"/tmp/generated_shelves/shelf{j}.usd") for j in range(num_shelfs)]


@configclass
class ShelvesEventCfg(EventCfg):
    reset_object = EventTerm(
        func=mdp.reset_asset_collision_free,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.2, 0.2], "y": [-0.4, 0.4], "z": [-0.25, 0.4],
                "roll":[-3.14, 3.14], "pitch":[-3.14, 3.14], "yaw": [-3.14, 3.14]
            },
            "velocity_range": {"x": [-0., 0.], "y": [-0., 0.], "z": [-0., 0.]},
            "collision_check_asset_cfg": SceneEntityCfg("object"),
            "collision_check_against_asset_cfg": [SceneEntityCfg("table"), SceneEntityCfg("robot")] 
        },
    )


@configclass
class DexSuiteShelvesEnvCfg(DexSuiteReorientEnvCfg):
    events = ShelvesEventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        super().__post_init__()
        self.scene.table = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/table",
            spawn=sim_utils.MultiAssetSpawnerCfg(
                assets_cfg=generate_shelf_usds(50),
                random_choice=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.775, 0.5, 0.0), rot=(0.7071068, 0, 0, -0.7071068)),
        )
        self.episode_length_s = 0.5
