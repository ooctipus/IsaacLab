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
            thickness_range=(0.02, 0.03),
            row_range=(3, 5),
            col_range=(2, 5),
            seed=i
        )
    return [sim_utils.UsdFileCfg(usd_path=f"/tmp/generated_shelves/shelf{j}.usd") for j in range(num_shelfs)]


@configclass
class ShevlesCommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.ObjectUniformPoseCommandCfg(
        asset_name="robot",
        object_name="object",
        resampling_time_range=(3.0, 5.0),
        debug_vis=False,
        ranges=mdp.ObjectUniformPoseCommandCfg.Ranges(
            pos_x=(-0.7, -0.375), pos_y=(-0.4, 0.4), pos_z=(0.1, 0.75), roll=(-3.14, 3.14), pitch=(-3.14, 3.14), yaw=(0., 0.)
        ),
    )


@configclass
class ShelvesEventCfg(EventCfg):
    
    reset_robot = EventTerm(
        func=mdp.reset_asset_collision_free,
        mode="reset",
        params={
            "reset_term": EventTerm(
                func=mdp.chained_reset_terms,
                mode="reset",
                params={
                    "terms":{
                        "reset_root": EventTerm(
                            func=mdp.reset_root_state_uniform,
                            mode="reset",
                            params={
                                "pose_range": {"x": [-0., 0.], "y": [-0., 0.], "yaw": [-0., 0.]},
                                "velocity_range": {"x": [-0., 0.], "y": [-0., 0.], "z": [-0., 0.]},
                                "asset_cfg": SceneEntityCfg("robot"),
                            },
                        ),
                        "reset_robot_joints": EventTerm(
                            func=mdp.reset_joints_by_offset,
                            mode="reset",
                            params={
                                "position_range": [-0.60, 0.60],
                                "velocity_range": [0., 0.],
                            },
                        ),
                        "reset_robot_shoulder_joints": EventTerm(
                            func=mdp.reset_joints_by_offset,
                            mode="reset",
                            params={
                                "asset_cfg": SceneEntityCfg("robot", joint_names="iiwa7_joint_4"),
                                "position_range": [0.2, 0.55],
                                "velocity_range": [0., 0.],
                            },
                        ),
                        "reset_robot_wrist_joint": EventTerm(
                            func=mdp.reset_joints_by_offset,
                            mode="reset",
                            params={
                                "asset_cfg": SceneEntityCfg("robot", joint_names="iiwa7_joint_7"),
                                "position_range": [-3, 3],
                                "velocity_range": [0., 0.],
                            },
                        )
                    }
                },
            ),
            "collision_check_asset_cfg": SceneEntityCfg("robot"),
            "collision_check_against_asset_cfg": [SceneEntityCfg("table")] 
        },
    )

    reset_object = EventTerm(
        func=mdp.reset_asset_collision_free,
        mode="reset",
        params={
            "reset_term": EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {
                        "x": [-0.2, 0.2], "y": [-0.5, 0.5], "z": [0.0, 0.4],
                        "roll":[-3.14, 3.14], "pitch":[-3.14, 3.14], "yaw": [-3.14, 3.14]
                    },
                    "velocity_range": {"x": [-0., 0.], "y": [-0., 0.], "z": [-0., 0.]},
                    "asset_cfg": SceneEntityCfg("object"),
                },
            ),
            "collision_check_asset_cfg": SceneEntityCfg("object"),
            "collision_check_against_asset_cfg": [SceneEntityCfg("table"), SceneEntityCfg("robot")] 
        },
    )


@configclass
class DexSuiteShelvesReorientEnvCfg(DexSuiteReorientEnvCfg):
    events: ShelvesEventCfg = ShelvesEventCfg()
    commands: ShevlesCommandsCfg = ShevlesCommandsCfg()

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


class DexSuiteShelvesPlaceEnvCfg(DexSuiteShelvesReorientEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        self.rewards.orientation_tracking = None  # no orientation reward
        if self.curriculum is not None:
            self.rewards.success.params["rot_std"] = None  # make success reward not consider orientation
            self.curriculum.adr.params["rot_tol"] = None  # make adr not tracking orientation
