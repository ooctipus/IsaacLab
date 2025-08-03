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


LENGTH_RANGE = (1.4, 1.4)
DEPTH_RANGE = (0.4, 0.4)
HEIGHT_RANGE = (1.1, 1.1)

def generate_shelf_usds(num_shelfs) -> list[sim_utils.SpawnerCfg]:
    for i in range(num_shelfs):
        build_shelf_usd(
            f"/tmp/generated_shelves/shelf{i}.usd",
            length_range=LENGTH_RANGE,
            depth_range=DEPTH_RANGE,
            height_range=HEIGHT_RANGE,
            thickness_range=(0.02, 0.03),
            row_range=(3, 4),
            col_range=(2, 4),
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
            pos_x=(-0.925, -0.275), pos_y=(-0.4, 0.4), pos_z=(0.1, 0.75), roll=(-3.14, 3.14), pitch=(-3.14, 3.14), yaw=(0., 0.)
        ),
    )
    
    
    # object_pose = mdp.ObjectUniformTableTopRestPoseCommandCfg(
    #     asset_name="robot",
    #     object_name="object",
    #     table_name="table",
    #     resampling_time_range=(3.0, 5.0),
    #     debug_vis=False,
    #     ranges=mdp.ObjectUniformTableTopRestPoseCommandCfg.Ranges(
    #         pos_x=(LENGTH_RANGE[1] * 0.2, LENGTH_RANGE[1] * 0.8),
    #         pos_y=(DEPTH_RANGE[1] * 0.25, DEPTH_RANGE[1] * 0.75),
    #         pos_z=(0.1, HEIGHT_RANGE[1]),
    #         roll=(-3.14, 3.14),
    #         pitch=(-3.14, 3.14),
    #         yaw=(0., 0.)
    #     ),
    #     num_samples=30
    # )


@configclass
class ShelvesEventCfg(EventCfg):
    
    reset_scene = EventTerm(
        func=mdp.reset_accumulator,
        mode="reset",
        params={
            "reset_assets": ["table", "object", "robot"],
            "acceptance_conditions": {
                "object_collision_free": mdp.CollisionAnalyzerCfg(
                    num_points=32,
                    max_dist=0.5,
                    asset_cfg=SceneEntityCfg("object"),
                    obstacle_cfgs=[SceneEntityCfg("table")]
                ),
                "collision_robot_free": mdp.CollisionAnalyzerCfg(
                    num_points=32,
                    max_dist=0.5,
                    asset_cfg=SceneEntityCfg("robot", body_names=["iiwa7_link_(5|6|7|ee)", "allegro_mount", "palm_link", "(index|middle|ring|thumb).*"]),
                    obstacle_cfgs=[SceneEntityCfg("table"), SceneEntityCfg("object")]
                )
            },
            "reset_term": EventTerm(
                func=mdp.chained_reset_terms,
                mode="reset",
                params={
                    "terms":{
                        # BUG: REPORT THAT KINEMATIC ENABLED ASSEST RESET DOESN'T WORK IN RENDERING
                        # SO I SET IT TO 0 FOR NOW
                        "reset_table": EventTerm(
                            func=mdp.reset_root_state_uniform,
                            mode="reset",
                            params={
                                "pose_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [0.0, 0.0]},
                                "velocity_range": {},
                                "asset_cfg": SceneEntityCfg("table"),
                            },
                        ),
                        "reset_object": EventTerm(
                            func=mdp.reset_root_state_uniform,
                            mode="reset",
                            params={
                                "pose_range": {
                                    "x": [-0.2, 0.2], "y": [-0.2, 0.2], "z": [0.0, 0.4],
                                    "roll":[-3.14, 3.14], "pitch":[-3.14, 3.14], "yaw": [-3.14, 3.14]
                                },
                                "velocity_range": {"x": [-0., 0.], "y": [-0., 0.], "z": [-0., 0.]},
                                "asset_cfg": SceneEntityCfg("object"),
                            },
                        ),
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
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.975, 0.5, 0.0), rot=(0.7071068, 0, 0, -0.7071068)),
        )


class DexSuiteShelvesPlaceEnvCfg(DexSuiteShelvesReorientEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        self.rewards.orientation_tracking = None  # no orientation reward
        if self.curriculum is not None:
            self.rewards.success.params["rot_std"] = None  # make success reward not consider orientation
            self.curriculum.adr.params["rot_tol"] = None  # make adr not tracking orientation