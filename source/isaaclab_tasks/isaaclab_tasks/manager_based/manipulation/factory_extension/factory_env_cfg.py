# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Sequence
import re
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.envs import ManagerBasedRLEnvCfg

from . import factory_assets_cfg as assets
from . import mdp
from .tasks.assembly_key_points_cfg import to_pascal
from .tasks import assembly_object_key_points as kps
from .tasks import ManipulationKeyPointCfg, TaskKeyPointCfg, BaseTaskKeyPointCfg, SuccessCondition
from .tasks import nist_board_tasks_key_points_cfg as tasks
from .tasks import manipulation_task_key_points_cfg as manipulations
from .tasks import nist_board_resets_key_points_cfg as resets
from .mdp.data_cfg import KeyPointDataCfg as Kp
from .mdp.data_cfg import AlignmentDataCfg as Align
from .mdp.data_cfg import DataCfg as Data

"""
Base scene definition for Factory Tasks
"""


@configclass
class FactoryTasksCfg:
    # assembly tasks
    tasks: list[TaskKeyPointCfg] = [
        tasks.Rod16MMInsert(),
        tasks.Rod12MMInsert(),
        tasks.Rod8MMInsert(),
        tasks.Rod4MMInsert(),
        tasks.RectangularPeg16MMInsert(),
        tasks.RectangularPeg12MMInsert(),
        tasks.RectangularPeg8MMInsert(),
        tasks.RectangularPeg4MMInsert(),
        tasks.GearMeshLarge(),
        tasks.GearMeshSmall(),
        tasks.GearMeshMedium(),
        tasks.USBAInsert(),
        tasks.DSUBInsert(),
        tasks.WaterproofInsert(),
        tasks.RJ45Insert(),

        # tasks.Rod16MMAlign(),
        # tasks.Rod12MMAlign(),
        # tasks.Rod8MMAlign(),
        # tasks.Rod4MMAlign(),
        # tasks.RectangularPeg16MMAlign(),
        # tasks.RectangularPeg12MMAlign(),
        # tasks.RectangularPeg8MMAlign(),
        # tasks.RectangularPeg4MMAlign(),
        # tasks.GearLargeAlign(),
        # tasks.GearSmallAlign(),
        # tasks.GearMediumAlign(),
        # tasks.USBAAlign(),
        # tasks.DSUBAlign(),
        # tasks.WaterproofAlign(),
        # tasks.RJ45Align(),
        
        # tasks.NutThreadM16(),
        # tasks.NutThreadM12(),
        # tasks.NutThreadM8(),
        # tasks.NutThreadM4(),
    ]

    manipulations: list[ManipulationKeyPointCfg] = []

    resets: Sequence[BaseTaskKeyPointCfg] = [
        resets.KitTrayAndNistBoardReset(),
        # resets.KitTrayReset(),
        # resets.NistBoardReset(),
    ]

    def __post_init__(self) -> None:
        # remove attributes that are not in the valid set
        valid = set(self.asset_names)
        for reset in self.resets:
            kp = reset.key_points
            for attr in set(vars(kp)) - valid:
                delattr(kp, attr)

        # append manipulations assets
        pattern = re.compile(r"(plug|nut|rod|peg|_gear)")
        for asset in self.asset_names:
            if pattern.search(asset):
                self.manipulations.append(getattr(manipulations, to_pascal(asset))())

    @property
    def task_names(self) -> list[str]:
        return [type(task).__name__ for task in self.tasks]

    @property
    def tasks_dict(self) -> dict[str, TaskKeyPointCfg]:
        return {type(task).__name__: task for task in self.tasks}

    @property
    def manipulations_dict(self) -> dict[str, ManipulationKeyPointCfg]:
        return {type(manipulation).__name__: manipulation for manipulation in self.manipulations}

    @property
    def resets_dict(self) -> dict[str, BaseTaskKeyPointCfg]:
        return {type(reset).__name__: reset for reset in self.resets}

    @property
    def task_metrics(self) -> list[SuccessCondition]:
        return [task.success_condition for task in self.tasks]

    @property
    def manipulation_alignment_metric(self) -> list[SuccessCondition]:
        return [manipulation.success_condition for manipulation in self.manipulations]

    @property
    def asset_names(self) -> list[str]:
        all_assets: set[str] = set()
        for task in self.tasks + self.manipulations:
            all_assets |= task.asset_set()
        return list(all_assets)

    @property
    def assets(self) -> dict[str, RigidObjectCfg]:
        rigid_body_cfg = {"nist_board": assets.NIST_BOARD_CFG, "kit_tray": assets.KIT_TRAY_CFG}
        for task in self.tasks + self.manipulations:
            for asset in task.asset_set():
                rigid_body_cfg[asset] = getattr(assets, f"{asset.upper()}_CFG")
        return rigid_body_cfg

    @property
    def assets_diameter(self) -> list[float]:
        return [m.held_asset_diameter for m in self.manipulations]


ASSEMBLY_TASKS = FactoryTasksCfg()


@configclass
class FactorySceneCfg(InteractiveSceneCfg):
    """Configuration for a factory task scene."""

    # Lights
    dome_light = assets.DOME_LIGHT_CFG

    # Ground plane
    ground = assets.GROUND_CFG

    # Table
    table = assets.TABLE_CFG

    # "FIXED ASSETS"
    assets = RigidObjectCollectionCfg(rigid_objects=ASSEMBLY_TASKS.assets)

    # Robot Override
    robot: ArticulationCfg = MISSING  # type: ignore


@configclass
class FactoryActionsCfg:
    """Action specifications for Factory"""

    attachment_action = mdp.RigidObjectCollectionAttachmentActionCfg(
        asset_name="assets",
        asset_cfg=SceneEntityCfg("assets", object_collection_names=['robot_root', 'panda_hand'], preserve_order=True),
        attach_to_asset_cfg=SceneEntityCfg("robot", body_names=["panda_link0", "panda_hand"], preserve_order=True),
    )


@configclass
class FactoryObservationsCfg:
    """Observation specifications for Factory."""

    @configclass
    class PolicyCfg(ObsGroup):

        # task_encoding = ObsTerm(func=mdp.task_encoding)

        # object_point_cloud = ObsTerm(
        #     func=mdp.PointCloud,
        #     params={
        #         "robot_cfg": SceneEntityCfg("robot", preserve_order=True),
        #         "object_cfg":SceneEntityCfg("assets", object_collection_names="^(?!(?:nist_board|kit_tray|panda_hand|robot_root)$).*"),
        #         "num_points": 512,
        #         "visualize": True
        #     }
        # )

        task_alignment = ObsTerm(
            func=mdp.target_asset_pose_in_root_asset_frame_min_error,
            params={"alignment_cfg": Align(term="alignment_data")},
        )

        manipulation_alignment = ObsTerm(
            func=mdp.target_asset_pose_in_root_asset_frame_min_error,
            params={"alignment_cfg": Align(term="manipulation_alignment_data")},
        )

        prev_action = ObsTerm(func=mdp.last_action)

        joint_pos = ObsTerm(func=mdp.joint_pos)

        joint_vel = ObsTerm(func=mdp.joint_vel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: PolicyCfg = PolicyCfg()  # same as policy


@configclass
class FactoryEventCfg:
    """Events specifications for Factory"""

    # mode: startup
    # held_asset_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,  # type: ignore
    #     mode="startup",
    #     params={
    #         "static_friction_range": (0.75, 0.75),
    #         "dynamic_friction_range": (0.75, 0.75),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 64,
    #         "asset_cfg": SceneEntityCfg("held_asset"),
    #     },
    # )

    # fixed_asset_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,  # type: ignore
    #     mode="startup",
    #     params={
    #         "static_friction_range": (0.75, 0.75),
    #         "dynamic_friction_range": (0.75, 0.75),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 64,
    #         "asset_cfg": SceneEntityCfg("fixed_asset"),
    #     },
    # )

    robot_material = EventTerm(
        func=mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # mode: reset
    reset_env = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_board = EventTerm(
        func=mdp.reset_collection_asset,
        mode="reset",
        params={
            "pose_range": {"x": (-0.0000, 0.0000), "y": (-0.2500, -0.2500), "yaw": (-3.1415, 3.1415)},
            "offset": list(kps.NIST_BOARD_KEY_POINTS_CFG.nist_board_center.offsets)[0],
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("assets", object_collection_names="nist_board"),
        },
    )

    reset_tray = EventTerm(
        func=mdp.reset_collection_asset,
        mode="reset",
        params={
            "pose_range": {"x": (-0.0000, 0.0000), "y": (0.2500, 0.2500), "yaw": (-3.1415, 3.1415)},
            "offset": list(kps.KIT_TRAY_KEY_POINTS_CFG.kit_tray_center.offsets)[0],
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("assets", object_collection_names="kit_tray"),
        },
    )

    reset_fixed_asset = EventTerm(
        func=mdp.reset_assets,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "assets",
                object_collection_names=ASSEMBLY_TASKS.resets[0].key_points.ordered_names(),
                preserve_order=True
            ),
            "asset_reset_key_points_cfg": Kp(
                term="reset_key_points",
                kp_attr="key_points",
                kp_names=ASSEMBLY_TASKS.resets[0].key_points.ordered_names()
            ),
        },
    )

    reset_attachments = EventTerm(
        func=mdp.reset_attachments,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("assets", object_collection_names=['robot_root', 'panda_hand'], preserve_order=True),
            "attach_to_asset_cfg": SceneEntityCfg("robot", body_names=["panda_link0", "panda_hand"], preserve_order=True)
        }
    )


@configclass
class FactoryCommandsCfg:
    """Command for Factory"""

    task_command = mdp.AssemblyTaskCommandCfg(
        resampling_time_range=(1e6, 1e6),
        tasks=ASSEMBLY_TASKS.task_names,
        alignment_metric_cfg=Align(term="alignment_data"),
        reset_terms_when_resample={
            "reset_task_asset": EventTerm(
                mode="reset",
                func=mdp.reset_task_assets,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "assets",
                        object_collection_names=ASSEMBLY_TASKS.resets[0].key_points.ordered_names(),
                        preserve_order=True
                    ),
                    "asset_reset_spec_cfg": Kp(term="reset_key_points", is_spec=True, kp_attr=None),
                    "asset_reset_kp_offset_cfg": Kp(term="reset_key_points", is_spec=True, kp_attr="key_points_offset", kp_names=ASSEMBLY_TASKS.resets[0].key_points.ordered_names()),
                    "align_asset_id_cfg": Kp(term="task_key_points", kp_attr="key_points_asset_id", kp_names="asset_align"),
                },
            ),
            "reset_held_asset_in_gripper": EventTerm(
                mode="reset",
                func=mdp.ChainedResetTerms,
                params={
                    "terms": {
                        "reset_end_effector_around_fixed_asset": mdp.reset_end_effector_round_fixed_asset,
                        "reset_attachment": mdp.reset_attachments,
                        "grasp_held_asset": mdp.grasp_held_asset,
                        "reset_held_asset_in_hand": mdp.reset_held_asset_against,
                    },
                    "params": {
                        "reset_end_effector_around_fixed_asset": {
                            "ee_reset_point_cfg": Kp(term="task_key_points", kp_names="asset_align_against"),
                            "pose_range_b": {
                                "x": (-0.02, 0.02),
                                "y": (-0.02, 0.02),
                                "z": (0.05, 0.06),
                                "roll": (3.14, 3.14),
                                "yaw": (-0.78, 0.78),
                            },
                            "robot_ik_cfg": SceneEntityCfg("robot"),
                        },
                        "reset_attachment": {
                            "asset_cfg": SceneEntityCfg("assets", object_collection_names=['robot_root', 'panda_hand'], preserve_order=True),
                            "attach_to_asset_cfg": SceneEntityCfg("robot", body_names=["panda_link0", "panda_hand"], preserve_order=True)
                        },
                        "grasp_held_asset": {
                            "robot_cfg": SceneEntityCfg("robot", joint_names="end_effector_joints"),
                            "held_asset_diameter_cfg": Data(term="diameter_look_up")
                        },
                        "reset_held_asset_in_hand": {
                            "aligning_point_offset_cfg": Kp(term="manipulation_key_points", kp_attr="key_points_offset", kp_names="asset_grasp"),
                            "aligning_point_asset_id_cfg": Kp(term="manipulation_key_points", kp_attr="key_points_asset_id", kp_names="asset_grasp"),
                            "point_align_against_cfg": Kp(term="manipulation_key_points", kp_names="robot_object_held"),
                        },
                    },
                    "probability": 1.0,
                }
            ),
        },
    )


@configclass
class FactoryDataCfg:
    """Data for Factory"""

    task_key_points = mdp.KeyPointTrackerCfg(spec=ASSEMBLY_TASKS.tasks_dict, context_id_callback=mdp.task_id_callback)

    manipulation_key_points = mdp.KeyPointTrackerCfg(
        spec=ASSEMBLY_TASKS.manipulations_dict,
        context_id_callback=mdp.asset_id_callback_deterministic,
        context_id_param={
            "kp_asset_id_cfg": Kp(term="task_key_points", kp_attr="key_points_asset_id", kp_names="asset_align"),
            "asset_to_kps_cfg": Kp(term="manipulation_key_points", kp_attr="asset_id_key_points", kp_names="asset_grasp", is_spec=True)
        },
        # debug_vis=True,
    )

    reset_key_points = mdp.KeyPointTrackerCfg(
        spec=ASSEMBLY_TASKS.resets_dict,
        context_id_callback=mdp.random_context_callback,
        update_only_on_reset=True,
        context_id_param={
            "reset_kp_asset_id_cfg": Kp(term="reset_key_points", kp_attr="key_points_asset_id", is_spec=True),
        }
    )

    diameter_look_up = mdp.DiameterLookUpCfg(
        manipulation_diameters=ASSEMBLY_TASKS.assets_diameter,
        context_id_callback=mdp.asset_id_callback_deterministic,
        context_id_param={
            "kp_asset_id_cfg": Kp(term="task_key_points", kp_attr="key_points_asset_id", kp_names="asset_align"),
            "asset_to_kps_cfg": Kp(term="manipulation_key_points", kp_attr="asset_id_key_points", kp_names="asset_grasp", is_spec=True)
        },
    )

    # make sure alignment is after key points
    alignment_data = mdp.AlignmentMetricCfg(
        spec=ASSEMBLY_TASKS.task_metrics,
        context_id_callback=mdp.task_id_callback,
        align_kp_cfg=Kp(term="task_key_points", kp_names="asset_align"),
        align_against_kp_cfg=Kp(term="task_key_points", kp_names="asset_align_against")
    )

    manipulation_alignment_data = mdp.AlignmentMetricCfg(
        spec=ASSEMBLY_TASKS.manipulation_alignment_metric,
        context_id_callback=mdp.asset_id_callback_deterministic,
        context_id_param={
            "kp_asset_id_cfg": Kp(term="task_key_points", kp_attr="key_points_asset_id", kp_names="asset_align"),
            "asset_to_kps_cfg": Kp(term="manipulation_key_points", kp_attr="asset_id_key_points", kp_names="asset_grasp", is_spec=True)
        },
        align_kp_cfg=Kp(term="manipulation_key_points", kp_names="robot_object_held"),
        align_against_kp_cfg=Kp(term="manipulation_key_points", kp_names="asset_grasp")
    )


@configclass
class FactoryRewardsCfg:
    """Reward terms for Factory"""

    # penalties
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.0)

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.0)

    progress_reward = RewTerm(func=mdp.progress_reward_l2, weight=1.0, params={"alignment": Align(term="alignment_data")})

    success_reward = RewTerm(func=mdp.success_reward, weight=2.0, params={"alignment": Align(term="alignment_data")})


@configclass
class FactoryTerminationsCfg:
    """Termination terms for Factory."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    out_of_bounds = DoneTerm(
        func=mdp.out_of_bounds,
        params={
            "assets_cfg": SceneEntityCfg("assets"),
            "robot_cfg": SceneEntityCfg("robot"),
            "pos_range": ((-0.15, -0.675, -0.05), (1.0, 0.675, 1.08)),
        },
    )


##
# Environment configuration
##
@configclass
class FactoryEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the base Factory environment."""

    scene: FactorySceneCfg = FactorySceneCfg(num_envs=2, env_spacing=2.0)
    commands: FactoryCommandsCfg = FactoryCommandsCfg()
    data: FactoryDataCfg = FactoryDataCfg()
    observations: FactoryObservationsCfg = FactoryObservationsCfg()
    events: FactoryEventCfg = FactoryEventCfg()
    terminations: FactoryTerminationsCfg = FactoryTerminationsCfg()
    rewards: FactoryRewardsCfg = FactoryRewardsCfg()
    actions: FactoryActionsCfg = FactoryActionsCfg()
    # look at finger
    # viewer: ViewerCfg = ViewerCfg(
    #     eye=(0.0, 0.3, 0.1), lookat=(0.0, 0.0, -0.05), origin_type="asset_body", asset_name="robot", body_name="panda_fingertip_centered"
    # )
    # look at table
    viewer: ViewerCfg = ViewerCfg(
        eye=(1.5, 0.0, 0.4), lookat=(0.0, 0.0, 0.2), origin_type="asset_body", asset_name="robot", body_name="panda_link0"
    )

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 5
        self.episode_length_s = 120.0
        # simulation settings
        self.sim.dt = 0.05 / self.decimation  # 20hz
        self.sim.render_interval = self.decimation

        self.sim.physx.solver_type = 1
        self.sim.physx.max_position_iteration_count = 192  # Important to avoid interpenetration.
        self.sim.physx.max_velocity_iteration_count = 1
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_offset_threshold = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.sim.physx.gpu_max_rigid_patch_count = 2**23
        self.sim.physx.gpu_collision_stack_size = 2**31
        self.sim.physx.gpu_max_num_partitions = 1

        self.sim.render.enable_ambient_occlusion = True
        self.sim.render.enable_dlssg = True
