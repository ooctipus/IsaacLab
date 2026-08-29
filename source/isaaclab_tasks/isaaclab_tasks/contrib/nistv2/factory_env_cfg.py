# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RL environment for assembling the complete NIST board."""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.nist import factory_assets_cfg as assets
from isaaclab_tasks.contrib.nist.factory_env_cfg import FactoryEnvCfg
from isaaclab_tasks.contrib.nist.factory_presets import (
    EndEffectorBodyCfg,
    GripperGraspOffsetCfg,
    GripperJointNamesCfg,
    IKJointNamesCfg,
    JointEffortNamesCfg,
)
from isaaclab_tasks.contrib.nist.utils import (
    BetaSamplingStrategyCfg,
    CollisionAnalyzerCfg,
    SamplerCfg,
    ValueShiftSamplingStrategyCfg,
)
from isaaclab_tasks.utils import PresetCfg, SuccessMonitorCfg, preset

from . import mdp
from .board_layout import AssemblySetCfg, board_layout
from .newton_selection import NewtonBodySelectorCfg


@configclass
class FactoryBoardSceneCfg(InteractiveSceneCfg):
    """NIST board scene populated by :class:`FactoryBoardEnvCfg`."""

    num_envs: int = 4096
    env_spacing: float = 2.0

    ground = assets.GROUND_CFG
    table = assets.TABLE_CFG.replace(spawn=assets.TABLE_CFG.spawn.replace(fix_root_link=True))
    nistboard = assets.NISTBOARD_CFG.replace(spawn=assets.NISTBOARD_CFG.spawn.replace(fix_root_link=True))
    robot = assets.FRANKA_PANDA_NEWTON_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=assets.FRANKA_PANDA_NEWTON_CFG.spawn.replace(
            physics_material=assets.ROBOT_CONTACT_MATERIAL_CFG.newton_mjwarp,
        ),
    )
    dome_light = assets.DOMELIGHT_CFG


@configclass
class FactoryBoardObservationsCfg:
    """Variant policy state generalized over the active assembly slots."""

    @configclass
    class PolicyCfg(ObsGroup):
        held_asset_in_fixed_asset_frame = ObsTerm(func=mdp.held_asset_in_fixed_asset_frame, history_length=5)
        end_effector_vel_lin_ang_b = ObsTerm(
            func=mdp.asset_link_velocity_in_root_asset_frame,
            history_length=5,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names=EndEffectorBodyCfg()),  # type: ignore
                "root_asset_cfg": SceneEntityCfg("robot"),
                "target_asset_offset": GripperGraspOffsetCfg(),
            },
        )
        joint_pos = ObsTerm(func=mdp.joint_pos, history_length=5)
        prev_action = ObsTerm(func=mdp.last_action, history_length=5)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PerceptionCfg(ObsGroup):
        scene_point_cloud = ObsTerm(
            func=mdp.scene_point_cloud_b,
            clip=(-2.0, 2.0),
            params={"fixed_num_points": 256, "held_num_points": 256, "robot_num_points": 256, "flatten": True},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    perception: PerceptionCfg = PerceptionCfg()


@configclass
class FactoryBoardRewardsCfg:
    """Penalize control effort and reward the active assembly goal."""

    action_l2 = RewTerm(func=mdp.action_l2_clamped, weight=-1e-4)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clamped, weight=-1e-4)
    joint_effort = RewTerm(
        func=mdp.joint_torques_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointEffortNamesCfg())},  # type:ignore
        weight=-1e-4,
    )
    early_termination = RewTerm(func=mdp.is_terminated_term, params={"term_keys": "abnormal"}, weight=-0.01)
    success_reward = RewTerm(func=mdp.assembly_success_reward, weight=100.0)
    solver_reset_reward = RewTerm(func="isaaclab_newton.envs.mdp:zero_reward_on_solver_reset", weight=1.0)


@configclass
class FactoryBoardTerminationsCfg:
    """Terminate on task success or invalid board state."""

    time_out = DoneTerm(
        func=mdp.initial_unfinished_time_out,
        time_out=True,
        params={
            "enabled": preset(default=True, progress_goal=False),
            "seconds_per_asset": 14.0,
            "dynamic": preset(default=True, fixed_timeout=False),
            "fixed_horizon_s": None,
            "dynamic_env_count": None,
        },
    )
    assembly_contact_force = DoneTerm(func=mdp.assembly_contact_force)
    oob = DoneTerm(func=mdp.any_held_asset_out_of_bound)
    progress_context = DoneTerm(func=mdp.assembly_progress_context)
    abnormal = DoneTerm(
        func=mdp.joint_vel_out_of_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="panda_joint[1-7]")},
    )
    success = DoneTerm(func=mdp.success_termination)
    solver_reset_required = DoneTerm(func="isaaclab_newton.envs.mdp:solver_reset_required")


@configclass
class FactoryBoardCurriculumCfg:
    """Publish reset-bank curriculum metrics."""

    metrics = CurrTerm(func=mdp.BoardMetrics)


@configclass
class FactoryBoardEventCfg:
    """Generate reset states and own the shared Newton board state."""

    held_asset_material = EventTerm(
        func=mdp.randomize_rigid_body_materials,
        mode="startup",
        params={
            "static_friction_range": (0.4, 1.0),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "asset_cfgs": (),
        },
    )
    fixed_asset_material = EventTerm(
        func=mdp.randomize_rigid_body_materials,
        mode="startup",
        params={
            "static_friction_range": (0.4, 1.0),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "asset_cfgs": (),
        },
    )
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
    reset_board = EventTerm(
        func=mdp.board_reset,
        mode="reset",
        params={
            "robot_ik_cfg": SceneEntityCfg("robot", joint_names=IKJointNamesCfg(), body_names=EndEffectorBodyCfg()),
            "robot_gripper_cfg": SceneEntityCfg("robot", joint_names=GripperJointNamesCfg()),
            "gripper_grasp_offset": GripperGraspOffsetCfg(),
            "variant_names": (),
            "num_slots": 0,
            "spawn_all_sockets": False,
            "state_table_size": 32760,
            "unfinished_count": preset(default=None, unfinished_1=1),
            "progress_goal": preset(default=False, progress_goal=True),
            "fallen_state_table_size": None,
            "settle_steps": 20,
            "fixed_asset_pose_range": {"x": (0.075, 0.25), "y": (-0.25, 0.25), "yaw": (-3.14, 3.14)},
            "held_asset_in_bound_range": {"x": (0.05, 1.0), "y": (-0.675, 0.675), "z": (-0.05, 1.0)},
            "acceptance_conditions": {},
            "success_monitor_cfg": preset(default=SuccessMonitorCfg(monitored_history_len=5), success_estimator=None),
            "success_monitor_env_count": None,
            "sampling": SamplerCfg(
                strategies=[
                    BetaSamplingStrategyCfg(
                        target=0.66,
                        kappa=1.0,
                        weight=1.0,
                        success_rate_bind="success_rates",
                    ),
                    ValueShiftSamplingStrategyCfg(
                        weight=preset(default=0.0, value_shift=0.0005, value_shift_005=0.005, value_shift_05=0.05),
                        value_shift_bind="value_shifts",
                    ),
                ],
                eps=1.0e-4,
            ),
            "report": True,
        },
    )
    assembly_state = EventTerm(
        func=mdp.AssemblyState,
        mode="startup",
        params={
            "held_bodies": NewtonBodySelectorCfg(path=()),
            "fixed_bodies": NewtonBodySelectorCfg(path=()),
            "board_body": NewtonBodySelectorCfg(path=r".*/NistBoard(?:/.*)?"),
            "robot_root_body": NewtonBodySelectorCfg(path=r".*/Robot(?:/.*)?/panda_link0"),
            "contact_sensor_cfg": SceneEntityCfg("assembly_contact"),
            "contact_force_threshold": 50.0,
            "success_threshold": 0.001,
            "workspace": {"x": (0.0, 1.0), "y": (-0.675, 0.675), "z": (-0.05, 1.0)},
        },
    )


@configclass
class FactoryBoardPhysicsCfg(PresetCfg):
    """Newton MJWarp preset matching the single-pair Variant task."""

    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=3000,
            nconmax=600,
            impratio=1.0,
            cone="pyramidal",
            update_data_interval=2,
            ls_parallel=False,
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(
            broad_phase="sap",
            max_triangle_pairs=90_000_000,
            contact_reduction_hashtable_size_factor=0.02,
            rigid_contact_max=5_000_000,
            speculative_config=NewtonCollisionPipelineCfg.SpeculativeContactCfg(max_speculative_extension=0.01),
            sdf_all_shapes=NewtonCollisionPipelineCfg.SDFAllShapesCfg(
                sdf_max_resolution=256,
                sdf_narrow_band_inner=-0.005,
                sdf_narrow_band_outer=0.005,
            ),
        ),
        num_substeps=8,
        debug_mode=False,
        use_cuda_graph=True,
    )
    default = newton_mjwarp


@configclass
class FactoryBoardEnvCfg(FactoryEnvCfg):
    """Factory task with a configurable number of held assets and matching sockets."""

    assembly_set: AssemblySetCfg = AssemblySetCfg()
    scene: FactoryBoardSceneCfg = FactoryBoardSceneCfg()
    observations: FactoryBoardObservationsCfg = FactoryBoardObservationsCfg()
    events: FactoryBoardEventCfg = FactoryBoardEventCfg()
    curriculum: FactoryBoardCurriculumCfg = FactoryBoardCurriculumCfg()
    terminations: FactoryBoardTerminationsCfg = FactoryBoardTerminationsCfg()
    rewards: FactoryBoardRewardsCfg = FactoryBoardRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.sim.render_interval = self.decimation
        if not isinstance(self.sim.physics, NewtonCfg):
            self.sim.physics = FactoryBoardPhysicsCfg()

        layout = board_layout(self.assembly_set)
        scene = self.scene
        for name in tuple(vars(scene)):
            if name.startswith(("fixed_", "held_")) or name == "assembly_contact":
                delattr(scene, name)
        light = scene.__dict__.pop("dome_light")
        scene.clone_cfg = cloner.CloneCfg(valid_set=[[0, 0, 0, *row] for row in layout.clone_rows()])

        fixed_spawns = tuple(
            variant.fixed_asset.spawn.replace(
                fix_root_link=True,
                collision_props=assets.ASSEMBLY_SOCKET_COLLISION_PROPS_CFG.newton_mjwarp,
                physics_material=assets.ASSEMBLY_CONTACT_MATERIAL_CFG.newton_mjwarp,
            )
            for variant in layout.variants
        )
        for slot, name in enumerate(layout.fixed_asset_names):
            if layout.fixed_assets_are_variant_banks:
                fixed_asset = RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/{name}",
                    spawn=sim_utils.MultiAssetSpawnerCfg(
                        assets_cfg=[spawn.copy() for spawn in fixed_spawns],
                        activate_contact_sensors=True,
                        random_choice=False,
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0 + 0.1 * slot)),
                    mesh_variants_enabled=True,
                )
            else:
                variant = layout.variants[layout.fixture_variant_indices[slot]]
                fixed_asset = variant.fixed_asset.replace(
                    prim_path=f"{{ENV_REGEX_NS}}/{name}",
                    spawn=fixed_spawns[layout.fixture_variant_indices[slot]],
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0 + 0.1 * slot)),
                )
            setattr(scene, name, fixed_asset)

        held_spawns = tuple(
            variant.held_asset.spawn.replace(
                collision_props=assets.ASSEMBLY_PLUG_COLLISION_PROPS_CFG.newton_mjwarp,
                physics_material=assets.ASSEMBLY_CONTACT_MATERIAL_CFG.newton_mjwarp,
            )
            for variant in layout.variants
        )
        for slot, name in enumerate(layout.held_asset_names):
            setattr(
                scene,
                name,
                RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/{name}",
                    spawn=sim_utils.MultiAssetSpawnerCfg(
                        assets_cfg=[spawn.copy() for spawn in held_spawns],
                        activate_contact_sensors=True,
                        random_choice=False,
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0 + 0.1 * slot)),
                    mesh_variants_enabled=True,
                    mesh_variant_inertia_diagonal_offset=1.0e-5,
                ),
            )
        scene.assembly_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/held_.*/.*",
            filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/{name}/.*" for name in layout.fixed_asset_names],
            history_length=1,
            update_period=0.0,
        )
        scene.dome_light = light

        self.events.held_asset_material = self.events.held_asset_material.replace(
            params=self.events.held_asset_material.params
            | {"asset_cfgs": tuple(SceneEntityCfg(name) for name in layout.held_asset_names)}
        )
        self.events.fixed_asset_material = self.events.fixed_asset_material.replace(
            params=self.events.fixed_asset_material.params
            | {"asset_cfgs": tuple(SceneEntityCfg(name) for name in layout.fixed_asset_names)}
        )
        self.events.reset_board = self.events.reset_board.replace(
            params=self.events.reset_board.params
            | {
                "variant_names": layout.variant_names,
                "num_slots": layout.num_slots,
                "spawn_all_sockets": layout.spawn_all_sockets,
                "acceptance_conditions": {
                    **{
                        f"held_{slot:02d}_collision_free": CollisionAnalyzerCfg(
                            num_points=256,
                            max_dist=0.5,
                            min_dist=-0.0005,
                            asset_cfg=SceneEntityCfg(name),
                            obstacle_cfgs=[
                                *(SceneEntityCfg(fixed) for fixed in layout.fixed_asset_names),
                                *(SceneEntityCfg(other) for other in layout.held_asset_names if other != name),
                                SceneEntityCfg("robot"),
                                SceneEntityCfg("nistboard"),
                                SceneEntityCfg("table"),
                            ],
                        )
                        for slot, name in enumerate(layout.held_asset_names)
                    },
                    "robot_collision_free": CollisionAnalyzerCfg(
                        num_points=1024,
                        max_dist=0.5,
                        min_dist=-0.002,
                        asset_cfg=SceneEntityCfg(
                            "robot", body_names="panda_link[2-7]|panda_hand|panda_(left|right)finger"
                        ),
                        obstacle_cfgs=[
                            SceneEntityCfg("table"),
                            SceneEntityCfg("nistboard"),
                            *(SceneEntityCfg(name) for name in layout.fixed_asset_names),
                        ],
                    ),
                },
            }
        )
        self.events.assembly_state = self.events.assembly_state.replace(
            params=self.events.assembly_state.params
            | {
                "held_bodies": NewtonBodySelectorCfg(
                    path=tuple(rf".*/{name}(?:/.*)?" for name in layout.held_asset_names)
                ),
                "fixed_bodies": NewtonBodySelectorCfg(
                    path=tuple(rf".*/{name}(?:/.*)?" for name in layout.fixed_asset_names)
                ),
            }
        )
        if layout.num_slots > 1 or layout.spawn_all_sockets:
            physics = self.sim.physics if isinstance(self.sim.physics, NewtonCfg) else self.sim.physics.newton_mjwarp
            physics.solver_cfg.njmax = 3200
            physics.solver_cfg.nconmax = 2400
            physics.collision_cfg.include_static_kinematic_pairs = False
            physics.collision_cfg.contact_reduction_hashtable_size_factor = 0.04
            if layout.num_slots > 1:
                physics.solver_cfg.enable_sleeping = True
                physics.solver_cfg.sleep_tolerance = 0.003
                # K19 peaks near 51 cached rows per world and slot; retain measured headroom.
                physics.collision_cfg.sdf_contact_replay_max_per_world = 64 * layout.num_slots
        if layout.num_slots > 1 and self.events.reset_board.params["state_table_size"] == 32760:
            self.events.reset_board.params["state_table_size"] = 65536
        self.episode_length_s = 14.0 * layout.num_slots

    def play_mode(self) -> None:
        """Use a small reset bank for interactive evaluation."""
        self.scene.num_envs = 16
        self.events.reset_board.params["state_table_size"] = 128
        self.events.reset_board.params["fallen_state_table_size"] = 256
