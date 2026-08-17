# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.envs import mdp as newton_mdp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg

from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from . import mdp
from .assembly_variants import ASSEMBLY_VARIANT_NAMES
from .factory_presets import (
    EndEffectorBodyCfg,
    GripperGraspOffsetCfg,
    JointEffortNamesCfg,
    RobotActionsCfg,
)
from .factory_scenes_cfg import FactorySceneCfg
from .reset_env_cfg import ACCUMULATOR_RESET
from .utils import SamplerCfg, UniformSamplingStrategyCfg


@configclass
class FactoryObservationsCfg:
    """Observation specifications for Factory."""

    @configclass
    class PolicyCfg(ObsGroup):
        end_effector_vel_lin_ang_b = ObsTerm(
            func=mdp.asset_link_velocity_in_root_asset_frame,
            history_length=5,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names=EndEffectorBodyCfg()),  # type:ignore
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
            params={
                "fixed_asset_cfg": SceneEntityCfg("fixed_asset"),
                "held_asset_cfg": SceneEntityCfg("held_asset"),
                "robot_asset_cfg": SceneEntityCfg("robot"),
                "fixed_num_points": 256,
                "held_num_points": 256,
                "robot_num_points": 256,
                "flatten": True,
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    perception: PerceptionCfg = PerceptionCfg()


@configclass
class FactoryEventCfg:
    """Events specifications for Factory"""

    assembly_variants = EventTerm(
        func=mdp.AssemblyVariantContext,
        mode="startup",
        params={
            "fixed_asset_cfg": SceneEntityCfg("fixed_asset"),
            "held_asset_cfg": SceneEntityCfg("held_asset"),
            "variant_names": ASSEMBLY_VARIANT_NAMES,
        },
    )

    held_asset_material = EventTerm(
        func=mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.4, 1.0),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "asset_cfg": SceneEntityCfg("held_asset"),
        },
    )

    fixed_asset_material = EventTerm(
        func=mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.4, 1.0),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "asset_cfg": SceneEntityCfg("fixed_asset"),
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

    reset_strategies = ACCUMULATOR_RESET


@configclass
class FactoryRewardsCfg:
    """Reward terms for Factory. Success is terminal and carries the dominant weight."""

    action_l2 = RewTerm(func=mdp.action_l2_clamped, weight=-1e-4)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clamped, weight=-1e-4)
    joint_effort = RewTerm(
        func=mdp.joint_torques_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointEffortNamesCfg())},  # type:ignore
        weight=-1e-4,
    )
    early_termination = RewTerm(func=mdp.is_terminated_term, params={"term_keys": "abnormal"}, weight=-0.01)
    success_reward = RewTerm(func=mdp.success_reward, weight=100.0)
    solver_reset_reward = RewTerm(func=newton_mdp.zero_reward_on_solver_reset, weight=1.0)


@configclass
class FactoryTerminationsCfg:
    """Termination terms for Factory. Reaching the assembled pose ends the episode."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    assembly_contact_force = DoneTerm(
        func=mdp.assembly_contact_force,
        params={"threshold": 50.0, "sensor_cfg": SceneEntityCfg("assembly_contact")},
    )

    oob = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "asset_cfg": SceneEntityCfg("held_asset"),
            "in_bound_range": {"x": (0.0, 1.0), "y": (-0.675, 0.675), "z": (-0.05, 1.0)},
        },
    )

    progress_context = DoneTerm(
        func=mdp.progress_context,
        params={
            "success_threshold": 0.001,
            "held_asset_cfg": SceneEntityCfg("held_asset"),
            "fixed_asset_cfg": SceneEntityCfg("fixed_asset"),
        },
    )

    abnormal = DoneTerm(
        func=mdp.joint_vel_out_of_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="panda_joint[1-7]")},
    )

    success = DoneTerm(func=mdp.success_termination)
    solver_reset_required = DoneTerm(func=newton_mdp.solver_reset_required)


##
# Environment configuration
##


@configclass
class FactoryPhysicsCfg(PresetCfg):
    """Newton MJWarp configuration for runtime mesh variants."""

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
class FactoryBaseEnvCfg(ManagerBasedRLEnvCfg):
    """Homogeneous Factory environment with reset-selectable assembly pairs."""

    scene: FactorySceneCfg = FactorySceneCfg()
    observations: FactoryObservationsCfg = FactoryObservationsCfg()
    events: FactoryEventCfg = FactoryEventCfg()
    terminations: FactoryTerminationsCfg = FactoryTerminationsCfg()
    rewards: FactoryRewardsCfg = FactoryRewardsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(0.0, 0.8, 0.4), lookat=(0.0, 0.0, 0.4))
    actions: RobotActionsCfg = RobotActionsCfg()  # type: ignore

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        # Collision runs once per sim step and the solver runs num_substeps inside it,
        # so dt sets the collision rate: 0.01 s -> 100 Hz collide, x8 substeps ->
        # 800 Hz solver. Decimation is halved alongside it so the policy still acts
        # every 0.04 s; leaving it at 8 would have quietly halved the control rate.
        self.decimation = 4
        self.episode_length_s = 14.0
        self.sim.dt = 0.04 / self.decimation
        self.sim.render_interval = self.decimation
        self.sim.physics = FactoryPhysicsCfg()

        self.sim.physics_material.static_friction = 0.5
        self.sim.physics_material.dynamic_friction = 0.5

    def play_mode(self) -> None:
        """Narrow the reset curriculum for evaluation.

        Training samples a curriculum over several reset strategies and a large bank of
        stored states; a policy is instead scored on one strategy, drawn uniformly, so the
        number reflects the task rather than whatever the curriculum currently favors.

        Called by :func:`~isaaclab_tasks.utils.hydra.register_task` after preset resolution,
        so it edits the already-resolved terms.
        """

        # ``play`` reads the training default when ``--num_envs`` is omitted, and the training
        # default is sized for throughput, not for watching a policy.
        self.scene.num_envs = 128

        uniform = SamplerCfg(strategies=[UniformSamplingStrategyCfg(weight=1.0)], eps=0.0)

        reset = self.events.reset_strategies.params

        if "state_table_size" in reset:
            reset["state_table_size"] = 500
        if "sampling" in reset:
            reset["sampling"] = uniform

        scene_reset = reset.get("reset_term", self.events.reset_strategies)
        choice = scene_reset.params["terms"]["reset_strategies"].params
        choice["terms"] = {"start_random": choice["terms"]["start_random"]}
