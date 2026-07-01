# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct manager-group preset axes for motion imitation."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab_newton.sim.schemas import MujocoCollisionPropertiesCfg, NewtonMaterialPropertiesCfg
from isaaclab_physx.sensors import ContactSensorCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import mdp as isaaclab_mdp
from isaaclab.managers import ActionTermCfg, SceneEntityCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import SensorBaseCfg
from isaaclab.sim.schemas import CollisionBaseCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as UniformNoise

from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.smpl.smpl_constants import MUJOCO_BODY_NAMES

from ...mdp.commands.state_command.state_command_cfg import StateCommandCfg
from ..data import MotionSampleGrid
from ..frames import G1_HEAD_PARENT_BODY_NAME
from ..mdp import observations
from ..mdp.actions_cfg import MotionJointPositionActionCfg, MotionMujocoControlActionCfg
from ..mdp.commands import MotionStatePayloadCfg, MotionTaskTableCfg
from ..mdp.curriculums import MotionPenaltyScaleCurriculum
from ..mdp.events import MotionPushVelocity, set_smpl_body_mass_inertia
from ..mdp.reset_sources import G1ReferenceAndLieDownReset, SmplMocapAndFallReset
from ..mdp.runtime import motion_time_out
from ..trajectory.g1 import g1_lafan_frame_builder
from ..trajectory.g1_smpl import g1_smpl_humenv_frame_builder
from ..trajectory.smpl import smpl_humenv_frame_builder
from .presets import G1_CMU_PROFILE_CFG, G1_LAFAN_PROFILE_CFG, SMPL_CMU_PROFILE_CFG
from .robots import (
    G1_BEHAVIOR_BODY_NAMES,
    G1_BEHAVIOR_JOINT_NAMES,
    g1_reference_kinematics,
    smpl_reference_kinematics,
)
from .sources import G1_LAFAN_SOURCE_CFG, SMPL_CMU_SOURCE_CFG

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _g1_actor_noise(name: str) -> UniformNoise | None:
    """Project one declared actor-noise half-range into an observation term."""
    noise = G1_LAFAN_PROFILE_CFG.observation_noise
    if not noise.enabled:
        return None
    half_range = noise.uniform_half_ranges[name]
    return UniformNoise(n_min=-half_range, n_max=half_range)


def _smpl_mocap_and_fall_reset(env: ManagerBasedRLEnv) -> SmplMocapAndFallReset:
    """Project the SMPL reset profile into its concrete runtime transform."""
    return SmplMocapAndFallReset(
        env,
        random_actions_high_exclusive=SMPL_CMU_PROFILE_CFG.reset.fall_random_actions_high_exclusive,
        physics_dt_seconds=SMPL_CMU_PROFILE_CFG.timing.physics_dt,
        physics_steps_per_action=SMPL_CMU_PROFILE_CFG.timing.control_decimation,
    )


def _g1_reference_and_lie_down_reset(env: ManagerBasedRLEnv) -> G1ReferenceAndLieDownReset:
    """Project the G1 reset profile into its concrete runtime transform."""
    return G1ReferenceAndLieDownReset(
        env,
        lie_down_root_height_m=G1_LAFAN_PROFILE_CFG.reset.lie_down_root_height_m,
    )


def _motion_ground(
    friction: float,
    collision_props: CollisionBaseCfg | None = None,
    physics_material: RigidBodyMaterialBaseCfg | None = None,
) -> AssetBaseCfg:
    """Return one local infinite plane with native friction."""
    if physics_material is None:
        physics_material = RigidBodyMaterialBaseCfg(
            static_friction=friction,
            dynamic_friction=friction,
            restitution=0.0,
        )
    return AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.PlaneCfg(
            size=(200.0, 200.0),
            collision_props=collision_props or CollisionBaseCfg(collision_enabled=True),
            physics_material=physics_material,
        ),
    )


@configclass
class MotionGroundCfg(PresetCfg):
    """Ground material selected by the shared motion preset name."""

    default = _motion_ground(
        0.7,
        MujocoCollisionPropertiesCfg(
            collision_enabled=True,
            margin=0.001,
            solimp=(0.99, 0.99, 0.003, 0.5, 2.0),
            solref=(0.015, 1.0),
        ),
        NewtonMaterialPropertiesCfg(
            static_friction=0.7,
            dynamic_friction=0.7,
            restitution=0.0,
            torsional_friction=0.005,
            rolling_friction=0.0001,
        ),
    )
    smpl_cmu = default
    g1_lafan = _motion_ground(1.0)
    g1_cmu = g1_lafan


@configclass
class MotionContactSensorCfg(PresetCfg):
    """Optional contact sensor selected by the shared motion preset name."""

    default: SensorBaseCfg | None = None
    smpl_cmu: SensorBaseCfg | None = None
    g1_lafan = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=G1_LAFAN_PROFILE_CFG.timing.physics_dt,
        history_length=0,
    )
    g1_cmu = g1_lafan


@configclass
class MotionActionsCfg:
    """One coordinated articulation action term."""

    joint_position: ActionTermCfg = MISSING


_SMPL_ACTIONS = MotionActionsCfg(
    joint_position=MotionMujocoControlActionCfg(
        asset_name="robot",
        action_width=SMPL_CMU_PROFILE_CFG.routes.behavior_action_width,
    )
)
_G1_ACTIONS = MotionActionsCfg(
    joint_position=MotionJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(G1_BEHAVIOR_JOINT_NAMES),
        preserve_order=True,
        default_joint_offset_range=G1_LAFAN_PROFILE_CFG.randomization.default_joint_offset_range_rad,
    )
)


@configclass
class MotionActionsPresetsCfg(PresetCfg):
    """Action group selected by the shared motion preset name."""

    default: MotionActionsCfg = _SMPL_ACTIONS
    smpl_cmu: MotionActionsCfg = _SMPL_ACTIONS
    g1_lafan: MotionActionsCfg = _G1_ACTIONS
    g1_cmu: MotionActionsCfg = _G1_ACTIONS


@configclass
class SmplMotionObservationsCfg:
    """Native 358-wide SMPL observation route."""

    @configclass
    class PolicyCfg(ObsGroup):
        observation = ObsTerm(
            func=observations.smpl_body_observation,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=list(MUJOCO_BODY_NAMES),
                    preserve_order=True,
                )
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class G1MotionObservationsCfg:
    """Native G1 actor, history, and privileged routes."""

    @configclass
    class StateCfg(ObsGroup):
        joint_position = ObsTerm(
            func=observations.motion_joint_position,
            params={"action_name": "joint_position"},
            noise=_g1_actor_noise("joint_position_rad"),
        )
        joint_velocity = ObsTerm(
            func=observations.motion_joint_velocity,
            params={"action_name": "joint_position"},
            noise=_g1_actor_noise("joint_velocity_rad_s"),
        )
        projected_gravity = ObsTerm(
            func=observations.motion_projected_gravity,
            noise=_g1_actor_noise("projected_gravity"),
        )
        base_angular_velocity = ObsTerm(
            func=observations.motion_root_angular_velocity,
            noise=_g1_actor_noise("base_angular_velocity_rad_s"),
            scale=0.25,
        )

        def __post_init__(self) -> None:
            self.enable_corruption = G1_LAFAN_PROFILE_CFG.observation_noise.enabled
            self.concatenate_terms = True

    @configclass
    class LastActionCfg(ObsGroup):
        value = ObsTerm(func=observations.motion_last_action, params={"action_name": "joint_position"})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class HistoryActorCfg(ObsGroup):
        value = ObsTerm(func=observations.motion_history, params={"command_name": "motion"})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PrivilegedStateCfg(ObsGroup):
        value = ObsTerm(
            func=observations.g1_privileged_body_observation,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=list(G1_BEHAVIOR_BODY_NAMES), preserve_order=True),
                "parent_body_index": G1_BEHAVIOR_BODY_NAMES.index(G1_HEAD_PARENT_BODY_NAME),
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = G1_LAFAN_PROFILE_CFG.observation_noise.privileged_enabled
            self.concatenate_terms = True

    state: StateCfg = StateCfg()
    last_action: LastActionCfg = LastActionCfg()
    history_actor: HistoryActorCfg = HistoryActorCfg()
    privileged_state: PrivilegedStateCfg = PrivilegedStateCfg()


@configclass
class MotionObservationsPresetsCfg(PresetCfg):
    """Observation groups selected by the shared motion preset name."""

    default = SmplMotionObservationsCfg()
    smpl_cmu = default
    g1_lafan = G1MotionObservationsCfg()
    g1_cmu = g1_lafan


@configclass
class MotionCommandsCfg:
    """One motion descriptor command."""

    motion: StateCommandCfg = MISSING


_EVIDENCE_UNITS = {
    "penalty_torques": "(N*m)^2",
    "penalty_action_rate": "",
    "limits_dof_pos": "rad",
    "limits_dof_vel": "rad/s",
    "limits_torque": "N*m",
    "penalty_slippage": "m/s",
    "penalty_undesired_contact": "",
    "penalty_ankle_roll": "rad^2",
    "penalty_feet_ori": "",
    "feet_heading_alignment": "rad",
}
_EVIDENCE_ANCHORS = {
    "penalty_torques": "transition_action_and_reached_physics",
    "penalty_action_rate": "transition_action",
}
_G1_HISTORY_FIELDS = (
    ("processed_action", 29),
    ("base_angular_velocity", 3),
    ("joint_position", 29),
    ("joint_velocity", 29),
    ("projected_gravity", 3),
)


def _raw_evidence(profile):
    return tuple(
        MotionStatePayloadCfg.RawEvidenceCfg(
            name=name,
            unit=_EVIDENCE_UNITS[name],
            anchor=_EVIDENCE_ANCHORS.get(name, "transition_reached_physics"),
        )
        for name in profile.routes.raw_evidence
    )


def _motion_command(
    profile,
    *,
    source,
    frame_builder_factory,
    task_row_mode,
    reset_sources,
    reference_kinematics_factory,
    expert_sample_grid,
    reset_transform_factory,
    root_velocity_frame,
):
    history = profile.routes.history
    return MotionCommandsCfg(
        motion=StateCommandCfg(
            resampling_time_range=(1.0e9, 1.0e9),
            debug_vis=False,
            randomize_command_indices=True,
            states_relative=True,
            commands={},
            task_table=MotionTaskTableCfg(
                source=source,
                frame_builder_factory=frame_builder_factory,
                reference_kinematics_factory=reference_kinematics_factory,
                expert_sample_grid=expert_sample_grid,
                task_row_mode=task_row_mode,
                reset_sources=reset_sources,
            ),
            payload=MotionStatePayloadCfg(
                robot_asset_name="robot",
                reset_transform_factory=reset_transform_factory,
                root_velocity_frame=root_velocity_frame,
                transition_state_factory=profile.routes.transition_state_factory,
                step_fields=(),
                command_fields=(),
                episode_length_steps=profile.timing.applied_actions_before_timeout,
                history_fields=() if history is None else _G1_HISTORY_FIELDS,
                history_length=0 if history is None else history.length,
                raw_evidence=_raw_evidence(profile),
                auxiliary_evidence=profile.routes.auxiliary_evidence,
            ),
        )
    )


_SMPL_COMMANDS = _motion_command(
    SMPL_CMU_PROFILE_CFG,
    source=SMPL_CMU_SOURCE_CFG,
    frame_builder_factory=smpl_humenv_frame_builder,
    reference_kinematics_factory=smpl_reference_kinematics,
    expert_sample_grid=MotionSampleGrid.source_rows(),
    task_row_mode="source_frames",
    reset_sources=(
        ("motion", SMPL_CMU_PROFILE_CFG.reset.motion_frame_probability),
        ("fall", SMPL_CMU_PROFILE_CFG.reset.fall_probability),
    ),
    reset_transform_factory=_smpl_mocap_and_fall_reset,
    root_velocity_frame="link",
)
_G1_LAFAN_COMMANDS = _motion_command(
    G1_LAFAN_PROFILE_CFG,
    source=G1_LAFAN_SOURCE_CFG,
    frame_builder_factory=g1_lafan_frame_builder,
    reference_kinematics_factory=g1_reference_kinematics,
    expert_sample_grid=MotionSampleGrid.uniform_before_source_end(step_seconds=G1_LAFAN_PROFILE_CFG.timing.control_dt),
    task_row_mode="clip_time_ranges",
    reset_sources=(
        ("reference", 1.0 - G1_LAFAN_PROFILE_CFG.reset.lie_down_probability),
        ("lie_down", G1_LAFAN_PROFILE_CFG.reset.lie_down_probability),
    ),
    reset_transform_factory=_g1_reference_and_lie_down_reset,
    root_velocity_frame="center_of_mass",
)
_G1_CMU_COMMANDS = _motion_command(
    G1_CMU_PROFILE_CFG,
    source=SMPL_CMU_SOURCE_CFG,
    frame_builder_factory=g1_smpl_humenv_frame_builder,
    reference_kinematics_factory=g1_reference_kinematics,
    expert_sample_grid=MotionSampleGrid.uniform_before_source_end(step_seconds=G1_CMU_PROFILE_CFG.timing.control_dt),
    task_row_mode="clip_time_ranges",
    reset_sources=(
        ("reference", 1.0 - G1_CMU_PROFILE_CFG.reset.lie_down_probability),
        ("lie_down", G1_CMU_PROFILE_CFG.reset.lie_down_probability),
    ),
    reset_transform_factory=_g1_reference_and_lie_down_reset,
    root_velocity_frame="center_of_mass",
)


@configclass
class MotionCommandsPresetsCfg(PresetCfg):
    """Command/table/payload group selected by the shared motion preset name."""

    default: MotionCommandsCfg = _SMPL_COMMANDS
    smpl_cmu: MotionCommandsCfg = _SMPL_COMMANDS
    g1_lafan: MotionCommandsCfg = _G1_LAFAN_COMMANDS
    g1_cmu: MotionCommandsCfg = _G1_CMU_COMMANDS


@configclass
class MotionEventsCfg:
    """Optional physical randomization."""

    robot_material: EventTerm | None = None
    body_mass_inertia: EventTerm | None = None
    body_mass: EventTerm | None = None
    torso_com: EventTerm | None = None
    push: EventTerm | None = None


_SMPL_EVENTS = MotionEventsCfg(
    body_mass_inertia=EventTerm(
        func=set_smpl_body_mass_inertia,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
)
_randomization = G1_LAFAN_PROFILE_CFG.randomization
_G1_EVENTS = MotionEventsCfg(
    robot_material=EventTerm(
        func=isaaclab_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": _randomization.friction_range,
            "dynamic_friction_range": _randomization.friction_range,
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    ),
    body_mass=EventTerm(
        func=isaaclab_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": _randomization.body_mass_scale_range,
            "operation": "scale",
            "distribution": "uniform",
        },
    ),
    torso_com=EventTerm(
        func=isaaclab_mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {axis: _randomization.torso_com_offset_range_m for axis in "xyz"},
        },
    ),
    push=EventTerm(
        func=MotionPushVelocity,
        mode="interval",
        interval_range_s=(1.0, 1.0),
        is_global_time=True,
        resample_interval_on_reset=False,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "interval_seconds_integer_high_exclusive": (_randomization.push_interval_seconds_integer_high_exclusive),
            "linear_velocity_range_m_s": _randomization.push_linear_velocity_range_m_s,
            "angular_velocity_range_rad_s": _randomization.push_angular_velocity_range_rad_s,
        },
    ),
)


@configclass
class MotionEventsPresetsCfg(PresetCfg):
    """Event group selected by the shared motion preset name."""

    default: MotionEventsCfg = _SMPL_EVENTS
    smpl_cmu: MotionEventsCfg = _SMPL_EVENTS
    g1_lafan: MotionEventsCfg = _G1_EVENTS
    g1_cmu: MotionEventsCfg = _G1_EVENTS


@configclass
class MotionRewardsCfg:
    """One environment-owned immediate reward channel."""

    environment = RewTerm(
        func="isaaclab_tasks.core.multi_task.motion.mdp.runtime:motion_transition_reward",
        weight=1.0,
    )


@configclass
class MotionTerminationsCfg:
    """Timeout-only native reproduction boundary."""

    time_out: DoneTerm = MISSING  # type: ignore[assignment]


def _motion_terminations(profile) -> MotionTerminationsCfg:
    """Bind one profile's exact applied-action timeout edge."""
    return MotionTerminationsCfg(
        time_out=DoneTerm(
            func=motion_time_out,
            time_out=True,
            params={"applied_actions_before_timeout": profile.timing.applied_actions_before_timeout},
        )
    )


_SMPL_TERMINATIONS = _motion_terminations(SMPL_CMU_PROFILE_CFG)
_G1_LAFAN_TERMINATIONS = _motion_terminations(G1_LAFAN_PROFILE_CFG)
_G1_CMU_TERMINATIONS = _motion_terminations(G1_CMU_PROFILE_CFG)


@configclass
class MotionTerminationsPresetsCfg(PresetCfg):
    """Timeout group selected by the shared motion preset name."""

    default: MotionTerminationsCfg = _SMPL_TERMINATIONS
    smpl_cmu: MotionTerminationsCfg = _SMPL_TERMINATIONS
    g1_lafan: MotionTerminationsCfg = _G1_LAFAN_TERMINATIONS
    g1_cmu: MotionTerminationsCfg = _G1_CMU_TERMINATIONS


@configclass
class MotionCurriculumCfg:
    """Optional environment reward curriculum."""

    penalty_scale: CurrTerm | None = None


@configclass
class MotionCurriculumPresetsCfg(PresetCfg):
    """Curriculum group selected by the shared motion preset name."""

    default = MotionCurriculumCfg()
    smpl_cmu = default
    g1_lafan = MotionCurriculumCfg(penalty_scale=CurrTerm(func=MotionPenaltyScaleCurriculum))
    g1_cmu = g1_lafan


__all__ = [
    "G1MotionObservationsCfg",
    "MotionActionsCfg",
    "MotionActionsPresetsCfg",
    "MotionCommandsCfg",
    "MotionCommandsPresetsCfg",
    "MotionContactSensorCfg",
    "MotionCurriculumCfg",
    "MotionCurriculumPresetsCfg",
    "MotionEventsCfg",
    "MotionEventsPresetsCfg",
    "MotionGroundCfg",
    "MotionObservationsPresetsCfg",
    "MotionRewardsCfg",
    "MotionTerminationsCfg",
    "MotionTerminationsPresetsCfg",
    "SmplMotionObservationsCfg",
]
