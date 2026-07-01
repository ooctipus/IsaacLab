# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Atomic SMPL-CMU and G1-LAFAN motion-environment presets."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from ..mdp.runtime import make_g1_motion_runtime, make_smpl_motion_runtime
from .profiles import MotionProfileCfg

_G1_RAW_EVIDENCE = (
    "penalty_torques",
    "penalty_action_rate",
    "limits_dof_pos",
    "limits_dof_vel",
    "limits_torque",
    "penalty_slippage",
    "penalty_undesired_contact",
    "penalty_ankle_roll",
    "penalty_feet_ori",
    "feet_heading_alignment",
)
_G1_AUXILIARY_EVIDENCE = (
    "penalty_torques",
    "penalty_action_rate",
    "limits_dof_pos",
    "limits_torque",
    "penalty_undesired_contact",
    "penalty_feet_ori",
    "penalty_ankle_roll",
    "penalty_slippage",
)


SMPL_CMU_PROFILE_CFG = MotionProfileCfg(
    identifier="smpl_cmu",
    timing=MotionProfileCfg.TimingCfg(
        physics_dt=1.0 / 450.0,
        control_decimation=15,
        configured_horizon_steps=300,
        applied_actions_before_timeout=300,
    ),
    reset=MotionProfileCfg.MocapAndFallResetCfg(
        motion_frame_probability=0.8,
        fall_probability=0.2,
        fall_random_actions_high_exclusive=5,
    ),
    routes=MotionProfileCfg.RouteCfg(
        transition_state_factory=make_smpl_motion_runtime,
        behavior_action_width=69,
        actor_width=358,
        privileged_width=358,
        expert_width=358,
        forward_width=358,
        actor_fields=("policy",),
        privileged_fields=("policy",),
        expert_fields=("policy",),
        forward_fields=("policy",),
    ),
)
"""Source-faithful SMPL-CMU runtime profile."""


G1_LAFAN_PROFILE_CFG = MotionProfileCfg(
    identifier="g1_lafan",
    timing=MotionProfileCfg.TimingCfg(
        physics_dt=1.0 / 200.0,
        control_decimation=4,
        configured_horizon_steps=500,
        applied_actions_before_timeout=501,
    ),
    reset=MotionProfileCfg.ReferenceResetCfg(
        lie_down_probability=0.3,
        lie_down_root_height_m=0.5,
    ),
    routes=MotionProfileCfg.RouteCfg(
        transition_state_factory=make_g1_motion_runtime,
        behavior_action_width=29,
        actor_width=465,
        privileged_width=463,
        expert_width=527,
        forward_width=928,
        actor_fields=("state", "last_action", "history_actor"),
        privileged_fields=("privileged_state",),
        expert_fields=("state", "privileged_state"),
        forward_fields=("state", "privileged_state", "last_action", "history_actor"),
        raw_evidence=_G1_RAW_EVIDENCE,
        auxiliary_evidence=_G1_AUXILIARY_EVIDENCE,
        history=MotionProfileCfg.HistoryCfg(
            length=4,
            frame_width=93,
            layout="field_major_then_newest_first_time",
            sources=(
                "processed_action",
                "base_angular_velocity",
                "joint_position",
                "joint_velocity",
                "projected_gravity",
            ),
            include_reset_seed=False,
        ),
    ),
    observation_noise=MotionProfileCfg.ObservationNoiseCfg(
        enabled=True,
        uniform_half_ranges={
            "base_angular_velocity_rad_s": 0.2,
            "joint_position_rad": 0.01,
            "joint_velocity_rad_s": 0.5,
            "projected_gravity": 0.05,
        },
        privileged_enabled=False,
    ),
    randomization=MotionProfileCfg.RandomizationCfg(
        enabled=True,
        body_mass_scale_range=(0.95, 1.05),
        friction_range=(0.5, 1.25),
        torso_com_offset_range_m=(-0.02, 0.02),
        default_joint_offset_range_rad=(-0.02, 0.02),
        push_linear_velocity_range_m_s=(-0.5, 0.5),
        push_angular_velocity_range_rad_s=(-0.5, 0.5),
        push_interval_seconds_integer_high_exclusive=(1, 3),
    ),
)
"""Source-faithful G1-LAFAN runtime profile."""


G1_CMU_PROFILE_CFG = G1_LAFAN_PROFILE_CFG.copy()
G1_CMU_PROFILE_CFG.identifier = "g1_cmu"
"""G1 runtime profile paired with the real HumEnv SMPL-CMU source."""


@configclass
class MotionControlDecimationCfg(PresetCfg):
    """Control decimation projected from the selected profile authority."""

    default = SMPL_CMU_PROFILE_CFG.timing.control_decimation
    smpl_cmu = SMPL_CMU_PROFILE_CFG.timing.control_decimation
    g1_lafan = G1_LAFAN_PROFILE_CFG.timing.control_decimation
    g1_cmu = G1_CMU_PROFILE_CFG.timing.control_decimation


@configclass
class MotionEpisodeLengthSecondsCfg(PresetCfg):
    """Episode duration [s] projected from the selected profile authority."""

    default = SMPL_CMU_PROFILE_CFG.timing.nominal_horizon_seconds
    smpl_cmu = SMPL_CMU_PROFILE_CFG.timing.nominal_horizon_seconds
    g1_lafan = G1_LAFAN_PROFILE_CFG.timing.nominal_horizon_seconds
    g1_cmu = G1_CMU_PROFILE_CFG.timing.nominal_horizon_seconds


__all__ = [
    "G1_CMU_PROFILE_CFG",
    "G1_LAFAN_PROFILE_CFG",
    "SMPL_CMU_PROFILE_CFG",
    "MotionControlDecimationCfg",
    "MotionEpisodeLengthSecondsCfg",
]
