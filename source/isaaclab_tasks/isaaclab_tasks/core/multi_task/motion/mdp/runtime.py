# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-profile transition measurements behind one motion-runtime contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol

import torch

from isaaclab.utils.math import quat_apply, quat_apply_inverse, wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .commands import MotionStatePayload, MotionTransitionState
from .curriculums import MotionPenaltyScaleCurriculum

_G1_ENVIRONMENT_CHANNELS = {
    "penalty_torques": (-1.0e-6, True),
    "penalty_action_rate": (-0.5, True),
    "limits_dof_pos": (-10.0, True),
    "limits_dof_vel": (-5.0, False),
    "limits_torque": (-5.0, True),
    "penalty_undesired_contact": (-1.0, True),
    "penalty_feet_ori": (-0.1, True),
    "penalty_ankle_roll": (-0.5, True),
    "penalty_slippage": (-1.0, True),
    "feet_heading_alignment": (-0.1, False),
}


class MotionRuntime(MotionTransitionState, Protocol):
    """Complete transition runtime required by :class:`MotionImitationEnv`."""

    action_applied: torch.Tensor
    environment_reward: torch.Tensor
    auxiliary_evidence: torch.Tensor

    def capture_current(self, observations: Mapping[str, torch.Tensor]) -> None:
        """Capture immutable current-node fields before the next action."""

    def measure(self) -> None:
        """Measure the reached state before Same-Step final capture."""


def motion_time_out(env: ManagerBasedRLEnv, applied_actions_before_timeout: int) -> torch.Tensor:
    """Return environments at their source-faithful applied-action horizon."""
    return env.episode_length_buf >= applied_actions_before_timeout


class SmplMotionRuntime:
    """Zero-evidence runtime for the native SMPL profile."""

    def __init__(self, env: ManagerBasedRLEnv, payload: MotionStatePayload) -> None:
        self.env = env
        self.payload = payload
        self.action_applied = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.environment_reward = torch.zeros(env.num_envs, device=env.device)
        self.auxiliary_evidence = torch.empty(env.num_envs, 0, device=env.device)
        self.raw_evidence = payload.raw_evidence

    def capture_current(self, observations: Mapping[str, torch.Tensor]) -> None:
        """Accept the common pre-step capture call."""
        del observations

    def measure(self) -> None:
        """Advance motion time with a zero environment reward."""
        self.environment_reward.zero_()
        self.payload.record_step()

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset no profile-specific temporal state."""
        del env_ids


class G1MotionRuntime:
    """Measure native G1 history, raw physical evidence, and scalar reward."""

    def __init__(self, env: ManagerBasedRLEnv, payload: MotionStatePayload) -> None:
        from .actions import MotionJointPositionAction

        self.env = env
        self.payload = payload
        self.robot = env.scene["robot"]
        action = env.action_manager.get_term("joint_position")
        if not isinstance(action, MotionJointPositionAction):
            raise TypeError("G1MotionRuntime requires the motion joint-position action term.")
        self.action = action
        self.contact_sensor = env.scene.sensors["contact_forces"]

        self.foot_body_ids, foot_body_names = self.robot.find_bodies(
            ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
        )
        penalized_names = [
            name for name in self.robot.body_names if any(token in name for token in ("pelvis", "shoulder", "hip"))
        ]
        self.penalized_body_ids, penalized_body_names = self.robot.find_bodies(penalized_names, preserve_order=True)
        self.foot_sensor_ids, foot_sensor_names = self.contact_sensor.find_sensors(foot_body_names, preserve_order=True)
        self.penalized_sensor_ids, penalized_sensor_names = self.contact_sensor.find_sensors(
            penalized_body_names, preserve_order=True
        )
        if foot_sensor_names != foot_body_names or penalized_sensor_names != penalized_body_names:
            raise ValueError("Contact-sensor body order must match the requested articulation body order.")
        self.ankle_roll_joint_ids, _ = self.robot.find_joints(
            ["left_ankle_roll_joint", "right_ankle_roll_joint"], preserve_order=True
        )

        required_raw = _G1_ENVIRONMENT_CHANNELS.keys()
        if set(payload.raw_evidence_names) != set(required_raw):
            raise ValueError(f"G1 raw evidence must contain exactly {tuple(required_raw)}.")
        if any(spec.width != 1 for spec in payload.raw_evidence_specs):
            raise ValueError("G1 raw evidence channels must be scalar.")
        required_auxiliary = {
            name for name, (_, curriculum_scaled) in _G1_ENVIRONMENT_CHANNELS.items() if curriculum_scaled
        }
        if set(payload.auxiliary_evidence_names) != required_auxiliary:
            raise ValueError(f"G1 auxiliary evidence must contain exactly {tuple(required_auxiliary)}.")
        expected_history = (
            "processed_action",
            "base_angular_velocity",
            "joint_position",
            "joint_velocity",
            "projected_gravity",
        )
        if tuple(payload.history_fields) != expected_history:
            raise ValueError(f"G1 history fields must be ordered as {expected_history}.")
        penalty_curriculum = env.curriculum_manager.get_term("penalty_scale")
        if not isinstance(penalty_curriculum, MotionPenaltyScaleCurriculum):
            raise TypeError("G1MotionRuntime requires MotionPenaltyScaleCurriculum.")

        self.action_applied = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.environment_reward = torch.zeros(env.num_envs, device=env.device)
        self.raw_evidence = payload.raw_evidence
        self.auxiliary_evidence_names = payload.auxiliary_evidence_names
        self.auxiliary_evidence = torch.empty(env.num_envs, len(self.auxiliary_evidence_names), device=env.device)
        self.penalty_curriculum = penalty_curriculum
        self._gravity = torch.zeros(env.num_envs, 3, device=env.device)
        self._gravity[:, 2] = -1.0
        self._forward = torch.zeros(env.num_envs, 3, device=env.device)
        self._forward[:, 0] = 1.0

    def capture_current(self, observations: Mapping[str, torch.Tensor]) -> None:
        """Capture the exact noisy current actor facts before applying the next action."""
        state = observations["state"]
        last_action = observations["last_action"]
        if state.shape != (self.env.num_envs, 64) or last_action.shape != (self.env.num_envs, 29):
            raise ValueError("G1 current observations must contain state[64] and last_action[29].")
        fields = self.payload.history_fields
        fields["processed_action"].copy_(last_action)
        fields["joint_position"].copy_(state[:, :29])
        fields["joint_velocity"].copy_(state[:, 29:58])
        fields["projected_gravity"].copy_(state[:, 58:61])
        fields["base_angular_velocity"].copy_(state[:, 61:64])

    def measure(self) -> None:
        """Measure the reached physics state and record one logical edge."""
        joint_position = self.robot.data.joint_pos.torch
        joint_velocity = self.robot.data.joint_vel.torch
        torque = self.action.applied_torque
        processed_action = self.action.processed_actions
        previous_action = self.payload.history_fields["processed_action"]

        self.raw_evidence["penalty_torques"][:, 0] = torch.sum(torque.square(), dim=-1)
        self.raw_evidence["penalty_action_rate"][:, 0] = torch.sum(
            (previous_action - processed_action).square(), dim=-1
        )

        limits = self.robot.data.joint_pos_limits.torch
        midpoint = (limits[..., 0] + limits[..., 1]) * 0.5
        radius = (limits[..., 1] - limits[..., 0]) * 0.5 * 0.95
        lower = midpoint - radius
        upper = midpoint + radius
        position_excess = torch.clamp_max(joint_position - lower, 0.0).neg()
        position_excess.add_(torch.clamp_min(joint_position - upper, 0.0))
        self.raw_evidence["limits_dof_pos"][:, 0] = position_excess.sum(dim=-1)

        effort_limit = self.action.joint_effort_limit
        self.raw_evidence["limits_torque"][:, 0] = torch.clamp_min(torque.abs() - 0.95 * effort_limit, 0.0).sum(dim=-1)

        contact_force = self.contact_sensor.data.net_forces_w.torch
        penalized_force = contact_force[:, self.penalized_sensor_ids]
        self.raw_evidence["penalty_undesired_contact"][:, 0] = torch.any(
            penalized_force.abs() > 1.0, dim=(1, 2)
        ).float()

        foot_force = contact_force[:, self.foot_sensor_ids]
        contact = foot_force[..., 2] > 1.0
        foot_rotation = self.robot.data.body_link_quat_w.torch[:, self.foot_body_ids]
        foot_gravity = quat_apply_inverse(
            foot_rotation.reshape(-1, 4),
            self._gravity[:, None].expand(-1, 2, -1).reshape(-1, 3),
        ).view(self.env.num_envs, 2, 3)
        self.raw_evidence["penalty_feet_ori"][:, 0] = (
            torch.linalg.vector_norm(foot_gravity[..., :2], dim=-1) * contact
        ).sum(dim=-1)

        ankle_roll = joint_position[:, self.ankle_roll_joint_ids]
        self.raw_evidence["penalty_ankle_roll"][:, 0] = ankle_roll.square().sum(dim=-1)

        foot_velocity = self.robot.data.body_com_lin_vel_w.torch[:, self.foot_body_ids]
        self.raw_evidence["penalty_slippage"][:, 0] = (
            torch.linalg.vector_norm(foot_velocity, dim=-1) * (torch.linalg.vector_norm(foot_force, dim=-1) > 1.0)
        ).sum(dim=-1)

        velocity_limit = self.robot.data.joint_vel_limits.torch
        self.raw_evidence["limits_dof_vel"][:, 0] = torch.clamp(
            joint_velocity.abs() - 0.95 * velocity_limit,
            min=0.0,
            max=1.0,
        ).sum(dim=-1)

        foot_forward = quat_apply(
            foot_rotation.reshape(-1, 4),
            self._forward[:, None].expand(-1, 2, -1).reshape(-1, 3),
        ).view(self.env.num_envs, 2, 3)
        root_forward = quat_apply(self.robot.data.root_quat_w.torch, self._forward)
        foot_heading = torch.atan2(foot_forward[..., 1], foot_forward[..., 0])
        root_heading = torch.atan2(root_forward[:, 1], root_forward[:, 0])
        self.raw_evidence["feet_heading_alignment"][:, 0] = (
            wrap_to_pi(foot_heading - root_heading[:, None]).abs().sum(dim=-1)
        )

        self.environment_reward.zero_()
        for name in self.raw_evidence:
            coefficient, curriculum_scaled = _G1_ENVIRONMENT_CHANNELS[name]
            scale = self.penalty_curriculum.scale if curriculum_scaled else 1.0
            self.environment_reward.add_(self.raw_evidence[name][:, 0], alpha=coefficient * scale)

        for index, name in enumerate(self.auxiliary_evidence_names):
            self.auxiliary_evidence[:, index].copy_(self.raw_evidence[name][:, 0])
        self.payload.record_step()

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset no reached-edge outputs before Same-Step return.

        The payload clears cross-edge history itself. Evidence and scalar
        reward describe the just-completed edge and must survive the internal
        autoreset until the learner consumes the returned transition. The next
        :meth:`measure` call overwrites every row.
        """
        del env_ids


def make_smpl_motion_runtime(env: ManagerBasedRLEnv, payload: MotionStatePayload) -> SmplMotionRuntime:
    """Construct the SMPL runtime selected by its concrete profile."""
    return SmplMotionRuntime(env, payload)


def make_g1_motion_runtime(env: ManagerBasedRLEnv, payload: MotionStatePayload) -> G1MotionRuntime:
    """Construct the G1 runtime selected by its concrete profile."""
    return G1MotionRuntime(env, payload)


def motion_transition_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Measure one transition pre-final and return its scalar environment reward."""
    env._motion_runtime.measure()
    return env._motion_runtime.environment_reward
