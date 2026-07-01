# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused tests for motion actions, observations, history, and runtime evidence."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch
import warp as wp

from isaaclab.managers import CurriculumTermCfg, RewardManager, RewardTermCfg
from isaaclab.utils.math import quat_apply

from isaaclab_tasks.core.multi_task.motion.config.presets import G1_LAFAN_PROFILE_CFG
from isaaclab_tasks.core.multi_task.motion.config.robots import G1_BEHAVIOR_BODY_NAMES
from isaaclab_tasks.core.multi_task.motion.config.source_skeletons import g1_lafan_source_skeleton
from isaaclab_tasks.core.multi_task.motion.data import MotionClipIndex, MotionSampleGrid
from isaaclab_tasks.core.multi_task.motion.frames import G1_HEAD_FRAME_NAME, G1_HEAD_PARENT_BODY_NAME
from isaaclab_tasks.core.multi_task.motion.mdp.actions import (
    MotionJointPositionAction,
    MotionMujocoControlAction,
)
from isaaclab_tasks.core.multi_task.motion.mdp.actions_cfg import MotionMujocoControlActionCfg
from isaaclab_tasks.core.multi_task.motion.mdp.commands import MotionStatePayload, MotionTaskTable
from isaaclab_tasks.core.multi_task.motion.mdp.curriculums import MotionPenaltyScaleCurriculum
from isaaclab_tasks.core.multi_task.motion.mdp.observations import (
    g1_privileged_body_observation,
    g1_privileged_observation,
    motion_history,
    motion_joint_position,
    motion_joint_velocity,
    motion_last_action,
    motion_projected_gravity,
    motion_root_angular_velocity,
    smpl_body_observation,
    smpl_humenv_observation,
)
from isaaclab_tasks.core.multi_task.motion.mdp.runtime import (
    G1MotionRuntime,
    SmplMotionRuntime,
    motion_transition_reward,
)

_G1_RAW_NAMES = G1_LAFAN_PROFILE_CFG.routes.raw_evidence
_G1_AUXILIARY_NAMES = G1_LAFAN_PROFILE_CFG.routes.auxiliary_evidence
_G1_HISTORY_FIELDS = (
    ("processed_action", 29),
    ("base_angular_velocity", 3),
    ("joint_position", 29),
    ("joint_velocity", 29),
    ("projected_gravity", 3),
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _view(value: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(torch=value)


class _Robot:
    def __init__(self, num_envs: int) -> None:
        skeleton = g1_lafan_source_skeleton()
        self.joint_names = list(skeleton.joint_names)
        self.body_names = list(skeleton.body_names)
        self.num_joints = len(self.joint_names)
        num_bodies = len(self.body_names)

        joint_position = torch.zeros(num_envs, self.num_joints)
        joint_velocity = torch.zeros_like(joint_position)
        joint_limits = torch.empty(num_envs, self.num_joints, 2)
        joint_limits[..., 0] = -1.0
        joint_limits[..., 1] = 1.0
        body_position = torch.zeros(num_envs, num_bodies, 3)
        body_rotation = torch.zeros(num_envs, num_bodies, 4)
        body_rotation[..., 3] = 1.0
        body_linear_velocity = torch.zeros_like(body_position)
        body_angular_velocity = torch.zeros_like(body_position)
        root_rotation = torch.zeros(num_envs, 4)
        root_rotation[:, 3] = 1.0
        self.data = SimpleNamespace(
            default_joint_pos=_view(torch.zeros_like(joint_position)),
            joint_stiffness=_view(torch.full_like(joint_position, 40.0)),
            joint_damping=_view(torch.full_like(joint_position, 2.0)),
            joint_effort_limits=_view(torch.full_like(joint_position, 20.0)),
            joint_pos=_view(joint_position),
            joint_vel=_view(joint_velocity),
            joint_pos_limits=_view(joint_limits),
            joint_vel_limits=_view(torch.full_like(joint_position, 2.0)),
            body_pos_w=_view(torch.full_like(body_position, -300.0)),
            body_link_pos_w=_view(body_position),
            body_quat_w=_view(body_rotation),
            body_link_quat_w=_view(body_rotation),
            body_lin_vel_w=_view(torch.full_like(body_linear_velocity, -100.0)),
            body_link_lin_vel_w=_view(torch.full_like(body_linear_velocity, -101.0)),
            body_com_lin_vel_w=_view(body_linear_velocity),
            body_ang_vel_w=_view(torch.full_like(body_angular_velocity, -200.0)),
            body_link_ang_vel_w=_view(torch.full_like(body_angular_velocity, -201.0)),
            body_com_ang_vel_w=_view(body_angular_velocity),
            root_quat_w=_view(root_rotation),
            projected_gravity_b=_view(torch.tensor((0.0, 0.0, -1.0)).repeat(num_envs, 1)),
            root_ang_vel_w=_view(torch.zeros(num_envs, 3)),
            root_ang_vel_b=_view(torch.zeros(num_envs, 3)),
        )
        self.last_joint_target: torch.Tensor | None = None
        self.last_joint_ids: torch.Tensor | slice | None = None

    def find_bodies(self, names: list[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        del preserve_order
        return [self.body_names.index(name) for name in names], names

    def find_joints(self, names: list[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        del preserve_order
        return [self.joint_names.index(name) for name in names], names

    def set_joint_position_target_index(
        self,
        *,
        target: torch.Tensor,
        joint_ids: torch.Tensor | slice,
    ) -> None:
        self.last_joint_target = target.clone()
        self.last_joint_ids = joint_ids

    def write_root_link_pose_to_sim_index(self, *, root_pose: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.data.root_quat_w.torch[env_ids] = root_pose[:, 3:7]

    def write_root_link_velocity_to_sim_index(self, *, root_velocity: torch.Tensor, env_ids: torch.Tensor) -> None:
        del root_velocity, env_ids

    def write_root_com_velocity_to_sim_index(self, *, root_velocity: torch.Tensor, env_ids: torch.Tensor) -> None:
        del root_velocity, env_ids

    def write_joint_position_to_sim_index(self, *, position: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.data.joint_pos.torch[env_ids] = position

    def write_joint_velocity_to_sim_index(self, *, velocity: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.data.joint_vel.torch[env_ids] = velocity


class _ContactSensor:
    def __init__(self, body_names: list[str], contact_force: torch.Tensor) -> None:
        self.body_names = body_names
        self.data = SimpleNamespace(net_forces_w=_view(contact_force))

    def find_sensors(self, names: list[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        del preserve_order
        return [self.body_names.index(name) for name in names], names


class _Scene:
    def __init__(self, robot: _Robot, contact_force: torch.Tensor) -> None:
        self.robot = robot
        self.env_origins = torch.zeros(contact_force.shape[0], 3)
        self.sensors = {"contact_forces": _ContactSensor(robot.body_names, contact_force)}

    def __getitem__(self, name: str):
        if name == "robot":
            return self.robot
        return self.sensors[name]


class _Manager:
    def __init__(self, terms: dict[str, object]) -> None:
        self.terms = terms

    def get_term(self, name: str):
        return self.terms[name]


def _action(robot: _Robot, num_envs: int) -> MotionJointPositionAction:
    action = object.__new__(MotionJointPositionAction)
    action._asset = robot
    action._joint_ids = torch.arange(29)
    action._joint_ids_tensor = torch.arange(29)
    action._joint_names = tuple(robot.joint_names)
    action._raw_actions = torch.zeros(num_envs, 29)
    action._processed_actions = torch.zeros_like(action._raw_actions)
    action._joint_position = torch.empty_like(action._raw_actions)
    action._joint_velocity = torch.empty_like(action._raw_actions)
    action.default_joint_offset = torch.zeros_like(action._raw_actions)
    action.joint_position_target = torch.empty_like(action._raw_actions)
    action.joint_default_position = robot.data.default_joint_pos.torch[0].clone()
    action.joint_stiffness = robot.data.joint_stiffness.torch[0].clone()
    action.joint_damping = robot.data.joint_damping.torch[0].clone()
    action.joint_effort_limit = robot.data.joint_effort_limits.torch[0].clone()
    action.cfg = SimpleNamespace(
        normalize_to=5.0,
        action_clip=5.0,
        action_scale=0.25,
        default_joint_offset_range=(0.0, 0.0),
    )
    action.joint_target_gain = action.cfg.action_scale * action.joint_effort_limit / action.joint_stiffness
    action._applied_torque = torch.zeros_like(action._raw_actions)
    return action


def _estimated_torque(
    action: MotionJointPositionAction,
    target: torch.Tensor,
    position: torch.Tensor,
    velocity: torch.Tensor,
) -> torch.Tensor:
    torque = (target - position) * action.joint_stiffness - action.joint_damping * velocity
    return torch.clamp(torque, -action.joint_effort_limit, action.joint_effort_limit)


def _g1_table() -> MotionTaskTable:
    clip = MotionClipIndex.Clip(
        clip_id="clip",
        source_path="clip.tensor",
        frame_count=2,
        source_fps=30.0,
        split="train",
        tags=(),
        content_sha256=_hash("clip"),
    )
    index = MotionClipIndex(
        source_content_sha256=_hash("source"),
        skeleton_sha256=_hash("g1-source-skeleton"),
        semantic_level="robot_state",
        license="test-only",
        clips=(clip,),
    )
    body_rotation = torch.zeros(2, 31, 4)
    body_rotation[..., 3] = 1.0
    frames = MotionTaskTable.Frames(
        joint_position=torch.zeros(2, 29),
        joint_velocity=torch.zeros(2, 29),
        body_position=torch.zeros(2, 31, 3),
        body_rotation=body_rotation,
        body_linear_velocity=torch.zeros(2, 31, 3),
        body_angular_velocity=torch.zeros(2, 31, 3),
    )
    return MotionTaskTable.from_storage(
        index,
        frames,
        tuple(g1_lafan_source_skeleton().joint_names),
        (*g1_lafan_source_skeleton().body_names, G1_HEAD_FRAME_NAME),
        "g1_test_builder_v1",
        _hash("g1-test-builder-construction"),
        "clip_time_ranges",
        (("reference", 1.0),),
        MotionSampleGrid.uniform_before_source_end(step_seconds=0.02),
        seed=7,
    )


def _g1_fixture(
    num_envs: int,
    history_length: int = 2,
    *,
    raw_names: tuple[str, ...] = _G1_RAW_NAMES,
    auxiliary_names: tuple[str, ...] = _G1_AUXILIARY_NAMES,
) -> SimpleNamespace:
    table = _g1_table()
    robot = _Robot(num_envs)
    contact_force = torch.zeros(num_envs, len(robot.body_names), 3)
    scene = _Scene(robot, contact_force)
    action = _action(robot, num_envs)
    env = SimpleNamespace(
        device=torch.device("cpu"),
        num_envs=num_envs,
        step_dt=0.02,
        scene=scene,
        action_manager=_Manager({"joint_position": action}),
    )
    penalty_curriculum = MotionPenaltyScaleCurriculum(CurriculumTermCfg(func=MotionPenaltyScaleCurriculum), env)
    env.curriculum_manager = _Manager({"penalty_scale": penalty_curriculum})
    evidence_specs = tuple(
        SimpleNamespace(name=name, width=1, unit="", anchor="transition_reached_physics") for name in raw_names
    )
    payload_cfg = SimpleNamespace(
        robot_asset_name="robot",
        reset_transform_factory=None,
        step_fields=(),
        command_fields=(),
        episode_length_steps=64,
        history_fields=_G1_HISTORY_FIELDS,
        history_length=history_length,
        raw_evidence=evidence_specs,
        auxiliary_evidence=auxiliary_names,
        root_velocity_frame="center_of_mass",
    )
    payload = MotionStatePayload(
        SimpleNamespace(payload=payload_cfg, states_relative=False),
        env,
        table,
    )
    payload.bind(torch.arange(num_envs), torch.zeros(num_envs, dtype=torch.int64))
    runtime = G1MotionRuntime(env, payload)
    env.motion_runtime = runtime
    env.command_manager = _Manager({"motion": SimpleNamespace(payload=payload)})
    payload.attach_transition_state(runtime)
    return SimpleNamespace(
        env=env,
        robot=robot,
        action=action,
        payload=payload,
        runtime=runtime,
        contact_force=contact_force,
    )


def _set_reached_state(fixture: SimpleNamespace) -> None:
    robot = fixture.robot
    action = fixture.action
    num_envs = fixture.env.num_envs
    joint_position = robot.data.joint_pos.torch
    joint_velocity = robot.data.joint_vel.torch
    joint_position.zero_()
    joint_velocity.zero_()
    joint_position[:, 0] = 1.2
    joint_position[:, 5] = 0.2
    joint_position[:, 11] = -0.3
    joint_velocity[:, 0] = 3.0
    action.process_actions(torch.full((num_envs, 29), 0.2))

    fixture.contact_force.zero_()
    pelvis = robot.body_names.index("pelvis")
    left_foot = robot.body_names.index("left_ankle_roll_link")
    right_foot = robot.body_names.index("right_ankle_roll_link")
    fixture.contact_force[:, pelvis, 2] = 2.0
    fixture.contact_force[:, left_foot, 2] = 2.0
    fixture.contact_force[:, right_foot, 2] = 2.0
    action.apply_actions()
    robot.data.body_com_lin_vel_w.torch.zero_()
    robot.data.body_com_lin_vel_w.torch[:, left_foot, 0] = 3.0
    robot.data.body_com_lin_vel_w.torch[:, left_foot, 1] = 4.0


def _history_sources(state: torch.Tensor, last_action: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "processed_action": last_action,
        "base_angular_velocity": state[:, 61:64],
        "joint_position": state[:, :29],
        "joint_velocity": state[:, 29:58],
        "projected_gravity": state[:, 58:61],
    }


def test_payload_history_fields_require_a_positive_history_length() -> None:
    """A configured field layout cannot silently disable history storage."""
    with pytest.raises(
        ValueError,
        match="history_fields and a positive history_length must be configured together",
    ):
        _g1_fixture(1, history_length=0)


def test_g1_runtime_uses_payload_order_and_validates_only_channel_membership() -> None:
    """Runtime packing order comes from the payload while G1 still requires its complete named sets."""
    raw_order = tuple(reversed(_G1_RAW_NAMES))
    auxiliary_order = tuple(reversed(_G1_AUXILIARY_NAMES))
    fixture = _g1_fixture(
        1,
        raw_names=raw_order,
        auxiliary_names=auxiliary_order,
    )
    assert tuple(fixture.runtime.raw_evidence) == raw_order
    assert fixture.runtime.raw_evidence is fixture.payload.raw_evidence
    for name in raw_order:
        assert fixture.runtime.raw_evidence[name] is fixture.payload.raw_evidence[name]

    assert fixture.runtime.auxiliary_evidence_names == auxiliary_order
    assert fixture.payload.raw_evidence_names == raw_order

    with pytest.raises(ValueError, match="raw evidence must contain exactly"):
        _g1_fixture(1, raw_names=_G1_RAW_NAMES[:-1])
    with pytest.raises(ValueError, match="auxiliary evidence must contain exactly"):
        _g1_fixture(1, auxiliary_names=_G1_AUXILIARY_NAMES[:-1])


def test_g1_action_preserves_normalized_action_target_and_clipped_torque() -> None:
    """Behavior, processed, target, and PD-torque tensors must remain distinct."""
    robot = _Robot(2)
    action = _action(robot, 2)
    behavior = torch.linspace(-2.0, 2.0, 29).repeat(2, 1)
    offset = torch.linspace(-0.05, 0.05, 29).repeat(2, 1)
    action.default_joint_offset.copy_(offset)
    robot.data.joint_pos.torch.copy_(torch.linspace(-0.2, 0.2, 29).repeat(2, 1))
    robot.data.joint_vel.torch.copy_(torch.linspace(-1.0, 1.0, 29).repeat(2, 1))

    action.process_actions(behavior)
    expected_processed = torch.clamp(
        behavior * action.cfg.normalize_to,
        -action.cfg.action_clip,
        action.cfg.action_clip,
    )
    expected_target = action.joint_default_position + offset + expected_processed * action.joint_target_gain
    expected_torque = _estimated_torque(
        action,
        expected_target,
        robot.data.joint_pos.torch,
        robot.data.joint_vel.torch,
    )
    torch.testing.assert_close(action.raw_actions, behavior)
    torch.testing.assert_close(action.processed_actions, expected_processed)
    torch.testing.assert_close(action.joint_position_target, expected_target)

    action.apply_actions()
    torch.testing.assert_close(robot.last_joint_target, expected_target)
    torch.testing.assert_close(action.applied_torque, expected_torque)
    applied_torque = action.applied_torque.clone()
    torch.testing.assert_close(robot.last_joint_ids, torch.arange(29))
    robot.data.joint_pos.torch.add_(0.4)
    reached_torque = _estimated_torque(
        action,
        action.joint_position_target,
        robot.data.joint_pos.torch,
        robot.data.joint_vel.torch,
    )
    assert not torch.allclose(reached_torque, applied_torque)
    torch.testing.assert_close(action.applied_torque, applied_torque)

    action.reset(torch.tensor((0,)))
    assert not action.raw_actions[0].any()
    assert not action.processed_actions[0].any()
    torch.testing.assert_close(action.joint_position_target[0], action.joint_default_position)
    assert not action.applied_torque[0].any()


def test_g1_runtime_uses_last_applied_torque_instead_of_reached_state_recomputation() -> None:
    """Torque evidence must retain the estimate applied before the final physics substep."""
    fixture = _g1_fixture(1)
    _set_reached_state(fixture)
    applied_torque = fixture.action.applied_torque.clone()

    fixture.robot.data.joint_pos.torch.add_(0.4)
    fixture.robot.data.joint_vel.torch.sub_(0.7)
    reached_torque = _estimated_torque(
        fixture.action,
        fixture.action.joint_position_target,
        fixture.robot.data.joint_pos.torch,
        fixture.robot.data.joint_vel.torch,
    )
    assert not torch.allclose(reached_torque, applied_torque)

    state = torch.zeros(1, 64)
    fixture.runtime.capture_current({"state": state, "last_action": fixture.action.processed_actions})
    fixture.runtime.measure()
    torch.testing.assert_close(
        fixture.runtime.raw_evidence["penalty_torques"][:, 0], applied_torque.square().sum(dim=-1)
    )


def test_g1_runtime_records_field_major_history_and_all_raw_evidence() -> None:
    """Pre-final measurement preserves density evidence and excludes the reset seed from history."""
    fixture = _g1_fixture(2)
    _set_reached_state(fixture)
    state0 = torch.arange(64, dtype=torch.float32).repeat(2, 1)
    last_action0 = torch.full((2, 29), 0.25)
    fixture.runtime.capture_current({"state": state0, "last_action": last_action0})
    fixture.runtime.measure()

    assert tuple(fixture.runtime.raw_evidence) == _G1_RAW_NAMES
    assert tuple(fixture.payload.raw_evidence) == _G1_RAW_NAMES
    assert fixture.payload.raw_evidence_names == _G1_RAW_NAMES
    assert fixture.payload.auxiliary_evidence_names == _G1_AUXILIARY_NAMES
    assert fixture.runtime.auxiliary_evidence_names == _G1_AUXILIARY_NAMES
    assert fixture.runtime.auxiliary_evidence.shape == (2, len(_G1_AUXILIARY_NAMES))
    assert fixture.runtime.raw_evidence is fixture.payload.raw_evidence
    assert fixture.payload.raw_evidence_value.is_contiguous()
    assert fixture.payload.raw_evidence_value.shape == (2, len(_G1_RAW_NAMES))
    for name in _G1_RAW_NAMES:
        assert fixture.runtime.raw_evidence[name].shape == (2, 1)
        assert fixture.payload.raw_evidence[name] is fixture.runtime.raw_evidence[name]
        assert (
            fixture.payload.raw_evidence[name].untyped_storage().data_ptr()
            == fixture.payload.raw_evidence_value.untyped_storage().data_ptr()
        )

    torque = fixture.action.applied_torque
    expected = {
        "penalty_torques": torque.square().sum(dim=-1),
        "penalty_action_rate": (last_action0 - fixture.action.processed_actions).square().sum(dim=-1),
        "limits_dof_pos": torch.full((2,), 0.25),
        "limits_dof_vel": torch.ones(2),
        "limits_torque": torch.clamp_min(
            torque.abs() - 0.95 * fixture.action.joint_effort_limit,
            0.0,
        ).sum(dim=-1),
        "penalty_undesired_contact": torch.ones(2),
        "penalty_feet_ori": torch.zeros(2),
        "penalty_ankle_roll": torch.full((2,), 0.13),
        "penalty_slippage": torch.full((2,), 5.0),
        "feet_heading_alignment": torch.zeros(2),
    }
    for name in _G1_RAW_NAMES:
        torch.testing.assert_close(fixture.runtime.raw_evidence[name][:, 0], expected[name])
    for index, name in enumerate(_G1_AUXILIARY_NAMES):
        torch.testing.assert_close(fixture.runtime.auxiliary_evidence[:, index], expected[name])

    expected_reward = torch.zeros(2)
    coefficients = {
        "penalty_torques": -1.0e-6,
        "penalty_action_rate": -0.5,
        "limits_dof_pos": -10.0,
        "limits_dof_vel": -5.0,
        "limits_torque": -5.0,
        "penalty_undesired_contact": -1.0,
        "penalty_feet_ori": -0.1,
        "penalty_ankle_roll": -0.5,
        "penalty_slippage": -1.0,
        "feet_heading_alignment": -0.1,
    }
    for name, coefficient in coefficients.items():
        scale = fixture.runtime.penalty_curriculum.scale if name in _G1_AUXILIARY_NAMES else 1.0
        expected_reward.add_(expected[name], alpha=coefficient * scale)
    torch.testing.assert_close(fixture.runtime.environment_reward, expected_reward)

    # The reset observation was captured before the first edge, but the prior
    # eligibility mask is false. The first reached observation sees no seed.
    assert not fixture.payload.history_value.any()
    assert motion_history(fixture.env, "motion").data_ptr() == fixture.payload.history_value.data_ptr()

    first_reached_state = state0 + 1000.0
    first_reached_action = last_action0 + 2.0
    fixture.runtime.capture_current({"state": first_reached_state, "last_action": first_reached_action})
    fixture.runtime.measure()
    first_reached = _history_sources(first_reached_state, first_reached_action)
    expected_second_reached_history = torch.cat(
        tuple(
            torch.cat((first_reached[name], torch.zeros_like(first_reached[name])), dim=-1)
            for name, _ in _G1_HISTORY_FIELDS
        ),
        dim=-1,
    )
    torch.testing.assert_close(fixture.payload.history_value, expected_second_reached_history)

    preterminal_state = first_reached_state + 1000.0
    preterminal_action = first_reached_action + 2.0
    fixture.runtime.capture_current({"state": preterminal_state, "last_action": preterminal_action})
    fixture.runtime.measure()
    preterminal = _history_sources(preterminal_state, preterminal_action)
    expected_terminal_history = torch.cat(
        tuple(torch.cat((preterminal[name], first_reached[name]), dim=-1) for name, _ in _G1_HISTORY_FIELDS),
        dim=-1,
    )
    torch.testing.assert_close(fixture.payload.history_value, expected_terminal_history)


def test_reward_manager_integrates_motion_reward_density_exactly_once() -> None:
    """Motion runtime returns density while the standard manager applies one control-step dt."""
    density = torch.tensor((2.5, -4.0))
    measure_calls = 0

    def measure() -> None:
        nonlocal measure_calls
        measure_calls += 1

    runtime = SimpleNamespace(environment_reward=density, measure=measure)
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        sim=SimpleNamespace(is_playing=lambda: True),
        _motion_runtime=runtime,
    )
    manager = RewardManager(
        {"environment": RewardTermCfg(func=motion_transition_reward, weight=1.0)},
        env,
    )

    step_dt = 0.02
    integrated = manager.compute(step_dt)
    torch.testing.assert_close(runtime.environment_reward, density)
    torch.testing.assert_close(integrated, density * step_dt)
    assert measure_calls == 1


def test_g1_payload_reset_preserves_single_owner_terminal_evidence() -> None:
    """Same-step reset clears history while all completed-edge outputs survive."""
    fixture = _g1_fixture(4)
    _set_reached_state(fixture)
    state = torch.ones(4, 64)
    last_action = torch.full((4, 29), 0.25)
    fixture.runtime.capture_current({"state": state, "last_action": last_action})
    fixture.runtime.measure()
    fixture.runtime.capture_current({"state": state + 1.0, "last_action": last_action + 1.0})
    fixture.runtime.measure()
    final_history = motion_history(fixture.env, "motion").clone()
    assert final_history.any()
    history_before = final_history
    evidence_before = {name: value.clone() for name, value in fixture.payload.raw_evidence.items()}
    auxiliary_before = fixture.runtime.auxiliary_evidence.clone()
    reward_before = fixture.runtime.environment_reward.clone()

    reset_ids = torch.tensor((0, 2))
    fixture.payload.bind(reset_ids, torch.zeros(2, dtype=torch.int64))

    assert not fixture.payload.history_value[reset_ids].any()
    assert final_history[reset_ids].any()
    assert fixture.runtime.raw_evidence is fixture.payload.raw_evidence
    for name, value in fixture.payload.raw_evidence.items():
        torch.testing.assert_close(value, evidence_before[name])
    torch.testing.assert_close(fixture.runtime.auxiliary_evidence, auxiliary_before)
    torch.testing.assert_close(fixture.runtime.environment_reward, reward_before)
    kept = torch.tensor((1, 3))
    torch.testing.assert_close(fixture.payload.history_value[kept], history_before[kept])
    torch.testing.assert_close(fixture.runtime.auxiliary_evidence[kept], auxiliary_before[kept])
    torch.testing.assert_close(fixture.runtime.environment_reward[kept], reward_before[kept])
    for name, value in fixture.payload.raw_evidence.items():
        torch.testing.assert_close(value[kept], evidence_before[name][kept])


def test_motion_observations_match_frozen_smpl_and_g1_body_math() -> None:
    """Runtime body routes must remain exact wrappers around the frozen frame-builder equations."""
    num_envs = 3
    smpl_position = torch.arange(num_envs * 24 * 3, dtype=torch.float32).view(num_envs, 24, 3) * 0.01
    smpl_rotation = torch.zeros(num_envs, 24, 4)
    smpl_rotation[..., 3] = 1.0
    smpl_linear = smpl_position + 0.5
    smpl_angular = smpl_position - 0.25
    smpl_robot = SimpleNamespace(
        data=SimpleNamespace(
            body_pos_w=_view(torch.full_like(smpl_position, -300.0)),
            body_link_pos_w=_view(smpl_position),
            body_quat_w=_view(torch.full_like(smpl_rotation, -400.0)),
            body_link_quat_w=_view(smpl_rotation),
            body_lin_vel_w=_view(torch.full_like(smpl_linear, -100.0)),
            body_link_lin_vel_w=_view(smpl_linear),
            body_ang_vel_w=_view(torch.full_like(smpl_angular, -200.0)),
            body_link_ang_vel_w=_view(smpl_angular),
        )
    )
    smpl_env = SimpleNamespace(scene={"robot": smpl_robot})
    asset_cfg = SimpleNamespace(name="robot", body_ids=slice(None))
    torch.testing.assert_close(
        smpl_body_observation(smpl_env, asset_cfg),
        smpl_humenv_observation(smpl_position, smpl_rotation, smpl_linear, smpl_angular),
    )

    fixture = _g1_fixture(num_envs)
    robot = fixture.robot
    body_position = robot.data.body_link_pos_w.torch
    body_position.copy_(torch.arange(body_position.numel()).view_as(body_position) * 0.001)
    body_linear = robot.data.body_com_lin_vel_w.torch
    body_linear.copy_(body_position + 0.2)
    body_angular = robot.data.body_com_ang_vel_w.torch
    body_angular.copy_(body_position - 0.1)
    body_rotation = robot.data.body_link_quat_w.torch
    parent = 15
    offset = body_position.new_tensor((0.0, 0.0, 0.35)).expand(num_envs, 3)
    synthetic_position = body_position[:, parent] + quat_apply(body_rotation[:, parent], offset)
    synthetic_linear = body_linear[:, parent] + torch.cross(body_angular[:, parent], offset, dim=-1)
    expected_g1 = g1_privileged_observation(
        torch.cat((body_position, synthetic_position[:, None]), dim=1),
        torch.cat((body_rotation, body_rotation[:, parent : parent + 1]), dim=1),
        torch.cat((body_linear, synthetic_linear[:, None]), dim=1),
        torch.cat((body_angular, body_angular[:, parent : parent + 1]), dim=1),
    )
    torch.testing.assert_close(
        g1_privileged_body_observation(
            fixture.env,
            SimpleNamespace(name="robot", body_ids=list(range(30))),
            parent,
        ),
        expected_g1,
    )
    assert expected_g1.shape == (num_envs, 463)


def test_named_g1_observation_components_use_controller_and_robot_state() -> None:
    """Actor component routes must preserve units, frames, and normalized last action."""
    fixture = _g1_fixture(2)
    _set_reached_state(fixture)
    fixture.action.default_joint_offset.fill_(0.05)
    fixture.robot.data.root_ang_vel_b.torch.fill_(0.3)
    asset_cfg = SimpleNamespace(name="robot", joint_ids=slice(None))

    torch.testing.assert_close(
        motion_joint_position(fixture.env, "joint_position"),
        fixture.robot.data.joint_pos.torch
        - fixture.action.joint_default_position
        - fixture.action.default_joint_offset,
    )
    torch.testing.assert_close(motion_joint_velocity(fixture.env, "joint_position"), fixture.robot.data.joint_vel.torch)
    projected_gravity = motion_projected_gravity(fixture.env, asset_cfg)
    assert projected_gravity.data_ptr() == fixture.robot.data.projected_gravity_b.torch.data_ptr()
    torch.testing.assert_close(
        projected_gravity,
        torch.tensor((0.0, 0.0, -1.0)).repeat(2, 1),
    )
    assert (
        motion_root_angular_velocity(fixture.env, asset_cfg).data_ptr()
        == fixture.robot.data.root_ang_vel_b.torch.data_ptr()
    )
    assert motion_last_action(fixture.env, "joint_position").data_ptr() == fixture.action.processed_actions.data_ptr()


class _SmplPayload:
    def __init__(self) -> None:
        self.raw_evidence: dict[str, torch.Tensor] = {}
        self.record_calls = 0

    def record_step(self) -> None:
        self.record_calls += 1


@pytest.mark.parametrize("num_envs", (1, 16, 1024))
def test_smpl_runtime_has_zero_evidence_and_clone_invariant_shapes(num_envs: int) -> None:
    """The SMPL profile must keep the same empty evidence contract at every vector scale."""
    payload = _SmplPayload()
    env = SimpleNamespace(num_envs=num_envs, device=torch.device("cpu"), step_dt=1.0 / 30.0)
    runtime = SmplMotionRuntime(env, payload)
    runtime.environment_reward.fill_(7.0)
    runtime.measure()

    assert runtime.raw_evidence is payload.raw_evidence
    assert runtime.auxiliary_evidence.shape == (num_envs, 0)
    assert runtime.environment_reward.shape == (num_envs,)
    assert not runtime.environment_reward.any()
    assert payload.record_calls == 1


@pytest.mark.parametrize("num_envs", (1, 16, 1024))
def test_g1_runtime_history_evidence_and_observation_shapes_are_clone_invariant(num_envs: int) -> None:
    """G1 runtime tensors must obey one clone law at single, vector, and native scale."""
    fixture = _g1_fixture(num_envs)
    _set_reached_state(fixture)
    state = torch.arange(64, dtype=torch.float32).repeat(num_envs, 1)
    last_action = torch.full((num_envs, 29), 0.25)
    fixture.runtime.capture_current({"state": state, "last_action": last_action})
    fixture.runtime.measure()

    assert fixture.runtime.environment_reward.shape == (num_envs,)
    assert fixture.runtime.auxiliary_evidence.shape == (num_envs, 8)
    assert fixture.payload.history_value.shape == (num_envs, 2 * 93)
    assert g1_privileged_body_observation(
        fixture.env,
        SimpleNamespace(name="robot", body_ids=list(range(30))),
        G1_BEHAVIOR_BODY_NAMES.index(G1_HEAD_PARENT_BODY_NAME),
    ).shape == (num_envs, 463)
    for value in (
        fixture.runtime.environment_reward,
        fixture.runtime.auxiliary_evidence,
        fixture.payload.history_value,
        *fixture.runtime.raw_evidence.values(),
    ):
        torch.testing.assert_close(value, value[:1].expand_as(value))


class _NativeControlRegistry:
    def clear_debug_vis_callback(self, term) -> None:
        del term


def _native_control_env(destination) -> SimpleNamespace:
    class NativePhysicsManager:
        @classmethod
        def get_control(cls):
            return SimpleNamespace(mujoco=SimpleNamespace(ctrl=destination))

    return SimpleNamespace(
        num_envs=2,
        device="cpu",
        scene={"robot": object()},
        sim=SimpleNamespace(
            physics_manager=NativePhysicsManager,
            vis_marker_registry=_NativeControlRegistry(),
        ),
    )


def test_mujoco_control_action_reuses_one_torch_warp_view() -> None:
    """Native actions copy through one persistent same-device Warp view."""
    destination = wp.zeros(6, dtype=wp.float32, device="cpu")
    term = MotionMujocoControlAction(
        MotionMujocoControlActionCfg(asset_name="robot", action_width=3),
        _native_control_env(destination),
    )
    source_identity = id(term._control_source)

    first = torch.arange(6, dtype=torch.float32).view(2, 3)
    term.process_actions(first)
    term.apply_actions()
    torch.testing.assert_close(torch.from_numpy(destination.numpy()), first.view(-1))

    second = -first
    term.process_actions(second)
    term.apply_actions()
    assert id(term._control_source) == source_identity
    assert term.raw_actions.data_ptr() == term.processed_actions.data_ptr()
    torch.testing.assert_close(torch.from_numpy(destination.numpy()), second.view(-1))


def test_mujoco_control_action_rejects_native_control_shape_mismatch() -> None:
    """A missing actuator row is a construction error rather than a fallback."""
    destination = wp.zeros(5, dtype=wp.float32, device="cpu")
    cfg = MotionMujocoControlActionCfg(asset_name="robot", action_width=3)
    with pytest.raises(ValueError, match="control input shape"):
        MotionMujocoControlAction(cfg, _native_control_env(destination))
