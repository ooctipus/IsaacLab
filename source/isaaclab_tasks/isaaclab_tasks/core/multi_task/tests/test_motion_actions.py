# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exact numerical tests for motion action laws."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isaaclab.managers import ActionTerm

from isaaclab_tasks.core.multi_task.motion.config.robots import G1_MOTION_ARTICULATION_CFG
from isaaclab_tasks.core.multi_task.motion.config.robots.g1 import (
    _SIMULATOR_JOINT_NAMES as _G1_SIMULATOR_JOINT_NAMES,
)
from isaaclab_tasks.core.multi_task.motion.config.robots.g1 import (
    G1_BEHAVIOR_JOINT_NAMES as _G1_BEHAVIOR_JOINT_NAMES,
)
from isaaclab_tasks.core.multi_task.motion.mdp.actions import MotionJointPositionAction
from isaaclab_tasks.core.multi_task.motion.mdp.actions_cfg import (
    MotionJointPositionActionCfg,
    MotionMujocoControlActionCfg,
)


def _phase3_fixtures() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "scripts/reinforcement_learning/forward_backward/phase3/fixtures"
        if candidate.is_dir():
            return candidate
    raise RuntimeError("Phase 3 fixtures were not found from the repository test path.")


class _Asset:
    def __init__(self, joint_position: torch.Tensor, joint_velocity: torch.Tensor) -> None:
        self.data = SimpleNamespace(
            joint_pos=SimpleNamespace(torch=joint_position),
            joint_vel=SimpleNamespace(torch=joint_velocity),
        )
        self.target: torch.Tensor | None = None

    def set_joint_position_target_index(self, *, target: torch.Tensor, joint_ids: torch.Tensor) -> None:
        del joint_ids
        self.target = target.clone()


class _ConfiguredAsset(_Asset):
    def __init__(self, num_envs: int, joint_names: tuple[str, ...]) -> None:
        joint_count = len(joint_names)
        super().__init__(torch.zeros(num_envs, joint_count), torch.zeros(num_envs, joint_count))
        self.num_joints = joint_count
        self.joint_names = list(joint_names)
        self.data.default_joint_pos = SimpleNamespace(torch=torch.zeros(1, joint_count))
        self.data.joint_stiffness = SimpleNamespace(torch=torch.full((1, joint_count), 40.0))
        self.data.joint_damping = SimpleNamespace(torch=torch.full((1, joint_count), 2.0))
        self.data.joint_effort_limits = SimpleNamespace(torch=torch.full((1, joint_count), 20.0))
        self.target_joint_ids: slice | list[int] | None = None
        self.preserve_order: bool | None = None

    def find_joints(self, names: list[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        self.preserve_order = preserve_order
        if names == [".*"]:
            joint_ids = list(range(self.num_joints))
        else:
            joint_ids = [self.joint_names.index(name) for name in names]
        return joint_ids, [self.joint_names[index] for index in joint_ids]

    def set_joint_position_target_index(self, *, target: torch.Tensor, joint_ids: slice | list[int]) -> None:
        self.target_joint_ids = joint_ids
        super().set_joint_position_target_index(target=target, joint_ids=torch.arange(self.num_joints))


def test_motion_action_cfg_rejects_ignored_base_clip() -> None:
    """Custom motion actions must reject the base clip knob they do not implement."""
    with pytest.raises(ValueError, match="does not use ActionTermCfg.clip"):
        MotionJointPositionActionCfg(asset_name="robot", joint_names=[".*"], clip={".*": (-1.0, 1.0)})
    with pytest.raises(ValueError, match="does not use ActionTermCfg.clip"):
        MotionMujocoControlActionCfg(asset_name="robot", action_width=69, clip={".*": (-1.0, 1.0)})


@pytest.mark.parametrize("preserve_order", (False, True))
def test_motion_joint_action_owns_minimal_resolution_and_slices_natural_all_joint_order(
    preserve_order: bool,
) -> None:
    """Natural all-joint control must avoid indexed gathers and expose truthful action metadata."""
    assert MotionJointPositionAction.__bases__ == (ActionTerm,)
    asset = _ConfiguredAsset(3, ("joint_a", "joint_b", "joint_c"))
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        scene={"robot": asset},
    )
    cfg = MotionJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        preserve_order=preserve_order,
        normalize_to=5.0,
        action_clip=5.0,
        action_scale=0.25,
    )

    action = MotionJointPositionAction(cfg, env)
    action.process_actions(torch.ones(3, 3))
    action.apply_actions()

    assert action._joint_ids == slice(None)
    torch.testing.assert_close(action.joint_ids, torch.arange(3))
    assert asset.target_joint_ids == slice(None)
    assert asset.preserve_order is preserve_order
    assert action.action_dim == 3
    assert action.IO_descriptor.shape == (3,)
    assert action.IO_descriptor.dtype == "torch.float32"
    assert action.IO_descriptor.action_type == "JointPosition"
    assert action.IO_descriptor.extras["joint_names"] == asset.joint_names
    assert action.IO_descriptor.extras["normalize_to"] == 5.0
    assert action.IO_descriptor.extras["processed_action_clip"] == (-5.0, 5.0)


def test_g1_action_maps_declared_behavior_axis_to_live_articulation_once() -> None:
    """Behavior actions and observations must share one exact named joint axis."""
    asset = _ConfiguredAsset(2, _G1_SIMULATOR_JOINT_NAMES)
    asset.data.joint_pos.torch.copy_(torch.arange(29, dtype=torch.float32).repeat(2, 1))
    asset.data.joint_vel.torch.copy_(torch.arange(29, dtype=torch.float32).repeat(2, 1).neg_())
    env = SimpleNamespace(num_envs=2, device="cpu", scene={"robot": asset})
    cfg = MotionJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(_G1_BEHAVIOR_JOINT_NAMES),
        preserve_order=True,
    )

    action = MotionJointPositionAction(cfg, env)
    expected_ids = torch.tensor(
        [_G1_SIMULATOR_JOINT_NAMES.index(name) for name in _G1_BEHAVIOR_JOINT_NAMES],
        dtype=torch.int64,
    )

    assert action.joint_names == _G1_BEHAVIOR_JOINT_NAMES
    torch.testing.assert_close(action.joint_ids, expected_ids)
    assert action._joint_ids == expected_ids.tolist()
    assert asset.preserve_order
    position = action.joint_position
    velocity = action.joint_velocity
    torch.testing.assert_close(position, asset.data.joint_pos.torch.index_select(1, expected_ids))
    torch.testing.assert_close(velocity, asset.data.joint_vel.torch.index_select(1, expected_ids))
    assert action.joint_position.data_ptr() == position.data_ptr()
    assert action.joint_velocity.data_ptr() == velocity.data_ptr()


def test_g1_action_reset_writes_selected_nonzero_default_offsets() -> None:
    """A subset reset must retain its sampled default-pose offsets in persistent action state."""
    asset = _ConfiguredAsset(4, ("joint_a", "joint_b"))
    env = SimpleNamespace(num_envs=4, device="cpu", scene={"robot": asset})
    cfg = MotionJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        default_joint_offset_range=(0.25, 0.25),
    )
    action = MotionJointPositionAction(cfg, env)
    action._raw_actions.fill_(1.0)
    action._processed_actions.fill_(1.0)
    action._applied_torque.fill_(1.0)
    selected = torch.tensor((1, 3), dtype=torch.int64)

    action.reset(selected)

    expected_offsets = torch.zeros(4, 2)
    expected_offsets[selected] = 0.25
    torch.testing.assert_close(action.default_joint_offset, expected_offsets)
    torch.testing.assert_close(action.joint_position_target[selected], torch.full((2, 2), 0.25))
    torch.testing.assert_close(action.raw_actions[selected], torch.zeros(2, 2))
    torch.testing.assert_close(action.processed_actions[selected], torch.zeros(2, 2))
    torch.testing.assert_close(action.applied_torque[selected], torch.zeros(2, 2))


def test_g1_action_full_slice_reset_uses_full_buffers_without_scratch() -> None:
    """The manager's full slice must take the allocation-free whole-buffer reset path."""
    asset = _ConfiguredAsset(4, ("joint_a", "joint_b"))
    env = SimpleNamespace(num_envs=4, device="cpu", scene={"robot": asset})
    cfg = MotionJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        default_joint_offset_range=(0.25, 0.25),
    )
    action = MotionJointPositionAction(cfg, env)
    action._raw_actions.fill_(1.0)
    action._processed_actions.fill_(1.0)
    action._applied_torque.fill_(1.0)
    action._reset_default_joint_offset.fill_(-3.0)
    action._reset_joint_position_target.fill_(-5.0)

    action.reset(slice(None))

    torch.testing.assert_close(action.raw_actions, torch.zeros(4, 2))
    torch.testing.assert_close(action.processed_actions, torch.zeros(4, 2))
    torch.testing.assert_close(action.applied_torque, torch.zeros(4, 2))
    torch.testing.assert_close(action.default_joint_offset, torch.full((4, 2), 0.25))
    torch.testing.assert_close(action.joint_position_target, torch.full((4, 2), 0.25))
    torch.testing.assert_close(action._reset_default_joint_offset, torch.full((4, 2), -3.0))
    torch.testing.assert_close(action._reset_joint_position_target, torch.full((4, 2), -5.0))


def test_g1_action_matches_native_processed_target_and_torque() -> None:
    """The articulation-derived action law must match the frozen BFM trace elementwise."""
    path = _phase3_fixtures() / "g1_lafan_same_step_trace_v1.npz"
    with np.load(path, allow_pickle=False) as tensors:
        behavior = torch.from_numpy(tensors["behavior_action"]).flatten(0, 1)
        offset = torch.from_numpy(tensors["current_default_joint_offset"]).flatten(0, 1)
        joint_position = torch.from_numpy(tensors["current_qpos"])[..., 7:].flatten(0, 1)
        joint_velocity = torch.from_numpy(tensors["current_qvel"])[..., 6:].flatten(0, 1)

        asset_cfg = G1_MOTION_ARTICULATION_CFG
        assert tuple(asset_cfg.actuators["motion"].joint_names_expr) == _G1_SIMULATOR_JOINT_NAMES
        joint_default = torch.tensor(
            [asset_cfg.init_state.joint_pos[name] for name in _G1_BEHAVIOR_JOINT_NAMES],
            dtype=torch.float32,
        )
        action = object.__new__(MotionJointPositionAction)
        action.cfg = SimpleNamespace(normalize_to=5.0, action_clip=5.0, action_scale=0.25)
        action._asset = _Asset(joint_position, joint_velocity)
        action._joint_ids = torch.arange(29)
        action._joint_ids_tensor = torch.arange(29)
        action._raw_actions = torch.empty_like(behavior)
        action._processed_actions = torch.empty_like(behavior)
        action._joint_position = torch.empty_like(behavior)
        action._joint_velocity = torch.empty_like(behavior)
        action.default_joint_offset = offset
        action.joint_position_target = torch.empty_like(behavior)
        action._applied_torque = torch.empty_like(behavior)
        action.joint_default_position = joint_default
        action.joint_stiffness = torch.from_numpy(tensors["current_joint_stiffness"][0, 0])
        action.joint_damping = torch.from_numpy(tensors["current_joint_damping"][0, 0])
        action.joint_effort_limit = torch.from_numpy(tensors["current_joint_effort_limit"][0, 0])
        action.joint_target_gain = action.cfg.action_scale * action.joint_effort_limit / action.joint_stiffness

        action.process_actions(behavior)
        action.apply_actions()

        torch.testing.assert_close(
            action.processed_actions,
            torch.from_numpy(tensors["processed_action"]).flatten(0, 1),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            action.joint_position_target,
            torch.from_numpy(tensors["controller_target_joint_position"]).flatten(0, 1),
            rtol=0.0,
            atol=1.0e-7,
        )
        torch.testing.assert_close(
            action.applied_torque,
            torch.from_numpy(tensors["substep_applied_pd_torque"][:, :, 0]).flatten(0, 1),
            rtol=1.0e-6,
            atol=1.0e-5,
        )
        torch.testing.assert_close(action._asset.target, action.joint_position_target)
