# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused tests for the native MuJoCo action boundary."""

from types import SimpleNamespace

import pytest
import torch
import warp as wp

from isaaclab_tasks.core.multi_task.mdp import NativeMujocoControlAction, NativeMujocoControlActionCfg


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


def test_native_mujoco_control_reuses_one_torch_warp_view() -> None:
    """Native actions copy through one persistent same-device Warp view."""
    destination = wp.zeros(6, dtype=wp.float32, device="cpu")
    term = NativeMujocoControlAction(
        NativeMujocoControlActionCfg(asset_name="robot", action_width=3),
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


def test_native_mujoco_control_rejects_shape_mismatch() -> None:
    """A missing actuator row is a construction error rather than a fallback."""
    destination = wp.zeros(5, dtype=wp.float32, device="cpu")
    cfg = NativeMujocoControlActionCfg(asset_name="robot", action_width=3)
    with pytest.raises(ValueError, match="control input shape"):
        NativeMujocoControlAction(cfg, _native_control_env(destination))
