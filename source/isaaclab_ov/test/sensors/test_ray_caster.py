# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for OVPhysX RayCaster backend glue."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import torch
import warp as wp
from isaaclab_ov.sensors.ray_caster import ray_caster as ray_caster_module

from isaaclab.cloner import ClonePlan
from isaaclab.cloner.clone_plan import FrameLayout


class _FakeBinding:
    def __init__(self, prim_paths):
        self.prim_paths = prim_paths
        self.shape = (len(prim_paths), 7)

    def read(self, dst):
        pass

    def destroy(self):
        pass


class _FakePhysx:
    def __init__(self):
        self.calls = []

    def create_tensor_binding(self, *, prim_paths, tensor_type):
        self.calls.append((prim_paths, tensor_type))
        return _FakeBinding(prim_paths)


class _DummyRayCaster(ray_caster_module._OvPhysxRayCasterMixin):
    def __init__(self, physics_manager):
        self.cfg = SimpleNamespace(prim_path="/World/envs/env_[^/]+/Robot/base/ray")
        self._device = "cpu"
        self._num_envs = 3
        self._physics_manager = physics_manager


def test_initialize_pose_tracking_uses_exact_clone_plan_paths_without_destination_usd(monkeypatch):
    """RayCaster binds exact planned bodies even when destination USD prims do not exist."""
    fake_tensor_type = object()
    fake_tensor_types = SimpleNamespace(RIGID_BODY_POSE=fake_tensor_type)
    fake_physx = _FakePhysx()
    clone_mask = np.ones(3, dtype=np.bool_)
    frames = (
        FrameLayout(
            path="/World/envs/env_{}/Robot/base/ray",
            source_path="/World/envs/env_0/Robot/base/ray",
            parent_path="/World/envs/env_{}/Robot/base",
            row=0,
            env_id=None,
            body_path="/World/envs/env_{}/Robot/base",
            body_view_path="/World/envs/env_*/Robot/base",
            pose=(0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0),
            parent_body_path="/World/envs/env_{}/Robot/base",
            clone_mask=clone_mask,
        ),
    )
    plan = ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=torch.ones((1, 3), dtype=torch.bool),
        env_ids=torch.arange(3),
        is_complete=True,
        frame_prototypes=frames,
        _env_ids_cpu=(0, 1, 2),
    )
    context = SimpleNamespace(get_clone_plan=lambda: plan)

    monkeypatch.setitem(sys.modules, "isaaclab_ov.tensor_types", fake_tensor_types)
    monkeypatch.setattr(ray_caster_module.SimulationContext, "instance", staticmethod(lambda: context))

    sensor = _DummyRayCaster(SimpleNamespace(get_physx_instance=lambda: fake_physx))

    sensor._initialize_pose_tracking()

    assert fake_physx.calls == [([f"/World/envs/env_{i}/Robot/base" for i in range(3)], fake_tensor_type)]
    assert sensor.count == 3
    torch.testing.assert_close(
        wp.to_torch(sensor._offset_pos_wp),
        torch.tensor([[0.1, 0.2, 0.3]] * 3, dtype=torch.float32),
    )
    torch.testing.assert_close(
        wp.to_torch(sensor._offset_quat_wp),
        torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 3, dtype=torch.float32),
    )
