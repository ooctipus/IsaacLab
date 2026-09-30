# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for physics-manager CUDA device initialization."""

from types import SimpleNamespace

from isaaclab.physics import PhysicsManager
from isaaclab.physics import physics_manager as physics_manager_module


def test_bind_context_synchronizes_cuda_device(monkeypatch):
    """Binding selects the configured CUDA device before backend resources are registered."""
    devices = []
    monkeypatch.setattr(physics_manager_module, "set_cuda_device", devices.append)

    sim_context = SimpleNamespace(cfg=SimpleNamespace(physics=object(), device="cuda:2"))
    manager = SimpleNamespace()
    PhysicsManager._bind_context(manager, sim_context)

    assert devices == ["cuda:2"]
    assert manager._device == "cuda:2"


def test_bind_context_does_not_synchronize_cpu_device(monkeypatch):
    """CPU simulation must not initialize or select CUDA runtimes while binding."""
    devices = []
    monkeypatch.setattr(physics_manager_module, "set_cuda_device", devices.append)
    sim_context = SimpleNamespace(cfg=SimpleNamespace(physics=object(), device="cpu"))
    manager = SimpleNamespace()

    PhysicsManager._bind_context(manager, sim_context)

    assert devices == []
    assert manager._device == "cpu"
