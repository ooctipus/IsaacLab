# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Reset-selection tests for the fast terrain scanner."""

import torch
import warp as wp

from isaaclab.sensors import SensorBase

from isaaclab_tasks.core.multi_task.sensors import FastTerrainScanner, FastTerrainScannerCfg


def test_reset_env_mask_takes_priority_over_env_ids(monkeypatch):
    """The Warp mask must select reset rows when both selectors are supplied."""
    scanner = object.__new__(FastTerrainScanner)
    scanner._view_count = 3
    scanner._device = "cpu"
    scanner._drift_torch = torch.zeros((3, 3))
    scanner._ray_cast_drift_torch = torch.zeros((3, 3))
    scanner._initialize_handle = None
    scanner._invalidate_initialize_handle = None
    scanner._prim_deletion_handle = None
    scanner._debug_vis_handle = None
    scanner.cfg = FastTerrainScannerCfg()
    scanner.cfg.drift_range = (1.0, 1.0)
    scanner.cfg.ray_cast_drift_range = {"x": (2.0, 2.0), "y": (3.0, 3.0), "z": (4.0, 4.0)}
    env_mask = wp.array([False, True, False], dtype=wp.bool, device="cpu")
    monkeypatch.setattr(SensorBase, "reset", lambda *_args, **_kwargs: None)

    scanner.reset(env_ids=[0], env_mask=env_mask)

    expected_drift = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    expected_ray_cast_drift = torch.tensor([[0.0, 0.0, 0.0], [2.0, 3.0, 4.0], [0.0, 0.0, 0.0]])
    torch.testing.assert_close(scanner._drift_torch, expected_drift)
    torch.testing.assert_close(scanner._ray_cast_drift_torch, expected_ray_cast_drift)
