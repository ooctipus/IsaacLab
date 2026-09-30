# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Kit visualizer scene-partition behavior."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from isaaclab_visualizers.kit.kit_visualization_markers import KitVisualizationMarkers

from pxr import Sdf, Usd, UsdGeom


def test_kit_visualizer_registers_clone_context_by_type(monkeypatch) -> None:
    """Kit registers its simulation-owned clone context by backend type."""
    from isaaclab_visualizers.kit import KitVisualizer, KitVisualizerCfg

    from isaaclab.cloner import UsdReplicateContext
    from isaaclab.sim import SimulationContext

    stage = Usd.Stage.CreateInMemory()
    clone_context = object()
    simulation = SimpleNamespace(stage=stage, get_or_create_backend=MagicMock(return_value=clone_context))
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)

    visualizer = KitVisualizer(KitVisualizerCfg())

    assert visualizer._clone_ctx is clone_context
    simulation.get_or_create_backend.assert_called_once_with(
        UsdReplicateContext,
        stage,
        clone_role="scene",
    )


def test_marker_environment_ids_are_sticky_until_count_changes() -> None:
    """Omitted environment IDs should persist only while the marker count is unchanged."""
    stage = Usd.Stage.CreateInMemory()
    for env_id in range(2):
        env_prim = stage.DefinePrim(f"/World/envs/env_{env_id}", "Xform")
        env_prim.CreateAttribute("primvars:omni:scenePartition", Sdf.ValueTypeNames.Token).Set(f"env_{env_id}")
    instancer = UsdGeom.PointInstancer.Define(stage, "/World/Visuals/markers")
    markers = object.__new__(KitVisualizationMarkers)
    markers._sim = SimpleNamespace(get_setting=lambda _path: True)
    markers.stage = stage
    markers._instancer_manager = instancer
    markers._environment_ids = None
    markers._count = 0

    markers.visualize(
        translations=torch.zeros((2, 3)),
        orientations=None,
        scales=None,
        marker_indices=None,
        environment_ids=torch.tensor([0, 1]),
    )
    primvar = UsdGeom.PrimvarsAPI(instancer).GetPrimvar("omni:scenePartition")
    assert list(primvar.Get()) == ["env_0", "env_1"]

    markers.visualize(
        translations=torch.ones((2, 3)),
        orientations=None,
        scales=None,
        marker_indices=None,
    )
    assert list(primvar.Get()) == ["env_0", "env_1"]

    markers.visualize(
        translations=torch.ones((1, 3)),
        orientations=None,
        scales=None,
        marker_indices=None,
    )

    assert not primvar.GetAttr().HasAuthoredValueOpinion()
