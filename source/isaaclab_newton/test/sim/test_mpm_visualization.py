# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for clone-planned MPM point geometry."""

import numpy as np
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMPointsCfg

from pxr import Gf, Usd, UsdGeom, UsdShade

import isaaclab.sim as sim_utils


def test_grid_spawner_declares_local_point_geometry():
    """The MPM prototype itself contains the geometry every clone-plan consumer draws."""
    stage = Usd.Stage.CreateInMemory()
    cfg = MPMGridCfg(
        lower=(0.0, 0.0, 0.0),
        upper=(0.2, 0.1, 0.1),
        voxel_size=0.1,
        particle_placement="cell_center",
        radius=0.02,
        visual_color=(0.1, 0.2, 0.3),
    )
    with sim_utils.use_stage(stage):
        prim = cfg.func("/World/envs/env_0/Sand", cfg, translation=(1.0, 2.0, 3.0))

    points = UsdGeom.Points(prim)
    np.testing.assert_allclose(points.GetPointsAttr().Get(), ((0.05, 0.05, 0.05), (0.15, 0.05, 0.05)))
    np.testing.assert_allclose(points.GetWidthsAttr().Get(), (0.04, 0.04))
    assert points.GetDisplayColorAttr().Get() == [Gf.Vec3f(0.1, 0.2, 0.3)]
    assert tuple(UsdGeom.Xformable(prim).GetOrderedXformOps()[0].Get()) == (1.0, 2.0, 3.0)


def test_points_spawner_declares_material_on_the_planned_root():
    """Explicit particles and their material are authored without a visualizer-specific path."""
    stage = Usd.Stage.CreateInMemory()
    cfg = MPMPointsCfg(
        positions=((0.0, 0.0, 0.0), (0.0, 0.1, 0.0)),
        radius=(0.01, 0.02),
        visible=False,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 0.6)),
    )
    with sim_utils.use_stage(stage):
        prim = cfg.func("/World/envs/env_0/Fluid", cfg)

    points = UsdGeom.Points(prim)
    np.testing.assert_allclose(points.GetPointsAttr().Get(), cfg.positions)
    np.testing.assert_allclose(points.GetWidthsAttr().Get(), (0.02, 0.04))
    assert UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible
    assert UsdShade.MaterialBindingAPI(prim).GetDirectBindingRel().GetTargets() == [
        "/World/envs/env_0/Fluid/Looks/visualMaterial"
    ]
