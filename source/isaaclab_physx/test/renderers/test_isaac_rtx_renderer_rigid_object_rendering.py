# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac RTX adapter for the shared rigid-object rendering contract."""

import sys
from pathlib import Path

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True, enable_cameras=True).app

"""Rest everything follows."""

import pytest
from isaaclab_physx.renderers import IsaacRtxRendererCfg

import omni.replicator.core as rep

from isaaclab.sim import build_simulation_context

_CONTRACT_DIR = Path(__file__).resolve().parents[3] / "isaaclab" / "test" / "renderers"
if str(_CONTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTRACT_DIR))

from rigid_object_rendering_contract import (  # noqa: E402
    RigidObjectRenderingBackend,
    run_rigid_object_scale_and_pose_rendering_contract,
)


@pytest.mark.parametrize("with_articulation", [False, True])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_kinematic_rigid_object_scale_and_pose_are_rendered(device, with_articulation):
    """Kinematic PhysX transforms and root scale must reach Isaac RTX."""
    backend = RigidObjectRenderingBackend(
        name="isaac_rtx",
        simulation_context_factory=lambda: build_simulation_context(device=device, gravity_enabled=False),
        renderer_cfg_factory=IsaacRtxRendererCfg,
        with_articulation=with_articulation,
        cleanup=lambda: rep.vp_manager.destroy_hydra_textures("Replicator"),
    )
    run_rigid_object_scale_and_pose_rendering_contract(backend)
