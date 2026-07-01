# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVRTX adapter for the shared rigid-object rendering contract."""

import importlib.util
import sys
from pathlib import Path

import pytest

from isaaclab.sim import SimulationCfg, build_simulation_context

_CONTRACT_DIR = Path(__file__).resolve().parents[3] / "isaaclab" / "test" / "renderers"
if str(_CONTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTRACT_DIR))

from rigid_object_rendering_contract import (  # noqa: E402
    RigidObjectRenderingBackend,
    run_rigid_object_scale_and_pose_rendering_contract,
)

_REQUIRED_MODULES = ("isaaclab_ov", "ovrtx", "isaaclab_ovphysx", "ovphysx", "isaaclab_newton", "newton")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]
_SKIP_MISSING_OVRTX = pytest.mark.skipif(
    bool(_MISSING_MODULES),
    reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
)

if not _MISSING_MODULES:
    from isaaclab_newton.physics import NewtonManager  # noqa: E402
    from isaaclab_ov.renderers import OVRTXRendererCfg  # noqa: E402
    from isaaclab_ovphysx.physics import OvPhysxCfg  # noqa: E402
else:
    NewtonManager = None
    OVRTXRendererCfg = None
    OvPhysxCfg = None


@pytest.mark.isaacsim_ci
@_SKIP_MISSING_OVRTX
def test_kinematic_rigid_object_scale_and_pose_are_rendered():
    """Kinematic OVPhysX transforms and root scale must reach OVRTX."""
    device = "cuda:0"
    assert NewtonManager is not None
    assert OVRTXRendererCfg is not None
    assert OvPhysxCfg is not None
    backend = RigidObjectRenderingBackend(
        name="ovrtx (OVPhysX)",
        simulation_context_factory=lambda: build_simulation_context(
            device=device,
            sim_cfg=SimulationCfg(
                dt=1.0 / 60.0,
                device=device,
                gravity=(0.0, 0.0, 0.0),
                physics=OvPhysxCfg(),
            ),
        ),
        renderer_cfg_factory=OVRTXRendererCfg,
        cleanup=NewtonManager.clear,
    )
    run_rigid_object_scale_and_pose_rendering_contract(backend)
