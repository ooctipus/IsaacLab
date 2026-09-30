# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-dispatching FrameView."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils.backend_utils import FactoryBase

from .base_frame_view import BaseFrameView

if TYPE_CHECKING:
    from isaaclab.sim import SimulationContext


class FrameView(FactoryBase, BaseFrameView):
    """FrameView that dispatches to the active physics backend.

    An active :class:`~isaaclab.sim.SimulationContext` selects the implementation:

    - **PhysX**: :class:`~isaaclab_physx.sim.views.PhysxFrameView`
      (Warp-native, reads body poses through the scene-data provider).
    - **OVPhysX**: :class:`~isaaclab_ov.sim.views.OvPhysxFrameView`
      (Warp-native, reads body poses through the scene-data provider).
    - **Newton**: :class:`~isaaclab_newton.sim.views.NewtonSiteFrameView`
      (Warp-native, reads body poses through the scene-data provider).
    """

    _backend_class_names = {
        "physx": "PhysxFrameView",
        "ovphysx": "OvPhysxFrameView",
        "newton": "NewtonSiteFrameView",
    }

    @classmethod
    def _get_backend(cls, _prim_path, simulation_context: SimulationContext, *_args, **_kwargs) -> str:
        """Select the backend from the explicitly owned simulation context."""
        return simulation_context.physics_backend

    def __new__(cls, *args, **kwargs) -> BaseFrameView:
        """Create a new FrameView for the active physics backend."""
        return super().__new__(cls, *args, **kwargs)
