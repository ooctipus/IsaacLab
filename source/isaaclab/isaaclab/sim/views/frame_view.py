# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-dispatching FrameView.

``FrameView(path, device=...)`` automatically selects the right backend:
- PhysX: :class:`~isaaclab_physx.sim.views.FabricFrameView`
- Newton: :class:`~isaaclab_newton.sim.views.NewtonSiteFrameView`
"""

from __future__ import annotations

from isaaclab.utils.backend_utils import FactoryBase

from .base_frame_view import BaseFrameView


class FrameView(FactoryBase, BaseFrameView):
    """FrameView that dispatches to the active physics backend.

    Callers use ``FrameView(prim_path, device=device)`` and get the
    correct implementation automatically:

    - **PhysX / no backend**: :class:`~isaaclab_physx.sim.views.FabricFrameView`
      (Fabric GPU acceleration with USD fallback).
    - **OVPhysX**: :class:`~isaaclab_ovphysx.sim.views.OvPhysxFrameView`
      (Warp-native, reads body poses via an OVPhysX ``RIGID_BODY_POSE``
      tensor binding).
    - **Newton**: :class:`~isaaclab_newton.sim.views.NewtonSiteFrameView`
      (Warp-native, reads ``body_q`` from the Newton state).
    """

    _backend_class_names = {
        "physx": "FabricFrameView",
        "ovphysx": "OvPhysxFrameView",
        "newton": "NewtonSiteFrameView",
    }

    @classmethod
    def _get_backend(cls, *args, **kwargs) -> str:
        from isaaclab.sim.simulation_context import SimulationContext  # noqa: PLC0415

        ctx = SimulationContext.instance()
        if ctx is None:
            return "physx"
        manager_name = ctx.physics_manager.__name__.lower()
        if "newton" in manager_name:
            return "newton"
        if "ovphysx" in manager_name:
            return "ovphysx"
        return "physx"

    @classmethod
    def register_frame(cls, prim_path: str | list[str], stage: object | None = None) -> bool:
        """Pre-register a frame with the active physics backend before finalization.

        Backends that inject frame sites during replication (Newton) record the
        registration so a view constructed later for the same ``prim_path``
        initializes from the injected sites. Backends without pre-registration
        (PhysX, OVPhysX) ignore the call.

        Args:
            prim_path: User-facing frame path pattern, or list of patterns.
            stage: USD stage that contains the source prims. Defaults to the current stage.

        Returns:
            Whether the active backend recorded the registration.
        """
        from isaaclab.sim.simulation_context import SimulationContext  # noqa: PLC0415

        if SimulationContext.instance() is None:
            return False
        impl = cls.resolve_class(prim_path, stage=stage)
        backend_register = getattr(impl, "register_frame", None)
        if backend_register is None:
            return False
        return backend_register(prim_path, stage=stage)

    def __new__(cls, *args, **kwargs) -> BaseFrameView:
        """Create a new FrameView for the active physics backend."""
        return super().__new__(cls, *args, **kwargs)
