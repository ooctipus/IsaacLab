# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit / Omniverse viewport helpers (Isaac Sim specific).

These live in :mod:`isaaclab_physx` so :class:`~isaaclab.sim.SimulationContext` stays
backend-agnostic.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Throttle: only the first failure per process is warned at WARNING; subsequent
# failures drop to DEBUG so the per-frame tracking callback does not spam logs.
_warned_once = False


def set_kit_renderer_camera_view(
    eye: tuple[float, float, float] | list[float],
    target: tuple[float, float, float] | list[float],
    camera_prim_path: str = "/OmniverseKit_Persp",
) -> None:
    """Set the renderer/viewport camera pose by writing the USD prim transform directly.

    Bypasses ``isaacsim.core.rendering_manager.ViewportManager`` and
    ``isaacsim.core.utils.viewports.set_camera_view``: the former is only loaded by
    ``apps/isaaclab.python.kit`` and the latter was dropped from the headless app's
    dependency list in upstream's "Updates deprecated extensions" commit. Writing the
    xform via USD is portable across both interactive and headless app files.

    This does not broadcast to visualizers.
    """
    global _warned_once
    try:
        import omni.usd
        from pxr import Gf, Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        prim = stage.GetPrimAtPath(str(camera_prim_path))
        if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Xformable):
            return

        eye_v = Gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2]))
        target_v = Gf.Vec3d(float(target[0]), float(target[1]), float(target[2]))
        up_axis = UsdGeom.GetStageUpAxis(stage)
        up = Gf.Vec3d(0.0, 0.0, 1.0) if up_axis == "Z" else Gf.Vec3d(0.0, 1.0, 0.0)
        # ``SetLookAt`` builds world->camera; the prim's local-to-world xform is its inverse.
        cam_to_world = Gf.Matrix4d(1.0).SetLookAt(eye_v, target_v, up).GetInverse()

        xformable = UsdGeom.Xformable(prim)
        # The Kit perspective camera lives on the session layer; authoring requires an
        # explicit edit context, otherwise ``MakeMatrixXform`` raises "Accessed schema on
        # invalid prim" because the default edit target cannot resolve the existing ops.
        session_layer = stage.GetSessionLayer()
        with Usd.EditContext(stage, session_layer):
            existing_ops = xformable.GetOrderedXformOps()
            matrix_op = None
            if len(existing_ops) == 1 and existing_ops[0].GetOpType() == UsdGeom.XformOp.TypeTransform:
                matrix_op = existing_ops[0]
            else:
                # Replace the existing translate/rotate/scale split with a single matrix op
                # so subsequent calls hit the fast path above.
                xformable.SetXformOpOrder([])
                matrix_op = xformable.AddTransformOp()
            matrix_op.Set(cam_to_world)
    except (ImportError, ModuleNotFoundError) as exc:
        logger.debug("[kit_viewport] Renderer camera update skipped (no USD context): %s", exc)
    except Exception as exc:
        if not _warned_once:
            logger.warning("[kit_viewport] Renderer camera update failed: %s (further failures suppressed)", exc)
            _warned_once = True
        else:
            logger.debug("[kit_viewport] Renderer camera update failed: %s", exc)
