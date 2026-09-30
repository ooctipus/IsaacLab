# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for synchronizing XR anchor pose with a reference prim and XR config."""

from __future__ import annotations

import contextlib
import logging
import math
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

from isaaclab.scene_data import SceneDataFormat
from isaaclab.sim import SimulationContext

from .xr_cfg import XrAnchorRotationMode

with contextlib.suppress(ModuleNotFoundError):
    from pxr import Gf as pxrGf


class _PlannedFrameTransform:
    """Read one plan-declared frame through the simulation's scene-data provider."""

    def __init__(self, prim_path: str):
        sim = SimulationContext.instance()
        if sim is None:
            raise RuntimeError("A planned frame requires an active SimulationContext.")
        plan = sim.get_clone_plan()
        if plan is None or not plan.is_complete:
            raise RuntimeError("A planned frame requires a completed clone plan.")
        frames = plan.match_frames(prim_path)
        if len(frames) != 1:
            raise ValueError(f"Frame expression {prim_path!r} resolved {len(frames)} frames; expected exactly one.")

        self._frame = frames[0]
        self._provider = sim.get_scene_data_provider()
        self._body_index: int | None = None
        if self._frame.body_path is not None:
            body_paths = tuple(plan.iter_rigid_body_paths())
            try:
                self._body_index = body_paths.index(self._frame.body_path)
            except ValueError as exc:
                raise RuntimeError(f"Clone plan declares no body {self._frame.body_path!r}.") from exc

        self._local_matrix = np.eye(4, dtype=np.float64)
        self._local_matrix[:3, :3] = Rotation.from_quat(self._frame.pose[3:]).as_matrix()
        self._local_matrix[:3, 3] = self._frame.pose[:3]

    def world_matrix(self) -> np.ndarray:
        """Return the conventional column-vector world matrix for the planned frame."""
        if self._body_index is None:
            return self._local_matrix
        publication = self._provider.request_transforms(SceneDataFormat.HostTransposedMatrix44d)
        if publication is None or publication.matrices is None:
            raise RuntimeError(f"Physics published no transform for planned body {self._frame.body_path!r}.")
        return publication.matrices[self._body_index].T @ self._local_matrix


class XrAnchorSynchronizer:
    """Keeps the XR anchor prim aligned with a reference prim according to XR config."""

    def __init__(
        self,
        xr_core: Any,
        xr_cfg: Any,
        xr_anchor_headset_path: str,
        anchor_frame: _PlannedFrameTransform | None,
        anchor_layer_identifier: str,
    ):
        self._xr_core = xr_core
        self._xr_cfg = xr_cfg
        self._xr_anchor_headset_path = xr_anchor_headset_path
        self._anchor_frame = anchor_frame
        self.__anchor_headset_layer_identifier = anchor_layer_identifier

        self.__anchor_prim_initial_quat = None
        self.__anchor_prim_initial_height = None
        self.__smoothed_anchor_quat = None
        self.__last_anchor_quat = None
        self.__anchor_rotation_enabled = True

        # Cached anchor world transform (pos, quat_xyzw) set by sync_headset_to_anchor().
        self.__cached_world_pos: np.ndarray | None = None
        self.__cached_world_quat_xyzw: np.ndarray | None = None

    def reset(self):
        self.__anchor_prim_initial_quat = None
        self.__anchor_prim_initial_height = None
        self.__smoothed_anchor_quat = None
        self.__last_anchor_quat = None
        self.__anchor_rotation_enabled = True
        self.__cached_world_pos = None
        self.__cached_world_quat_xyzw = None
        self.sync_headset_to_anchor()

    def toggle_anchor_rotation(self):
        self.__anchor_rotation_enabled = not self.__anchor_rotation_enabled
        logger.info(f"XR: Toggling anchor rotation: {self.__anchor_rotation_enabled}")

    def get_world_transform(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return the anchor world transform.

        Returns the cached world transform that was computed by the most recent
        call to :meth:`sync_headset_to_anchor`. The reference frame itself is
        resolved from the clone plan and read through the scene-data provider.

        Returns:
            A ``(position, quat_xyzw)`` tuple of numpy float64 arrays,
            or ``None`` if :meth:`sync_headset_to_anchor` has not run yet.
        """
        if self.__cached_world_pos is not None and self.__cached_world_quat_xyzw is not None:
            return self.__cached_world_pos, self.__cached_world_quat_xyzw
        return None

    def sync_headset_to_anchor(self):
        """Sync XR anchor pose in USD for both dynamic and static anchoring.

        For **dynamic** anchoring (``anchor_prim_path`` is set), the reference
        prim's world pose is read through the scene-data provider and ``anchor_pos``
        is added as an offset. For **static** anchoring, ``anchor_pos`` is the world position.

        In both cases the function calls ``set_world_transform_matrix`` on the
        XR core so that the rendering anchor and the pipeline's
        ``world_T_anchor`` matrix are guaranteed to agree, and caches the
        world transform for :meth:`get_world_transform`.
        """
        reference_quat = None
        if self._anchor_frame is not None:
            reference_matrix = self._anchor_frame.world_matrix()
            reference_pos = reference_matrix[:3, 3].copy()
            reference_quat_xyzw = Rotation.from_matrix(reference_matrix[:3, :3]).as_quat()
            reference_quat = pxrGf.Quatd(reference_quat_xyzw[3], pxrGf.Vec3d(*reference_quat_xyzw[:3]))
            if self.__anchor_prim_initial_quat is None:
                self.__anchor_prim_initial_quat = reference_quat
            if self._xr_cfg.fixed_anchor_height:
                if self.__anchor_prim_initial_height is None:
                    self.__anchor_prim_initial_height = reference_pos[2]
                reference_pos[2] = self.__anchor_prim_initial_height
            pxr_anchor_pos = pxrGf.Vec3d(*reference_pos) + pxrGf.Vec3d(*self._xr_cfg.anchor_pos)
        else:
            pxr_anchor_pos = pxrGf.Vec3d(*self._xr_cfg.anchor_pos)

        x, y, z, w = self._xr_cfg.anchor_rot
        pxr_cfg_quat = pxrGf.Quatd(w, pxrGf.Vec3d(x, y, z))
        pxr_anchor_quat = pxr_cfg_quat

        if reference_quat is not None and self._xr_cfg.anchor_rotation_mode in (
            XrAnchorRotationMode.FOLLOW_PRIM,
            XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED,
        ):
            delta_quat = reference_quat * self.__anchor_prim_initial_quat.GetInverse()
            wq = delta_quat.GetReal()
            ix, iy, iz = delta_quat.GetImaginary()
            yaw = math.atan2(2.0 * (wq * iz + ix * iy), 1.0 - 2.0 * (iy * iy + iz * iz))
            pxr_anchor_quat = pxrGf.Quatd(math.cos(yaw * 0.5), pxrGf.Vec3d(0.0, 0.0, math.sin(yaw * 0.5)))
            pxr_anchor_quat = pxr_anchor_quat * pxr_cfg_quat

            if self._xr_cfg.anchor_rotation_mode == XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED:
                if self.__smoothed_anchor_quat is None:
                    self.__smoothed_anchor_quat = pxr_anchor_quat
                else:
                    dt = SimulationContext.instance().get_rendering_dt()
                    alpha = 1.0 - math.exp(-dt / max(self._xr_cfg.anchor_rotation_smoothing_time, 1e-6))
                    self.__smoothed_anchor_quat = pxrGf.Slerp(
                        min(1.0, max(0.05, alpha)), self.__smoothed_anchor_quat, pxr_anchor_quat
                    )
                    pxr_anchor_quat = self.__smoothed_anchor_quat

        if self.__anchor_rotation_enabled:
            pxr_final_quat = pxr_anchor_quat
            self.__last_anchor_quat = pxr_anchor_quat
        else:
            if self.__last_anchor_quat is None:
                self.__last_anchor_quat = pxr_anchor_quat
            pxr_final_quat = self.__last_anchor_quat
            self.__smoothed_anchor_quat = self.__last_anchor_quat

        pxr_mat = pxrGf.Matrix4d()
        pxr_mat.SetTranslateOnly(pxr_anchor_pos)
        pxr_mat.SetRotateOnly(pxr_final_quat)
        self.__cached_world_pos = np.asarray(pxr_anchor_pos, dtype=np.float64)
        fq_img = pxr_final_quat.GetImaginary()
        self.__cached_world_quat_xyzw = np.array(
            [fq_img[0], fq_img[1], fq_img[2], pxr_final_quat.GetReal()], dtype=np.float64
        )
        self._xr_core.set_world_transform_matrix(
            self._xr_anchor_headset_path, pxr_mat, self.__anchor_headset_layer_identifier
        )
