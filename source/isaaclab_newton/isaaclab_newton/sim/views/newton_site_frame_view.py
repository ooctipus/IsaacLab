# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-backed FrameView over clone-plan sites and SDP transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from isaaclab.scene_data import SceneDataFormat
from isaaclab.sim.views.base_frame_view import BaseFrameView
from isaaclab.sim.views.xform_space_writer import FrameViewLocalSpaceWriter, FrameViewWorldSpaceWriter
from isaaclab.utils.warp import ProxyArray

if TYPE_CHECKING:
    from isaaclab.cloner import ClonePlan
    from isaaclab.scene_data import SceneDataProvider
    from isaaclab.sim import SimulationContext

WORLD_BODY_INDEX = -1


@wp.kernel
def _compute_site_world_transforms(
    body_q: wp.array(dtype=wp.transformf),
    body_indices: wp.array(dtype=wp.int32),
    site_body: wp.array(dtype=wp.int32),
    site_local: wp.array(dtype=wp.transformf),
    indices: wp.array(dtype=wp.int32),
    out_pos: wp.array(dtype=wp.vec3f),
    out_quat: wp.array(dtype=wp.vec4f),
):
    """Compute world-space transforms for selected sites."""
    i = wp.tid()
    si = indices[i]
    bid = site_body[si]
    if bid == WORLD_BODY_INDEX:
        world = site_local[si]
    else:
        world = wp.transform_multiply(body_q[body_indices[bid]], site_local[si])
    out_pos[i] = wp.transform_get_translation(world)
    q = wp.transform_get_rotation(world)
    out_quat[i] = wp.vec4f(q[0], q[1], q[2], q[3])


@wp.kernel
def _compute_site_local_transforms(
    body_q: wp.array(dtype=wp.transformf),
    body_indices: wp.array(dtype=wp.int32),
    site_body: wp.array(dtype=wp.int32),
    site_local: wp.array(dtype=wp.transformf),
    parent_site_body: wp.array(dtype=wp.int32),
    parent_site_local: wp.array(dtype=wp.transformf),
    indices: wp.array(dtype=wp.int32),
    out_pos: wp.array(dtype=wp.vec3f),
    out_quat: wp.array(dtype=wp.vec4f),
):
    """Compute immediate-parent-relative transforms for selected sites."""
    i = wp.tid()
    si = indices[i]
    bid = site_body[si]
    prim_world = site_local[si]
    if bid != WORLD_BODY_INDEX:
        prim_world = wp.transform_multiply(body_q[body_indices[bid]], prim_world)
    parent_bid = parent_site_body[si]
    parent_world = parent_site_local[si]
    if parent_bid != WORLD_BODY_INDEX:
        parent_world = wp.transform_multiply(body_q[body_indices[parent_bid]], parent_world)
    local_tf = wp.transform_multiply(wp.transform_inverse(parent_world), prim_world)
    out_pos[i] = wp.transform_get_translation(local_tf)
    q = wp.transform_get_rotation(local_tf)
    out_quat[i] = wp.vec4f(q[0], q[1], q[2], q[3])


@wp.kernel
def _write_site_local_from_world_poses(
    body_q: wp.array(dtype=wp.transformf),
    body_indices: wp.array(dtype=wp.int32),
    site_body: wp.array(dtype=wp.int32),
    indices: wp.array(dtype=wp.int32),
    world_pos: wp.array(dtype=wp.vec3f),
    world_quat: wp.array(dtype=wp.vec4f),
    site_local: wp.array(dtype=wp.transformf),
):
    """Update local offsets so selected sites reach desired world poses."""
    i = wp.tid()
    si = indices[i]
    w_pos = world_pos[i]
    w_q = world_quat[i]
    desired_world = wp.transform(w_pos, wp.quatf(w_q[0], w_q[1], w_q[2], w_q[3]))

    bid = site_body[si]
    if bid == WORLD_BODY_INDEX:
        site_local[si] = desired_world
    else:
        site_local[si] = wp.transform_multiply(wp.transform_inverse(body_q[body_indices[bid]]), desired_world)


@wp.kernel
def _write_site_local_from_local_poses(
    body_q: wp.array(dtype=wp.transformf),
    body_indices: wp.array(dtype=wp.int32),
    site_body: wp.array(dtype=wp.int32),
    parent_site_body: wp.array(dtype=wp.int32),
    parent_site_local: wp.array(dtype=wp.transformf),
    indices: wp.array(dtype=wp.int32),
    local_pos: wp.array(dtype=wp.vec3f),
    local_quat: wp.array(dtype=wp.vec4f),
    site_local: wp.array(dtype=wp.transformf),
):
    """Update local offsets so selected sites reach immediate-parent-relative poses."""
    i = wp.tid()
    si = indices[i]
    l_pos = local_pos[i]
    l_q = local_quat[i]
    local = wp.transform(l_pos, wp.quatf(l_q[0], l_q[1], l_q[2], l_q[3]))
    parent_bid = parent_site_body[si]
    parent_world = parent_site_local[si]
    if parent_bid != WORLD_BODY_INDEX:
        parent_world = wp.transform_multiply(body_q[body_indices[parent_bid]], parent_world)
    desired_world = wp.transform_multiply(parent_world, local)
    bid = site_body[si]
    if bid == WORLD_BODY_INDEX:
        site_local[si] = desired_world
    else:
        site_local[si] = wp.transform_multiply(wp.transform_inverse(body_q[body_indices[bid]]), desired_world)


@wp.kernel
def _gather_xform_scales(
    site_xform_scale: wp.array(dtype=wp.vec3f),
    indices: wp.array(dtype=wp.int32),
    out_scales: wp.array(dtype=wp.vec3f),
):
    """Gather per-site xform scales."""
    i = wp.tid()
    out_scales[i] = site_xform_scale[indices[i]]


@wp.kernel
def _scatter_xform_scales(
    indices: wp.array(dtype=wp.int32),
    new_scales: wp.array(dtype=wp.vec3f),
    site_xform_scale: wp.array(dtype=wp.vec3f),
):
    """Scatter per-site xform scales."""
    i = wp.tid()
    site_xform_scale[indices[i]] = new_scales[i]


class NewtonSiteFrameView(BaseFrameView):
    """Batched Newton site view for non-physics frames.

    The public construction contract matches the generic :class:`FrameView`:
    callers provide a prim expression and the backend resolves each source prim
    relative to its immediate clone-plan parent.
    """

    def __init__(
        self,
        prim_path: str | list[str],
        simulation_context: SimulationContext,
        device: str = "cpu",
        validate_xform_ops: bool = True,
    ):
        """Construct the Newton site frame view before replication.

        Args:
            prim_path: User-facing frame path pattern, or list of patterns.
            simulation_context: Active simulation composition root.
            device: Warp device for GPU arrays.
            validate_xform_ops: Unused backend-agnostic argument.
        """
        del simulation_context, validate_xform_ops

        self._prim_paths = [prim_path] if isinstance(prim_path, str) else list(prim_path)
        self._prim_path = prim_path if isinstance(prim_path, str) else ", ".join(self._prim_paths)
        self._device = device
        self._prims = []
        self._frames = ()
        self._count = 0

    def initialize(self, plan: ClonePlan, scene_data_provider: SceneDataProvider) -> None:
        """Bind planned sites and SDP transforms after replication."""
        self._scene_data_provider = scene_data_provider

        if plan is None or not plan.is_complete:
            raise RuntimeError("NewtonSiteFrameView requires a completed clone plan.")
        self._frames = tuple(frame for path in self._prim_paths for frame in plan.match_frames(path))
        if len({frame.path for frame in self._frames}) != len(self._frames):
            raise ValueError(f"Frame expressions overlap in {self._prim_paths!r}.")
        for frame in self._frames:
            if frame.body_path == frame.path:
                raise ValueError(f"FrameView prim {frame.path!r} is a Newton physics body.")
        body_indices = {path: index for index, path in enumerate(plan.iter_rigid_body_paths())}
        site_bodies: list[int] = []
        parent_bodies: list[int] = []
        site_locals: list[list[float]] = []
        parent_locals: list[list[float]] = []
        site_scales: list[tuple[float, float, float]] = []

        for frame in self._frames:
            if frame.body_path is None:
                site_bodies.append(WORLD_BODY_INDEX)
            else:
                try:
                    site_bodies.append(body_indices[frame.body_path])
                except KeyError as exc:
                    raise ValueError(f"FrameView planned body {frame.body_path!r} has no SDP transform.") from exc
            parent_bodies.append(
                WORLD_BODY_INDEX if frame.parent_body_path is None else body_indices[frame.parent_body_path]
            )
            site_locals.append(list(frame.pose))
            parent_locals.append(list(frame.parent_pose))
            site_scales.append(frame.scale)

        self._physics_bound = any(index != WORLD_BODY_INDEX for index in (*site_bodies, *parent_bodies))
        self._empty_body_q = wp.zeros(1, dtype=wp.transformf, device=self._device)
        self._empty_body_indices = wp.zeros(1, dtype=wp.int32, device=self._device)
        self._create_buffers(site_bodies, parent_bodies, site_locals, parent_locals, site_scales)

    def _current_body_q(self) -> tuple[wp.array, wp.array]:
        """Return native rigid-body transforms and their canonical gather indices through SDP."""
        transforms = self._scene_data_provider.request_transforms(SceneDataFormat.IndexedTransform)
        if transforms is None or transforms.transforms is None or transforms.source_indices is None:
            if self._physics_bound:
                raise RuntimeError("NewtonSiteFrameView requires initialized physics transforms.")
            return self._empty_body_q, self._empty_body_indices
        return transforms.transforms, transforms.source_indices

    def _create_buffers(
        self,
        site_bodies: list[int],
        parent_bodies: list[int],
        site_locals: list[list[float]],
        parent_locals: list[list[float]],
        site_scales: list[tuple[float, float, float]],
    ) -> None:
        """Allocate view buffers from body indices and local transforms."""
        self._count = len(site_bodies)
        device = self._device
        self._site_body = wp.array(site_bodies, dtype=wp.int32, device=device)
        self._parent_site_body = wp.array(parent_bodies, dtype=wp.int32, device=device)
        self._site_local = wp.array([wp.transform(*x) for x in site_locals], dtype=wp.transformf, device=device)
        self._parent_site_local = wp.array(
            [wp.transform(*x) for x in parent_locals], dtype=wp.transformf, device=device
        )
        self._site_xform_scale = wp.array([wp.vec3f(*scale) for scale in site_scales], dtype=wp.vec3f, device=device)
        self._site_indices = wp.array(list(range(self._count)), dtype=wp.int32, device=device)
        self._pos_buf = wp.zeros(self._count, dtype=wp.vec3f, device=device)
        self._quat_buf = wp.zeros(self._count, dtype=wp.vec4f, device=device)
        self._local_pos_buf = wp.zeros(self._count, dtype=wp.vec3f, device=device)
        self._local_quat_buf = wp.zeros(self._count, dtype=wp.vec4f, device=device)
        self._scale_buf = wp.zeros(self._count, dtype=wp.vec3f, device=device)
        self._pos_ta = ProxyArray(self._pos_buf)
        self._quat_ta = ProxyArray(self._quat_buf)
        self._local_pos_ta = ProxyArray(self._local_pos_buf)
        self._local_quat_ta = ProxyArray(self._local_quat_buf)
        self._scale_ta = ProxyArray(self._site_xform_scale)

    @property
    def prims(self) -> list:
        """List of USD prims being managed by this view.

        Newton site views do not retain USD prim handles.
        """
        return self._prims

    @property
    def count(self) -> int:
        """Number of frames in this view."""
        return self._count

    @property
    def device(self) -> str:
        """Device where arrays are allocated."""
        return self._device

    # ------------------------------------------------------------------
    # Writer factory hooks (pass-through; Newton has no separate Fabric storage)
    # ------------------------------------------------------------------

    def _make_world_space_writer(self) -> FrameViewWorldSpaceWriter:
        return _NewtonWorldSpaceWriter(self)

    def _make_local_space_writer(self) -> FrameViewLocalSpaceWriter:
        return _NewtonLocalSpaceWriter(self)

    # ------------------------------------------------------------------
    # Backend hooks
    # ------------------------------------------------------------------

    def _get_world_poses_impl(self, indices: wp.array | None = None) -> tuple[ProxyArray, ProxyArray]:
        """Get world-space positions and orientations."""
        body_q, body_indices = self._current_body_q()
        site_indices = self._site_indices if indices is None else indices
        n = self.count if indices is None else len(indices)
        pos_buf = self._pos_buf if indices is None else wp.zeros(n, dtype=wp.vec3f, device=self._device)
        quat_buf = self._quat_buf if indices is None else wp.zeros(n, dtype=wp.vec4f, device=self._device)

        wp.launch(
            _compute_site_world_transforms,
            dim=n,
            inputs=[body_q, body_indices, self._site_body, self._site_local, site_indices],
            outputs=[pos_buf, quat_buf],
            device=self._device,
        )
        if indices is None:
            return self._pos_ta, self._quat_ta
        return ProxyArray(pos_buf), ProxyArray(quat_buf)

    def _apply_world_pose_write(
        self,
        positions: wp.array | None = None,
        orientations: wp.array | None = None,
        indices: wp.array | None = None,
    ) -> None:
        """Set world-space positions and/or orientations."""
        if positions is None and orientations is None:
            return

        body_q, body_indices = self._current_body_q()
        if positions is None or orientations is None:
            cur_pos_ta, cur_quat_ta = self._get_world_poses_impl(indices)
            if positions is None:
                positions = cur_pos_ta.warp
            if orientations is None:
                orientations = cur_quat_ta.warp

        site_indices = self._site_indices if indices is None else indices
        n = self.count if indices is None else len(indices)
        wp.launch(
            _write_site_local_from_world_poses,
            dim=n,
            inputs=[body_q, body_indices, self._site_body, site_indices, positions, orientations, self._site_local],
            device=self._device,
        )

    def _get_local_poses_impl(self, indices: wp.array | None = None) -> tuple[ProxyArray, ProxyArray]:
        """Get immediate-parent-relative positions and orientations."""
        body_q, body_indices = self._current_body_q()
        site_indices = self._site_indices if indices is None else indices
        n = self.count if indices is None else len(indices)
        pos_buf = self._local_pos_buf if indices is None else wp.zeros(n, dtype=wp.vec3f, device=self._device)
        quat_buf = self._local_quat_buf if indices is None else wp.zeros(n, dtype=wp.vec4f, device=self._device)

        wp.launch(
            _compute_site_local_transforms,
            dim=n,
            inputs=[
                body_q,
                body_indices,
                self._site_body,
                self._site_local,
                self._parent_site_body,
                self._parent_site_local,
                site_indices,
            ],
            outputs=[pos_buf, quat_buf],
            device=self._device,
        )
        if indices is None:
            return self._local_pos_ta, self._local_quat_ta
        return ProxyArray(pos_buf), ProxyArray(quat_buf)

    def _apply_local_pose_write(
        self,
        translations: wp.array | None = None,
        orientations: wp.array | None = None,
        indices: wp.array | None = None,
    ) -> None:
        """Set immediate-parent-relative translations and/or orientations."""
        if translations is None and orientations is None:
            return
        body_q, body_indices = self._current_body_q()

        if translations is None or orientations is None:
            cur_pos_ta, cur_quat_ta = self._get_local_poses_impl(indices)
            if translations is None:
                translations = cur_pos_ta.warp
            if orientations is None:
                orientations = cur_quat_ta.warp

        site_indices = self._site_indices if indices is None else indices
        n = self.count if indices is None else len(indices)
        wp.launch(
            _write_site_local_from_local_poses,
            dim=n,
            inputs=[
                body_q,
                body_indices,
                self._site_body,
                self._parent_site_body,
                self._parent_site_local,
                site_indices,
                translations,
                orientations,
                self._site_local,
            ],
            device=self._device,
        )

    # ------------------------------------------------------------------
    # Scales
    # ------------------------------------------------------------------

    def _get_world_scales_impl(self, indices: wp.array | None = None) -> ProxyArray:
        """Get per-site world xform scales.

        These are transform scales, matching the USD FrameView scale API.  They
        are intentionally separate from Newton collision shape geometry sizes.
        """
        if indices is None:
            return self._scale_ta
        n = len(indices)
        out = wp.zeros(n, dtype=wp.vec3f, device=self._device)
        wp.launch(
            _gather_xform_scales,
            dim=n,
            inputs=[self._site_xform_scale, indices],
            outputs=[out],
            device=self._device,
        )
        return ProxyArray(out)

    def _get_local_scales_impl(self, indices: wp.array | None = None) -> ProxyArray:
        """Get per-site local xform scales.

        These are transform scales, matching the USD FrameView scale API.  They
        are intentionally separate from Newton collision shape geometry sizes.
        """
        return self._get_world_scales_impl(indices)

    def _apply_world_scale_write(self, scales: wp.array, indices: wp.array | None = None) -> None:
        """Set per-site world xform scales.

        These update transform scale state only.
        """
        if indices is None:
            indices = self._site_indices
        n = self.count if indices is self._site_indices else len(indices)
        wp.launch(
            _scatter_xform_scales,
            dim=n,
            inputs=[indices, scales, self._site_xform_scale],
            device=self._device,
        )

    def _apply_local_scale_write(self, scales: wp.array, indices: wp.array | None = None) -> None:
        """Set per-site local xform scales.

        These update transform scale state only.
        """
        self._apply_world_scale_write(scales, indices)


# ----------------------------------------------------------------------
# Pass-through writer classes
# ----------------------------------------------------------------------


class _NewtonWorldSpaceWriter(FrameViewWorldSpaceWriter):
    """Newton world-space writer: pass-through to backend ``_apply_*`` hooks."""

    def set_poses(self, positions=None, orientations=None, indices=None) -> None:
        self._view._apply_world_pose_write(positions, orientations, indices)  # type: ignore[attr-defined]

    def set_scales(self, scales, indices=None) -> None:
        self._view._apply_world_scale_write(scales, indices)  # type: ignore[attr-defined]

    def get_poses(self, indices=None) -> tuple[ProxyArray, ProxyArray]:
        return self._view._get_world_poses_impl(indices)  # type: ignore[attr-defined]

    def get_scales(self, indices=None) -> ProxyArray:
        return self._view._get_world_scales_impl(indices)  # type: ignore[attr-defined]


class _NewtonLocalSpaceWriter(FrameViewLocalSpaceWriter):
    """Newton local-space writer: pass-through to backend ``_apply_*`` hooks."""

    def set_poses(self, positions=None, orientations=None, indices=None) -> None:
        self._view._apply_local_pose_write(positions, orientations, indices)  # type: ignore[attr-defined]

    def set_scales(self, scales, indices=None) -> None:
        self._view._apply_local_scale_write(scales, indices)  # type: ignore[attr-defined]

    def get_poses(self, indices=None) -> tuple[ProxyArray, ProxyArray]:
        return self._view._get_local_poses_impl(indices)  # type: ignore[attr-defined]

    def get_scales(self, indices=None) -> ProxyArray:
        return self._view._get_local_scales_impl(indices)  # type: ignore[attr-defined]
