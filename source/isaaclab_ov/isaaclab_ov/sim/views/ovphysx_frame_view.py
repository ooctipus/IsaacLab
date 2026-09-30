# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVPhysX-backed FrameView over clone-plan sites and SDP transforms."""

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
    site_body: wp.array(dtype=wp.int32),
    site_local: wp.array(dtype=wp.transformf),
    out_pos: wp.array(dtype=wp.vec3f),
    out_quat: wp.array(dtype=wp.vec4f),
):
    """Compute world-space transforms for every site in the view.

    For each site *i*, computes ``world = body_q[site_body[i]] * site_local[i]``
    and splits the result into position and quaternion outputs.  When
    ``site_body[i] == -1`` the site is world-attached and ``site_local[i]`` is
    returned directly.

    Args:
        body_q: Rigid-body world transforms from the OVPhysX-backed Newton state,
            shape ``[num_bodies]``.
        site_body: Per-site body index (flat model-level), shape ``[num_sites]``.
            ``-1`` indicates a world-attached site.
        site_local: Per-site local offset relative to its parent body, shape ``[num_sites]``.
        out_pos: Output world positions [m], shape ``[num_sites]``.
        out_quat: Output world orientations as ``(qx, qy, qz, qw)``, shape ``[num_sites]``.
    """
    i = wp.tid()
    bid = site_body[i]
    if bid == -1:
        world = site_local[i]
    else:
        world = wp.transform_multiply(body_q[bid], site_local[i])
    out_pos[i] = wp.transform_get_translation(world)
    q = wp.transform_get_rotation(world)
    out_quat[i] = wp.vec4f(q[0], q[1], q[2], q[3])


@wp.kernel
def _compute_site_world_transforms_indexed(
    body_q: wp.array(dtype=wp.transformf),
    site_body: wp.array(dtype=wp.int32),
    site_local: wp.array(dtype=wp.transformf),
    indices: wp.array(dtype=wp.int32),
    out_pos: wp.array(dtype=wp.vec3f),
    out_quat: wp.array(dtype=wp.vec4f),
):
    """Indexed variant of :func:`_compute_site_world_transforms`."""
    i = wp.tid()
    si = indices[i]
    bid = site_body[si]
    if bid == -1:
        world = site_local[si]
    else:
        world = wp.transform_multiply(body_q[bid], site_local[si])
    out_pos[i] = wp.transform_get_translation(world)
    q = wp.transform_get_rotation(world)
    out_quat[i] = wp.vec4f(q[0], q[1], q[2], q[3])


@wp.kernel
def _gather_scales(scales: wp.array(dtype=wp.vec3f), indices: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.vec3f)):
    """Gather selected view-owned frame scales."""
    i = wp.tid()
    out[i] = scales[indices[i]]


@wp.kernel
def _scatter_scales(
    scales: wp.array(dtype=wp.vec3f), indices: wp.array(dtype=wp.int32), values: wp.array(dtype=wp.vec3f)
):
    """Scatter selected view-owned frame scales."""
    i = wp.tid()
    scales[indices[i]] = values[i]


@wp.kernel
def _write_site_local_from_world_poses(
    body_q: wp.array(dtype=wp.transformf),
    site_body: wp.array(dtype=wp.int32),
    world_pos: wp.array(dtype=wp.vec3f),
    world_quat: wp.array(dtype=wp.vec4f),
    site_local: wp.array(dtype=wp.transformf),
):
    """Update site local offsets so that sites reach desired world poses.

    For each site *i*, sets ``site_local[i] = inv(body_q[bid]) * desired_world``
    so that subsequent reads produce the requested world pose.  Does **not**
    modify ``body_q``.  World-attached sites (``site_body[i] == -1``) receive
    the desired world transform directly.

    Args:
        body_q: Rigid-body world transforms, shape ``[num_bodies]``.
        site_body: Per-site body index, shape ``[num_sites]``.
        world_pos: Desired world positions [m], shape ``[num_sites]``.
        world_quat: Desired world orientations as ``(qx, qy, qz, qw)``, shape ``[num_sites]``.
        site_local: Per-site local offset (modified in-place), shape ``[num_sites]``.
    """
    i = wp.tid()
    w_pos = world_pos[i]
    w_q = world_quat[i]
    desired_world = wp.transform(w_pos, wp.quatf(w_q[0], w_q[1], w_q[2], w_q[3]))
    bid = site_body[i]
    if bid == -1:
        site_local[i] = desired_world
    else:
        site_local[i] = wp.transform_multiply(wp.transform_inverse(body_q[bid]), desired_world)


@wp.kernel
def _write_site_local_from_world_poses_indexed(
    body_q: wp.array(dtype=wp.transformf),
    site_body: wp.array(dtype=wp.int32),
    indices: wp.array(dtype=wp.int32),
    world_pos: wp.array(dtype=wp.vec3f),
    world_quat: wp.array(dtype=wp.vec4f),
    site_local: wp.array(dtype=wp.transformf),
):
    """Indexed variant of :func:`_write_site_local_from_world_poses`."""
    i = wp.tid()
    si = indices[i]
    w_pos = world_pos[i]
    w_q = world_quat[i]
    desired_world = wp.transform(w_pos, wp.quatf(w_q[0], w_q[1], w_q[2], w_q[3]))
    bid = site_body[si]
    if bid == -1:
        site_local[si] = desired_world
    else:
        site_local[si] = wp.transform_multiply(wp.transform_inverse(body_q[bid]), desired_world)


@wp.kernel
def _compute_site_local_transforms(
    body_q: wp.array(dtype=wp.transformf),
    site_body: wp.array(dtype=wp.int32),
    site_local: wp.array(dtype=wp.transformf),
    parent_site_body: wp.array(dtype=wp.int32),
    parent_site_local: wp.array(dtype=wp.transformf),
    out_pos: wp.array(dtype=wp.vec3f),
    out_quat: wp.array(dtype=wp.vec4f),
):
    """Compute parent-relative transforms for every site in the view.

    For each site *i*, computes the world pose of both the site and its USD
    parent, then returns ``inv(parent_world) * prim_world``.  World-attached
    sites/parents use ``site_local`` / ``parent_site_local`` directly.
    """
    i = wp.tid()
    prim_bid = site_body[i]
    if prim_bid == -1:
        prim_world = site_local[i]
    else:
        prim_world = wp.transform_multiply(body_q[prim_bid], site_local[i])
    parent_bid = parent_site_body[i]
    if parent_bid == -1:
        parent_world = parent_site_local[i]
    else:
        parent_world = wp.transform_multiply(body_q[parent_bid], parent_site_local[i])
    local_tf = wp.transform_multiply(wp.transform_inverse(parent_world), prim_world)
    out_pos[i] = wp.transform_get_translation(local_tf)
    q = wp.transform_get_rotation(local_tf)
    out_quat[i] = wp.vec4f(q[0], q[1], q[2], q[3])


@wp.kernel
def _compute_site_local_transforms_indexed(
    body_q: wp.array(dtype=wp.transformf),
    site_body: wp.array(dtype=wp.int32),
    site_local: wp.array(dtype=wp.transformf),
    parent_site_body: wp.array(dtype=wp.int32),
    parent_site_local: wp.array(dtype=wp.transformf),
    indices: wp.array(dtype=wp.int32),
    out_pos: wp.array(dtype=wp.vec3f),
    out_quat: wp.array(dtype=wp.vec4f),
):
    """Indexed variant of :func:`_compute_site_local_transforms`."""
    i = wp.tid()
    si = indices[i]
    prim_bid = site_body[si]
    if prim_bid == -1:
        prim_world = site_local[si]
    else:
        prim_world = wp.transform_multiply(body_q[prim_bid], site_local[si])
    parent_bid = parent_site_body[si]
    if parent_bid == -1:
        parent_world = parent_site_local[si]
    else:
        parent_world = wp.transform_multiply(body_q[parent_bid], parent_site_local[si])
    local_tf = wp.transform_multiply(wp.transform_inverse(parent_world), prim_world)
    out_pos[i] = wp.transform_get_translation(local_tf)
    q = wp.transform_get_rotation(local_tf)
    out_quat[i] = wp.vec4f(q[0], q[1], q[2], q[3])


@wp.kernel
def _write_site_local_from_local_poses(
    body_q: wp.array(dtype=wp.transformf),
    site_body: wp.array(dtype=wp.int32),
    parent_site_body: wp.array(dtype=wp.int32),
    parent_site_local: wp.array(dtype=wp.transformf),
    local_pos: wp.array(dtype=wp.vec3f),
    local_quat: wp.array(dtype=wp.vec4f),
    site_local: wp.array(dtype=wp.transformf),
):
    """Update site local offsets so that sites reach desired parent-relative poses."""
    i = wp.tid()
    parent_bid = parent_site_body[i]
    if parent_bid == -1:
        parent_world = parent_site_local[i]
    else:
        parent_world = wp.transform_multiply(body_q[parent_bid], parent_site_local[i])
    l_pos = local_pos[i]
    l_q = local_quat[i]
    local_tf = wp.transform(l_pos, wp.quatf(l_q[0], l_q[1], l_q[2], l_q[3]))
    desired_world = wp.transform_multiply(parent_world, local_tf)
    bid = site_body[i]
    if bid == -1:
        site_local[i] = desired_world
    else:
        site_local[i] = wp.transform_multiply(wp.transform_inverse(body_q[bid]), desired_world)


@wp.kernel
def _write_site_local_from_local_poses_indexed(
    body_q: wp.array(dtype=wp.transformf),
    site_body: wp.array(dtype=wp.int32),
    parent_site_body: wp.array(dtype=wp.int32),
    parent_site_local: wp.array(dtype=wp.transformf),
    indices: wp.array(dtype=wp.int32),
    local_pos: wp.array(dtype=wp.vec3f),
    local_quat: wp.array(dtype=wp.vec4f),
    site_local: wp.array(dtype=wp.transformf),
):
    """Indexed variant of :func:`_write_site_local_from_local_poses`."""
    i = wp.tid()
    si = indices[i]
    parent_bid = parent_site_body[si]
    if parent_bid == -1:
        parent_world = parent_site_local[si]
    else:
        parent_world = wp.transform_multiply(body_q[parent_bid], parent_site_local[si])
    l_pos = local_pos[i]
    l_q = local_quat[i]
    local_tf = wp.transform(l_pos, wp.quatf(l_q[0], l_q[1], l_q[2], l_q[3]))
    desired_world = wp.transform_multiply(parent_world, local_tf)
    bid = site_body[si]
    if bid == -1:
        site_local[si] = desired_world
    else:
        site_local[si] = wp.transform_multiply(wp.transform_inverse(body_q[bid]), desired_world)


class OvPhysxFrameView(BaseFrameView):
    """Batched prim view for non-physics prims tracked as sites on OVPhysX bodies.

    Each planned frame is bound at init to the rigid-body ancestor and fixed pose
    declared by :class:`~isaaclab.cloner.ClonePlan`. A body-less frame is attached
    to the world (``body_index = WORLD_BODY_INDEX``).

    Body world poses are requested from the simulation's scene-data provider.

    World poses are computed on GPU as ``body_q[body_index] * site_local`` via
    a Warp kernel, with the world-attached branch returning ``site_local``
    directly. Both :meth:`set_world_poses` and :meth:`set_local_poses` update
    the view-owned ``site_local`` buffer -- neither writes to the physics state.

    Scales are view-owned metadata and never mutate physics or a USD stage.

    Getters return :class:`~isaaclab.utils.warp.ProxyArray`.  Setters
    accept ``wp.array``.

    """

    def __init__(
        self,
        prim_path: str,
        simulation_context: SimulationContext,
        device: str = "cpu",
        validate_xform_ops: bool = True,
    ):
        """Construct the OVPhysX site-based frame view before replication.

        Args:
            prim_path: USD prim path pattern (may contain regex).
            simulation_context: Active simulation composition root.
            device: Warp device for GPU arrays (e.g. ``"cuda:0"``).
            validate_xform_ops: Unused backend-agnostic argument.
        """
        del simulation_context, validate_xform_ops
        self._prim_path = prim_path
        self._device = device
        self._frames = ()

    def initialize(self, plan: ClonePlan, scene_data_provider: SceneDataProvider) -> None:
        """Bind planned sites and SDP transforms after replication."""
        self._scene_data_provider = scene_data_provider

        if plan is None or not plan.is_complete:
            raise RuntimeError("OvPhysxFrameView requires a completed clone plan.")
        self._frames = plan.match_frames(self._prim_path)
        for frame in self._frames:
            if frame.body_path == frame.path:
                raise ValueError(f"OvPhysxFrameView planned frame {frame.path!r} is a rigid body.")

        path_to_row = {path: index for index, path in enumerate(plan.iter_rigid_body_paths())}
        site_bodies = [
            path_to_row[frame.body_path] if frame.body_path is not None else WORLD_BODY_INDEX for frame in self._frames
        ]
        parent_bodies = [
            path_to_row[frame.parent_body_path] if frame.parent_body_path is not None else WORLD_BODY_INDEX
            for frame in self._frames
        ]

        device = self._device
        self._physics_bound = any(index != WORLD_BODY_INDEX for index in (*site_bodies, *parent_bodies))
        self._empty_body_q = wp.zeros(1, dtype=wp.transformf, device=device)
        self._site_body = wp.array(site_bodies, dtype=wp.int32, device=device)
        self._site_local = wp.array(
            [wp.transform(*frame.pose) for frame in self._frames], dtype=wp.transformf, device=device
        )
        self._parent_site_body = wp.array(parent_bodies, dtype=wp.int32, device=device)
        self._parent_site_local = wp.array(
            [wp.transform(*frame.parent_pose) for frame in self._frames], dtype=wp.transformf, device=device
        )
        self._scales = wp.array([wp.vec3f(*frame.scale) for frame in self._frames], dtype=wp.vec3f, device=device)

        count = len(self._frames)
        self._pos_buf = wp.zeros(count, dtype=wp.vec3f, device=device)
        self._quat_buf = wp.zeros(count, dtype=wp.vec4f, device=device)
        self._local_pos_buf = wp.zeros(count, dtype=wp.vec3f, device=device)
        self._local_quat_buf = wp.zeros(count, dtype=wp.vec4f, device=device)
        self._pos_ta = ProxyArray(self._pos_buf)
        self._quat_ta = ProxyArray(self._quat_buf)
        self._local_pos_ta = ProxyArray(self._local_pos_buf)
        self._local_quat_ta = ProxyArray(self._local_quat_buf)
        self._scales_ta = ProxyArray(self._scales)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def prims(self) -> list:
        """Return no USD handles; the view is bound entirely from the clone plan."""
        return []

    @property
    def prim_paths(self) -> list[str]:
        """Exact destination frame paths in clone-plan order."""
        return [frame.path for frame in self._frames]

    @property
    def count(self) -> int:
        """Number of sites in this view (one per binding row, or per matched prim in world-only mode)."""
        return len(self._frames)

    @property
    def device(self) -> str:
        """Device where arrays are allocated (``"cpu"`` or ``"cuda:0"``)."""
        return self._device

    def _current_body_q(self) -> wp.array:
        """Return the current rigid-body transforms through SDP."""
        transforms = self._scene_data_provider.request_transforms(SceneDataFormat.Transform)
        if transforms is None or transforms.transforms is None:
            if self._physics_bound:
                raise RuntimeError("OvPhysxFrameView requires initialized physics transforms.")
            return self._empty_body_q
        return transforms.transforms

    # ------------------------------------------------------------------
    # World / local pose APIs (Tasks 5 & 6)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # World poses
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Writer factory hooks (pass-through; OvPhysX has no separate Fabric storage)
    # ------------------------------------------------------------------

    def _make_world_space_writer(self) -> FrameViewWorldSpaceWriter:
        return _OvPhysxWorldSpaceWriter(self)

    def _make_local_space_writer(self) -> FrameViewLocalSpaceWriter:
        return _OvPhysxLocalSpaceWriter(self)

    # ------------------------------------------------------------------
    # Backend hooks
    # ------------------------------------------------------------------

    def _get_world_poses_impl(self, indices: wp.array | None = None) -> tuple[ProxyArray, ProxyArray]:
        """Get world-space positions and orientations."""
        body_q = self._current_body_q()

        if indices is not None:
            n = len(indices)
            pos_buf = wp.zeros(n, dtype=wp.vec3f, device=self._device)
            quat_buf = wp.zeros(n, dtype=wp.vec4f, device=self._device)
            wp.launch(
                _compute_site_world_transforms_indexed,
                dim=n,
                inputs=[body_q, self._site_body, self._site_local, indices],
                outputs=[pos_buf, quat_buf],
                device=self._device,
            )
            return ProxyArray(pos_buf), ProxyArray(quat_buf)

        wp.launch(
            _compute_site_world_transforms,
            dim=self.count,
            inputs=[body_q, self._site_body, self._site_local],
            outputs=[self._pos_buf, self._quat_buf],
            device=self._device,
        )
        return self._pos_ta, self._quat_ta

    def _apply_world_pose_write(
        self,
        positions: wp.array | None = None,
        orientations: wp.array | None = None,
        indices: wp.array | None = None,
    ) -> None:
        """Set world-space positions and/or orientations."""
        if positions is None and orientations is None:
            return
        body_q = self._current_body_q()

        if positions is None or orientations is None:
            cur_pos_ta, cur_quat_ta = self._get_world_poses_impl(indices)
            if positions is None:
                positions = cur_pos_ta.warp
            if orientations is None:
                orientations = cur_quat_ta.warp

        if indices is not None:
            wp.launch(
                _write_site_local_from_world_poses_indexed,
                dim=len(indices),
                inputs=[body_q, self._site_body, indices, positions, orientations, self._site_local],
                device=self._device,
            )
        else:
            wp.launch(
                _write_site_local_from_world_poses,
                dim=self.count,
                inputs=[body_q, self._site_body, positions, orientations, self._site_local],
                device=self._device,
            )

    # ------------------------------------------------------------------
    # Local poses (parent-relative)
    # ------------------------------------------------------------------

    def _get_local_poses_impl(self, indices: wp.array | None = None) -> tuple[ProxyArray, ProxyArray]:
        """Get parent-relative positions and orientations."""
        body_q = self._current_body_q()

        if indices is not None:
            n = len(indices)
            pos_buf = wp.zeros(n, dtype=wp.vec3f, device=self._device)
            quat_buf = wp.zeros(n, dtype=wp.vec4f, device=self._device)
            wp.launch(
                _compute_site_local_transforms_indexed,
                dim=n,
                inputs=[
                    body_q,
                    self._site_body,
                    self._site_local,
                    self._parent_site_body,
                    self._parent_site_local,
                    indices,
                ],
                outputs=[pos_buf, quat_buf],
                device=self._device,
            )
            return ProxyArray(pos_buf), ProxyArray(quat_buf)

        wp.launch(
            _compute_site_local_transforms,
            dim=self.count,
            inputs=[
                body_q,
                self._site_body,
                self._site_local,
                self._parent_site_body,
                self._parent_site_local,
            ],
            outputs=[self._local_pos_buf, self._local_quat_buf],
            device=self._device,
        )
        return self._local_pos_ta, self._local_quat_ta

    def _apply_local_pose_write(
        self,
        translations: wp.array | None = None,
        orientations: wp.array | None = None,
        indices: wp.array | None = None,
    ) -> None:
        """Set parent-relative translations and/or orientations."""
        if translations is None and orientations is None:
            return
        body_q = self._current_body_q()

        if translations is None or orientations is None:
            cur_pos_ta, cur_quat_ta = self._get_local_poses_impl(indices)
            if translations is None:
                translations = cur_pos_ta.warp
            if orientations is None:
                orientations = cur_quat_ta.warp

        if indices is not None:
            wp.launch(
                _write_site_local_from_local_poses_indexed,
                dim=len(indices),
                inputs=[
                    body_q,
                    self._site_body,
                    self._parent_site_body,
                    self._parent_site_local,
                    indices,
                    translations,
                    orientations,
                    self._site_local,
                ],
                device=self._device,
            )
        else:
            wp.launch(
                _write_site_local_from_local_poses,
                dim=self.count,
                inputs=[
                    body_q,
                    self._site_body,
                    self._parent_site_body,
                    self._parent_site_local,
                    translations,
                    orientations,
                    self._site_local,
                ],
                device=self._device,
            )

    # ------------------------------------------------------------------
    # View-owned scales
    # ------------------------------------------------------------------

    def _get_local_scales_impl(self, indices: wp.array | None = None) -> ProxyArray:
        """Get view-owned frame scales."""
        if indices is None:
            return self._scales_ta
        output = wp.empty(len(indices), dtype=wp.vec3f, device=self._device)
        wp.launch(
            _gather_scales,
            dim=len(indices),
            inputs=[self._scales, indices],
            outputs=[output],
            device=self._device,
        )
        return ProxyArray(output)

    def _get_world_scales_impl(self, indices: wp.array | None = None) -> ProxyArray:
        """Get view-owned frame scales."""
        return self._get_local_scales_impl(indices)

    def _apply_local_scale_write(self, scales: wp.array, indices: wp.array | None = None) -> None:
        """Set view-owned frame scales."""
        if indices is None:
            wp.copy(self._scales, scales)
        else:
            wp.launch(_scatter_scales, dim=len(indices), inputs=[self._scales, indices, scales], device=self._device)

    def _apply_world_scale_write(self, scales: wp.array, indices: wp.array | None = None) -> None:
        """Set view-owned frame scales."""
        self._apply_local_scale_write(scales, indices)


# ----------------------------------------------------------------------
# Pass-through writer classes
# ----------------------------------------------------------------------


class _OvPhysxWorldSpaceWriter(FrameViewWorldSpaceWriter):
    """OvPhysX world-space writer: pass-through to backend ``_apply_*`` hooks."""

    def set_poses(self, positions=None, orientations=None, indices=None) -> None:
        self._view._apply_world_pose_write(positions, orientations, indices)  # type: ignore[attr-defined]

    def set_scales(self, scales, indices=None) -> None:
        self._view._apply_world_scale_write(scales, indices)  # type: ignore[attr-defined]

    def get_poses(self, indices=None) -> tuple[ProxyArray, ProxyArray]:
        return self._view._get_world_poses_impl(indices)  # type: ignore[attr-defined]

    def get_scales(self, indices=None) -> ProxyArray:
        return self._view._get_world_scales_impl(indices)  # type: ignore[attr-defined]


class _OvPhysxLocalSpaceWriter(FrameViewLocalSpaceWriter):
    """OvPhysX local-space writer: pass-through to backend ``_apply_*`` hooks."""

    def set_poses(self, positions=None, orientations=None, indices=None) -> None:
        self._view._apply_local_pose_write(positions, orientations, indices)  # type: ignore[attr-defined]

    def set_scales(self, scales, indices=None) -> None:
        self._view._apply_local_scale_write(scales, indices)  # type: ignore[attr-defined]

    def get_poses(self, indices=None) -> tuple[ProxyArray, ProxyArray]:
        return self._view._get_local_poses_impl(indices)  # type: ignore[attr-defined]

    def get_scales(self, indices=None) -> ProxyArray:
        return self._view._get_local_scales_impl(indices)  # type: ignore[attr-defined]
