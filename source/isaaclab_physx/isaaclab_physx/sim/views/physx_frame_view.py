# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX FrameView over clone-plan frames and SDP transforms."""

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

_WORLD_BODY_INDEX = -1


@wp.kernel
def _get_world_poses(
    body_q: wp.array(dtype=wp.transformf),
    site_body: wp.array(dtype=wp.int32),
    site_local: wp.array(dtype=wp.transformf),
    indices: wp.array(dtype=wp.int32),
    positions: wp.array(dtype=wp.vec3f),
    orientations: wp.array(dtype=wp.vec4f),
):
    i = wp.tid()
    site = indices[i]
    body = site_body[site]
    world = site_local[site]
    if body != _WORLD_BODY_INDEX:
        world = wp.transform_multiply(body_q[body], world)
    positions[i] = wp.transform_get_translation(world)
    q = wp.transform_get_rotation(world)
    orientations[i] = wp.vec4f(q[0], q[1], q[2], q[3])


@wp.kernel
def _get_local_poses(
    body_q: wp.array(dtype=wp.transformf),
    site_body: wp.array(dtype=wp.int32),
    site_local: wp.array(dtype=wp.transformf),
    parent_body: wp.array(dtype=wp.int32),
    parent_local: wp.array(dtype=wp.transformf),
    indices: wp.array(dtype=wp.int32),
    positions: wp.array(dtype=wp.vec3f),
    orientations: wp.array(dtype=wp.vec4f),
):
    i = wp.tid()
    site = indices[i]
    body = site_body[site]
    world = site_local[site]
    if body != _WORLD_BODY_INDEX:
        world = wp.transform_multiply(body_q[body], world)
    parent = parent_local[site]
    parent_index = parent_body[site]
    if parent_index != _WORLD_BODY_INDEX:
        parent = wp.transform_multiply(body_q[parent_index], parent)
    local = wp.transform_multiply(wp.transform_inverse(parent), world)
    positions[i] = wp.transform_get_translation(local)
    q = wp.transform_get_rotation(local)
    orientations[i] = wp.vec4f(q[0], q[1], q[2], q[3])


@wp.kernel
def _set_world_poses(
    body_q: wp.array(dtype=wp.transformf),
    site_body: wp.array(dtype=wp.int32),
    indices: wp.array(dtype=wp.int32),
    positions: wp.array(dtype=wp.vec3f),
    orientations: wp.array(dtype=wp.vec4f),
    site_local: wp.array(dtype=wp.transformf),
):
    i = wp.tid()
    site = indices[i]
    q = orientations[i]
    world = wp.transform(positions[i], wp.quatf(q[0], q[1], q[2], q[3]))
    body = site_body[site]
    if body == _WORLD_BODY_INDEX:
        site_local[site] = world
    else:
        site_local[site] = wp.transform_multiply(wp.transform_inverse(body_q[body]), world)


@wp.kernel
def _set_local_poses(
    body_q: wp.array(dtype=wp.transformf),
    site_body: wp.array(dtype=wp.int32),
    parent_body: wp.array(dtype=wp.int32),
    parent_local: wp.array(dtype=wp.transformf),
    indices: wp.array(dtype=wp.int32),
    positions: wp.array(dtype=wp.vec3f),
    orientations: wp.array(dtype=wp.vec4f),
    site_local: wp.array(dtype=wp.transformf),
):
    i = wp.tid()
    site = indices[i]
    q = orientations[i]
    local = wp.transform(positions[i], wp.quatf(q[0], q[1], q[2], q[3]))
    parent = parent_local[site]
    parent_index = parent_body[site]
    if parent_index != _WORLD_BODY_INDEX:
        parent = wp.transform_multiply(body_q[parent_index], parent)
    world = wp.transform_multiply(parent, local)
    body = site_body[site]
    if body == _WORLD_BODY_INDEX:
        site_local[site] = world
    else:
        site_local[site] = wp.transform_multiply(wp.transform_inverse(body_q[body]), world)


@wp.kernel
def _gather_scales(
    scales: wp.array(dtype=wp.vec3f), indices: wp.array(dtype=wp.int32), output: wp.array(dtype=wp.vec3f)
):
    i = wp.tid()
    output[i] = scales[indices[i]]


@wp.kernel
def _scatter_scales(
    indices: wp.array(dtype=wp.int32), values: wp.array(dtype=wp.vec3f), scales: wp.array(dtype=wp.vec3f)
):
    i = wp.tid()
    scales[indices[i]] = values[i]


class PhysxFrameView(BaseFrameView):
    """Plan-bound PhysX frame view whose only dynamic input comes through SDP."""

    def __init__(
        self,
        prim_path: str | list[str],
        simulation_context: SimulationContext,
        device: str = "cpu",
        validate_xform_ops: bool = True,
    ):
        """Construct the view before the clone plan is finalized.

        Args:
            prim_path: Full-path expression or expressions for planned frames.
            simulation_context: Active simulation composition root.
            device: Warp device for view arrays.
            validate_xform_ops: Unused backend-agnostic argument.
        """
        del simulation_context, validate_xform_ops
        self._prim_paths = [prim_path] if isinstance(prim_path, str) else list(prim_path)
        self._device = device
        self._frames = ()
        self._count = 0

    def initialize(self, plan: ClonePlan, scene_data_provider: SceneDataProvider) -> None:
        """Bind exact planned frames and the shared scene-data provider."""
        if plan is None or not plan.is_complete:
            raise RuntimeError("PhysxFrameView requires a completed clone plan.")
        self._frames = tuple(frame for path in self._prim_paths for frame in plan.match_frames(path))
        if len({frame.path for frame in self._frames}) != len(self._frames):
            raise ValueError(f"Frame expressions overlap in {self._prim_paths!r}.")
        if any(frame.body_path == frame.path for frame in self._frames):
            raise ValueError("PhysxFrameView manages non-body frames; use a rigid-body view for physics bodies.")

        body_rows = {path: index for index, path in enumerate(plan.iter_rigid_body_paths())}
        try:
            site_bodies = [
                _WORLD_BODY_INDEX if frame.body_path is None else body_rows[frame.body_path] for frame in self._frames
            ]
            parent_bodies = [
                _WORLD_BODY_INDEX if frame.parent_body_path is None else body_rows[frame.parent_body_path]
                for frame in self._frames
            ]
        except KeyError as exc:
            raise ValueError(f"Planned frame body {exc.args[0]!r} has no SDP transform.") from exc

        self._scene_data_provider = scene_data_provider
        self._physics_bound = any(index != _WORLD_BODY_INDEX for index in (*site_bodies, *parent_bodies))
        self._empty_body_q = wp.zeros(1, dtype=wp.transformf, device=self._device)
        self._create_buffers(site_bodies, parent_bodies)

    @property
    def count(self) -> int:
        """Number of planned frames in the view."""
        return self._count

    @property
    def device(self) -> str:
        """Device where arrays are allocated."""
        return self._device

    @property
    def prims(self) -> list:
        """Return no USD handles; this view is bound entirely from the clone plan."""
        return []

    @property
    def prim_paths(self) -> list[str]:
        """Exact destination paths in plan order."""
        return [frame.path for frame in self._frames]

    def _make_world_space_writer(self) -> FrameViewWorldSpaceWriter:
        return _PhysxWorldSpaceWriter(self)

    def _make_local_space_writer(self) -> FrameViewLocalSpaceWriter:
        return _PhysxLocalSpaceWriter(self)

    def _get_world_poses_impl(self, indices: wp.array | None = None) -> tuple[ProxyArray, ProxyArray]:
        body_q = self._current_body_q()
        indices = self._indices if indices is None else indices
        count = len(indices)
        positions = self._positions if indices is self._indices else wp.empty(count, dtype=wp.vec3f, device=self.device)
        orientations = (
            self._orientations if indices is self._indices else wp.empty(count, dtype=wp.vec4f, device=self.device)
        )
        wp.launch(
            _get_world_poses,
            dim=count,
            inputs=[body_q, self._site_body, self._site_local, indices],
            outputs=[positions, orientations],
            device=self.device,
        )
        if indices is self._indices:
            return self._positions_proxy, self._orientations_proxy
        return ProxyArray(positions), ProxyArray(orientations)

    def _get_local_poses_impl(self, indices: wp.array | None = None) -> tuple[ProxyArray, ProxyArray]:
        body_q = self._current_body_q()
        indices = self._indices if indices is None else indices
        count = len(indices)
        positions = (
            self._local_positions if indices is self._indices else wp.empty(count, dtype=wp.vec3f, device=self.device)
        )
        orientations = (
            self._local_orientations
            if indices is self._indices
            else wp.empty(count, dtype=wp.vec4f, device=self.device)
        )
        wp.launch(
            _get_local_poses,
            dim=count,
            inputs=[
                body_q,
                self._site_body,
                self._site_local,
                self._parent_body,
                self._parent_local,
                indices,
            ],
            outputs=[positions, orientations],
            device=self.device,
        )
        if indices is self._indices:
            return self._local_positions_proxy, self._local_orientations_proxy
        return ProxyArray(positions), ProxyArray(orientations)

    def _get_world_scales_impl(self, indices: wp.array | None = None) -> ProxyArray:
        if indices is None:
            return self._scales_proxy
        output = wp.empty(len(indices), dtype=wp.vec3f, device=self.device)
        wp.launch(
            _gather_scales, dim=len(indices), inputs=[self._scales, indices], outputs=[output], device=self.device
        )
        return ProxyArray(output)

    def _get_local_scales_impl(self, indices: wp.array | None = None) -> ProxyArray:
        return self._get_world_scales_impl(indices)

    def _apply_world_pose_write(
        self,
        positions: wp.array | None = None,
        orientations: wp.array | None = None,
        indices: wp.array | None = None,
    ) -> None:
        if positions is None and orientations is None:
            return
        if positions is None or orientations is None:
            current_positions, current_orientations = self._get_world_poses_impl(indices)
            positions = current_positions.warp if positions is None else positions
            orientations = current_orientations.warp if orientations is None else orientations
        indices = self._indices if indices is None else indices
        wp.launch(
            _set_world_poses,
            dim=len(indices),
            inputs=[self._current_body_q(), self._site_body, indices, positions, orientations, self._site_local],
            device=self.device,
        )

    def _apply_local_pose_write(
        self,
        translations: wp.array | None = None,
        orientations: wp.array | None = None,
        indices: wp.array | None = None,
    ) -> None:
        if translations is None and orientations is None:
            return
        if translations is None or orientations is None:
            current_positions, current_orientations = self._get_local_poses_impl(indices)
            translations = current_positions.warp if translations is None else translations
            orientations = current_orientations.warp if orientations is None else orientations
        indices = self._indices if indices is None else indices
        wp.launch(
            _set_local_poses,
            dim=len(indices),
            inputs=[
                self._current_body_q(),
                self._site_body,
                self._parent_body,
                self._parent_local,
                indices,
                translations,
                orientations,
                self._site_local,
            ],
            device=self.device,
        )

    def _apply_world_scale_write(self, scales: wp.array, indices: wp.array | None = None) -> None:
        indices = self._indices if indices is None else indices
        wp.launch(_scatter_scales, dim=len(indices), inputs=[indices, scales, self._scales], device=self.device)

    def _apply_local_scale_write(self, scales: wp.array, indices: wp.array | None = None) -> None:
        self._apply_world_scale_write(scales, indices)

    def _current_body_q(self) -> wp.array:
        if not self._physics_bound:
            return self._empty_body_q
        transforms = self._scene_data_provider.request_transforms(SceneDataFormat.Transform)
        if transforms is None or transforms.transforms is None:
            raise RuntimeError("PhysxFrameView requires initialized physics transforms.")
        return transforms.transforms

    def _create_buffers(self, site_bodies: list[int], parent_bodies: list[int]) -> None:
        device = self.device
        self._count = len(self._frames)
        self._site_body = wp.array(site_bodies, dtype=wp.int32, device=device)
        self._parent_body = wp.array(parent_bodies, dtype=wp.int32, device=device)
        self._site_local = wp.array(
            [wp.transform(*frame.pose) for frame in self._frames], dtype=wp.transformf, device=device
        )
        self._parent_local = wp.array(
            [wp.transform(*frame.parent_pose) for frame in self._frames], dtype=wp.transformf, device=device
        )
        self._scales = wp.array([wp.vec3f(*frame.scale) for frame in self._frames], dtype=wp.vec3f, device=device)
        self._indices = wp.array(list(range(self.count)), dtype=wp.int32, device=device)
        self._positions = wp.empty(self.count, dtype=wp.vec3f, device=device)
        self._orientations = wp.empty(self.count, dtype=wp.vec4f, device=device)
        self._local_positions = wp.empty(self.count, dtype=wp.vec3f, device=device)
        self._local_orientations = wp.empty(self.count, dtype=wp.vec4f, device=device)
        self._positions_proxy = ProxyArray(self._positions)
        self._orientations_proxy = ProxyArray(self._orientations)
        self._local_positions_proxy = ProxyArray(self._local_positions)
        self._local_orientations_proxy = ProxyArray(self._local_orientations)
        self._scales_proxy = ProxyArray(self._scales)


class _PhysxWorldSpaceWriter(FrameViewWorldSpaceWriter):
    """Pass world-space operations to the plan-bound view."""

    def set_poses(self, positions=None, orientations=None, indices=None) -> None:
        self._view._apply_world_pose_write(positions, orientations, indices)

    def set_scales(self, scales, indices=None) -> None:
        self._view._apply_world_scale_write(scales, indices)

    def get_poses(self, indices=None) -> tuple[ProxyArray, ProxyArray]:
        return self._view._get_world_poses_impl(indices)

    def get_scales(self, indices=None) -> ProxyArray:
        return self._view._get_world_scales_impl(indices)


class _PhysxLocalSpaceWriter(FrameViewLocalSpaceWriter):
    """Pass local-space operations to the plan-bound view."""

    def set_poses(self, positions=None, orientations=None, indices=None) -> None:
        self._view._apply_local_pose_write(positions, orientations, indices)

    def set_scales(self, scales, indices=None) -> None:
        self._view._apply_local_scale_write(scales, indices)

    def get_poses(self, indices=None) -> tuple[ProxyArray, ProxyArray]:
        return self._view._get_local_poses_impl(indices)

    def get_scales(self, indices=None) -> ProxyArray:
        return self._view._get_local_scales_impl(indices)
