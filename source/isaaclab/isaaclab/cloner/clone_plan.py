# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The :class:`ClonePlan` value type and the constructors that build one.

A plan is the whole description of a replication layout: which prototypes exist, where each
one is cloned to, and which envs each one populates. It is built once, queried through
:mod:`~isaaclab.cloner.query`, and executed by :class:`~isaaclab.cloner.ReplicateSession`.

Two constructors cover the ways a layout is specified:

* :func:`make_clone_plan` — the layout is derived from an explicit asset-cfg manifest, expanding
  multi-asset spawners into per-variant prototypes.
* :func:`make_valid_clone_combinations` — restricts which variant combinations
  :func:`make_clone_plan` may draw from, weighted per combination.
"""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

import isaaclab.sim as sim_utils

from .cloner_cfg import DEFAULT_ENV_TEMPLATE, CloneCfg, InclusionSet, expand_env_regex_ns
from .cloner_strategies import sequential
from .path import match, relative_to


def _group_by_env(entries: Iterable[Any]) -> dict[int | None, tuple[Any, ...]]:
    return {env_id: tuple(group) for env_id, group in itertools.groupby(entries, key=lambda entry: entry.env_id)}


@dataclass(frozen=True, eq=False, slots=True)
class FrameLayout:
    """One exact or prototype transformable frame declared by a clone-plan row."""

    path: str
    """Destination prim path, with ``{}`` only on a replicated prototype."""

    source_path: str
    """Exact authored prototype prim path."""

    parent_path: str | None
    """Destination parent path, with ``{}`` only on a replicated prototype."""

    row: int
    """Clone-plan row that owns the frame."""

    env_id: int | None
    """Destination environment id, or ``None`` for a prototype/global frame."""

    body_path: str | None = None
    """Destination rigid-body ancestor, or ``None`` for a world-attached frame."""

    body_view_path: str | None = None
    """Native rigid-body view pattern, or ``None`` for a world-attached frame."""

    pose: tuple[float, float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    """Frame pose relative to ``body_path``, or in destination world space when body-less."""

    parent_body_path: str | None = None
    """Rigid-body ancestor of ``parent_path``, or ``None`` when world-attached."""

    parent_pose: tuple[float, float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    """Parent pose relative to ``parent_body_path``, or in destination world space when body-less."""

    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Authored local transform scale."""

    clone_mask: np.ndarray | None = None
    """Read-only populated clone columns for a prototype, or ``None`` for an exact/global frame."""


@dataclass(frozen=True, eq=False, slots=True)
class RigidBodyLayout:
    """One exact or prototype rigid body declared by a clone-plan row."""

    path: str
    """Destination prim path, with ``{}`` only on a replicated prototype."""

    view_path: str
    """Native-view path pattern derived from the row's destination template."""

    row: int
    """Clone-plan row that owns the body."""

    env_id: int | None
    """Destination environment id, or ``None`` for a prototype/global body."""

    name: str
    """Authored body name, including a schema name override when present."""

    source_path: str | None = None
    """Exact authored prototype rigid-body path."""

    contact_report: bool = False
    """Whether the prototype declares the PhysX contact-report schema."""

    clone_mask: np.ndarray | None = None
    """Read-only populated clone columns for a prototype, or ``None`` for an exact/global body."""


@dataclass(frozen=True, eq=False, slots=True)
class GeometryLayout:
    """One exact or prototype geometry declared from a requested clone-plan target."""

    path: str
    """Destination geometry path, with ``{}`` only on a replicated prototype."""

    source_path: str
    """Exact authored prototype geometry path."""

    row: int
    """Clone-plan row that owns the geometry."""

    vertices: np.ndarray
    """Scale-baked vertices [m], shape ``[N, 3]``."""

    faces: np.ndarray
    """Triangle vertex indices, shape ``[M, 3]``."""

    frame: FrameLayout
    """Frame that owns this geometry's fixed transform and rigid-body binding."""

    collision: bool = False
    """Whether the prototype geometry has collision enabled."""

    view_path: str = ""
    """Native-view path pattern derived from the row's destination template."""

    collision_approximation: str | None = None
    """Authored mesh-collision approximation, or ``None`` for non-mesh colliders."""

    heightfield: tuple[str, float] | None = None
    """Tagged source root and raster resolution [m], or ``None`` for ordinary geometry."""

    clone_mask: np.ndarray | None = None
    """Read-only populated clone columns for a prototype, or ``None`` for exact/global geometry."""


@dataclass(frozen=True, slots=True)
class NewtonActuatorLayout:
    """One parsed Newton actuator declaration owned by its target joint."""

    controller_class: type
    """Resolved Newton controller class."""

    controller_arguments: tuple[tuple[str, Any], ...]
    """Resolved immutable controller arguments."""

    component_arguments: tuple[tuple[type, tuple[tuple[str, Any], ...]], ...]
    """Resolved immutable delay and clamping component declarations."""


@dataclass(frozen=True, eq=False, slots=True)
class JointLayout:
    """One exact or prototype enabled articulation joint."""

    path: str
    """Destination joint path, with ``{}`` only on a replicated prototype."""

    name: str
    """Authored joint name, including a schema name override when present."""

    parent_path: str | None
    """Destination body-0 path, or ``None`` for a world attachment."""

    child_path: str
    """Destination child-body path, with ``{}`` only on a replicated prototype."""

    tendon_type: str | None
    """Tendon family declared on the joint: ``"fixed"``, ``"spatial"``, or ``None``."""

    pose: tuple[float, float, float, float, float, float, float]
    """Child-side joint pose as ``(tx, ty, tz, qx, qy, qz, qw)``."""

    newton_actuator: NewtonActuatorLayout | None = None
    """Parsed Newton actuator targeting this joint, or ``None``."""


@dataclass(frozen=True, eq=False, slots=True)
class ArticulationLayout:
    """One exact or prototype articulation root and its enabled child-side joints."""

    root_path: str
    """Destination articulation-root path, with ``{}`` only on a replicated prototype."""

    view_path: str
    """Native-view path pattern derived from the row's destination template."""

    row: int
    """Clone-plan row that owns the articulation."""

    joints: tuple[JointLayout, ...]
    """Enabled joints in prototype traversal order."""

    bodies: tuple[RigidBodyLayout, ...]
    """Connected rigid bodies in parent-before-child depth-first order."""

    clone_mask: np.ndarray | None = None
    """Read-only populated clone columns for a prototype, or ``None`` for an exact/global articulation."""


@dataclass(frozen=True, eq=False, slots=True)
class DeformableLayout:
    """One deformable body declared by a clone-plan row."""

    root_path: str
    """Exact deformable-body root path."""

    sim_mesh_path: str
    """Exact simulation-mesh prim path."""

    vis_mesh_path: str
    """Exact visual-mesh prim path updated from the published nodes."""

    view_path: str
    """Native-view path pattern derived from the row's destination template."""

    deformable_type: str
    """Native deformable view family, ``"volume"`` or ``"surface"``."""

    vertex_count: int
    """Unpadded simulation-node count."""

    vis_vertex_count: int
    """Number of vertices in the mesh drawn for this body."""

    point_indices: np.ndarray | None
    """Per-visual-vertex simulation-node indices, or ``None`` for a direct slice."""

    point_weights: np.ndarray | None
    """Per-visual-vertex interpolation weights, or ``None`` for a direct slice."""

    row: int
    """Clone-plan row that owns the body."""

    env_id: int | None
    """Destination environment id, or ``None`` for a global body."""

    material_path: str | None = None
    """Exact bound deformable-material path, or ``None`` when no material is bound."""

    material_view_path: str | None = None
    """Native material-view path pattern, or ``None`` when no material is bound."""

    source_path: str | None = None
    """Exact authored prototype deformable-root path."""

    vertices: np.ndarray | None = None
    """Simulation vertices baked into the deformable parent's frame [m], shape ``[N, 3]``."""

    indices: np.ndarray | None = None
    """Surface triangle or volume tetrahedron indices."""

    material_attributes: tuple[tuple[str, Any], ...] = ()
    """Authored deformable-material attribute values."""


@dataclass(frozen=True, eq=False, slots=True)
class CableLayout:
    """One open linear cable declared by a clone-plan row."""

    path: str
    """Exact destination ``BasisCurves`` prim path."""

    view_path: str
    """Native-view path pattern derived from the row's destination template."""

    segment_count: int
    """Number of simulated cable segments."""

    row: int
    """Clone-plan row that owns the cable."""

    env_id: int | None
    """Destination environment id, or ``None`` for a global cable."""


@dataclass(frozen=True, eq=False, slots=True)
class PointCloudLayout:
    """One authored ``UsdGeom.Points`` prim declared by a clone-plan row."""

    path: str
    """Exact destination point-cloud path."""

    count: int
    """Number of authored points."""

    row: int
    """Clone-plan row that owns the point cloud."""

    env_id: int | None
    """Destination environment id, or ``None`` for a global point cloud."""


@dataclass(frozen=True, eq=False, slots=True)
class PointBinding:
    """One plan-owned mapping from a native point slice to a drawable point array."""

    path: str
    """Exact drawable destination path."""

    source_offset: int
    """First point in the native SDP publication."""

    source_count: int
    """Number of native published points owned by this destination."""

    output_offset: int
    """First point in the flattened drawable output."""

    output_count: int
    """Number of points in the drawable destination."""

    source_indices: np.ndarray | None
    """Local native indices gathered for each output, or ``None`` for a direct slice."""

    weights: np.ndarray | None
    """Interpolation weights for each output, or ``None`` for a direct slice."""


@dataclass(frozen=True, eq=False, slots=True)
class SurfaceGripperLayout:
    """One Isaac SurfaceGripper declared by a clone-plan row."""

    path: str
    """Exact destination gripper path."""

    view_path: str
    """Native-view path pattern derived from the row's destination template."""

    row: int
    """Clone-plan row that owns the gripper."""

    env_id: int | None
    """Destination environment id, or ``None`` for a global gripper."""

    max_grip_distance: float | None
    """Authored maximum grip distance [m]."""

    coaxial_force_limit: float | None
    """Authored coaxial force limit [N]."""

    shear_force_limit: float | None
    """Authored shear force limit [N]."""

    retry_interval: float | None
    """Authored retry interval [s]."""


@dataclass(frozen=True, eq=False)
class ClonePlan:
    """Immutable replication plan, completed once with its authored prototype topology."""

    sources: tuple[str, ...]
    """Source prim paths, one per replication row."""

    destinations: tuple[str, ...]
    """Destination path templates with ``"{}"`` for the env id, one per row."""

    clone_mask: np.ndarray
    """Boolean array ``[len(sources), num_clones]``; ``True`` if env ``j`` comes from row ``i``."""

    env_ids: np.ndarray | None = None
    """Integer array ``[num_clones]`` of target env ids.

    Optional for plans used only with :func:`~isaaclab.cloner.query.iter_sources` or
    :func:`~isaaclab.cloner.query.path_to_source`; required by the replication session.
    """

    positions: np.ndarray | None = None
    """Per-env world positions [m], shape ``[num_clones, 3]``, or ``None``."""

    env_template: str = DEFAULT_ENV_TEMPLATE
    """Environment path template whose ``{}`` marks the environment index."""

    cfg_rows: dict[int, tuple[int, ...]] = field(default_factory=dict)
    """``id(cfg)`` to the row indices the cfg owns."""

    global_paths: tuple[str, ...] = ()
    """Unique shared-scene roots represented by non-replicated rows."""

    semantic_tags: tuple[tuple[tuple[str, str], ...], ...] = ()
    """Cfg-declared semantic tags, one tuple per replication row.

    An empty tuple means the plan carries no semantic metadata. Consumers must not recover missing
    labels by inspecting the cloned stage: semantics embedded only in an asset file are outside the
    plan contract.
    """

    geometry_requests: tuple[str, ...] = ()
    """Target expressions whose geometry the completed plan must declare."""

    root_layer_identifier: str | None = None
    """Root layer that owns the planned scene, captured by the replication session."""

    is_complete: bool = False
    """Whether authored prototype topology has been declared on this plan."""

    frame_prototypes: tuple[FrameLayout, ...] = ()
    """Transformable prototype facts with destination templates and clone-column masks."""

    rigid_body_prototypes: tuple[RigidBodyLayout, ...] = ()
    """Rigid-body prototype facts in row then articulation-topology order."""

    geometry_prototypes: tuple[GeometryLayout, ...] = ()
    """Requested geometry prototype facts with destination templates and clone-column masks."""

    articulation_prototypes: tuple[ArticulationLayout, ...] = ()
    """Articulation prototype facts in clone-plan row order."""

    deformables: tuple[DeformableLayout, ...] = ()
    """Deformable bodies in global then clone-plan environment order."""

    cables: tuple[CableLayout, ...] = ()
    """Cables in global then clone-plan environment order."""

    point_clouds: tuple[PointCloudLayout, ...] = ()
    """Authored point clouds in global then clone-plan environment order."""

    surface_grippers: tuple[SurfaceGripperLayout, ...] = ()
    """Surface grippers in global then clone-plan environment order."""

    _env_ids_cpu: tuple[int, ...] = ()
    """Environment ids cached on the host in clone-plan column order."""

    _positions_cpu: np.ndarray | None = None
    """Clone positions [m] in environment-column order, shape ``[num_clones, 3]``."""

    _source_positions: tuple[tuple[float, float, float] | None, ...] = ()
    """Source environment position [m] for each clone-plan row."""

    def iter_rigid_body_paths(self) -> Iterable[str]:
        """Yield exact rigid-body paths in canonical SDP transform order."""
        self._require_complete()
        for prototype in self.rigid_body_prototypes:
            if prototype.clone_mask is None:
                yield prototype.path
        for column, env_id in enumerate(self._env_ids_cpu):
            for prototype in self.rigid_body_prototypes:
                if prototype.clone_mask is not None and prototype.clone_mask[column]:
                    yield prototype.path.format(env_id)

    def _match_frame_prototypes(self, path_expr: str) -> tuple[tuple[FrameLayout, np.ndarray | None], ...]:
        """Return prototype frames and only the clone columns selected by ``path_expr``."""
        self._require_complete()
        template_matches: dict[str, tuple[re.Pattern, np.ndarray | None] | None] = {}
        matched = []
        for prototype in self.frame_prototypes:
            destination = self.destinations[prototype.row]
            if destination not in template_matches:
                template_match = match(path_expr, destination)
                if template_match is None:
                    template_matches[destination] = None
                elif "{}" not in destination:
                    template_matches[destination] = (re.compile(template_match.suffix), None)
                else:
                    instance = re.compile(template_match.instance)
                    instance_mask = np.fromiter(
                        (instance.fullmatch(str(env_id)) is not None for env_id in self._env_ids_cpu),
                        dtype=np.bool_,
                        count=len(self._env_ids_cpu),
                    )
                    if np.all(instance_mask):
                        instance_mask = None
                    else:
                        instance_mask.setflags(write=False)
                    template_matches[destination] = (re.compile(template_match.suffix), instance_mask)
            template_match = template_matches[destination]
            if template_match is None:
                continue
            suffix_pattern, instance_mask = template_match
            suffix = relative_to(prototype.path, destination)
            if suffix is None or suffix_pattern.fullmatch(suffix) is None:
                continue
            if prototype.clone_mask is None:
                matched.append((prototype, None))
            else:
                clone_mask = prototype.clone_mask if instance_mask is None else prototype.clone_mask & instance_mask
                if np.any(clone_mask):
                    if instance_mask is not None:
                        clone_mask.setflags(write=False)
                    matched.append((prototype, clone_mask))
        return tuple(matched)

    def _materialize_frame(self, prototype: FrameLayout, column: int | None) -> FrameLayout:
        """Materialize one exact frame from a prototype and optional clone column."""
        env_id = None if column is None else self._env_ids_cpu[column]

        def exact(path: str | None) -> str | None:
            return None if path is None or env_id is None else path.format(env_id)

        path = prototype.path if env_id is None else prototype.path.format(env_id)
        parent_path = prototype.parent_path if env_id is None else exact(prototype.parent_path)
        body_path = prototype.body_path if env_id is None else exact(prototype.body_path)
        parent_body_path = prototype.parent_body_path if env_id is None else exact(prototype.parent_body_path)
        pose = prototype.pose
        parent_pose = prototype.parent_pose
        needs_projection = prototype.body_path is None or (
            prototype.parent_path is not None and prototype.parent_body_path is None
        )
        if column is not None and self._positions_cpu is not None and needs_projection:
            source_position = self._source_positions[prototype.row]
            if source_position is None:
                raise ValueError(f"Cannot project planned frame {path!r} into its destination world.")
            delta = self._positions_cpu[column, :3] - source_position
            if prototype.body_path is None:
                pose = (*map(float, np.asarray(pose[:3]) + delta), *pose[3:])
            if prototype.parent_path is not None and prototype.parent_body_path is None:
                parent_pose = (*map(float, np.asarray(parent_pose[:3]) + delta), *parent_pose[3:])
        return replace(
            prototype,
            path=path,
            parent_path=parent_path,
            env_id=env_id,
            body_path=body_path,
            pose=pose,
            parent_body_path=parent_body_path,
            parent_pose=parent_pose,
            clone_mask=None,
        )

    def _materialize_rigid_body(self, prototype: RigidBodyLayout, column: int | None) -> RigidBodyLayout:
        """Materialize one exact rigid body from a prototype and optional clone column."""
        env_id = None if column is None else self._env_ids_cpu[column]
        return replace(
            prototype,
            path=prototype.path if env_id is None else prototype.path.format(env_id),
            env_id=env_id,
            clone_mask=None,
        )

    def _materialize_articulation(self, prototype: ArticulationLayout, column: int | None) -> ArticulationLayout:
        """Materialize one exact articulation and its joints and bodies."""
        env_id = None if column is None else self._env_ids_cpu[column]

        def exact(path: str | None) -> str | None:
            return path if path is None or env_id is None else path.format(env_id)

        return replace(
            prototype,
            root_path=exact(prototype.root_path),
            joints=tuple(
                replace(
                    joint,
                    path=exact(joint.path),
                    parent_path=exact(joint.parent_path),
                    child_path=exact(joint.child_path),
                )
                for joint in prototype.joints
            ),
            bodies=tuple(self._materialize_rigid_body(body, column) for body in prototype.bodies),
            clone_mask=None,
        )

    def _materialize_geometry(self, prototype: GeometryLayout, column: int | None) -> GeometryLayout:
        """Materialize one exact geometry and its already-declared frame."""
        env_id = None if column is None else self._env_ids_cpu[column]
        return replace(
            prototype,
            path=prototype.path if env_id is None else prototype.path.format(env_id),
            frame=self._materialize_frame(prototype.frame, column),
            clone_mask=None,
        )

    @property
    def point_stream_names(self) -> tuple[str, ...]:
        """Return point streams declared by the plan."""
        self._require_complete()
        return tuple(
            name
            for name, populated in (("points", self.deformables or self.point_clouds), ("cables", self.cables))
            if populated
        )

    def point_bindings(self, name: str = "points") -> tuple[PointBinding, ...]:
        """Return native-to-drawable point mappings in canonical plan order."""
        self._require_complete()
        if name == "points":
            entries = [
                (
                    entry.vis_mesh_path,
                    entry.vertex_count,
                    entry.vis_vertex_count,
                    entry.point_indices,
                    entry.point_weights,
                    entry.row,
                    entry.env_id,
                    0 if entry.deformable_type == "surface" else 1,
                    index,
                )
                for index, entry in enumerate(self.deformables)
            ]
            for index, entry in enumerate(self.point_clouds):
                entries.append((entry.path, entry.count, entry.count, None, None, entry.row, entry.env_id, 2, index))
        elif name == "cables":
            entries = []
            for index, entry in enumerate(self.cables):
                count = entry.segment_count + 1
                entries.append((entry.path, count, count, None, None, entry.row, entry.env_id, 0, index))
        else:
            raise KeyError(f"Clone plan declares no {name!r} point stream.")
        if not entries:
            return ()

        env_order = {env_id: index for index, env_id in enumerate(self._env_ids_cpu)}

        def order(entry) -> tuple[int, int, int, int]:
            path, _source_count, _output_count, _indices, _weights, row, env_id, family, index = entry
            if env_id is not None and env_id not in env_order:
                raise ValueError(f"Point destination {path!r} has undeclared environment id {env_id}.")
            return (-1 if env_id is None else env_order[env_id], row, family, index)

        source_offset = 0
        output_offset = 0
        bindings = []
        for path, source_count, output_count, indices, weights, *_ in sorted(entries, key=order):
            if source_count <= 0 or output_count <= 0:
                raise ValueError(f"Point destination {path!r} must contain at least one point.")
            if indices is None or weights is None:
                if indices is not None or weights is not None or source_count != output_count:
                    raise ValueError(f"Direct point binding {path!r} must preserve its native point count.")
            else:
                indices = np.asarray(indices, dtype=np.int32)
                weights = np.asarray(weights, dtype=np.float32)
                if indices.shape != (output_count, 4) or weights.shape != (output_count, 4):
                    raise ValueError(f"Point mapping for {path!r} does not match its {output_count} outputs.")
                if np.any(indices < 0) or np.any(indices >= source_count):
                    raise ValueError(f"Point mapping for {path!r} reads outside its native source slice.")
                if not np.all(np.isfinite(weights)) or not np.allclose(weights.sum(axis=1), 1.0, atol=1.0e-5):
                    raise ValueError(f"Point mapping for {path!r} has invalid interpolation weights.")
            bindings.append(
                PointBinding(
                    path,
                    source_offset,
                    source_count,
                    output_offset,
                    output_count,
                    indices,
                    weights,
                )
            )
            source_offset += source_count
            output_offset += output_count
        if len({binding.path for binding in bindings}) != len(bindings):
            raise ValueError("Clone plan declares a point destination more than once.")
        return tuple(bindings)

    def match_frames(self, path_expr: str) -> tuple[FrameLayout, ...]:
        """Return the exact planned frames matching a full-path regular expression."""
        frames = []
        for prototype, clone_mask in self._match_frame_prototypes(path_expr):
            if clone_mask is None:
                frames.append(self._materialize_frame(prototype, None))
            else:
                frames.extend(self._materialize_frame(prototype, int(column)) for column in np.flatnonzero(clone_mask))
        if not frames:
            raise ValueError(f"Frame expression {path_expr!r} is not covered by the clone plan.")
        env_order = {env_id: column for column, env_id in enumerate(self._env_ids_cpu)}
        return tuple(
            sorted(frames, key=lambda frame: (-1 if frame.env_id is None else env_order[frame.env_id], frame.row))
        )

    def match_rigid_body_subtrees(self, path_expr: str) -> tuple[RigidBodyLayout, ...]:
        """Return the single planned rigid body inside every matched frame."""
        matched = []
        env_order = {env_id: column for column, env_id in enumerate(self._env_ids_cpu)}
        for frame in self.match_frames(path_expr):
            column = None if frame.env_id is None else env_order[frame.env_id]
            candidates = []
            for prototype in self.rigid_body_prototypes:
                if column is None:
                    if prototype.clone_mask is not None:
                        continue
                elif prototype.clone_mask is None or not prototype.clone_mask[column]:
                    continue
                path = prototype.path if column is None else prototype.path.format(frame.env_id)
                if path == frame.path or path.startswith(frame.path + "/"):
                    candidates.append(self._materialize_rigid_body(prototype, column))
            if len(candidates) != 1:
                raise ValueError(
                    f"Frame {frame.path!r} contains {len(candidates)} planned rigid bodies; expected exactly one."
                )
            matched.append(candidates[0])
        if len(matched) != len({entry.path for entry in matched}):
            raise ValueError(f"Rigid-body expression {path_expr!r} resolves the same body more than once.")
        return tuple(matched)

    def match_rigid_body(self, path_expr: str) -> RigidBodyLayout:
        """Return one compatible rigid-body prototype without expanding its clones."""
        targets = self._match_frame_prototypes(path_expr)
        if not targets:
            raise ValueError(f"Frame expression {path_expr!r} is not covered by the clone plan.")
        matched: list[tuple[RigidBodyLayout, np.ndarray | None]] = []
        selected = np.zeros(len(self._env_ids_cpu), dtype=np.int16)
        covered = np.zeros_like(selected)
        selected_global = covered_global = 0
        for frame, frame_mask in targets:
            selected_global += frame_mask is None
            if frame_mask is not None:
                selected += frame_mask
            for body in self.rigid_body_prototypes:
                if relative_to(body.path, frame.path) is None:
                    continue
                if frame_mask is None and body.clone_mask is None:
                    covered_global += 1
                    matched.append((body, None))
                elif frame_mask is not None and body.clone_mask is not None:
                    active = frame_mask & body.clone_mask
                    if np.any(active):
                        covered += active
                        matched.append((body, active))
        if selected_global > 1 or (selected_global and np.any(selected)) or np.any(selected > 1):
            raise ValueError(f"Rigid-body expression {path_expr!r} selects multiple frames per clone.")
        if covered_global != selected_global or not np.array_equal(covered, selected):
            raise ValueError(
                f"Rigid-body expression {path_expr!r} does not resolve exactly one body per selected clone."
            )
        if len({body.view_path for body, _ in matched}) != 1:
            raise ValueError(f"Rigid object {path_expr!r} has incompatible clone-plan body paths.")
        return min(matched, key=lambda entry: (-1 if entry[1] is None else int(entry[1].argmax()), entry[0].row))[0]

    def match_contact_bodies(self, path_expr: str) -> tuple[RigidBodyLayout, ...]:
        """Return contact-reporting bodies selected by a parent path and descendant leaf expression."""
        separator = re.sub(r"\[\^?[^]]*\]", lambda result: "\0" * len(result.group()), path_expr).rfind("/")
        pattern = re.compile(path_expr[:separator] + r"/(?:[^/]+/)*" + path_expr[separator + 1 :])
        matched = tuple(
            self._materialize_rigid_body(body, column)
            for column in (None, *range(len(self._env_ids_cpu)))
            for body in self.rigid_body_prototypes
            if body.contact_report
            and (column is None) == (body.clone_mask is None)
            and (column is None or body.clone_mask[column])
            and pattern.fullmatch(body.path if column is None else body.path.format(self._env_ids_cpu[column]))
        )
        if not matched:
            raise ValueError(f"Contact expression {path_expr!r} matches no planned contact-reporting body.")
        return matched

    def match_geometry_prototypes(
        self, path_expr: str
    ) -> tuple[tuple[FrameLayout, tuple[GeometryLayout, ...], np.ndarray | None], ...]:
        """Return matching target prototypes, their geometry prototypes, and selected clone masks."""
        groups = []
        targets = self._match_frame_prototypes(path_expr)
        if not targets:
            raise ValueError(f"Frame expression {path_expr!r} is not covered by the clone plan.")
        for target, target_mask in targets:
            geometries = tuple(
                geometry
                for geometry in self.geometry_prototypes
                if relative_to(geometry.path, target.path) is not None
                and (
                    (target_mask is None and geometry.clone_mask is None)
                    or (
                        target_mask is not None
                        and geometry.clone_mask is not None
                        and np.any(target_mask & geometry.clone_mask)
                    )
                )
            )
            if not geometries:
                raise ValueError(f"Ray-cast target {target.path!r} has no planned geometry.")
            groups.append((target, geometries, target_mask))
        return tuple(groups)

    def match_geometry_targets(self, path_expr: str) -> tuple[tuple[FrameLayout, tuple[GeometryLayout, ...]], ...]:
        """Return exact target frames and their planned descendant geometry."""
        groups = []
        seen: set[str] = set()
        env_order = {env_id: column for column, env_id in enumerate(self._env_ids_cpu)}
        for target in self.match_frames(path_expr):
            column = None if target.env_id is None else env_order[target.env_id]
            geometries = []
            for prototype in self.geometry_prototypes:
                if column is None:
                    if prototype.clone_mask is not None:
                        continue
                elif prototype.clone_mask is None or not prototype.clone_mask[column]:
                    continue
                path = prototype.path if column is None else prototype.path.format(target.env_id)
                if path == target.path or path.startswith(target.path + "/"):
                    geometries.append(self._materialize_geometry(prototype, column))
            if not geometries:
                raise ValueError(f"Ray-cast target {target.path!r} has no planned geometry.")
            duplicate = seen.intersection(entry.path for entry in geometries)
            if duplicate:
                raise ValueError(f"Ray-cast expression {path_expr!r} overlaps target geometry {sorted(duplicate)!r}.")
            seen.update(entry.path for entry in geometries)
            groups.append((target, tuple(geometries)))
        return tuple(groups)

    def match_articulations(self, path_expr: str) -> tuple[ArticulationLayout, ...]:
        """Return the single planned articulation inside each matched frame."""
        matched = []
        env_order = {env_id: column for column, env_id in enumerate(self._env_ids_cpu)}
        for frame in self.match_frames(path_expr):
            column = None if frame.env_id is None else env_order[frame.env_id]
            candidates = []
            for prototype in self.articulation_prototypes:
                if column is None:
                    if prototype.clone_mask is not None:
                        continue
                elif prototype.clone_mask is None or not prototype.clone_mask[column]:
                    continue
                root_path = prototype.root_path if column is None else prototype.root_path.format(frame.env_id)
                if root_path == frame.path or root_path.startswith(frame.path + "/"):
                    candidates.append(self._materialize_articulation(prototype, column))
            if len(candidates) != 1:
                raise ValueError(
                    f"Frame {frame.path!r} contains {len(candidates)} planned articulations; expected exactly one."
                )
            matched.append(candidates[0])
        if len(matched) != len({entry.root_path for entry in matched}):
            raise ValueError(f"Articulation expression {path_expr!r} resolves the same root more than once.")
        if len({entry.view_path for entry in matched}) != 1:
            raise ValueError(f"Articulation expression {path_expr!r} has incompatible clone-plan root paths.")
        return tuple(matched)

    def match_articulation(self, path_expr: str) -> ArticulationLayout:
        """Return one compatible representative articulation without expanding its clones."""
        targets = self._match_frame_prototypes(path_expr)
        if not targets:
            raise ValueError(f"Frame expression {path_expr!r} is not covered by the clone plan.")
        matched: list[tuple[ArticulationLayout, np.ndarray | None]] = []
        selected = np.zeros(len(self._env_ids_cpu), dtype=np.int16)
        covered = np.zeros_like(selected)
        selected_global = covered_global = 0
        for frame, frame_mask in targets:
            selected_global += frame_mask is None
            if frame_mask is not None:
                selected += frame_mask
            for articulation in self.articulation_prototypes:
                if relative_to(articulation.root_path, frame.path) is None:
                    continue
                if frame_mask is None and articulation.clone_mask is None:
                    covered_global += 1
                    matched.append((articulation, None))
                elif frame_mask is not None and articulation.clone_mask is not None:
                    active = frame_mask & articulation.clone_mask
                    if np.any(active):
                        covered += active
                        matched.append((articulation, active))
        if selected_global > 1 or (selected_global and np.any(selected)) or np.any(selected > 1):
            raise ValueError(f"Articulation expression {path_expr!r} selects multiple frames per clone.")
        if covered_global != selected_global or not np.array_equal(covered, selected):
            raise ValueError(
                f"Articulation expression {path_expr!r} does not resolve exactly one articulation per selected clone."
            )
        if len({articulation.view_path for articulation, _ in matched}) != 1:
            raise ValueError(f"Articulation expression {path_expr!r} has incompatible clone-plan root paths.")

        signatures = {
            tuple((joint.name, joint.newton_actuator) for joint in articulation.joints if joint.newton_actuator)
            for articulation, _ in matched
        }
        if len(signatures) != 1:
            raise ValueError(f"Articulation expression {path_expr!r} has incompatible Newton actuator declarations.")

        prototype, clone_mask = min(
            matched,
            key=lambda entry: (-1 if entry[1] is None else int(entry[1].argmax()), entry[0].row),
        )
        column = None if clone_mask is None else int(clone_mask.argmax())
        return self._materialize_articulation(prototype, column)

    def match_deformables(
        self,
        deformable_type: str,
        paths: Sequence[str],
        entries: Sequence[DeformableLayout] | None = None,
    ) -> tuple[DeformableLayout, ...]:
        """Return declared deformables in native-view order, accepting only exact root or mesh paths."""
        self._require_complete()
        if entries is None:
            entries = tuple(entry for entry in self.deformables if entry.deformable_type == deformable_type)
        else:
            entries = tuple(entries)
        if any(entry.deformable_type != deformable_type for entry in entries):
            raise ValueError(f"Expected only {deformable_type} deformables.")
        by_path: dict[str, DeformableLayout] = {}
        for entry in entries:
            for path in (entry.root_path, entry.sim_mesh_path, entry.vis_mesh_path):
                owner = by_path.setdefault(path, entry)
                if owner is not entry:
                    raise ValueError(f"Clone plan assigns deformable path {path!r} to multiple bodies.")
        try:
            ordered = tuple(by_path[path] for path in paths)
        except KeyError as exc:
            raise ValueError(f"Native {deformable_type} view returned undeclared path {exc.args[0]!r}.") from exc
        roots = tuple(entry.root_path for entry in ordered)
        expected = {entry.root_path for entry in entries}
        if len(roots) != len(set(roots)) or set(roots) != expected:
            raise ValueError(f"Native {deformable_type} view does not match the clone plan: {tuple(paths)!r}.")
        return ordered

    def match_deformable_subtrees(self, path_expr: str) -> tuple[DeformableLayout, ...]:
        """Return the single planned deformable body inside every matched frame."""
        matched = []
        deformables_by_env = _group_by_env(self.deformables)
        for frame in self.match_frames(path_expr):
            candidates = tuple(
                entry
                for entry in deformables_by_env.get(frame.env_id, ())
                if entry.root_path == frame.path or entry.root_path.startswith(frame.path + "/")
            )
            if len(candidates) != 1:
                raise ValueError(
                    f"Frame {frame.path!r} contains {len(candidates)} planned deformables; expected exactly one."
                )
            matched.append(candidates[0])
        if len(matched) != len({entry.root_path for entry in matched}):
            raise ValueError(f"Deformable expression {path_expr!r} resolves the same body more than once.")
        facts = {(entry.view_path, entry.deformable_type, entry.material_view_path) for entry in matched}
        if len(facts) != 1:
            raise ValueError(f"Deformable expression {path_expr!r} has incompatible clone-plan views.")
        return tuple(matched)

    def match_surface_grippers(self, path_expr: str) -> tuple[SurfaceGripperLayout, ...]:
        """Return exact planned surface grippers matching a full-path regular expression."""
        self._require_complete()
        pattern = re.compile(path_expr)
        matched = tuple(entry for entry in self.surface_grippers if pattern.fullmatch(entry.path) is not None)
        if not matched:
            raise ValueError(f"Surface-gripper expression {path_expr!r} is not covered by the clone plan.")
        if len({entry.view_path for entry in matched}) != 1:
            raise ValueError(f"Surface-gripper expression {path_expr!r} has incompatible clone-plan paths.")
        return matched

    def match_cables(self, path_expr: str) -> tuple[CableLayout, ...]:
        """Return the single planned cable inside every matched frame."""
        matched = []
        cables_by_env = _group_by_env(self.cables)
        for frame in self.match_frames(path_expr):
            candidates = tuple(
                entry
                for entry in cables_by_env.get(frame.env_id, ())
                if entry.path == frame.path or entry.path.startswith(frame.path + "/")
            )
            if len(candidates) != 1:
                raise ValueError(
                    f"Frame {frame.path!r} contains {len(candidates)} planned cables; expected exactly one."
                )
            matched.append(candidates[0])
        if len(matched) != len({entry.path for entry in matched}):
            raise ValueError(f"Cable expression {path_expr!r} resolves the same cable more than once.")
        if len({entry.view_path for entry in matched}) != 1 or len({entry.segment_count for entry in matched}) != 1:
            raise ValueError(f"Cable expression {path_expr!r} has incompatible clone-plan topology.")
        return tuple(matched)

    def _require_complete(self) -> None:
        """Reject topology queries before prototype authoring completes."""
        if not self.is_complete:
            raise RuntimeError("Clone-plan topology is unavailable before prototype authoring completes.")


def grid_transforms(N: int, spacing: float = 1.0, up_axis: str = "z") -> tuple[np.ndarray, np.ndarray]:
    """Create centered grid transforms as host arrays.

    Args:
        N: Number of instances.
        spacing: Distance between neighboring grid positions [m].
        up_axis: Up axis for positions (``"z"``, ``"y"``, or ``"x"``).

    Returns:
        Positions [m], shape ``[N, 3]``, and identity xyzw orientations, shape ``[N, 4]``.
    """
    num_rows = int(math.ceil(N / math.sqrt(N)))
    num_cols = int(math.ceil(N / num_rows))
    ii, jj = np.meshgrid(np.arange(num_rows, dtype=np.float32), np.arange(num_cols, dtype=np.float32), indexing="ij")
    ii = ii.reshape(-1)[:N]
    jj = jj.reshape(-1)[:N]
    x = -(ii - (num_rows - 1) / 2) * spacing
    y = (jj - (num_cols - 1) / 2) * spacing
    zero = np.zeros(N, dtype=np.float32)
    if up_axis.lower() == "z":
        positions = np.stack((x, y, zero), axis=1)
    elif up_axis.lower() == "y":
        positions = np.stack((x, zero, y), axis=1)
    else:
        positions = np.stack((zero, x, y), axis=1)
    orientations = np.zeros((N, 4), dtype=np.float32)
    orientations[:, 3] = 1.0
    return positions.astype(np.float32, copy=False), orientations


def num_spawn_variants(spawn_cfg: Any) -> int:
    """Return the number of variants declared by one spawner configuration."""
    if isinstance(spawn_cfg, sim_utils.MultiAssetSpawnerCfg):
        return len(spawn_cfg.assets_cfg)
    if isinstance(spawn_cfg, sim_utils.MultiUsdFileCfg):
        return 1 if isinstance(spawn_cfg.usd_path, str) else len(spawn_cfg.usd_path)
    return 1


def _variant_semantic_tags(spawn_cfg: Any) -> tuple[tuple[tuple[str, str], ...], ...]:
    """Return the semantic tags each spawn variant authors on its planned root."""
    if isinstance(spawn_cfg, sim_utils.MultiAssetSpawnerCfg):
        variants = spawn_cfg.assets_cfg
        shared = getattr(spawn_cfg, "semantic_tags", None) or []
    else:
        variants = [spawn_cfg]
        shared = []
    if isinstance(spawn_cfg, sim_utils.MultiUsdFileCfg):
        variants = [spawn_cfg] * num_spawn_variants(spawn_cfg)
    return tuple(
        tuple((semantic_type.replace(" ", "_"), value.replace(" ", "_")) for semantic_type, value in tags)
        for variant in variants
        for tags in [(getattr(variant, "semantic_tags", None) or []) + shared]
    )


def make_valid_clone_combinations(
    asset_names: Sequence[str],
    variant_counts: Sequence[int],
    clone_combinations: Sequence[InclusionSet] | None = None,
    *,
    all_asset_names: Sequence[str] | None = None,
) -> np.ndarray:
    """Build the legal prototype-combination array.

    Each combination contributes rows in proportion to its weight, split evenly across its
    spawn variants and interleaved round-robin.

    Args:
        asset_names: Clone-planned scene asset names, one per array column.
        variant_counts: Number of spawn variants per clone-planned asset.
        clone_combinations: Legal clone combinations. Assets absent from every entry are active
            in every combination. ``None`` uses the full Cartesian product.
        all_asset_names: Optional complete scene binding names used to validate combination entries.

    Returns:
        Integer array ``[num_combinations, num_assets]``; ``-1`` marks an absent asset.
    """
    if len(asset_names) != len(variant_counts):
        raise ValueError(f"Expected one variant count per asset, got {len(variant_counts)} and {len(asset_names)}.")
    if not asset_names:
        raise ValueError("Expected at least one asset name.")
    if any(count <= 0 for count in variant_counts):
        raise ValueError("Variant counts must be positive.")
    if not clone_combinations:
        return np.asarray(list(itertools.product(*[range(count) for count in variant_counts])), dtype=np.int64)

    clone_asset_names = set(asset_names)
    known_assets = set(all_asset_names) if all_asset_names is not None else clone_asset_names
    combination_assets = []
    for combination in clone_combinations:
        if combination.weight < 0:
            raise ValueError("Clone combination weights must be non-negative.")
        unknown_assets = sorted(set(combination.assets) - known_assets)
        if unknown_assets:
            raise ValueError(f"Unknown assets in clone combination: {unknown_assets}.")
        combination_assets.append(set(combination.assets) & clone_asset_names)

    claimed_assets = set().union(*combination_assets) if combination_assets else set()
    expanded: list[tuple[int, list[tuple[int, ...]]]] = []
    for combination, active_assets in zip(clone_combinations, combination_assets, strict=True):
        if combination.weight == 0:
            continue
        variant_ranges = [
            range(count) if name not in claimed_assets or name in active_assets else (-1,)
            for name, count in zip(asset_names, variant_counts, strict=True)
        ]
        expanded.append((combination.weight, list(itertools.product(*variant_ranges))))
    if not expanded:
        raise ValueError("Clone combinations produced no valid clone rows.")

    common_multiple = math.lcm(*(len(variants) for _, variants in expanded))
    rows = []
    cursors = [0] * len(expanded)
    for _ in range(common_multiple):
        for index, (weight, variants) in enumerate(expanded):
            for _ in range(weight):
                rows.append(variants[cursors[index] % len(variants)])
                cursors[index] += 1
    return np.asarray(rows, dtype=np.int64)


def make_clone_plan(
    cfgs: Iterable[Any],
    num_clones: int,
    env_spacing: float,
    *,
    global_paths: Iterable[str] = (),
    geometry_prim_paths: Iterable[str] = (),
    clone_strategy: Callable[[np.ndarray, int], np.ndarray] = sequential,
    valid_set: np.ndarray | None = None,
    env_template: str = DEFAULT_ENV_TEMPLATE,
) -> ClonePlan:
    """Build one flat plan from an explicit asset and sensor configuration manifest.

    Each top-level entry must directly declare ``prim_path`` or ``mesh_prim_paths``.
    The cloner never discovers environment or scene configuration types.

    Args:
        cfgs: Flat prim-authoring asset and sensor configurations.
        num_clones: Number of target environments.
        env_spacing: Distance between neighboring environment origins [m].
        global_paths: Additional shared prim roots not represented by a configuration.
        geometry_prim_paths: Prim-path expressions whose geometry the completed plan must declare.
        clone_strategy: Prototype-to-environment assignment function.
        valid_set: Optional legal prototype combinations, shape ``[num_combinations, num_assets]``.
        env_template: Environment path template whose ``{}`` marks the environment index.

    Returns:
        The preliminary plan consumed by scene construction and completed before replication.
    """
    cfgs = tuple(cfgs)
    groups: list[tuple[Any, Any, str, int]] = []
    globals_: list[tuple[Any | None, str, Any | None]] = []
    geometry_requests = [expand_env_regex_ns(path, env_template) for path in geometry_prim_paths]
    owned_paths: set[str] = set()

    for cfg in cfgs:
        try:
            fields = vars(cfg)
        except TypeError as error:
            raise TypeError(f"Clone participants must be prim-authoring cfgs, got {type(cfg).__name__}.") from error
        if "prim_path" not in fields and "mesh_prim_paths" not in fields:
            raise TypeError(f"Clone participants must be prim-authoring cfgs, got {type(cfg).__name__}.")
        for target in fields.get("mesh_prim_paths", ()):
            target_expr = target if isinstance(target, str) else getattr(target, "prim_expr", None)
            if target_expr is not None:
                geometry_requests.append(expand_env_regex_ns(target_expr, env_template))
        prim_path = fields.get("prim_path")
        if prim_path is None:
            continue
        if not isinstance(prim_path, str):
            raise TypeError(f"{type(cfg).__name__}.prim_path must be a string.")
        prim_path = expand_env_regex_ns(prim_path, env_template)
        matched = match(prim_path, env_template)
        destination = prim_path if matched is None else env_template + matched.suffix
        spawn_cfg = fields.get("spawn")
        if spawn_cfg is None and matched is not None:
            continue
        if destination in owned_paths:
            raise ValueError(f"Multiple cfgs author the same clone-plan destination: {destination!r}.")
        owned_paths.add(destination)
        if matched is None:
            globals_.append((cfg, prim_path, spawn_cfg))
            continue
        count = num_spawn_variants(spawn_cfg)
        if count <= 0:
            raise ValueError(f"Spawner at {prim_path!r} must have at least one variant.")
        groups.append((cfg, spawn_cfg, destination, count))

    roots: list[str] = []
    for path in global_paths:
        if not isinstance(path, str):
            raise TypeError(f"Global clone paths must be strings, got {type(path).__name__}.")
        if any(path == root or path.startswith(root + "/") for root in roots):
            continue
        roots = [root for root in roots if not root.startswith(path + "/")]
        roots.append(path)
    for path in roots:
        if path not in owned_paths:
            globals_.append((None, path, None))
            owned_paths.add(path)

    env_ids = np.arange(num_clones, dtype=np.int64)
    positions = grid_transforms(num_clones, env_spacing)[0]
    group_sizes = [count for _, _, _, count in groups]
    if groups:
        if valid_set is None:
            combinations = np.asarray(list(itertools.product(*[range(size) for size in group_sizes])), dtype=np.int64)
        else:
            combinations = np.asarray(valid_set)
        if not np.issubdtype(combinations.dtype, np.integer):
            raise ValueError("valid_set must contain integer prototype indices.")
        combinations = combinations.astype(np.int64, copy=False)
        if combinations.ndim != 2 or combinations.shape[0] == 0 or combinations.shape[1] != len(groups):
            raise ValueError(f"valid_set must have shape [N, {len(groups)}], got {combinations.shape}.")
        sizes = np.asarray(group_sizes, dtype=np.int64)[None, :]
        if np.any((combinations < -1) | ((combinations >= sizes) & (combinations != -1))):
            raise ValueError("valid_set contains prototype indices outside [-1, group_size).")
        chosen = np.asarray(clone_strategy(combinations, num_clones))
        if not np.issubdtype(chosen.dtype, np.integer):
            raise ValueError("clone_strategy result must contain integer prototype indices.")
        chosen = chosen.astype(np.int64, copy=False)
        if chosen.shape != (num_clones, len(groups)):
            raise ValueError(f"clone_strategy result must have shape {(num_clones, len(groups))}, got {chosen.shape}.")

        offsets = np.asarray([0, *itertools.accumulate(group_sizes[:-1])], dtype=np.int64)
        active = chosen >= 0
        flat_rows = (chosen + offsets).reshape(-1)
        flat_cols = np.broadcast_to(np.arange(num_clones)[:, None], chosen.shape).reshape(-1)
        active_flat = active.reshape(-1)
        clone_mask = np.zeros((sum(group_sizes), num_clones), dtype=np.bool_)
        clone_mask[flat_rows[active_flat], flat_cols[active_flat]] = True
    else:
        clone_mask = np.zeros((0, num_clones), dtype=np.bool_)

    sources: list[str] = []
    destinations: list[str] = []
    semantic_tags: list[tuple[tuple[str, str], ...]] = []
    cfg_rows: dict[int, tuple[int, ...]] = {}
    row = 0
    for cfg, spawn_cfg, destination, count in groups:
        cfg_rows[id(cfg)] = tuple(range(row, row + count))
        group_mask = clone_mask[row : row + count]
        source_env_ids = group_mask.argmax(axis=1)
        active_variants = group_mask.any(axis=1)
        for index, (source_env_id, is_active) in enumerate(zip(source_env_ids, active_variants, strict=True)):
            sources.append(destination.format(int(source_env_id) if is_active else index))
            destinations.append(destination)
        semantic_tags.extend(_variant_semantic_tags(spawn_cfg))
        row += count

    if globals_:
        shared = np.zeros((len(globals_), num_clones), dtype=np.bool_)
        clone_mask = np.concatenate((clone_mask, shared), axis=0)
    for cfg, path, spawn_cfg in globals_:
        if cfg is not None:
            cfg_rows[id(cfg)] = (len(sources),)
        sources.append(path)
        destinations.append(path)
        semantic_tags.append(() if spawn_cfg is None else _variant_semantic_tags(spawn_cfg)[0])

    return ClonePlan(
        sources=tuple(sources),
        destinations=tuple(destinations),
        clone_mask=clone_mask,
        env_ids=env_ids,
        positions=positions,
        env_template=env_template,
        cfg_rows=cfg_rows,
        global_paths=tuple(path for _, path, _ in globals_),
        semantic_tags=tuple(semantic_tags),
        geometry_requests=tuple(dict.fromkeys(geometry_requests)),
    )


def clone_plan_from_env_0(
    clone_cfg: CloneCfg,
    asset_cfgs: Iterable[Any],
    num_envs: int,
    env_spacing: float,
    *,
    positions: np.ndarray | None = None,
) -> ClonePlan:
    """Build and publish a homogeneous plan from a flat configuration manifest."""
    if not isinstance(clone_cfg, CloneCfg):
        raise TypeError(f"clone_cfg must be CloneCfg, got {type(clone_cfg).__name__}.")
    if clone_cfg.clone_combinations:
        raise ValueError("clone_plan_from_env_0 requires a homogeneous CloneCfg.")

    asset_cfgs = tuple(asset_cfgs)
    for cfg in asset_cfgs:
        try:
            fields = vars(cfg)
        except TypeError as error:
            raise TypeError(f"Asset entries must directly declare prim_path, got {type(cfg).__name__}.") from error
        if "prim_path" not in fields and "mesh_prim_paths" not in fields:
            raise TypeError(f"Asset entries must directly declare prim_path, got {type(cfg).__name__}.")
        spawn = fields.get("spawn")
        if spawn is not None and num_spawn_variants(spawn) != 1:
            raise ValueError("clone_plan_from_env_0 requires single-variant spawners.")

    sim = sim_utils.SimulationContext.instance()
    if sim is None:
        raise RuntimeError("Clone planning requires an active SimulationContext.")
    if sim.get_clone_plan() is not None:
        raise RuntimeError("A SimulationContext owns exactly one clone lifecycle.")

    plan = make_clone_plan(
        asset_cfgs,
        num_envs,
        env_spacing,
        clone_strategy=clone_cfg.clone_strategy,
        env_template=clone_cfg.clone_template,
    )
    plan = replace(plan, root_layer_identifier=sim.stage.GetRootLayer().identifier)
    if positions is not None:
        positions = np.asarray(positions, dtype=np.float32)
        if positions.shape != (num_envs, 3):
            raise ValueError(f"positions must have shape {(num_envs, 3)}, got {positions.shape}.")
        plan = replace(plan, positions=positions)
    sim.set_clone_plan(plan)
    return plan
