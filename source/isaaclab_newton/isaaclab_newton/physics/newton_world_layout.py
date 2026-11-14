# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build a cloned scene's Newton model from one finalized world per world prototype.

The clone plan already knows which prototype every world copies. The cloner therefore composes a *seed* builder: the
shared world ``-1`` and one unplaced world per world prototype. Newton finalizes only that seed, so every per-entity
derivation (collision data, SDF, meshes, inertia, custom attributes) runs once per prototype. :func:`build_model` then
expands each seed array to the full scene by the plan's layout, driven by Newton's attribute metadata
(:attr:`newton.Model.attribute_specs`): rows are gathered from their prototype's seed world, entity references are
offset by the referencing world's start in the referenced domain, world ids and world starts are recomputed, and each
world is placed by its transform with the arithmetic :meth:`newton.ModelBuilder.replicate` uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp
from newton import JointType, Model, ModelBuilder
from newton._src.core.host_transform import transform_inverse_rows, transform_multiply_rows
from newton._src.sim.graph_coloring import combine_independent_coloring_plan
from newton._src.sim.shape_contact_pairs import _ShapeContactPairs

from isaaclab.physics import PhysicsCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from collections.abc import Callable

_FREQUENCY = Model.AttributeFrequency
_IDENTITY = np.asarray((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), dtype=np.float32)

_BUILTIN_COUNTS = {
    frequency: name
    for frequency, name in Model._ATTRIBUTE_FREQUENCY_COUNT_ATTRS.items()
    if frequency not in (_FREQUENCY.WORLD, _FREQUENCY.CONTACT, _FREQUENCY.CONTACT_RIGID, _FREQUENCY.CONTACT_SOFT)
}
"""Built-in row domains and the model attribute holding each one's count."""

_KIND_REFERENCES = {
    "mujoco:equality_constraint_target": (
        "mujoco:equality_constraint_target_kind",
        {1: _FREQUENCY.JOINT, 2: _FREQUENCY.CONSTRAINT_MIMIC},
    ),
}
"""Attributes whose rows reference the domain named by a kind attribute; Newton's builder remaps them by name too."""

_ROW_POLICIES = ("start", "end", "range_start")
"""Compaction policies whose values index the referenced domain relative to the row's own world."""


class NewtonWorldLayout:
    """Which world prototype each cloned world copies and where it is placed.

    The cloner fills it while composing the seed builder; the backend reads it on every model build. A layout
    without worlds (:attr:`seed_world` is ``None``) finalizes the builder as it is.
    """

    def __init__(self, cfg: NewtonWorldLayoutCfg | None = None):
        self.seed_world: np.ndarray | None = None
        """Seed world (``0..T-1``) each cloned world copies, shape [num_worlds], dtype int32."""
        self.xforms: np.ndarray | None = None
        """Placement of each cloned world as ``(px, py, pz, qx, qy, qz, qw)``, shape [num_worlds, 7], dtype float32."""
        self.prefixes: np.ndarray | None = None
        """Label prefix (environment path) of each cloned world, shape [num_worlds]."""
        self.seed_prefixes: tuple[str, ...] = ()
        """Label prefix of each seed world."""
        self.env_ids: np.ndarray | None = None
        """Environment id of each cloned world, substituted for ``{}`` in string attributes."""

    def assign(
        self,
        seed_world: np.ndarray,
        xforms: np.ndarray,
        prefixes: np.ndarray,
        seed_prefixes: tuple[str, ...],
        env_ids: np.ndarray,
    ) -> None:
        """Record the cloned worlds of the seed builder."""
        self.seed_world = np.ascontiguousarray(seed_world, dtype=np.int32)
        self.xforms = np.ascontiguousarray(xforms, dtype=np.float32).reshape(-1, 7)
        self.prefixes = np.asarray(prefixes, dtype=np.dtypes.StringDType())
        self.seed_prefixes = tuple(seed_prefixes)
        self.env_ids = np.asarray(env_ids)

    def build_model(self, builder: ModelBuilder, device: Any = None) -> Model:
        """Finalize the seed builder and expand it to every cloned world.

        Args:
            builder: Seed builder holding the shared world and one world per world prototype.
            device: Warp device of the model.

        Returns:
            The scene model, equal to finalizing a builder that holds every cloned world.
        """
        model = builder.finalize(device=device)
        if self.seed_world is not None:
            _expand(model, self, builder)
        return model

    def deformable_starts(self, builder: ModelBuilder) -> dict[str, int]:
        """Return the first expanded particle of every surface and volume deformable, by expanded label.

        Args:
            builder: Seed builder.
        """
        labels = [*builder.surface_label, *builder.volume_label]
        starts = [*builder._surface_particle_start, *builder._volume_particle_start]
        if self.seed_world is None or not labels:
            return dict(zip(labels, starts, strict=True))
        world = np.asarray(builder.particle_world, dtype=np.int64)
        particles = _domain(world, self.seed_world, "particle")
        result = {}
        for label, start in zip(labels, starts, strict=True):
            seed = int(world[start]) if start < len(world) else -1
            if seed < 0:
                result[label] = start + (particles.tail_shift if start >= particles.seed_start[-1] else 0)
                continue
            worlds = np.flatnonzero(self.seed_world == seed)
            prefix = self.seed_prefixes[seed] + "/"
            suffix = label[len(prefix) :] if label.startswith(prefix) else None
            moved = (start - particles.seed_start[seed] + particles.start[worlds]).tolist()
            for world_index, world_start in zip(worlds.tolist(), moved, strict=True):
                result[label if suffix is None else f"{self.prefixes[world_index]}/{suffix}"] = world_start
        return result


@configclass
class NewtonWorldLayoutCfg:
    """Share the world layout of the Newton builder for one physics selection, retained across hard resets."""

    class_type: Callable[[NewtonWorldLayoutCfg], NewtonWorldLayout] | type = NewtonWorldLayout
    """Factory of the layout; :func:`~isaaclab.utils.instantiate` calls it."""
    physics_cfg: PhysicsCfg = field(kw_only=True, metadata={"copy": False})
    """Selected physics settings, as in :class:`~isaaclab_newton.physics.NewtonBuilderCfg`."""


@dataclass
class _Domain:
    """Seed blocks of one row domain and where they land in the expanded model."""

    world: np.ndarray
    """Seed world of each seed row (``-1`` for shared rows)."""
    prefix: int
    """Shared rows before the first world row."""
    seed_start: np.ndarray
    """First seed row of each seed world, shape [T + 1] (the last entry starts the shared tail)."""
    seed_total: int
    start: np.ndarray
    """First expanded row of each cloned world, shape [N + 1] (the last entry starts the shared tail)."""
    total: int
    gather: np.ndarray
    """Seed row of each expanded row."""
    row_world: np.ndarray
    """Cloned world of each expanded row (``-1`` for shared rows)."""

    @property
    def tail_shift(self) -> int:
        return self.total - self.seed_total


def _get(model: Model, name: str) -> Any:
    if ":" in name:
        namespace, attr = name.split(":", 1)
        return getattr(getattr(model, namespace, None), attr, None)
    return getattr(model, name, None)


def _set(model: Model, name: str, value: Any) -> None:
    if ":" in name:
        namespace, attr = name.split(":", 1)
        setattr(getattr(model, namespace), attr, value)
    else:
        setattr(model, name, value)


def _host(value: Any) -> np.ndarray:
    return value.numpy() if isinstance(value, wp.array) else np.asarray(value)


def _like(value: Any, data: np.ndarray, device: Any) -> Any:
    """Return ``data`` in the container type of ``value`` (Warp array, NumPy array or list)."""
    if isinstance(value, wp.array):
        return wp.array(np.ascontiguousarray(data), dtype=value.dtype, device=device, requires_grad=value.requires_grad)
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(data, dtype=value.dtype)
    return data.tolist()


def _count(model: Model, frequency: Any) -> int | None:
    if isinstance(frequency, _FREQUENCY):
        name = _BUILTIN_COUNTS.get(frequency)
        return None if name is None else int(getattr(model, name))
    return model.custom_frequency_counts.get(frequency)


def _specs(model: Model) -> dict[str, Model.AttributeSpec]:
    return {name: spec for name, spec in model.attribute_specs.items() if not spec.deprecated and spec.alias_of is None}


def _seed_worlds(model: Model, specs: dict[str, Model.AttributeSpec], counts: dict[Any, int]) -> dict[Any, np.ndarray]:
    """Find the seed world of every row of every domain from the model's metadata."""
    worlds: dict[Any, np.ndarray] = {}
    for name, spec in specs.items():
        frequency = spec.frequency
        if frequency not in counts or frequency in worlds:
            continue
        value = _get(model, name)
        if value is None:
            continue
        if spec.compaction_policy == "world_start":
            starts = _host(value).astype(np.int64)
            world = np.full(counts[frequency], -1, dtype=np.int64)
            seeds = len(starts) - 2
            world[starts[0] : starts[seeds]] = np.repeat(np.arange(seeds), np.diff(starts[: seeds + 1]))
            worlds[frequency] = world
        elif Model._normalize_attribute_reference(spec.references) == _FREQUENCY.WORLD and spec.row_width == 1:
            data = _host(value)
            if data.ndim == 1 and len(data) == counts[frequency]:
                worlds[frequency] = data.astype(np.int64)
    # Rows without their own world inherit it from the rows that own them (start arrays) or that they reference.
    changed = True
    while changed:
        changed = False
        for name, spec in specs.items():
            references = Model._normalize_attribute_reference(spec.references)
            frequency = spec.frequency
            if references is None or frequency not in counts:
                continue
            value = _get(model, name)
            if value is None:
                continue
            if spec.compaction_policy == "start" and frequency in worlds and references not in worlds:
                if references not in counts:
                    continue
                starts = _host(value).astype(np.int64)[: counts[frequency]]
                ends = np.append(starts[1:], counts[references])
                worlds[references] = np.repeat(worlds[frequency], ends - starts)
                changed = True
            elif spec.compaction_policy == "generic" and frequency not in worlds and references in worlds:
                data = _host(value).astype(np.int64).reshape(counts[frequency], -1)
                if counts[frequency]:
                    worlds[frequency] = worlds[references][data[:, 0]]
                    changed = True
    if counts.get(_FREQUENCY.ACTUATOR) and _FREQUENCY.ACTUATOR not in worlds:
        worlds[_FREQUENCY.ACTUATOR] = _actuator_worlds(model, worlds, counts[_FREQUENCY.ACTUATOR])
    for frequency, count in counts.items():
        if count and frequency not in worlds:
            raise ValueError(f"Newton metadata does not place the rows of {frequency!r} in worlds.")
        worlds.setdefault(frequency, np.empty(0, dtype=np.int64))
    return worlds


def _domain(world: np.ndarray, seed_world: np.ndarray, frequency: Any) -> _Domain:
    """Lay out one domain's expanded rows: shared prefix, each cloned world's seed block, shared tail."""
    seed_count = int(seed_world.max(initial=-1)) + 1
    local = np.flatnonzero(world >= 0)
    prefix = int(local[0]) if len(local) else len(world)
    sizes = np.bincount(world[local], minlength=seed_count) if len(local) else np.zeros(seed_count, dtype=np.int64)
    seed_start = prefix + np.concatenate(([0], np.cumsum(sizes)))
    tail = len(world) - seed_start[-1]
    expected = np.repeat(np.arange(seed_count), sizes)
    if not np.array_equal(world[prefix : seed_start[-1]], expected) or np.any(world[seed_start[-1] :] != -1):
        raise ValueError(f"Rows of {frequency!r} are not grouped as shared, seed worlds in order, shared.")
    world_sizes = sizes[seed_world]
    start = prefix + np.concatenate(([0], np.cumsum(world_sizes)))
    total = int(start[-1] + tail)
    worlds = len(seed_world)
    row_world = np.full(total, -1, dtype=np.int64)
    row_world[prefix : start[-1]] = np.repeat(np.arange(worlds), world_sizes)
    gather = np.arange(total, dtype=np.int64)
    in_world = slice(prefix, int(start[-1]))
    gather[in_world] += np.repeat(seed_start[seed_world] - start[:-1], world_sizes)
    gather[start[-1] :] -= total - len(world)
    return _Domain(world, prefix, seed_start, len(world), start, total, gather, row_world)


def _offset(
    values: np.ndarray, row_world: np.ndarray, target: _Domain, seed_world: np.ndarray, by_row: bool
) -> np.ndarray:
    """Map seed indices into ``target`` to expanded indices for expanded rows of cloned worlds ``row_world``.

    ``values`` has one row per expanded row (gathered). A world row's index into its seed world block moves to its own
    world's block; shared tail indices move past the expanded worlds; shared prefix indices and negative sentinels stay.
    ``by_row`` places range boundaries (start/end policies) by the row's world instead of by the value.
    """
    values = values.astype(np.int64)
    if values.size == 0:
        return values
    shaped = values.reshape(len(row_world), -1)
    out = shaped.copy()
    world = row_world[:, None]
    tail_start = target.seed_start[-1]
    if by_row:
        in_world = (world >= 0) & (shaped >= 0)
        in_tail = (world < 0) & (shaped >= tail_start)
    else:
        in_tail = shaped >= tail_start
        in_world = (shaped >= target.prefix) & ~in_tail
    out[in_tail] += target.tail_shift
    if np.any(in_world):
        cloned = np.broadcast_to(world, shaped.shape)[in_world]
        if np.any(cloned < 0):
            raise ValueError("A shared row references a seed world's row, which has no single expanded counterpart.")
        seeds = seed_world[cloned]
        if not by_row:
            owner = np.searchsorted(target.seed_start, shaped[in_world], side="right") - 1
            if np.any(owner != seeds):
                raise ValueError("A world row references another world's row.")
        out[in_world] += target.start[cloned] - target.seed_start[seeds]
    return out.reshape(values.shape)


def _labels(values: list, rows: _Domain, layout: NewtonWorldLayout, seed_world: np.ndarray) -> list:
    """Move each world row's label from its seed world's prefix to its own world's prefix.

    Each seed world's labels are split once into (carries the prefix, text after it); every world then joins its own
    prefix to its prototype's suffixes.
    """
    seeds = []
    for seed, prefix in enumerate(layout.seed_prefixes):
        rooted = prefix + "/"
        block = values[rows.seed_start[seed] : rows.seed_start[seed + 1]]
        seeds.append([(True, v[len(rooted) :]) if v.startswith(rooted) else (False, v) for v in block])
    prefixes = np.strings.add(layout.prefixes, "/").tolist()
    worlds = [
        prefix + text if moved else text
        for prefix, seed in zip(prefixes, seed_world.tolist(), strict=True)
        for moved, text in seeds[seed]
    ]
    return [*values[: rows.prefix], *worlds, *values[rows.seed_start[-1] :]]


def _format_slots(values: list, rows: _Domain, layout: NewtonWorldLayout) -> list:
    """Substitute each row's environment id for the ``{}`` world slot of string values."""
    full = np.asarray(values, dtype=object)[rows.gather]
    is_text = np.fromiter((isinstance(v, str) and "{}" in v for v in full), dtype=bool, count=len(full))
    if np.any(is_text):
        env = layout.env_ids[rows.row_world[is_text]].astype(str)
        text = np.asarray(full[is_text], dtype=np.dtypes.StringDType())
        full[is_text] = np.strings.replace(text, "{}", env.astype(np.dtypes.StringDType())).astype(object)
    return full.tolist()


def _runs(seed_world: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of worlds copying one prototype, as :meth:`ModelBuilder.replicate` was called per run."""
    bounds = np.r_[0, np.flatnonzero(np.diff(seed_world)) + 1, len(seed_world)]
    return list(zip(bounds[:-1].tolist(), bounds[1:].tolist(), strict=True))


def _rotate_vectors(rotation: np.ndarray, vectors: np.ndarray) -> None:
    """Rotate ``vectors`` in place with the cross-product form :meth:`ModelBuilder.replicate` uses for velocities."""
    q = np.asarray(rotation, dtype=np.float32)
    xyz = q[..., :3]
    w = q[..., 3:]
    t = 2.0 * np.cross(xyz, vectors)
    vectors += np.cross(xyz, t) + w * t


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors ``v`` by xyzw quaternions ``q``, as the cloner rotates particles."""
    axis, angle_w = q[..., :3], q[..., 3:4]
    t = 2.0 * np.cross(axis, v)
    return (v + angle_w * t + np.cross(axis, t)).astype(np.float32)


def _place(arrays: dict[str, np.ndarray], domains: dict[Any, _Domain], seed: Model, layout: NewtonWorldLayout) -> None:
    """Place every world's rows by its transform, run by run, with :meth:`ModelBuilder.replicate`'s arithmetic."""
    xforms, seed_world = layout.xforms, layout.seed_world
    moved_world = ~np.all(xforms == _IDENTITY, axis=1)
    rotated_world = moved_world & ~np.all(xforms[:, 3:] == _IDENTITY[3:], axis=1)
    bodies, shapes, joints = domains[_FREQUENCY.BODY], domains[_FREQUENCY.SHAPE], domains[_FREQUENCY.JOINT]
    coords, dofs, particles = (
        domains[_FREQUENCY.JOINT_COORD],
        domains[_FREQUENCY.JOINT_DOF],
        domains[_FREQUENCY.PARTICLE],
    )
    tets = domains[_FREQUENCY.TETRAHEDRON]
    seed_body = _host(seed.shape_body)
    seed_type, seed_parent = _host(seed.joint_type), _host(seed.joint_parent)
    seed_q_start, seed_qd_start = _host(seed.joint_q_start), _host(seed.joint_qd_start)

    def block(domain: _Domain, seed: int) -> np.ndarray:
        """Seed rows of one seed world, relative to its block start."""
        return np.arange(domain.seed_start[seed + 1] - domain.seed_start[seed])

    for start, stop in _runs(seed_world):
        seed = int(seed_world[start])
        worlds = np.arange(start, stop)
        moved = worlds[moved_world[start:stop]]
        moved_xforms = xforms[moved]
        rotated = rotated_world[moved]
        translations_only = not np.any(rotated)
        offsets = np.where(moved_world[start:stop, None], xforms[start:stop, :3], np.float32(0.0))

        local = block(particles, seed)
        if "particle_q" in arrays and len(local):
            arrays["particle_q"][particles.start[worlds, None] + local[None]] += offsets[:, None]
            # The cloner rotates particles and tet rest frames, which the builder's composition omits.
            rotating = worlds[rotated_world[start:stop]]
            seed_rows = particles.seed_start[seed] + local
            for world in rotating.tolist():
                q, target = xforms[world], particles.start[world] + local
                arrays["particle_q"][target] = _quat_rotate(q[3:], arrays["_seed_particle_q"][seed_rows]) + q[:3]
                arrays["particle_qd"][target] = _quat_rotate(q[3:], arrays["_seed_particle_qd"][seed_rows])
                tet = block(tets, seed)
                if len(tet):
                    poses = arrays["_seed_tet_poses"][tets.seed_start[seed] + tet]
                    arrays["tet_poses"][tets.start[world] + tet] = _quat_rotate(q[3:], poses)

        if len(moved) and len(block(shapes, seed)):
            local = block(shapes, seed)
            static = local[seed_body[shapes.seed_start[seed] + local] == -1]
            if len(static):
                source = arrays["_seed_shape_transform"][shapes.seed_start[seed] + static]
                transformed = transform_multiply_rows(moved_xforms[:, None], source[None])
                arrays["shape_transform"][shapes.start[moved, None] + static[None]] = transformed

        local = block(joints, seed)
        if len(moved) and len(local):
            rows = joints.seed_start[seed] + local
            parents, types = seed_parent[rows], seed_type[rows]
            nonfree = local[(parents == -1) & (types != int(JointType.FREE))]
            if len(nonfree):
                source = arrays["_seed_joint_X_p"][joints.seed_start[seed] + nonfree]
                transformed = transform_multiply_rows(moved_xforms[:, None], source[None])
                arrays["joint_X_p"][joints.start[moved, None] + nonfree[None]] = transformed
            free = local[(parents == -1) & (types == int(JointType.FREE))]
            if len(free):
                seed_rows = joints.seed_start[seed] + free
                q_local = seed_q_start[seed_rows] - coords.seed_start[seed]
                qd_local = seed_qd_start[seed_rows] - dofs.seed_start[seed]
                q_index = q_local[:, None] + np.arange(7)
                qd_index = qd_local[:, None] + np.arange(6)
                seed_q = arrays["_seed_joint_q"][coords.seed_start[seed] :]
                xform_prev = seed_q[q_index]
                X_pj = arrays["_seed_joint_X_p"][seed_rows]
                X_pj_inv = transform_inverse_rows(X_pj)
                xform_local = transform_multiply_rows(
                    transform_multiply_rows(X_pj_inv[None], moved_xforms[:, None]), X_pj[None]
                )
                targets = coords.start[moved, None, None] + q_index[None]
                arrays["joint_q"][targets] = transform_multiply_rows(xform_local, xform_prev[None])
                if np.any(rotated):
                    qd_targets = dofs.start[moved[rotated], None, None] + qd_index[None]
                    twist = arrays["joint_qd"][qd_targets].reshape((*qd_targets.shape[:2], 2, 3))
                    _rotate_vectors(xform_local[rotated, :, None, 3:], twist)
                    arrays["joint_qd"][qd_targets] = twist.reshape(qd_targets.shape)

        local = block(bodies, seed)
        if len(local):
            rows = bodies.start[worlds, None] + local[None]
            if translations_only:
                if np.any(offsets):
                    arrays["body_q"][rows, :3] += offsets[:, None]
            elif len(moved):
                source = arrays["_seed_body_q"][bodies.seed_start[seed] + local]
                arrays["body_q"][bodies.start[moved, None] + local[None]] = transform_multiply_rows(
                    moved_xforms[:, None], source[None]
                )
            if np.any(rotated):
                twists = arrays["body_qd"][bodies.start[moved[rotated], None] + local[None]].reshape(
                    (int(np.count_nonzero(rotated)), len(local), 2, 3)
                )
                _rotate_vectors(moved_xforms[rotated, None, None, 3:], twists)
                arrays["body_qd"][bodies.start[moved[rotated], None] + local[None]] = twists.reshape(-1, len(local), 6)


_PLACED = (
    "particle_q",
    "particle_qd",
    "tet_poses",
    "shape_transform",
    "joint_X_p",
    "joint_q",
    "joint_qd",
    "body_q",
    "body_qd",
)
"""Attributes :func:`_place` moves to each world's transform."""


def _expand(model: Model, layout: NewtonWorldLayout, builder: ModelBuilder) -> None:
    """Expand a finalized seed model in place to every cloned world of ``layout``."""
    device = model.device
    seed_world = layout.seed_world
    num_worlds = len(seed_world)
    if model.world_count != int(seed_world.max(initial=-1)) + 1:
        raise ValueError(f"The seed holds {model.world_count} worlds; the layout uses {seed_world.max() + 1}.")
    if model.tendon_count or model.muscle_count or model._deformable_visual_meshes or model.batteries:
        raise NotImplementedError("Expanding tendons, muscles, batteries or deformable visual meshes is not supported.")
    transformed = [key for key, attr in builder.custom_attributes.items() if attr.reference_value_transformer]
    if any(len(_host(_get(model, key))) for key in transformed if _get(model, key) is not None):
        raise NotImplementedError(f"Custom attributes {transformed} remap their values with builder callbacks.")
    specs = _specs(model)
    counts = {frequency: count for frequency in _BUILTIN_COUNTS if (count := _count(model, frequency)) is not None}
    counts.update(model.custom_frequency_counts)
    worlds = _seed_worlds(model, specs, counts)
    domains = {frequency: _domain(worlds[frequency], seed_world, frequency) for frequency in counts}

    # Rows: gather each attribute from its seed world and offset the entity indices it holds.
    arrays: dict[str, np.ndarray] = {}
    containers: dict[str, Any] = {}
    for name, spec in specs.items():
        frequency = spec.frequency
        if frequency not in domains or spec.compaction_policy in ("passthrough", "color_groups"):
            continue
        value = _get(model, name)
        if value is None:
            continue
        rows = domains[frequency]
        if spec.compaction_policy == "world_start":
            starts = np.concatenate((rows.start, [rows.total]))
            if name == "shape_world_start":
                starts = np.minimum(starts, rows.total)
            _set(model, name, _like(value, starts, device))
            continue
        if rows.seed_total == 0:
            continue  # nothing to expand; empty storage may keep a sentinel row
        references = Model._normalize_attribute_reference(spec.references)
        if isinstance(value, list) and references is None:
            if len(value) != rows.seed_total:
                raise ValueError(f"List attribute {name!r} does not have one value per row.")
            if value and all(isinstance(v, str) for v in value):
                is_label = name.endswith("_label") or name in model.custom_frequency_label_attributes.values()
                expanded = _labels(value, rows, layout, seed_world) if is_label else _format_slots(value, rows, layout)
            else:
                objects = np.empty(len(value), dtype=object)
                objects[:] = value
                expanded = objects[rows.gather].tolist()
            _set(model, name, expanded)
            continue
        data = _host(value)
        sentinel = spec.compaction_policy == "start" and len(data) == rows.seed_total + 1
        body = data[: rows.seed_total] if sentinel else data
        width = spec.row_width
        if body.shape[0] not in (rows.seed_total, rows.seed_total * width):
            raise ValueError(f"Attribute {name!r} has {body.shape[0]} rows for {rows.seed_total} {frequency!r} rows.")
        flat = body.shape[0] != rows.seed_total
        shaped = body.reshape(rows.seed_total, width, *body.shape[1:]) if flat else body
        full = shaped[rows.gather]
        if references == _FREQUENCY.WORLD:
            world = rows.row_world.reshape(-1, *[1] * (full.ndim - 1))
            full = np.where(full >= 0, np.broadcast_to(world, full.shape), full).astype(full.dtype)
        elif name in _KIND_REFERENCES:
            kind_name, kinds = _KIND_REFERENCES[name]
            kind = _host(_get(model, kind_name))[rows.gather]
            for code, target in kinds.items():
                if target in domains and np.any(kind == code):
                    hit = kind == code
                    full[hit] = _offset(full[hit], rows.row_world[hit], domains[target], seed_world, False)
        elif references is not None:
            if references not in domains:
                raise ValueError(f"Attribute {name!r} references {references!r}, which has no row count.")
            by_row = spec.compaction_policy in _ROW_POLICIES
            full = _offset(full, rows.row_world, domains[references], seed_world, by_row).astype(full.dtype)
        if flat:
            full = full.reshape(-1, *body.shape[1:])
        if sentinel:
            full = np.concatenate((full, np.asarray([domains[references].total], dtype=full.dtype)))
        if name in _PLACED:
            arrays[name] = np.array(full, dtype=np.float32)
            arrays["_seed_" + name] = np.asarray(body, dtype=np.float32)
            containers[name] = value
        else:
            _set(model, name, _like(value, full, device))
    _place(arrays, domains, model, layout)
    for name, value in containers.items():
        _set(model, name, _like(value, arrays[name], device))

    # Counts and per-world values.
    for frequency, name in _BUILTIN_COUNTS.items():
        if frequency in domains:
            setattr(model, name, domains[frequency].total)
    model.custom_frequency_counts = {key: domains[key].total for key in model.custom_frequency_counts}
    model.world_count = num_worlds
    gravity = _host(model.gravity)
    model.gravity = _like(model.gravity, np.concatenate((gravity[seed_world], gravity[-1:])), device)
    equality = getattr(model, "mujoco", None)
    if equality is not None and hasattr(equality, "equality_constraint_count"):
        frequency = "mujoco:equality_constraint"
        if frequency in domains:
            rows = domains[frequency]
            equality.equality_constraint_count = rows.total
            starts = np.concatenate((rows.start, [rows.total]))
            equality.equality_constraint_world_start = _like(equality.equality_constraint_world_start, starts, device)
    for key in list(model.custom_frequency_articulation):
        label = builder.custom_frequencies[key].articulation_owner_attribute
        model.custom_frequency_articulation[key] = _get(model, label)

    _expand_shape_structures(model, domains, seed_world, device)
    _expand_articulation_schedule(model, domains, seed_world, device)
    _expand_actuators(model, domains, seed_world, device)
    _expand_coloring(model, domains, seed_world, device)

    # Derived from the expanded rows as finalize derives them.
    if model.particle_count > 1 and model.particle_max_radius > 0.0 and model.particle_grid is None:
        with wp.ScopedDevice(device):
            model.particle_grid = wp.HashGrid(128, 128, 128)
    model.bvh_build_shapes(
        model,
        bvh_constructor=builder.default_bvh_cfg.shape_constructor,
        shape_flags=builder.default_bvh_cfg.shape_flags,
    )
    model.bvh_build_particles(model)


def _per_world(seed_rows: np.ndarray, seed_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gather rows owned by seed worlds into every cloned world, in world order.

    Args:
        seed_rows: Owning seed world of each row (``0..T-1``).
        seed_world: Seed world of each cloned world.

    Returns:
        The row each expanded row copies and the cloned world that owns it.
    """
    seeds = int(seed_world.max(initial=-1)) + 1
    order = np.argsort(seed_rows, kind="stable")
    sizes = np.bincount(seed_rows, minlength=seeds)
    seed_offsets = np.concatenate(([0], np.cumsum(sizes)))
    world_sizes = sizes[seed_world]
    world_offsets = np.concatenate(([0], np.cumsum(world_sizes)))
    index = np.arange(world_offsets[-1]) - np.repeat(world_offsets[:-1] - seed_offsets[seed_world], world_sizes)
    return order[index], np.repeat(np.arange(len(seed_world)), world_sizes)


def _expand_shape_structures(model: Model, domains: dict[Any, _Domain], seed_world: np.ndarray, device: Any) -> None:
    """Expand the shape structures finalize derives once per model: body shape lists, filter and contact pairs."""
    shapes, bodies = domains[_FREQUENCY.SHAPE], domains[_FREQUENCY.BODY]
    shape_body = _host(model.shape_body)
    order = np.argsort(shape_body, kind="stable")
    keys = np.arange(-1, bodies.total)
    bounds = np.searchsorted(shape_body[order], np.r_[keys, bodies.total]).tolist()
    members = order.tolist()
    model.body_shapes = dict(zip(keys.tolist(), map(members.__getitem__, map(slice, bounds[:-1], bounds[1:]))))

    packed = model._shape_collision_filter_pairs._packed
    if len(packed):
        pairs = np.stack(((packed >> 32), (packed & 0xFFFFFFFF)), axis=1).astype(np.int64)
        owner = np.maximum(shapes.world[pairs[:, 0]], shapes.world[pairs[:, 1]])
        shared = np.flatnonzero(owner < 0)
        local = np.flatnonzero(owner >= 0)
        rows, row_world = _per_world(owner[local], seed_world)
        expanded = np.concatenate(
            (
                _offset(pairs[shared], np.full(len(shared), -1), shapes, seed_world, by_row=False),
                _offset(pairs[local][rows], row_world, shapes, seed_world, by_row=False),
            )
        )
        codes = (expanded[:, 0] << 32) | expanded[:, 1]
        codes.sort(kind="stable")
        model._set_shape_collision_filter_packed(codes)
    seed_counts = model._shape_contact_pair_data.counts
    pair_data = _ShapeContactPairs(
        shape_body,
        _host(model.shape_world),
        _host(model.shape_collision_group),
        _host(model.shape_flags),
        model._shape_collision_filter_pairs._packed,
        model.world_count,
    )
    # Pair counts are per world slot (shared pairs first); each world counts as its prototype's seed world.
    counts = np.concatenate((seed_counts[:1], seed_counts[1:][seed_world]))
    counts.setflags(write=False)
    pair_data.__dict__["counts"] = counts
    model._shape_contact_pair_data = pair_data
    model._shape_contact_pair_counts.clear()
    model._shape_contact_pair_counts[None] = counts
    model._shape_contact_pair_count = int(counts.sum())

    # Per-shape fields of the geometry structures; their tables of distinct geometry stay shared.
    for structure, names in (
        (getattr(model, "_mesh_cells", None), ("shape_grid",)),
        (getattr(getattr(model, "_mesh_topology", None), "data", None), ("shape_base", "shape_radius")),
    ):
        for name in names if structure is not None else ():
            value = getattr(structure, name)
            if isinstance(value, wp.array) and len(value) == shapes.seed_total:
                setattr(structure, name, _like(value, value.numpy()[shapes.gather], device))


def _expand_articulation_schedule(
    model: Model, domains: dict[Any, _Domain], seed_world: np.ndarray, device: Any
) -> None:
    """Expand the forward-kinematics schedules finalize derives per articulation."""
    joints = domains[_FREQUENCY.JOINT]
    order = getattr(model, "_fk_joint_order", None)
    if order is not None:
        full = order.numpy()[joints.gather]
        model._fk_joint_order = _like(order, _offset(full, joints.row_world, joints, seed_world, by_row=False), device)
    level_start = getattr(model, "_fk_articulation_level_start", None)
    if level_start is None:
        return
    articulations = domains[_FREQUENCY.ARTICULATION]

    def gather_ranges(starts: np.ndarray, owners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rows of the ranges ``starts[owners]..starts[owners + 1]``, and each range's size."""
        sizes = np.diff(starts)[owners]
        firsts = np.concatenate(([0], np.cumsum(sizes)))[:-1]
        return np.repeat(starts[:-1][owners], sizes) + np.arange(sizes.sum()) - np.repeat(firsts, sizes), sizes

    level_rows, levels = gather_ranges(level_start.numpy().astype(np.int64), articulations.gather)
    level_world = np.repeat(articulations.row_world, levels)
    slot_rows, slots = gather_ranges(model._fk_level_joint_start.numpy().astype(np.int64), level_rows)
    slot_world = np.repeat(level_world, slots)
    level_joints = model._fk_level_joints.numpy()[slot_rows]
    model._fk_articulation_level_start = _like(level_start, np.concatenate(([0], np.cumsum(levels))), device)
    model._fk_level_joint_start = _like(model._fk_level_joint_start, np.concatenate(([0], np.cumsum(slots))), device)
    model._fk_level_joints = _like(
        model._fk_level_joints, _offset(level_joints, slot_world, joints, seed_world, by_row=False), device
    )
    model._fk_level_parent_pos = _like(
        model._fk_level_parent_pos, model._fk_level_parent_pos.numpy()[slot_rows], device
    )


def _actuator_worlds(model: Model, worlds: dict[Any, np.ndarray], count: int) -> np.ndarray:
    """Seed world of each actuator row: the world of the degree of freedom it drives."""
    world = np.full(count, -1, dtype=np.int64)
    for actuator in model.actuators:
        if actuator.actuator_indices is not None:
            dofs = worlds[model.get_attribute_frequency(actuator.state_vel_attr)]
            world[actuator.actuator_indices.numpy()] = dofs[actuator.indices.numpy()]
    return world


def _expand_actuators(model: Model, domains: dict[Any, _Domain], seed_world: np.ndarray, device: Any) -> None:
    """Expand each actuator's rows (one per driven degree of freedom) to every world.

    An actuator's rows are owned by the world of the degree of freedom they drive. Its index arrays index the model
    arrays it reads and writes (named by its ``*_attr`` fields) and are offset like entity references; every other
    per-row array of the actuator and its drive, clamping, motor and input processors is gathered.
    """
    for actuator in model.actuators:
        if any(vars(processor) for processor in actuator.input_processors):
            raise NotImplementedError("Expanding actuator input processors is not supported.")
        count = actuator.num_actuators
        velocity = domains[model.get_attribute_frequency(actuator.state_vel_attr)]
        rows = _domain(velocity.world[actuator.indices.numpy()], seed_world, "actuator rows")
        index_domains = {
            "indices": actuator.state_vel_attr,
            "pos_indices": actuator.state_pos_attr,
            "target_pos_indices": actuator.control_target_pos_attr,
            "effort_indices": actuator.control_output_attr,
        }
        expanded: dict[int, wp.array] = {}

        def gather(value: wp.array, target: _Domain | None) -> wp.array:
            if id(value) not in expanded:
                full = value.numpy()[rows.gather]
                if target is not None:
                    full = _offset(full, rows.row_world, target, seed_world, False).astype(full.dtype)
                expanded[id(value)] = _like(value, full, device)
            return expanded[id(value)]

        for name, attr in index_domains.items():
            value = getattr(actuator, name)
            setattr(actuator, name, gather(value, domains[model.get_attribute_frequency(attr)]))
        if actuator.actuator_indices is not None:
            actuator.actuator_indices = gather(actuator.actuator_indices, domains[_FREQUENCY.ACTUATOR])
        seen: set[int] = set()
        parts = [actuator, actuator.drive, actuator.motor, *(actuator.clamping or ()), actuator._effort_mode]
        for part in parts:
            if part is None or id(part) in seen:
                continue
            seen.add(id(part))
            for name, value in list(vars(part).items()):
                if isinstance(value, wp.array) and value.ndim >= 1 and value.shape[0] == count:
                    setattr(part, name, gather(value, None))
                elif type(value) is int and value == count and name.endswith("num_actuators"):
                    setattr(part, name, rows.total)


def _expand_coloring(model: Model, domains: dict[Any, _Domain], seed_world: np.ndarray, device: Any) -> None:
    """Combine every world's coloring into the particle and body color groups.

    Each world keeps its prototype's coloring (the seed coloring restricted to its seed world); the worlds' groups are
    paired with :meth:`newton.ModelBuilder.replicate`'s balancing. The result is a valid coloring, but not the one a
    coloring pass over all worlds at once (as solvers that color the builder run before finalize) would choose.
    """
    for kind, frequency in (("particle", _FREQUENCY.PARTICLE), ("body", _FREQUENCY.BODY)):
        groups = getattr(model, f"{kind}_color_groups")
        if not groups:
            continue
        rows = domains[frequency]
        host = [group.numpy().astype(np.int64) for group in groups]
        shared = [group[rows.world[group] < 0] for group in host]
        shared = [np.where(group >= rows.seed_start[-1], group + rows.tail_shift, group) for group in shared]
        plan = [(len(group), [(group, None)]) for group in shared if len(group)]
        local = [
            [group[rows.world[group] == seed] - rows.seed_start[seed] for group in host]
            for seed in range(len(rows.seed_start) - 1)
        ]
        local = [[group for group in seed_groups if len(group)] for seed_groups in local]
        for world, seed in enumerate(seed_world.tolist()):
            start = int(rows.start[world])
            plan = combine_independent_coloring_plan(plan, [(len(group), [(group, start)]) for group in local[seed]])
        merged = [
            np.concatenate([group if offset is None else group + offset for group, offset in chunks])
            for _, chunks in plan
        ]
        colors = np.empty(rows.total, dtype=np.int64)
        for color, group in enumerate(merged):
            colors[group] = color
        setattr(model, f"{kind}_color_groups", [wp.array(group, dtype=int, device=device) for group in merged])
        setattr(model, f"{kind}_colors", wp.array(colors, dtype=int, device=device))
