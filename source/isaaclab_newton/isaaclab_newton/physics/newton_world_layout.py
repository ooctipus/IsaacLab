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

import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import warp as wp
from newton import JointType, Model, ModelBuilder
from newton._src.sim.shape_contact_pairs import _ShapeContactPairs
from newton.solvers import SolverMuJoCo
from newton.utils import combine_independent_coloring_plan, transform_inverse_rows, transform_multiply_rows

from isaaclab.physics import PhysicsCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from collections.abc import Callable

_FREQUENCY = Model.AttributeFrequency
_IDENTITY = np.asarray((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), dtype=np.float32)

_UNCOUNTED = (_FREQUENCY.ONCE, _FREQUENCY.WORLD, _FREQUENCY.CONTACT, _FREQUENCY.CONTACT_RIGID, _FREQUENCY.CONTACT_SOFT)
"""Frequencies without entity rows: a single value, the worlds themselves, and the contact capacities."""

_TRN = SolverMuJoCo.TrnType
_KIND_REFERENCES = {
    "mujoco:equality_constraint_target": (
        "mujoco:equality_constraint_target_kind",
        {1: (_FREQUENCY.JOINT,), 2: (_FREQUENCY.CONSTRAINT_MIMIC,)},
    ),
    "mujoco:actuator_trnid": (
        "mujoco:actuator_trntype",
        {
            _TRN.JOINT: (_FREQUENCY.JOINT_DOF, None),
            _TRN.JOINT_IN_PARENT: (_FREQUENCY.JOINT_DOF, None),
            _TRN.TENDON: ("mujoco:tendon", None),
            _TRN.SITE: (_FREQUENCY.SHAPE, _FREQUENCY.SHAPE),
            _TRN.BODY: (_FREQUENCY.BODY, None),
            _TRN.SLIDERCRANK: (_FREQUENCY.SHAPE, _FREQUENCY.SHAPE),
        },
    ),
}
"""Attributes whose columns reference the domains a per-row kind attribute names; Newton's builder remaps them by name
too (``None`` keeps a column)."""

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

    gather_device: dict[Any, torch.Tensor] = field(default_factory=dict)
    """:attr:`gather` on each device it was needed on."""

    @property
    def tail_shift(self) -> int:
        return self.total - self.seed_total

    def gather_on(self, device: torch.device) -> torch.Tensor:
        if device not in self.gather_device:
            self.gather_device[device] = torch.as_tensor(self.gather, device=device)
        return self.gather_device[device]


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


_DEVICE_GATHER_TYPES = {
    np.dtype(t) for t in (np.float16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64, np.uint8, np.bool_)
}
"""Element types gathered on the array's device; others (unsigned wide integers, structs) are gathered on the host."""


def _gather_on_device(value: wp.array, rows: _Domain, width: int) -> wp.array | None:
    """Gather a Warp array's rows on its device into a new array, or ``None`` when its type is gathered on the host."""
    scalar = getattr(value.dtype, "_wp_scalar_type_", value.dtype)
    if wp.types.type_is_struct(value.dtype) or np.dtype(wp.dtype_to_numpy(scalar)) not in _DEVICE_GATHER_TYPES:
        return None
    seed = wp.to_torch(value, requires_grad=False)
    shape = seed.shape
    if shape[0] != rows.seed_total * width:
        raise ValueError(
            f"An attribute declares {width} values per row but stores {shape[0]} for {rows.seed_total} rows."
        )
    seed = seed.reshape(rows.seed_total, width, *shape[1:]) if width > 1 else seed
    total = rows.total * width
    out = wp.empty((total, *value.shape[1:]), dtype=value.dtype, device=value.device, requires_grad=value.requires_grad)
    target = wp.to_torch(out, requires_grad=False).reshape(rows.total, *seed.shape[1:])
    torch.index_select(seed, 0, rows.gather_on(seed.device), out=target)
    return out


def _like(value: Any, data: np.ndarray, device: Any) -> Any:
    """Return ``data`` in the container type of ``value`` (Warp array, NumPy array or list)."""
    if isinstance(value, wp.array):
        return wp.array(np.ascontiguousarray(data), dtype=value.dtype, device=device, requires_grad=value.requires_grad)
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(data, dtype=value.dtype)
    return data.tolist()


def _counts(model: Model, specs: dict[str, Model.AttributeSpec]) -> dict[Any, int]:
    """Row count of every built-in, custom and derived frequency the model declares."""
    frequencies = [frequency for frequency in _FREQUENCY if frequency not in _UNCOUNTED]
    frequencies += [spec.frequency for spec in specs.values() if isinstance(spec.frequency, str)]
    frequencies += list(model.custom_frequency_counts)
    return {frequency: model.attribute_frequency_count(frequency) for frequency in dict.fromkeys(frequencies)}


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
        elif Model._normalize_attribute_reference(spec.references) == _FREQUENCY.WORLD:
            data = _host(value)
            if data.shape != (counts[frequency],):
                raise ValueError(f"World attribute {name!r} has shape {data.shape} for {counts[frequency]} rows.")
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

    ``values`` has one row per expanded row (gathered). A world row's index into its seed world's block moves to its
    own world's block; shared tail indices move past the expanded worlds; shared prefix indices and negative sentinels
    stay. ``by_row`` treats every non-negative index of a world row as its own block's (range boundaries of start and
    end policies may sit on the block's end).
    """
    values = values.astype(np.int64)
    if values.size == 0:
        return values
    shaped = values.reshape(len(row_world), -1)
    local = (row_world >= 0)[:, None]
    world = np.maximum(row_world, 0)
    seed = seed_world[world]
    low, high = target.seed_start[seed][:, None], target.seed_start[seed + 1][:, None]
    shift = (target.start[world] - target.seed_start[seed])[:, None]
    tail = shaped >= target.seed_start[-1]
    if by_row:
        inside = local & (shaped >= 0)
        tail &= ~local
    else:
        inside = local & (shaped >= low) & (shaped < high)
        if np.any(~inside & ~tail & (shaped >= target.prefix)):
            raise ValueError("A row references a seed world's row that is not its own world's.")
    return (shaped + np.where(inside, shift, np.where(tail, target.tail_shift, 0))).reshape(values.shape)


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


def _is_label(model: Model, name: str) -> bool:
    """Whether replication names a string attribute per world: built-in labels and custom frequencies' labels."""
    if ":" not in name:
        return name.endswith("_label")
    return name in model.custom_frequency_label_attributes.values() or name == "mujoco:equality_constraint_label"


def _has_world_slots(model: Model, name: str, spec: Model.AttributeSpec) -> bool:
    """Whether the cloner fills ``{}`` world slots of a string attribute: a custom attribute gives its rows a world."""
    if name == "isaaclab:visual_material_path":
        return True
    return any(
        other.frequency == spec.frequency
        and ":" in key
        and Model._normalize_attribute_reference(other.references) == _FREQUENCY.WORLD
        for key, other in model.attribute_specs.items()
    )


def _gather_list(values: list, rows: _Domain) -> list:
    """Copy each row's value from its seed row."""
    objects = np.empty(len(values), dtype=object)
    objects[:] = values
    return objects[rows.gather].tolist()


def _format_slots(values: list, rows: _Domain, layout: NewtonWorldLayout) -> list:
    """Substitute each row's environment id for the ``{}`` world slot of string values."""
    seed = np.asarray(values, dtype=object)
    full = seed[rows.gather]
    has_slot = np.fromiter((isinstance(v, str) and "{}" in v for v in seed), dtype=bool, count=len(seed))
    is_text = has_slot[rows.gather]
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


def _expand_rows(
    model: Model,
    name: str,
    spec: Model.AttributeSpec,
    value: Any,
    domains: dict[Any, _Domain],
    layout: NewtonWorldLayout,
) -> np.ndarray | list | None:
    """Return one row attribute expanded to every world: rows gathered from their seed world, references offset.

    Returns ``None`` for an attribute without rows to expand (empty storage may keep a sentinel row).
    """
    seed_world = layout.seed_world
    rows = domains[spec.frequency]
    if spec.compaction_policy == "world_start":
        starts = np.concatenate((rows.start, [rows.total]))
        return np.minimum(starts, rows.total) if name == "shape_world_start" else starts
    references = Model._normalize_attribute_reference(spec.references)
    if rows.seed_total == 0:
        if spec.compaction_policy == "start":
            return np.asarray([domains[references].total], dtype=_host(value).dtype)
        return None
    if references is None and isinstance(value, wp.array) and name not in _PLACED and name not in _KIND_REFERENCES:
        gathered = _gather_on_device(value, rows, spec.row_width)
        if gathered is not None:
            return gathered
    if isinstance(value, list) and references is None:
        if len(value) != rows.seed_total:
            raise ValueError(f"List attribute {name!r} does not have one value per row.")
        if value and all(isinstance(v, str) for v in value):
            if _is_label(model, name):
                return _labels(value, rows, layout, seed_world)
            return (
                _format_slots(value, rows, layout) if _has_world_slots(model, name, spec) else _gather_list(value, rows)
            )
        return _gather_list(value, rows)
    data = _host(value)
    # A start attribute ends with the referenced domain's count; a row of a wider attribute stores its values flat.
    sentinel = spec.compaction_policy == "start"
    stored = rows.seed_total * spec.row_width + sentinel
    if data.shape[0] != stored:
        raise ValueError(f"Attribute {name!r} stores {data.shape[0]} values; its metadata declares {stored}.")
    body = data[: rows.seed_total * spec.row_width]
    flat = spec.row_width > 1
    full = (body.reshape(rows.seed_total, spec.row_width, *body.shape[1:]) if flat else body)[rows.gather]
    if references == _FREQUENCY.WORLD:
        world = rows.row_world.reshape(-1, *[1] * (full.ndim - 1))
        full = np.where(full >= 0, np.broadcast_to(world, full.shape), full).astype(full.dtype)
    elif name in _KIND_REFERENCES:
        kind_name, kinds = _KIND_REFERENCES[name]
        kind = _host(_get(model, kind_name))[rows.gather]
        columns = full.reshape(len(full), -1)
        for code, targets in kinds.items():
            hit = kind == int(code)
            for column, target in enumerate(targets):
                if target is not None and np.any(hit):
                    columns[hit, column] = _offset(
                        columns[hit, column], rows.row_world[hit], domains[target], seed_world, False
                    )
    elif references is not None:
        if references not in domains:
            raise ValueError(f"Attribute {name!r} references {references!r}, which has no row count.")
        by_row = spec.compaction_policy in _ROW_POLICIES
        full = _offset(full, rows.row_world, domains[references], seed_world, by_row).astype(full.dtype)
    if flat:
        full = full.reshape(-1, *body.shape[1:])
    if sentinel:
        full = np.concatenate((full, np.asarray([domains[references].total], dtype=full.dtype)))
    return full


def _expand_all_rows(
    model: Model,
    specs: dict[str, Model.AttributeSpec],
    domains: dict[Any, _Domain],
    layout: NewtonWorldLayout,
    arrays: dict[str, np.ndarray],
    containers: dict[str, Any],
) -> None:
    """Expand every row and world attribute in place; collect the attributes :func:`_place` moves in ``arrays``."""
    device = model.device
    for name, spec in specs.items():
        frequency = spec.frequency
        if frequency == _FREQUENCY.WORLD:
            _expand_world_values(model, name, layout.seed_world, device)
            continue
        if frequency not in domains or spec.compaction_policy == "color_groups":
            continue
        value = _get(model, name)
        if value is None:
            continue
        full = _expand_rows(model, name, spec, value, domains, layout)
        if full is None:
            continue
        if name in _PLACED:
            arrays[name] = np.array(full, dtype=np.float32)
            arrays["_seed_" + name] = np.asarray(_host(value)[: domains[frequency].seed_total], dtype=np.float32)
            containers[name] = value
        else:
            _set(model, name, full if isinstance(full, (list, wp.array)) else _like(value, full, device))


def _expand(model: Model, layout: NewtonWorldLayout, builder: ModelBuilder) -> None:
    """Expand a finalized seed model in place to every cloned world of ``layout``."""
    device = model.device
    seed_world = layout.seed_world
    num_worlds = len(seed_world)
    if model.world_count != int(seed_world.max(initial=-1)) + 1:
        raise ValueError(f"The seed holds {model.world_count} worlds; the layout uses {seed_world.max() + 1}.")
    if model.tendon_count or model.muscle_count or model._deformable_visual_meshes or model.batteries:
        raise NotImplementedError("Expanding tendons, muscles, batteries or deformable visual meshes is not supported.")
    transformed = [
        key
        for key, attr in builder.custom_attributes.items()
        if attr.reference_value_transformer and key not in _KIND_REFERENCES
    ]
    if any(len(_host(_get(model, key))) for key in transformed if _get(model, key) is not None):
        raise NotImplementedError(f"Custom attributes {transformed} remap their values with builder callbacks.")
    specs = _specs(model)
    counts = _counts(model, specs)
    worlds = _seed_worlds(model, specs, counts)
    domains = {frequency: _domain(worlds[frequency], seed_world, frequency) for frequency in counts}

    # Rows: gather each attribute from its seed world and offset the entity indices it holds. Device gathers run on
    # Warp's stream, ordered after finalize and before the kernels that read the expanded model.
    arrays: dict[str, np.ndarray] = {}
    containers: dict[str, Any] = {}
    stream = torch.cuda.stream(wp.stream_to_torch(device)) if device.is_cuda else contextlib.nullcontext()
    with stream:
        _expand_all_rows(model, specs, domains, layout, arrays, containers)
    _place(arrays, domains, model, layout)
    for name, value in containers.items():
        _set(model, name, _like(value, arrays[name], device))

    # Counts and per-world values.
    for frequency, rows in domains.items():
        model.set_attribute_frequency_count(frequency, rows.total)
    model.world_count = num_worlds
    equality = getattr(model, "mujoco", None)
    if equality is not None and hasattr(equality, "equality_constraint_count"):
        frequency = "mujoco:equality_constraint"
        if frequency in domains:
            rows = domains[frequency]
            equality.equality_constraint_count = rows.total
            starts = np.concatenate((rows.start, [rows.total]))
            equality.equality_constraint_world_start = _like(equality.equality_constraint_world_start, starts, device)
    for key in list(model.custom_frequency_articulation):
        # Owners of an empty frequency stay the empty array finalize made; others are the expanded attribute.
        owners = _get(model, builder.custom_frequencies[key].articulation_owner_attribute)
        if owners is not None:
            model.custom_frequency_articulation[key] = owners

    _expand_shape_structures(model, domains, seed_world, device)
    _expand_actuators(model, domains, seed_world, device)
    _expand_coloring(model, domains, seed_world, device)

    # Derived from the expanded rows as finalize derives them.
    model.build_soft_mesh_adjacency()
    if model.particle_count > 1 and model.particle_max_radius > 0.0 and model.particle_grid is None:
        with wp.ScopedDevice(device):
            model.particle_grid = wp.HashGrid(128, 128, 128)
    model.bvh_build_shapes(
        model,
        bvh_constructor=builder.default_bvh_cfg.shape_constructor,
        shape_flags=builder.default_bvh_cfg.shape_flags,
    )
    model.bvh_build_particles(model)


def _expand_world_values(model: Model, name: str, seed_world: np.ndarray, device: Any) -> None:
    """Give every world its seed world's value; a trailing value of the global world ``-1`` stays last."""
    value = _get(model, name)
    if value is None:
        return
    data = _host(value) if not isinstance(value, list) else np.asarray(value, dtype=object)
    seeds = int(seed_world.max(initial=-1)) + 1
    if len(data) != seeds + 1:
        raise ValueError(
            f"World attribute {name!r} has {len(data)} values for {seeds} seed worlds and the global world."
        )
    full = np.concatenate((data[seed_world], data[seeds:]))
    _set(model, name, full.tolist() if isinstance(value, list) else _like(value, full, device))


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
        model._set_shape_collision_filter_packed(_expand_pair_codes(packed, shapes, seed_world))
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
        # A processor's joint parameters index joint_qd-shaped (``*_indices``) and joint_q-shaped (``*_pos_indices``)
        # arrays.
        for processor in actuator.input_processors:
            for parameter in processor.JOINT_PARAMS:
                for suffix, frequency in (("_indices", _FREQUENCY.JOINT_DOF), ("_pos_indices", _FREQUENCY.JOINT_COORD)):
                    value = getattr(processor, parameter + suffix, None)
                    if isinstance(value, wp.array):
                        setattr(processor, parameter + suffix, gather(value, domains[frequency]))
        seen: set[int] = set()
        parts = [
            actuator,
            actuator.drive,
            actuator.motor,
            *(actuator.clamping or ()),
            *actuator.input_processors,
            actuator._effort_mode,
        ]
        for part in parts:
            if part is None or id(part) in seen:
                continue
            seen.add(id(part))
            for name, value in list(vars(part).items()):
                if isinstance(value, wp.array) and value.ndim >= 1 and value.shape[0] == count:
                    setattr(part, name, gather(value, None))
                elif type(value) is int and value == count and name.endswith("num_actuators"):
                    setattr(part, name, rows.total)
        # Drives, input processors and motors allocate their working buffers and row indices for the actuator count,
        # as the actuator's construction does. Clampings keep their shared tables: their finalize only loads them.
        requires_grad = actuator.requires_grad
        actuator.drive.finalize(device, rows.total)
        for processor in actuator.input_processors:
            processor.finalize(device, rows.total, requires_grad=requires_grad)
        if actuator.motor is not None:
            actuator.motor.finalize(device, rows.total, requires_grad=requires_grad)


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


def _expand_pair_codes(codes: np.ndarray, shapes: _Domain, seed_world: np.ndarray) -> np.ndarray:
    """Expand sorted packed shape-pair codes ``(first << 32) | second`` (first < second) to every world, sorted.

    A pair touching a seed world repeats in each world of its prototype. Pairs whose first shape is in a world come in
    world order, which keeps them sorted, and are written per prototype as one broadcast add; pairs whose first shape
    is shared are sorted once; shared tail pairs move past the worlds.
    """
    first, second = codes >> 32, codes & 0xFFFFFFFF
    tail_start, tail_shift = shapes.seed_start[-1], shapes.tail_shift
    first_world, second_world = shapes.world[first], shapes.world[second]
    shifts = shapes.start[:-1] - shapes.seed_start[seed_world]
    # Shared first shape: unchanged, or paired with a world shape (once per world of that prototype).
    shared = np.flatnonzero((first < shapes.prefix) & (second_world < 0))
    leading = [codes[shared] + np.where(second[shared] >= tail_start, tail_shift, 0)]
    for seed in np.flatnonzero(np.bincount(second_world[first < shapes.prefix] + 1)[1:]).tolist():
        pairs = codes[(first < shapes.prefix) & (second_world == seed)]
        worlds = np.flatnonzero(seed_world == seed)
        leading.append((pairs[None, :] + shifts[worlds, None]).ravel())
    leading = np.sort(np.concatenate(leading))
    trailing = codes[first >= tail_start] + (tail_shift << 32) + tail_shift
    # World first shape: the second is in the same world or in the shared tail.
    seeds = int(seed_world.max(initial=-1)) + 1
    blocks = [codes[first_world == seed] for seed in range(seeds)]
    to_tail = [(block & 0xFFFFFFFF) >= tail_start for block in blocks]
    sizes = np.asarray([len(block) for block in blocks], dtype=np.int64)[seed_world]
    offsets = len(leading) + np.concatenate(([0], np.cumsum(sizes)))
    out = np.empty(int(offsets[-1]) + len(trailing), dtype=np.int64)
    out[: len(leading)] = leading
    out[offsets[-1] :] = trailing
    both = np.int64((1 << 32) + 1)
    for seed, (block, tail) in enumerate(zip(blocks, to_tail, strict=True)):
        worlds = np.flatnonzero(seed_world == seed)
        if not len(block) or not len(worlds):
            continue
        world_shifts = shifts[worlds, None]
        if len(worlds) == len(seed_world):
            values = out[offsets[0] : offsets[-1]].reshape(len(worlds), len(block))
            np.add(block[None, :], world_shifts * both, out=values)
            if np.any(tail):
                values[:, tail] += tail_shift - world_shifts
        else:
            values = block[None, :] + world_shifts * both
            if np.any(tail):
                values[:, tail] += tail_shift - world_shifts
            out[offsets[worlds, None] + np.arange(len(block))] = values
    return out
