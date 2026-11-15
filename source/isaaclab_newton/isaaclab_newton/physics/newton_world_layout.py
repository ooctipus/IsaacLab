# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build a cloned scene's Newton model from one finalized world per world prototype.

The clone plan already knows which prototype every world copies. The cloner therefore composes a *seed* builder: the
shared world ``-1`` and one unplaced world per world prototype. Newton finalizes only that seed, so every per-entity
derivation (collision data, SDF, meshes, inertia, custom attributes) runs once per prototype. :func:`build_model` then
expands each seed array to the full scene by the plan's layout, driven by Newton's attribute metadata
(:attr:`newton.Model.attribute_specs` and the specs of the objects the model owns): each domain's rows take their
world from :meth:`newton.Model.attribute_frequency_world`, rows are gathered from their prototype's seed world, entity
references (also those whose domain depends on a row's kind) are offset by the referencing world's start in the
referenced domain, labels take each world's prefix, world ids and world starts are recomputed, and each world is
placed by its transform with :func:`newton.utils.place_copies`, the placement :meth:`newton.ModelBuilder.replicate`
uses. The objects the model owns (actuators and their components, batteries, deformable visual meshes) expand their
declared arrays the same way. Structures finalize derives once per model (body shape lists, contact pairs, coloring,
adjacency, BVHs) are derived again for the expanded model.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import warp as wp
from newton import Model, ModelBuilder
from newton.geometry import DeformableVisualMesh
from newton.utils import combine_world_colorings, place_copies

from isaaclab.physics import PhysicsCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from collections.abc import Callable

_FREQUENCY = Model.AttributeFrequency

_UNCOUNTED = (_FREQUENCY.ONCE, _FREQUENCY.WORLD, _FREQUENCY.CONTACT, _FREQUENCY.CONTACT_RIGID, _FREQUENCY.CONTACT_SOFT)
"""Frequencies without entity rows: a single value, the worlds themselves, and the contact capacities."""

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
        starts = [*builder.surface_particle_start, *builder.volume_particle_start]
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


def _has_world_slots(model: Model, name: str, spec: Model.AttributeSpec) -> bool:
    """Whether the cloner fills ``{}`` world slots of a string attribute: a custom attribute gives its rows a world."""
    if name == "isaaclab:visual_material_path":
        return True
    return any(
        other.frequency == spec.frequency and ":" in key and other.reference_domain == _FREQUENCY.WORLD
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


def _place(
    arrays: dict[str, np.ndarray],
    seeds: dict[str, np.ndarray],
    specs: dict[str, Model.AttributeSpec],
    domains: dict[Any, _Domain],
    layout: NewtonWorldLayout,
) -> None:
    """Place each run of worlds of one prototype with Newton's placement of copied worlds.

    A run's worlds lie one after another in every domain, as the copies :meth:`ModelBuilder.replicate` lays out; each
    placement reads the run's seed block, with start attributes relative to the block.
    """
    placed = {name: specs[name] for name in arrays}
    inputs = dict.fromkeys(name for key in placed for name in (key, *specs[key].placement.inputs))
    for start, stop in _runs(layout.seed_world):
        seed = int(layout.seed_world[start])
        source = {}
        for name in inputs:
            spec = specs[name]
            rows = domains[spec.frequency]
            block = seeds[name][rows.seed_start[seed] : rows.seed_start[seed + 1]]
            if spec.compaction_policy == "start":
                block = block - domains[spec.reference_domain].seed_start[seed]
            source[name] = block
        for name, (targets, values) in place_copies(placed, source, layout.xforms[start:stop]).items():
            arrays[name][domains[specs[name].frequency].start[start] + targets] = values


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
        return np.concatenate((rows.start, [rows.total]))
    references = spec.reference_domain
    if rows.seed_total == 0:
        if spec.compaction_policy == "start":
            return np.asarray([domains[references].total], dtype=_host(value).dtype)
        return None
    if references is None and isinstance(value, wp.array) and spec.placement is None and spec.kind_references is None:
        gathered = _gather_on_device(value, rows, spec.row_width)
        if gathered is not None:
            return gathered
    if isinstance(value, list) and references is None:
        if len(value) != rows.seed_total:
            raise ValueError(f"List attribute {name!r} does not have one value per row.")
        if value and all(isinstance(v, str) for v in value):
            if spec.label:
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
    elif spec.kind_references is not None:
        kinds = _host(_get(model, spec.kind_references.kind))[rows.gather]
        columns = full.reshape(len(full), -1)
        for code, targets in spec.kind_references.targets:
            hit = kinds == code
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
    seeds: dict[str, np.ndarray],
) -> None:
    """Expand every row and world attribute in place.

    The attributes :func:`_place` moves are collected in ``arrays`` and the seed values placements read in ``seeds``.
    """
    device = model.device
    placement_inputs = {
        name for key, spec in specs.items() if spec.placement is not None for name in (key, *spec.placement.inputs)
    }
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
        if name in placement_inputs:
            seeds[name] = _host(value)
        full = _expand_rows(model, name, spec, value, domains, layout)
        if full is None:
            continue
        if spec.placement is not None:
            arrays[name] = np.array(full, dtype=np.float32)
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
    specs = _specs(model)
    counts = _counts(model, specs)
    domains = {
        frequency: _domain(model.attribute_frequency_world(frequency), seed_world, frequency) for frequency in counts
    }

    # Rows: gather each attribute from its seed world and offset the entity indices it holds. Device gathers run on
    # Warp's stream, ordered after finalize and before the kernels that read the expanded model.
    arrays: dict[str, np.ndarray] = {}
    containers: dict[str, Any] = {}
    seeds: dict[str, np.ndarray] = {}
    stream = torch.cuda.stream(wp.stream_to_torch(device)) if device.is_cuda else contextlib.nullcontext()
    with stream:
        _expand_all_rows(model, specs, domains, layout, arrays, containers, seeds)
    _place(arrays, seeds, specs, domains, layout)
    for name, value in containers.items():
        _set(model, name, _like(value, arrays[name], device))

    # Counts and per-world values.
    for frequency, rows in domains.items():
        model.set_attribute_frequency_count(frequency, rows.total)
    model.world_count = num_worlds

    _expand_shape_structures(model, domains, seed_world)
    with stream:
        _expand_actuators(model, domains, layout)
        _expand_batteries(model, domains, layout)
    _expand_visual_meshes(model, domains, layout)
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


def _expand_shape_structures(model: Model, domains: dict[Any, _Domain], seed_world: np.ndarray) -> None:
    """Expand the shape structures finalize derives once per model: body shape lists, filter and contact pairs, and
    the shape rows of the mesh structures."""
    shapes = domains[_FREQUENCY.SHAPE]
    model.build_body_shapes()
    codes = model.shape_collision_filter_pair_codes
    if len(codes):
        model.set_shape_collision_filter_pair_codes(_expand_pair_codes(codes, shapes, seed_world))
    model.build_shape_contact_pairs()
    model.take_shape_rows(shapes.gather)


def _object_frequency_world(
    frequency: str, specs: dict[str, Model.AttributeSpec], values: dict[str, Any], domains: dict[Any, _Domain]
) -> np.ndarray | None:
    """Seed world of each row of an object's ``frequency``, or ``None`` while a domain it depends on is not laid out.

    Rows take their world from the frequency's world attribute, from an owner's start attribute, or from the rows the
    frequency's first reference attribute indexes, as :meth:`newton.Model.attribute_frequency_world` resolves a model's.
    """
    own = [(name, spec) for name, spec in specs.items() if spec.frequency == frequency and values[name] is not None]
    for name, spec in own:
        if spec.reference_domain == _FREQUENCY.WORLD:
            return _host(values[name]).astype(np.int64)
    for name, spec in specs.items():
        if spec.compaction_policy == "start" and spec.reference_domain == frequency:
            owner = domains.get(spec.frequency)
            return None if owner is None else np.repeat(owner.world, np.diff(_host(values[name])))
    for name, spec in own:
        if spec.reference_domain is not None:
            target = domains.get(spec.reference_domain)
            if target is None:
                return None
            rows = _host(values[name]).astype(np.int64)
            return np.where(rows >= 0, target.world[np.maximum(rows, 0)], -1)
    raise ValueError(f"No attribute declares the world of the rows of {frequency!r}.")


def _expand_object(
    model: Model,
    specs: dict[str, Model.AttributeSpec],
    values: dict[str, Any],
    domains: dict[Any, _Domain],
    layout: NewtonWorldLayout,
    expanded: dict[tuple[int, Any], tuple[Any, Any]],
) -> dict[str, Any]:
    """Expand the declared arrays of an object the model owns to every world.

    ``domains`` gains the object's own row domains. An array shared by several attributes with the same reference
    domain stays shared: ``expanded`` maps each expanded seed array, by identity and reference domain, to the seed
    array (kept alive so its identity stays unique) and its copy.
    """
    pending = list(dict.fromkeys(spec.frequency for spec in specs.values() if spec.frequency not in domains))
    while pending:
        worlds = {frequency: _object_frequency_world(frequency, specs, values, domains) for frequency in pending}
        ready = [frequency for frequency, world in worlds.items() if world is not None]
        if not ready:
            raise ValueError(f"The worlds of the rows of {pending} depend on each other.")
        for frequency in ready:
            domains[frequency] = _domain(worlds[frequency], layout.seed_world, frequency)
            pending.remove(frequency)
    arrays = {}
    for name, spec in specs.items():
        value = values[name]
        key = (id(value), spec.reference_domain)
        if value is not None and key not in expanded:
            full = _expand_rows(model, name, spec, value, domains, layout)
            copy = value if full is None else full if isinstance(full, wp.array) else _like(value, full, model.device)
            expanded[key] = (value, copy)
        arrays[name] = None if value is None else expanded[key][1]
    return arrays


def _expand_actuators(model: Model, domains: dict[Any, _Domain], layout: NewtonWorldLayout) -> None:
    """Expand every actuator's slots to every world by the declared metadata of the actuator and its components.

    Each per-slot array of the actuator and its components is replaced by its expanded copy, and the actuator is
    finalized again for its new slot count.
    """
    for actuator in model.actuators:
        local = dict(domains)
        expanded: dict[tuple[int, Any], tuple[Any, Any]] = {}
        for part in (actuator, *actuator.components):
            specs = part.attribute_specs()
            values = {name: getattr(part, name) for name in specs}
            for name, value in _expand_object(model, specs, values, local, layout, expanded).items():
                setattr(part, name, value)
        actuator.finalize()


def _expand_batteries(model: Model, domains: dict[Any, _Domain], layout: NewtonWorldLayout) -> None:
    """Build every battery object again from its arrays expanded by its declared metadata.

    A battery belongs to its world and powers actuators of that world; its arrays are its constructor's arguments.
    """
    for index, battery in enumerate(model.batteries):
        specs = battery.attribute_specs()
        values = {name: getattr(battery, name) for name in specs}
        arrays = _expand_object(model, specs, values, dict(domains), layout, {})
        model.batteries[index] = type(battery)(**arrays, world_count=model.world_count)
        model.batteries[index].reset(model)


def _expand_visual_meshes(model: Model, domains: dict[Any, _Domain], layout: NewtonWorldLayout) -> None:
    """Copy every world's deformable visual meshes to the worlds of its prototype by their declared metadata.

    A copy keeps its mesh's shared geometry; its vertices' parents move to the copy's world in the domain the mesh's
    kind names, the body-local offsets of world-fixed vertices are placed by the world's transform, and its label
    takes the world's prefix.
    """
    meshes = model.deformable_visual_meshes
    if not meshes:
        return
    specs = DeformableVisualMesh.attribute_specs()
    placement = {"local_offsets": specs["local_offsets"]}
    targets = dict(specs["parent"].kind_references.targets)
    rows = _domain(np.asarray([mesh.world for mesh in meshes], dtype=np.int64), layout.seed_world, "visual mesh")
    labels = _labels([mesh.label for mesh in meshes], rows, layout, layout.seed_world)
    copies = []
    for index, (seed, world) in enumerate(zip(rows.gather.tolist(), rows.row_world.tolist(), strict=True)):
        mesh = meshes[seed]
        parent = mesh.parent.numpy()
        (domain,) = targets[int(mesh.kind)]
        moved = _offset(parent, np.full(len(parent), world), domains[domain], layout.seed_world, False)
        local_offsets = mesh.local_offsets
        if local_offsets is not None and world >= 0:
            local = local_offsets.numpy().copy()
            placed = place_copies(placement, {"local_offsets": local, "parent": parent}, layout.xforms[world])
            if placed:
                (row_targets, values) = placed["local_offsets"]
                local[row_targets.reshape(-1)] = values.reshape(-1, 3)
                local_offsets = _like(local_offsets, local, model.device)
        copies.append(
            DeformableVisualMesh(
                mesh.kind,
                mesh.rest_vertices,
                mesh.indices,
                _like(mesh.parent, moved.astype(np.int32), model.device),
                weights=mesh.weights,
                local_offsets=local_offsets,
                uvs=mesh.uvs,
                texture=mesh.texture,
                world=world,
                label=labels[index],
                index=index,
            )
        )
    model.deformable_visual_meshes = copies


def _expand_coloring(model: Model, domains: dict[Any, _Domain], seed_world: np.ndarray, device: Any) -> None:
    """Combine every world's coloring into the particle and body color groups.

    Each world keeps its prototype's coloring (the seed coloring restricted to its seed world); the worlds' groups are
    combined by :func:`newton.utils.combine_world_colorings`, as :meth:`newton.ModelBuilder.replicate` combines them.
    The result is a valid coloring, but not the one a coloring pass over all worlds at once (as solvers that color the
    builder run before finalize) would choose.
    """
    for kind, frequency in (("particle", _FREQUENCY.PARTICLE), ("body", _FREQUENCY.BODY)):
        groups = getattr(model, f"{kind}_color_groups")
        if not groups:
            continue
        rows = domains[frequency]
        host = [group.numpy().astype(np.int64) for group in groups]
        shared = [group[rows.world[group] < 0] for group in host]
        shared = [np.where(group >= rows.seed_start[-1], group + rows.tail_shift, group) for group in shared]
        local = [
            [group[rows.world[group] == seed] - rows.seed_start[seed] for group in host]
            for seed in range(len(rows.seed_start) - 1)
        ]
        local = [[group for group in seed_groups if len(group)] for seed_groups in local]
        copies = ((local[seed], int(rows.start[world])) for world, seed in enumerate(seed_world.tolist()))
        merged = combine_world_colorings([group for group in shared if len(group)], copies)
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
