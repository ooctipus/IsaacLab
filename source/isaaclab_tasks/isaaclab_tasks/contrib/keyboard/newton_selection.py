# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-local indexing of Newton model frequencies and episode participation.

Selections retain explicit native bindings and indices, never copies of simulation
state. Ordinary solver sleeping does not change membership. Dense gathers are an
explicit policy boundary; compact indices and world offsets are available to Warp
terms without a host synchronization.
"""

from __future__ import annotations

import re
from dataclasses import MISSING, fields, is_dataclass
from typing import Any, Literal

import numpy as np
import torch
import warp as wp
from newton import Control, Model, State
from newton.solvers import SolverMuJoCo

from isaaclab.utils import configclass

BODY = "body"
JOINT_COORD = "joint_coord"
JOINT_DOF = "joint_dof"


def bind_selectors(value, resolve):
    """Resolve declarative task selections recursively before constructing manager terms."""
    if isinstance(value, NewtonSelectorCfg):
        return resolve(value)
    if isinstance(value, dict):
        for key, item in value.items():
            value[key] = bind_selectors(item, resolve)
    elif isinstance(value, (tuple, list)):
        return type(value)(bind_selectors(item, resolve) for item in value)
    elif is_dataclass(value):
        for field in fields(value):
            setattr(value, field.name, bind_selectors(getattr(value, field.name), resolve))
    return value


@wp.struct
class _ScalarSource:
    values: wp.array[float]


@wp.struct
class NewtonScalarField:
    """Borrow native float32 arrays at a selection's dense policy boundary.

    Storage belongs to the selection and its native owners. Reacquire after a
    group rebind; episode mask updates remain visible through the borrowed view.
    """

    sources: wp.array[_ScalarSource]
    source_ids: wp.array[int]
    ids: wp.array2d[int]
    active: wp.array2d[bool]


@wp.func
def scalar_field_read(field: NewtonScalarField, world: int, slot: int) -> float:
    """Read one selected scalar, returning zero for an excluded policy slot."""
    value = float(0.0)
    if field.active[world, slot]:
        source = field.sources[field.source_ids[world]]
        value = source.values[field.ids[world, slot]]
    return value


@wp.func
def scalar_field_write(field: NewtonScalarField, world: int, slot: int, value: float):
    """Write one selected scalar, preserving native values in excluded slots."""
    if field.active[world, slot]:
        source = field.sources[field.source_ids[world]]
        source.values[field.ids[world, slot]] = value


@wp.struct
class _PoseSource:
    values: wp.array[wp.transform]


@wp.kernel
def _gather_scalars(
    sources: wp.array[_ScalarSource],
    source_ids: wp.array[int],
    ids: wp.array2d[int],
    active: wp.array2d[bool],
    fill: float,
    out: wp.array2d[float],
):
    world, slot = wp.tid()
    value = fill
    if active[world, slot]:
        source_id = source_ids[world]
        source = sources[source_id]
        index = ids[world, slot]
        value = source.values[index]
    out[world, slot] = value


@wp.kernel
def _gather_poses(
    sources: wp.array[_PoseSource],
    source_ids: wp.array[int],
    ids: wp.array2d[int],
    active: wp.array2d[bool],
    fill: float,
    out: wp.array2d[wp.transform],
):
    world, slot = wp.tid()
    value = wp.transform(wp.vec3(fill), wp.quat(fill, fill, fill, fill))
    if active[world, slot]:
        source_id = source_ids[world]
        source = sources[source_id]
        index = ids[world, slot]
        value = source.values[index]
    out[world, slot] = value


@wp.kernel
def _scatter_values(
    sources: wp.array[Any],
    source_ids: wp.array[int],
    ids: wp.array2d[int],
    active: wp.array2d[bool],
    worlds: wp.array[int],
    values: wp.array2d[Any],
):
    row, slot = wp.tid()
    world = worlds[row]
    if active[world, slot]:
        source_id = source_ids[world]
        source = sources[source_id]
        index = ids[world, slot]
        source.values[index] = values[row, slot]


@configclass
class NewtonSelectorCfg:
    """Match full model labels, preserving pattern order and model order within each world.

    Overlapping patterns select an entity only once. Global entities (world -1)
    are excluded. ``count_per_world`` validates static topology, before masking.
    Joint patterns expand into all coordinates or DOFs of each matched joint.
    """

    frequency: Literal["body", "joint_coord", "joint_dof"] = MISSING
    path: str | tuple[str, ...] | list[str] = MISSING
    count_per_world: int | None = None
    dense_width: int | None = None
    """Explicit policy width for a group of differently sized native selections."""


@wp.kernel
def _count_active(
    starts: wp.array[int],
    bodies: wp.array[int],
    body_active: wp.array[bool],
    world_active: wp.array[bool],
    counts: wp.array[int],
):
    world = wp.tid()
    count = int(0)
    if world_active[world]:
        for i in range(starts[world], starts[world + 1]):
            if body_active[bodies[i]]:
                count += 1
    counts[world] = count


@wp.kernel
def _compact_active(
    starts: wp.array[int],
    ids: wp.array[int],
    bodies: wp.array[int],
    body_active: wp.array[bool],
    world_active: wp.array[bool],
    world_start: wp.array[int],
    freq_ids: wp.array[int],
    env_ids: wp.array[int],
    slot_ids: wp.array[int],
    active: wp.array[bool],
):
    world = wp.tid()
    dst = world_start[world]
    for i in range(starts[world], starts[world + 1]):
        enabled = world_active[world] and body_active[bodies[i]]
        active[i] = enabled
        if enabled:
            freq_ids[dst] = ids[i]
            env_ids[dst] = world
            slot_ids[dst] = i - starts[world]
            dst += 1


class NewtonSelection:
    """Static model binding with a compact, episode-filtered device representation.

    Only entries before ``world_start[-1]`` in ``freq_ids/env_ids/slot_ids`` are
    valid. Storage and pointers remain stable across membership changes. Empty
    worlds have equal adjacent offsets. ``dense`` explicitly pads excluded slots.
    """

    # Runtime binding storage stays out of the manager's configuration serializer.
    # The instance dictionary contains only the original declarative selector.
    __slots__ = (
        "__dict__",
        "owner",
        "counts",
        "capacity",
        "ids",
        "body_ids",
        "starts",
        "freq_ids",
        "env_ids",
        "slot_ids",
        "world_start",
        "_counts",
        "active",
        "joint_ids",
        "_dense_width",
        "_dense_ids",
        "_dense_active",
        "_scalar_views",
        "_scalar_source_ids",
    )

    def __init__(self, owner, cfg, rows=None, bodies=None, *, source: NewtonSelection | None = None):
        self.frequency = cfg.frequency
        self.path = cfg.path
        self.count_per_world = cfg.count_per_world
        self.dense_width = cfg.dense_width
        self.owner = owner
        if source is None:
            self.counts = tuple(map(len, rows))
            self.ids = wp.array([i for row in rows for i in row], dtype=wp.int32, device=owner.model.device)
            self.body_ids = wp.array(bodies, dtype=wp.int32, device=owner.model.device)
            self.starts = wp.array(np.cumsum([0, *self.counts]), dtype=wp.int32, device=owner.model.device)
        else:
            self.counts = (source.width,) * owner.model.world_count
            worlds = owner.world_ids.to(torch.int32)[:, None]
            stride = {
                BODY: source.owner.model.body_count,
                JOINT_COORD: source.owner.model.joint_coord_count,
                JOINT_DOF: source.owner.model.joint_dof_count,
            }[cfg.frequency]
            self.ids = wp.from_torch((wp.to_torch(source.ids)[None] + worlds * stride).flatten())
            self.body_ids = wp.from_torch(
                (wp.to_torch(source.body_ids)[None] + worlds * source.owner.model.body_count).flatten()
            )
            self.starts = wp.from_torch(
                torch.arange(owner.model.world_count + 1, dtype=torch.int32, device=worlds.device) * source.width
            )
        self.capacity = sum(self.counts)
        self.freq_ids = wp.empty_like(self.ids)
        self.env_ids = wp.empty_like(self.ids)
        self.slot_ids = wp.empty_like(self.ids)
        self.world_start = wp.zeros(owner.model.world_count + 1, dtype=wp.int32, device=owner.model.device)
        self._counts = wp.zeros_like(self.world_start)
        self.active = wp.zeros(self.capacity, dtype=wp.bool, device=owner.model.device)
        self.joint_ids: wp.array | None = None
        if source is not None and source.joint_ids is not None:
            self.joint_ids = wp.from_torch(
                (wp.to_torch(source.joint_ids)[None] + worlds * source.owner.model.joint_count).flatten()
            )
        self._dense_width = self.counts[0] if self.counts and len(set(self.counts)) == 1 else None
        self._dense_ids = None
        self._dense_active = None
        self._scalar_views = {}
        self._scalar_source_ids = None
        if self._dense_width is not None:
            self._dense_ids = wp.to_torch(self.ids).reshape(len(self.counts), self._dense_width)
            self._dense_active = wp.to_torch(self.active).reshape(len(self.counts), self._dense_width)
        self.refresh()

    def refresh(self) -> None:
        """Rebuild membership on device without changing topology or allocating storage."""
        owner = self.owner
        wp.launch(
            _count_active,
            dim=len(self.counts),
            inputs=[self.starts, self.body_ids, owner.body_active, owner.world_active],
            outputs=[self._counts],
            device=owner.model.device,
        )
        wp.utils.array_scan(self._counts, self.world_start, inclusive=False)
        wp.launch(
            _compact_active,
            dim=len(self.counts),
            inputs=[self.starts, self.ids, self.body_ids, owner.body_active, owner.world_active, self.world_start],
            outputs=[self.freq_ids, self.env_ids, self.slot_ids, self.active],
            device=owner.model.device,
        )

    def __deepcopy__(self, memo):
        # Manager configs copy their parameters; runtime bindings belong to one env.
        memo[id(self)] = self
        return self

    def dense_ids(self) -> torch.Tensor:
        """Return static IDs at a uniform policy boundary; reject ragged topology."""
        if self._dense_ids is None:
            raise ValueError(
                "Dense selection requires equal static counts per world; use compact IDs for ragged terms."
            )
        return self._dense_ids

    @property
    def width(self) -> int:
        """Number of slots at this selection's explicit dense boundary."""
        if self._dense_width is None:
            raise ValueError("Dense selection requires equal static counts per world.")
        return self._dense_width

    def joint_types(self) -> torch.Tensor:
        """Return the native joint types underlying selected coordinates or DOFs."""
        if self.joint_ids is None:
            raise ValueError("Joint types require a coordinate or DOF selection.")
        return wp.to_torch(self.owner.model.joint_type)[wp.to_torch(self.joint_ids)].reshape(-1, self.width)

    @property
    def use_coord_layout_targets(self) -> bool:
        """Whether joint-position controls use coordinate rather than DOF indices."""
        return self.owner.model.use_coord_layout_targets

    @property
    def native_bindings(self):
        """Yield model-scoped indices and their logical world IDs for native kernels."""
        return ((self, self.owner.world_ids),)

    def dense_active(self) -> torch.Tensor:
        """Return participation in stable per-world slot order."""
        if self._dense_active is None:
            raise ValueError("Dense selection requires equal static counts per world.")
        return self._dense_active

    def dense(self, values: wp.array, fill: float = 0.0) -> torch.Tensor:
        """Gather an array in this frequency and pad excluded slots with ``fill``."""
        selected = wp.to_torch(values)[self.dense_ids()]
        active = self.dense_active()
        while active.ndim < selected.ndim:
            active = active.unsqueeze(-1)
        return torch.where(active, selected, fill)

    def read_state(self, attribute: str, fill: float = 0.0) -> torch.Tensor:
        """Gather a native state attribute through this explicit model binding."""
        if self.owner.state is None:
            raise RuntimeError("This selection has no bound Newton state.")
        return self.dense(getattr(self.owner.state, attribute), fill)

    def scalar_field(self, source: Literal["state", "model", "control"], attribute: str) -> NewtonScalarField:
        """Borrow a float32 field for a fused device term, without gathering state.

        Native array storage must remain stable while this binding is in use.
        Reads mask excluded slots to zero; writes leave their native values intact.
        """
        key = (source, attribute)
        if key not in self._scalar_views:
            if source not in ("state", "model", "control"):
                raise ValueError(f"Unknown native field source: {source!r}.")
            values = getattr(getattr(self.owner, source), attribute)
            if values.dtype != wp.float32 or values.ndim != 1:
                raise TypeError("Scalar fields require a one-dimensional float32 native array.")
            shape = (len(self.counts), self.width)
            if self._scalar_source_ids is None:
                self._scalar_source_ids = wp.zeros(len(self.counts), dtype=wp.int32, device=values.device)
            descriptor = _ScalarSource()
            descriptor.values = values
            view = NewtonScalarField()
            view.sources = wp.array([descriptor], dtype=_ScalarSource, device=values.device)
            view.source_ids = self._scalar_source_ids
            view.ids, view.active = self.ids.reshape(shape), self.active.reshape(shape)
            self._scalar_views[key] = view
        return self._scalar_views[key]

    def read_model(self, attribute: str, fill: float = 0.0) -> torch.Tensor:
        """Gather a native model attribute in this selection's frequency."""
        return self.dense(getattr(self.owner.model, attribute), fill)

    def write_state(self, attribute: str, values: torch.Tensor, env_ids=None) -> None:
        """Write participating selected state entries; values use the selected world order."""
        if self.owner.state is None:
            raise RuntimeError("This selection has no bound Newton state.")
        rows = slice(None) if env_ids is None else env_ids
        target = wp.to_torch(getattr(self.owner.state, attribute))
        ids, active = self.dense_ids()[rows], self.dense_active()[rows]
        while active.ndim < values.ndim:
            active = active.unsqueeze(-1)
        target[ids] = torch.where(active, values, target[ids])

    def write_control(self, attribute: str, values: torch.Tensor, env_ids=None) -> None:
        """Write participating selected control entries in stable world/slot order."""
        if self.owner.control is None:
            raise RuntimeError("This selection has no bound Newton control.")
        target = wp.to_torch(getattr(self.owner.control, attribute))
        worlds = slice(None) if env_ids is None else env_ids
        ids = self.dense_ids()[worlds]
        target[ids] = torch.where(self.dense_active()[worlds], values, target[ids])


class NewtonSelections:
    """One task's model bindings and authoritative episode participation masks."""

    def __init__(
        self,
        model: Model,
        *,
        state: State | None = None,
        control: Control | None = None,
        solver: SolverMuJoCo | None = None,
        source: NewtonSelections | None = None,
    ):
        self.model = model
        self.state = state
        self.control = control
        self.solver = solver
        self.world_ids = torch.arange(model.world_count, dtype=torch.long, device=str(model.device))
        self._source = source
        self.body_active = wp.ones(model.body_count, dtype=wp.bool, device=model.device)
        self.world_active = wp.ones(model.world_count, dtype=wp.bool, device=model.device)
        self._bindings: dict[tuple, NewtonSelection] = {}
        self._retired = False
        if source is not None:
            if source.model.world_count != 1 or model._replication_source is not source.model:
                raise ValueError("Selection replication requires the exact prepared one-world source model.")
            roots = source.root_joint_ids[None]
            expanded = roots + self.world_ids[:, None] * source.model.joint_count
            self.root_joint_ids = torch.where(roots >= 0, expanded, -1).flatten()
            return
        self._body_world = model.body_world.numpy()
        self._joint_world = model.joint_world.numpy()
        self._joint_child = model.joint_child.numpy()
        self._q_start = model.joint_q_start.numpy()
        self._qd_start = model.joint_qd_start.numpy()
        self.root_joint_ids = torch.full((model.body_count,), -1, dtype=torch.int64, device=str(model.device))
        roots = np.flatnonzero(model.joint_parent.numpy() == -1)
        self.root_joint_ids[torch.as_tensor(self._joint_child[roots], device=str(model.device))] = torch.as_tensor(
            roots, device=str(model.device)
        )

    def resolve(self, cfg: NewtonSelectorCfg) -> NewtonSelection:
        """Resolve a declarative selector once against this finalized model."""
        if self._retired:
            raise RuntimeError("Cannot resolve selections from a retired owner.")
        patterns = (cfg.path,) if isinstance(cfg.path, str) else tuple(cfg.path)
        key = (cfg.frequency, patterns, cfg.count_per_world, cfg.dense_width)
        if key in self._bindings:
            return self._bindings[key]
        if self._source is not None:
            selection = NewtonSelection(self, cfg, source=self._source.resolve(cfg))
            self._bindings[key] = selection
            return selection
        if cfg.frequency not in (BODY, JOINT_COORD, JOINT_DOF):
            raise ValueError(f"Unknown Newton frequency: {cfg.frequency!r}")
        labels = self.model.body_label if cfg.frequency == BODY else self.model.joint_label
        worlds = self._body_world if cfg.frequency == BODY else self._joint_world
        rows = [[] for _ in range(self.model.world_count)]
        owners = [[] for _ in rows]
        seen = set()
        for pattern in patterns:
            regex = re.compile(pattern)
            matched = False
            for entity, (label, world) in enumerate(zip(labels, worlds, strict=True)):
                if world < 0 or not regex.fullmatch(label):
                    continue
                matched = True
                if entity in seen:
                    continue
                seen.add(entity)
                if cfg.frequency == BODY:
                    indices = [entity]
                    body = entity
                else:
                    starts = self._q_start if cfg.frequency == JOINT_COORD else self._qd_start
                    indices = range(int(starts[entity]), int(starts[entity + 1]))
                    body = int(self._joint_child[entity])
                rows[world].extend(indices)
                owners[world].extend([body] * len(indices))
            if not matched:
                raise ValueError(f"Selector {pattern!r} matched no {cfg.frequency} entities.")
        if cfg.count_per_world is not None and any(len(row) != cfg.count_per_world for row in rows):
            raise ValueError(
                f"Expected {cfg.count_per_world} {cfg.frequency} entries per world; got {list(map(len, rows))}."
            )
        selection = NewtonSelection(self, cfg, rows, [b for row in owners for b in row])
        if cfg.frequency != BODY:
            starts = self._q_start if cfg.frequency == JOINT_COORD else self._qd_start
            inverse = np.repeat(np.arange(self.model.joint_count), np.diff(starts))
            flat_ids = np.array([i for row in rows for i in row], dtype=np.int32)
            selection.joint_ids = wp.array(inverse[flat_ids], dtype=wp.int32, device=self.model.device)
        self._bindings[key] = selection
        return selection

    def retire(self) -> None:
        """Release cached bindings and reject new ones after native consumers finish.

        Already returned selections remain valid and retain their native resources.
        Clearing this owner's cache breaks their reference cycle without changing
        that standalone lifetime contract. Repeated retirement is harmless.
        """
        self._retired = True
        self._bindings.clear()

    def refresh(self) -> None:
        """Refresh all bound selections after episode masks are changed at reset."""
        for selection in self._bindings.values():
            selection.refresh()


class NewtonSelectionGroup:
    """Compose model-scoped selections into stable logical policy rows.

    Native bindings retain compact, exact topology. Only this explicit dense
    boundary pads missing slots. GPU descriptors address each native array
    directly; this object never assembles a synthetic Newton model or state.
    """

    __slots__ = (
        "__dict__",
        "_parts",
        "_num_envs",
        "_width",
        "_worlds",
        "_ids",
        "_active",
        "_source_ids",
        "_tables",
        "_scalar_views",
        "counts",
        "joint_ids",
        "_device",
    )

    def __init__(self, cfg: NewtonSelectorCfg, parts, num_envs: int):
        self.frequency, self.path = cfg.frequency, cfg.path
        self.count_per_world, self.dense_width = cfg.count_per_world, cfg.dense_width
        self._num_envs = num_envs
        self._width = cfg.dense_width or cfg.count_per_world or max(part.width for part, _ in parts)
        self._device = parts[0][0].owner.model.device
        self._worlds = wp.from_torch(torch.arange(num_envs, dtype=torch.int32, device=str(self._device)))
        self._ids = wp.empty((num_envs, self._width), dtype=wp.int32, device=self._device)
        self._active = wp.empty((num_envs, self._width), dtype=wp.bool, device=self._device)
        self._source_ids = wp.empty(num_envs, dtype=wp.int32, device=self._device)
        self._tables = {}
        # Counts here describe policy slots. Native counts remain on each binding.
        self.counts = (self._width,) * num_envs
        self.rebind(parts)

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    @property
    def width(self) -> int:
        """Number of slots at the dense policy boundary."""
        return self._width

    @property
    def native_bindings(self):
        """Model-scoped selections paired with stable logical world IDs."""
        return self._parts

    @property
    def use_coord_layout_targets(self) -> bool:
        """Require one compatible control convention across the composed models."""
        modes = {part.use_coord_layout_targets for part, _ in self._parts}
        if len(modes) != 1:
            raise ValueError("Grouped controls require the same target layout in every model.")
        return modes.pop()

    def rebind(self, parts) -> None:
        """Publish replacement native bindings after their old GPU consumers finish."""
        if sum(len(worlds) for _, worlds in parts) != self._num_envs:
            raise ValueError("Grouped selections must cover every logical world exactly once.")
        if any(part.width > self._width for part, _ in parts):
            raise ValueError("A native selection exceeds the configured dense policy width.")
        self._parts = tuple(parts)
        # Logical rows can move while the borrowed native arrays stay unchanged.
        self._tables = {
            key: entry
            for key, entry in self._tables.items()
            if len(entry[2]) == len(self._parts)
            and all(
                getattr(getattr(part.owner, key[0]), key[1]) is array
                for (part, _), array in zip(self._parts, entry[2], strict=True)
            )
        }
        self._scalar_views = {}
        self._ids.fill_(-1)
        self._active.zero_()
        ids, active, source_ids = map(wp.to_torch, (self._ids, self._active, self._source_ids))
        coverage = torch.zeros(self._num_envs, dtype=torch.int32, device=str(self._device))
        for source, (part, worlds) in enumerate(parts):
            if part.frequency != self.frequency or len(worlds) != len(part.counts):
                raise ValueError("Native selection frequency/world count does not match its logical binding.")
            ids[worlds, : part.width] = part.dense_ids()
            active[worlds, : part.width] = part.dense_active()
            source_ids[worlds] = source
            coverage.index_add_(0, worlds, torch.ones_like(worlds, dtype=torch.int32))
        torch._assert_async(
            torch.all(coverage == 1), "Grouped selections require disjoint, complete logical world IDs."
        )
        self.joint_ids = (
            None
            if self.frequency == BODY
            else wp.from_torch(torch.cat([wp.to_torch(part.joint_ids) for part, _ in parts]))
        )

    def dense_active(self) -> torch.Tensor:
        """Return participation in stable logical-world/policy-slot order."""
        return wp.to_torch(self._active)

    def refresh(self) -> None:
        """Refresh logical membership after native owners refresh their authoritative masks."""
        active = wp.to_torch(self._active)
        for part, worlds in self._parts:
            active[worlds, : part.width] = part.dense_active()

    def joint_types(self) -> torch.Tensor:
        """Read joint types once in logical-world order for scalar-joint validation."""
        values = torch.zeros((self._num_envs, self.width), dtype=torch.int32, device=str(self._device))
        for part, worlds in self._parts:
            values[worlds, : part.width] = part.joint_types()
        return values

    def _table(self, source: str, attribute: str):
        key = (source, attribute)
        if key not in self._tables:
            arrays = [getattr(getattr(part.owner, source), attribute) for part, _ in self._parts]
            dtype = arrays[0].dtype
            if any(array.dtype != dtype for array in arrays):
                raise TypeError("A grouped attribute must use one native dtype.")
            if dtype == wp.float32:
                struct = _ScalarSource
            elif dtype == wp.transform:
                struct = _PoseSource
            else:
                raise TypeError(f"Unsupported grouped attribute dtype: {dtype}.")
            descriptors = []
            for array in arrays:
                descriptor = struct()
                descriptor.values = array
                descriptors.append(descriptor)
            self._tables[key] = (wp.array(descriptors, dtype=struct, device=self._device), dtype, tuple(arrays))
        return self._tables[key][:2]

    def _read(self, source: str, attribute: str, fill: float):
        table, dtype = self._table(source, attribute)
        out = wp.empty((self._num_envs, self.width), dtype=dtype, device=self._device)
        kernel = _gather_scalars if dtype == wp.float32 else _gather_poses
        wp.launch(
            kernel,
            dim=out.shape,
            inputs=[table, self._source_ids, self._ids, self._active, fill],
            outputs=[out],
            device=self._device,
        )
        return wp.to_torch(out)

    def scalar_field(self, source: Literal["state", "model", "control"], attribute: str) -> NewtonScalarField:
        """Borrow a float32 field across populations; reacquire after ``rebind``."""
        key = (source, attribute)
        if key not in self._scalar_views:
            if source not in ("state", "model", "control"):
                raise ValueError(f"Unknown native field source: {source!r}.")
            table, dtype = self._table(source, attribute)
            if dtype != wp.float32:
                raise TypeError("Scalar fields require a one-dimensional float32 native array.")
            view = NewtonScalarField()
            view.sources, view.source_ids = table, self._source_ids
            view.ids, view.active = self._ids, self._active
            self._scalar_views[key] = view
        return self._scalar_views[key]

    def read_state(self, attribute: str, fill: float = 0.0) -> torch.Tensor:
        """Gather a state attribute across native populations in one GPU launch."""
        return self._read("state", attribute, fill)

    def read_model(self, attribute: str, fill: float = 0.0) -> torch.Tensor:
        """Gather a model attribute across native populations in one GPU launch."""
        return self._read("model", attribute, fill)

    def _write(self, source: str, attribute: str, values: torch.Tensor, env_ids=None) -> None:
        table, dtype = self._table(source, attribute)
        worlds = self._worlds if env_ids is None else wp.from_torch(env_ids.to(torch.int32))
        if values.shape[:2] != (len(worlds), self.width):
            raise ValueError("Grouped writes require one policy-width row per selected logical world.")
        wp.launch(
            _scatter_values,
            dim=(len(worlds), self.width),
            inputs=[
                table,
                self._source_ids,
                self._ids,
                self._active,
                worlds,
                wp.from_torch(values.contiguous(), dtype=dtype),
            ],
            device=self._device,
        )

    def write_state(self, attribute: str, values: torch.Tensor, env_ids=None) -> None:
        """Write selected logical rows directly into their native population state."""
        self._write("state", attribute, values, env_ids)

    def write_control(self, attribute: str, values: torch.Tensor, env_ids=None) -> None:
        """Write policy rows directly into their native population control arrays."""
        self._write("control", attribute, values, env_ids)
