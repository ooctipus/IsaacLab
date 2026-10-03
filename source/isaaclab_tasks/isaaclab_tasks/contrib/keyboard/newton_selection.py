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

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import torch
import warp as wp
from newton import Control, Model, State
from newton.solvers import SolverMuJoCo

BODY = "body"
JOINT_COORD = "joint_coord"
JOINT_DOF = "joint_dof"


def _validate_field_domain(model: Model, index_domain: str, attribute: str) -> None:
    """Match a schema field to its index domain before borrowing its values."""
    expected = Model.AttributeFrequency[index_domain.upper()]
    actual = model.get_attribute_frequency(attribute)
    if actual != expected:
        raise ValueError(
            f"Field {attribute!r} has index domain {getattr(actual, 'name', actual)}; selection uses {expected.name}."
        )


def _validate_write_env_indices(env_ids: torch.Tensor | None, num_envs: int, device: torch.device) -> int:
    """Validate an injective environment subset; the complete domain needs no scan."""
    if env_ids is None:
        return num_envs
    if (
        not isinstance(env_ids, torch.Tensor)
        or env_ids.ndim != 1
        or env_ids.dtype != torch.long
        or env_ids.device != device
    ):
        raise ValueError("Environment indices must be a one-dimensional int64 tensor on the selection device.")
    torch._assert_async(((env_ids >= 0) & (env_ids < num_envs)).all(), "Environment indices are outside the task.")
    seen = torch.zeros(num_envs, dtype=torch.int32, device=device)
    seen.scatter_add_(0, env_ids, torch.ones_like(env_ids, dtype=torch.int32))
    torch._assert_async((seen <= 1).all(), "Environment indices must be unique.")
    return len(env_ids)


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
def scalar_field_active(field: NewtonScalarField, world: int, slot: int) -> bool:
    """Whether a selected scalar participates in the current task episode."""
    return field.active[world, slot]


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


@wp.struct
class NewtonPoseField:
    """Borrow read-only body poses [m, xyzw] with the selection's episode mask.

    This is a derived observation, not a generalized coordinate write interface.
    Native storage must remain stable; reacquire after a group rebind.
    """

    sources: wp.array[_PoseSource]
    source_ids: wp.array[int]
    ids: wp.array2d[int]
    active: wp.array2d[bool]


@wp.func
def pose_field_active(field: NewtonPoseField, world: int, slot: int) -> bool:
    if world < 0 or world >= field.active.shape[0] or slot < 0 or slot >= field.active.shape[1]:
        return False
    return field.active[world, slot]


@wp.func
def pose_field_read(field: NewtonPoseField, world: int, slot: int) -> wp.transform:
    """Read one pose; excluded slots have zero position and zero quaternion."""
    value = wp.transform(wp.vec3(0.0), wp.quat(0.0, 0.0, 0.0, 0.0))
    if pose_field_active(field, world, slot):
        source = field.sources[field.source_ids[world]]
        value = source.values[field.ids[world, slot]]
    return value


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
    # The empty instance dictionary deliberately carries no config provenance.
    __slots__ = (
        "__dict__",
        "owner",
        "index_domain",
        "world_selection_counts",
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
        "_pose_views",
        "_source_ids",
    )

    def __init__(self, owner, index_domain: str, rows, bodies):
        self.index_domain = index_domain
        self.owner = owner
        self.world_selection_counts = tuple(map(len, rows))
        self.ids = wp.array([i for row in rows for i in row], dtype=wp.int32, device=owner.model.device)
        self.body_ids = wp.array(bodies, dtype=wp.int32, device=owner.model.device)
        self.starts = wp.array(np.cumsum([0, *self.world_selection_counts]), dtype=wp.int32, device=owner.model.device)
        self.capacity = sum(self.world_selection_counts)
        self.freq_ids = wp.empty_like(self.ids)
        self.env_ids = wp.empty_like(self.ids)
        self.slot_ids = wp.empty_like(self.ids)
        self.world_start = wp.zeros(owner.model.world_count + 1, dtype=wp.int32, device=owner.model.device)
        self._counts = wp.zeros_like(self.world_start)
        self.active = wp.zeros(self.capacity, dtype=wp.bool, device=owner.model.device)
        self.joint_ids: wp.array | None = None
        self._dense_width = (
            self.world_selection_counts[0]
            if self.world_selection_counts and len(set(self.world_selection_counts)) == 1
            else None
        )
        self._dense_ids = None
        self._dense_active = None
        self._scalar_views = {}
        self._pose_views = {}
        self._source_ids = None
        if self._dense_width is not None:
            self._dense_ids = wp.to_torch(self.ids).reshape(len(self.world_selection_counts), self._dense_width)
            self._dense_active = wp.to_torch(self.active).reshape(len(self.world_selection_counts), self._dense_width)
        self.refresh()

    def refresh(self) -> None:
        """Rebuild membership on device without changing topology or allocating storage."""
        owner = self.owner
        wp.launch(
            _count_active,
            dim=len(self.world_selection_counts),
            inputs=[self.starts, self.body_ids, owner.body_active, owner.world_active],
            outputs=[self._counts],
            device=owner.model.device,
        )
        wp.utils.array_scan(self._counts, self.world_start, inclusive=False)
        wp.launch(
            _compact_active,
            dim=len(self.world_selection_counts),
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

    def active_counts(self) -> torch.Tensor:
        """Current episode-participating count per model world (GPU int32)."""
        return wp.to_torch(self._counts)[:-1]

    @property
    def dense_shape(self) -> tuple[int, int]:
        """Environment and slot extents at the explicit dense policy boundary."""
        return len(self.world_selection_counts), self.width

    def joint_types(self) -> torch.Tensor:
        """Return the native joint types underlying selected coordinates or DOFs."""
        if self.joint_ids is None:
            raise ValueError("Joint types require a coordinate or DOF selection.")
        return wp.to_torch(self.owner.model.joint_type)[wp.to_torch(self.joint_ids)].reshape(
            len(self.world_selection_counts), self.width
        )

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
        """Gather an array in this index domain and pad excluded slots with ``fill``."""
        selected = wp.to_torch(values)[self.dense_ids()]
        active = self.dense_active()
        while active.ndim < selected.ndim:
            active = active.unsqueeze(-1)
        return torch.where(active, selected, fill)

    def read_state(self, attribute: str, fill: float = 0.0) -> torch.Tensor:
        """Gather a native state attribute through this explicit model binding."""
        if self.owner.state is None:
            raise RuntimeError("This selection has no bound Newton state.")
        _validate_field_domain(self.owner.model, self.index_domain, attribute)
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
            _validate_field_domain(self.owner.model, self.index_domain, attribute)
            values = getattr(getattr(self.owner, source), attribute)
            if values.dtype != wp.float32 or values.ndim != 1:
                raise TypeError("Scalar fields require a one-dimensional float32 native array.")
            shape = (len(self.world_selection_counts), self.width)
            if self._source_ids is None:
                self._source_ids = wp.zeros(len(self.world_selection_counts), dtype=wp.int32, device=values.device)
            descriptor = _ScalarSource()
            descriptor.values = values
            view = NewtonScalarField()
            view.sources = wp.array([descriptor], dtype=_ScalarSource, device=values.device)
            view.source_ids = self._source_ids
            view.ids, view.active = self.ids.reshape(shape), self.active.reshape(shape)
            self._scalar_views[key] = view
        return self._scalar_views[key]

    def pose_field(self, source: Literal["state", "model"], attribute: str) -> NewtonPoseField:
        """Borrow body transforms without materializing a dense pose tensor."""
        key = source, attribute
        if key not in self._pose_views:
            if source not in ("state", "model") or self.index_domain != BODY:
                raise ValueError("Pose fields require a body selection and state/model source.")
            _validate_field_domain(self.owner.model, self.index_domain, attribute)
            values = getattr(getattr(self.owner, source), attribute)
            if values.dtype != wp.transform or values.ndim != 1:
                raise TypeError("Pose fields require a one-dimensional transform array.")
            if self._source_ids is None:
                self._source_ids = wp.zeros(self.dense_shape[0], dtype=wp.int32, device=values.device)
            descriptor = _PoseSource()
            descriptor.values = values
            field = NewtonPoseField()
            field.sources = wp.array([descriptor], dtype=_PoseSource, device=values.device)
            field.source_ids = self._source_ids
            field.ids, field.active = self.ids.reshape(self.dense_shape), self.active.reshape(self.dense_shape)
            self._pose_views[key] = field
        return self._pose_views[key]

    def read_model(self, attribute: str, fill: float = 0.0) -> torch.Tensor:
        """Gather a native model attribute in this selection's index domain."""
        _validate_field_domain(self.owner.model, self.index_domain, attribute)
        return self.dense(getattr(self.owner.model, attribute), fill)

    def _write(self, source: str, attribute: str, values: torch.Tensor, env_ids) -> None:
        native = getattr(self.owner, source)
        if native is None:
            raise RuntimeError(f"This selection has no bound Newton {source}.")
        _validate_field_domain(self.owner.model, self.index_domain, attribute)
        target = wp.to_torch(getattr(native, attribute))
        count = _validate_write_env_indices(env_ids, len(self.world_selection_counts), target.device)
        expected = (count, self.width, *target.shape[1:])
        if (
            not isinstance(values, torch.Tensor)
            or values.shape != expected
            or values.dtype != target.dtype
            or values.device != target.device
        ):
            raise ValueError("Writes require matching native dtype, device and policy-width value rows.")
        if count == 0:
            return
        rows = slice(None) if env_ids is None else env_ids
        ids, active = self.dense_ids()[rows], self.dense_active()[rows]
        while active.ndim < values.ndim:
            active = active.unsqueeze(-1)
        target[ids] = torch.where(active, values, target[ids])

    def write_state(self, attribute: str, values: torch.Tensor, env_ids=None) -> None:
        """Write participating state entries for a unique, in-range environment subset."""
        self._write("state", attribute, values, env_ids)

    def write_control(self, attribute: str, values: torch.Tensor, env_ids=None) -> None:
        """Write participating controls for a unique, in-range environment subset."""
        self._write("control", attribute, values, env_ids)


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
        self.source = source
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

    def bind(self, index_domain: str, ids: Sequence[int] | np.ndarray) -> NewtonSelection:
        """Bind ordered integer indices in this model, grouped by world.

        IDs are global model indices, including for replicated models. Duplicate,
        out-of-range and global-world IDs are rejected; empty bindings are valid.
        """
        if self._retired:
            raise RuntimeError("Cannot bind selections from a retired owner.")
        if index_domain not in (BODY, JOINT_COORD, JOINT_DOF):
            raise ValueError(f"Unknown Newton index domain: {index_domain!r}")
        values = np.asarray(ids)
        if values.ndim != 1 or ((values.size or isinstance(ids, np.ndarray)) and values.dtype.kind not in "iu"):
            raise ValueError("Selection IDs must be a one-dimensional integer sequence.")
        if not isinstance(ids, np.ndarray) and any(isinstance(value, (bool, np.bool_)) for value in ids):
            raise ValueError("Selection IDs must be integers, not booleans.")
        count = {
            BODY: self.model.body_count,
            JOINT_COORD: self.model.joint_coord_count,
            JOINT_DOF: self.model.joint_dof_count,
        }[index_domain]
        if np.any(values < 0) or np.any(values >= count) or len(np.unique(values)) != len(values):
            raise ValueError("Selection IDs must be unique valid model indices.")
        values = values.astype(np.int32)
        key = (index_domain, tuple(map(int, values)))
        if key in self._bindings:
            return self._bindings[key]
        topology = self if self.source is None else self.source
        stride = {
            BODY: topology.model.body_count,
            JOINT_COORD: topology.model.joint_coord_count,
            JOINT_DOF: topology.model.joint_dof_count,
        }[index_domain]
        local = values if self.source is None else values % stride
        if index_domain == BODY:
            bodies = local
            joints = None
            worlds = topology._body_world[local]
        else:
            starts = topology._q_start if index_domain == JOINT_COORD else topology._qd_start
            joints = np.searchsorted(starts[1:], local, side="right").astype(np.int32)
            bodies = topology._joint_child[joints]
            worlds = topology._joint_world[joints]
        if np.any(worlds < 0):
            raise ValueError("Selections exclude global-world entities.")
        if self.source is not None:
            worlds = values // stride
            bodies = bodies + worlds * topology.model.body_count
            if joints is not None:
                joints = joints + worlds * topology.model.joint_count
        rows = [values[worlds == world].tolist() for world in range(self.model.world_count)]
        body_rows = [bodies[worlds == world].tolist() for world in range(self.model.world_count)]
        selection = NewtonSelection(self, index_domain, rows, [body for row in body_rows for body in row])
        if joints is not None:
            joint_rows = [joints[worlds == world] for world in range(self.model.world_count)]
            selection.joint_ids = wp.array(np.concatenate(joint_rows), dtype=wp.int32, device=self.model.device)
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
        "index_domain",
        "_parts",
        "_num_envs",
        "_width",
        "_worlds",
        "_ids",
        "_active",
        "_source_ids",
        "_tables",
        "_scalar_views",
        "_pose_views",
        "world_selection_counts",
        "world_bindings",
        "_device",
    )

    def __init__(self, index_domain: str, parts, num_envs: int, *, policy_width: int | None = None):
        self.index_domain = index_domain
        self._num_envs = num_envs
        if not parts:
            raise ValueError("Grouped selections require at least one numeric native binding.")
        self._width = max(part.width for part, _ in parts) if policy_width is None else policy_width
        if isinstance(self._width, bool) or not isinstance(self._width, int) or self._width < 0:
            raise ValueError("Policy width must be a nonnegative integer.")
        self._device = parts[0][0].owner.model.device
        self._worlds = wp.from_torch(torch.arange(num_envs, dtype=torch.int32, device=str(self._device)))
        self._ids = wp.empty((num_envs, self._width), dtype=wp.int32, device=self._device)
        self._active = wp.empty((num_envs, self._width), dtype=wp.bool, device=self._device)
        self._source_ids = wp.empty(num_envs, dtype=wp.int32, device=self._device)
        self._tables = {}
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
    def dense_shape(self) -> tuple[int, int]:
        """Logical environment and slot extents at the dense policy boundary."""
        return self._num_envs, self.width

    @property
    def use_coord_layout_targets(self) -> bool:
        """Require one compatible control convention across the composed models."""
        modes = {part.use_coord_layout_targets for part, _ in self._parts}
        if len(modes) != 1:
            raise ValueError("Grouped controls require the same target layout in every model.")
        return modes.pop()

    def rebind(self, parts) -> None:
        """Publish a bijective world placement after old GPU consumers finish.

        Borrowed world-index tensors must remain unchanged until the next rebind.
        Distinct logical environments cannot alias one physical model world.
        """
        parts = tuple(parts)
        if sum(len(worlds) for _, worlds in parts) != self._num_envs:
            raise ValueError("Grouped selections must cover every logical world exactly once.")
        if any(part.width > self._width for part, _ in parts):
            raise ValueError("A native selection exceeds the configured dense policy width.")
        counts, world_bindings = [0] * self._num_envs, [None] * self._num_envs
        for part, worlds in parts:
            if (
                part.index_domain != self.index_domain
                or part.owner.model.device != self._device
                or worlds.ndim != 1
                or worlds.dtype != torch.long
                or worlds.device != torch.device(str(self._device))
                or len(worlds) != len(part.world_selection_counts)
            ):
                raise ValueError("Selection index domain/world placement does not match its logical binding.")
            for row, (world, count) in enumerate(zip(worlds.tolist(), part.world_selection_counts, strict=True)):
                if not 0 <= world < self._num_envs or world_bindings[world] is not None:
                    raise ValueError("Grouped selections require disjoint, complete logical world IDs.")
                world_bindings[world] = (part.owner, row)
                counts[world] = count
        if len(set(world_bindings)) != self._num_envs:
            raise ValueError("Grouped selections cannot alias one physical model world across environments.")
        self._parts = parts
        self.world_selection_counts = tuple(counts)
        # Immutable host relation for preparation checks; never read GPU placement in an MDP term.
        self.world_bindings = tuple(world_bindings)
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
        self._pose_views = {}
        self._ids.fill_(-1)
        self._active.zero_()
        ids, active, source_ids = map(wp.to_torch, (self._ids, self._active, self._source_ids))
        for source, (part, worlds) in enumerate(parts):
            ids[worlds, : part.width] = part.dense_ids()
            active[worlds, : part.width] = part.dense_active()
            source_ids[worlds] = source

    def dense_active(self) -> torch.Tensor:
        """Return participation in stable logical-world/policy-slot order."""
        return wp.to_torch(self._active)

    def refresh(self) -> None:
        """Refresh logical membership after native owners refresh their authoritative masks."""
        active = wp.to_torch(self._active)
        for part, worlds in self._parts:
            active[worlds, : part.width] = part.dense_active()

    def active_counts(self) -> torch.Tensor:
        """Current participating count per logical policy actor (GPU int64)."""
        return self.dense_active().sum(dim=1)

    def joint_types(self) -> torch.Tensor:
        """Read joint types once in logical-world order for scalar-joint validation."""
        values = torch.zeros((self._num_envs, self.width), dtype=torch.int32, device=str(self._device))
        for part, worlds in self._parts:
            values[worlds, : part.width] = part.joint_types()
        return values

    def _table(self, source: str, attribute: str):
        key = (source, attribute)
        if key not in self._tables:
            for part, _ in self._parts:
                _validate_field_domain(part.owner.model, self.index_domain, attribute)
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

    def pose_field(self, source: Literal["state", "model"], attribute: str) -> NewtonPoseField:
        """Borrow body transforms across populations; reacquire after ``rebind``."""
        key = source, attribute
        if key not in self._pose_views:
            if source not in ("state", "model") or self.index_domain != BODY:
                raise ValueError("Pose fields require a body selection and state/model source.")
            table, dtype = self._table(source, attribute)
            if dtype != wp.transform:
                raise TypeError("Pose fields require native transform arrays.")
            field = NewtonPoseField()
            field.sources, field.source_ids = table, self._source_ids
            field.ids, field.active = self._ids, self._active
            self._pose_views[key] = field
        return self._pose_views[key]

    def read_model(self, attribute: str, fill: float = 0.0) -> torch.Tensor:
        """Gather a model attribute across native populations in one GPU launch."""
        return self._read("model", attribute, fill)

    def _write(self, source: str, attribute: str, values: torch.Tensor, env_ids=None) -> None:
        target = wp.to_torch(getattr(getattr(self._parts[0][0].owner, source), attribute))
        count = _validate_write_env_indices(env_ids, self._num_envs, target.device)
        expected = (count, self.width, *target.shape[1:])
        if (
            not isinstance(values, torch.Tensor)
            or values.shape != expected
            or values.dtype != target.dtype
            or values.device != target.device
        ):
            raise ValueError("Writes require matching native dtype, device and policy-width value rows.")
        table, dtype = self._table(source, attribute)
        if count == 0:
            return
        worlds = self._worlds if env_ids is None else wp.from_torch(env_ids.to(torch.int32).contiguous())
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
