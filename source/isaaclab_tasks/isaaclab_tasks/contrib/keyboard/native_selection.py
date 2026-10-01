# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task selections over native prototype fields and stable world handles.

The runtime owns identities, placement and physical storage. This module owns
immutable selected-column maps and episode participation at the policy boundary.
Native fields are borrowed strided views; gathers never become physics storage.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import numpy as np
import torch
import warp as wp
from mujoco_warp._src.support import contact_force_fn
from mujoco_warp._src.types import vec5
from newton import Model
from newton.solvers import SolverMuJoCo
from newton.worlds import WorldDirectoryData, world_handle_at, world_location

from .newton_selection import (
    BODY,
    JOINT_COORD,
    JOINT_DOF,
    NewtonSelections,
    scalar_field_active,
    scalar_field_read,
    scalar_field_write,
)


@wp.struct
class _NativeCapacity:
    ready_rows: wp.array[int]


@wp.struct
class NativePlacement:
    actor_ids: wp.array[int]
    actor_generations: wp.array[wp.uint64]
    actor_active: wp.array[bool]
    directory: WorldDirectoryData
    capacities: wp.array[_NativeCapacity]


@wp.func
def _native_location(placement: NativePlacement, actor: int):
    prototype, row = int(-1), int(-1)
    # An actor index can arrive through a caller-authored kernel, independently
    # of the checked Python write API. Reject it before any descriptor access.
    if actor < 0 or actor >= placement.actor_ids.shape[0]:
        return prototype, row
    if placement.actor_active[actor]:
        candidate, local_row, valid = world_location(
            placement.directory, placement.actor_ids[actor], placement.actor_generations[actor]
        )
        if valid:
            capacity = placement.capacities[candidate]
            if local_row < capacity.ready_rows[0]:
                prototype, row = candidate, local_row
    return prototype, row


@wp.struct
class _NativeScalarSource:
    values: wp.array2d[float]
    broadcast_rows: int


@wp.struct
class NativeScalarField:
    """Direct native scalar access; task coordinates subtract ``offsets`` on read."""

    placement: NativePlacement
    sources: wp.array[_NativeScalarSource]
    columns: wp.array2d[int]
    offsets: wp.array2d[float]
    task_active: wp.array2d[bool]


@wp.func(module=scalar_field_active.module, name=scalar_field_active.key)
def scalar_field_active(field: NativeScalarField, world: int, slot: int) -> bool:
    prototype, row = _native_location(field.placement, world)
    if prototype < 0:
        return False
    if slot < 0 or slot >= field.task_active.shape[1]:
        return False
    if not field.task_active[world, slot]:
        return False
    return field.columns[prototype, slot] >= 0


@wp.func(module=scalar_field_read.module, name=scalar_field_read.key)
def scalar_field_read(field: NativeScalarField, world: int, slot: int) -> float:
    prototype, row = _native_location(field.placement, world)
    if prototype < 0:
        return 0.0
    if slot < 0 or slot >= field.task_active.shape[1]:
        return 0.0
    if not field.task_active[world, slot]:
        return 0.0
    column = field.columns[prototype, slot]
    if column < 0:
        return 0.0
    source = field.sources[prototype]
    if source.broadcast_rows != 0:
        row = 0
    return source.values[row, column] - field.offsets[prototype, slot]


@wp.func(module=scalar_field_write.module, name=scalar_field_write.key)
def scalar_field_write(field: NativeScalarField, world: int, slot: int, value: float):
    prototype, row = _native_location(field.placement, world)
    if prototype < 0:
        return
    if slot < 0 or slot >= field.task_active.shape[1]:
        return
    if not field.task_active[world, slot]:
        return
    column = field.columns[prototype, slot]
    if column >= 0:
        source = field.sources[prototype]
        # Immutable model fields have no writable physical row.
        if source.broadcast_rows == 0:
            source.values[row, column] = value + field.offsets[prototype, slot]


@wp.struct
class _NativePoseSource:
    position: wp.array2d[wp.vec3]
    quaternion: wp.array2d[wp.quat]
    broadcast_rows: int
    wxyz: int


@wp.kernel
def _gather_scalars(field: NativeScalarField, fill: float, out: wp.array2d[float]):
    actor, selected = wp.tid()
    value = fill
    if scalar_field_active(field, actor, selected):
        value = scalar_field_read(field, actor, selected)
    out[actor, selected] = value


@wp.kernel
def _gather_poses(
    placement: NativePlacement,
    sources: wp.array[_NativePoseSource],
    columns: wp.array2d[int],
    task_active: wp.array2d[bool],
    fill: float,
    out: wp.array2d[wp.transform],
):
    actor, selected = wp.tid()
    value = wp.transform(wp.vec3(fill), wp.quat(fill, fill, fill, fill))
    prototype, row = _native_location(placement, actor)
    if prototype >= 0 and task_active[actor, selected]:
        column = columns[prototype, selected]
        if column >= 0:
            source = sources[prototype]
            if source.broadcast_rows != 0:
                row = 0
            rotation = source.quaternion[row, column]
            if source.wxyz != 0:
                rotation = wp.quat(rotation[1], rotation[2], rotation[3], rotation[0])
            value = wp.transform(source.position[row, column], rotation)
    out[actor, selected] = value


@wp.kernel
def _selection_active(
    placement: NativePlacement,
    columns: wp.array2d[int],
    task_active: wp.array2d[bool],
    out: wp.array2d[bool],
):
    actor, selected = wp.tid()
    prototype, row = _native_location(placement, actor)
    active = bool(False)
    if prototype >= 0 and task_active[actor, selected]:
        active = columns[prototype, selected] >= 0
    out[actor, selected] = active


@wp.kernel
def _joint_types(placement: NativePlacement, values: wp.array2d[int], out: wp.array2d[int]):
    actor, selected = wp.tid()
    prototype, row = _native_location(placement, actor)
    value = int(0)
    if prototype >= 0:
        value = values[prototype, selected]
    out[actor, selected] = value


@wp.kernel
def _scatter(field: NativeScalarField, actors: wp.array[int], values: wp.array2d[float]):
    row, selected = wp.tid()
    scalar_field_write(field, actors[row], selected, values[row, selected])


class NativeSelection:
    """Numeric prototype-local columns placed through stable world handles."""

    # Manager config serialization excludes runtime storage and retains no query.
    __slots__ = (
        "__dict__",
        "owner",
        "parts",
        "width",
        "static_counts",
        "frequency",
        "_indices",
        "_columns",
        "_ids",
        "task_active",
        "_active",
        "_fields",
        "_poses",
        "joint_ids",
        "_joint_types",
        "_contact_forces",
        "_contact_visible",
        "_contact_generations",
        "_contact_maps",
    )

    def __init__(self, owner, frequency: str, parts, *, policy_width: int | None = None):
        self.owner, self.parts = owner, tuple(parts)
        self.frequency = frequency
        self.width = max(part.width for part in parts) if policy_width is None else policy_width
        if isinstance(self.width, bool) or not isinstance(self.width, int) or self.width < 0:
            raise ValueError("Policy width must be a nonnegative integer.")
        if any(part.width > self.width for part in parts):
            raise ValueError("Native selection exceeds the configured policy width.")
        # Static topology counts are per prototype; width is the padded policy extent.
        self.static_counts = tuple(part.width for part in parts)
        self._indices = [part.ids.numpy() for part in parts]
        self._columns = np.full((len(parts), self.width), -1, np.int32)
        for prototype, ids in enumerate(self._indices):
            self._columns[prototype, : len(ids)] = ids
        self._ids = wp.array(self._columns, dtype=int, device=owner.device)
        self.task_active = wp.ones((owner.num_envs, self.width), dtype=bool, device=owner.device)
        self._active = wp.empty_like(self.task_active)
        self._fields, self._poses = {}, {}
        self._contact_forces = None
        self.joint_ids = None
        if self.frequency != BODY:
            self.joint_ids = wp.array(
                np.concatenate([part.joint_ids.numpy() for part in parts]), dtype=int, device=owner.device
            )
            values = np.zeros_like(self._columns)
            for prototype, part in enumerate(parts):
                values[prototype, : part.width] = part.owner.model.joint_type.numpy()[part.joint_ids.numpy()]
            self._joint_types = wp.array(values, dtype=int, device=owner.device)

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    @property
    def use_coord_layout_targets(self) -> bool:
        modes = {part.use_coord_layout_targets for part in self.parts}
        if len(modes) != 1:
            raise ValueError("Native prototype controls require one target layout convention.")
        return modes.pop()

    def dense_active(self) -> torch.Tensor:
        """Participation by policy actor and slot; ordinary solver sleep is unrelated."""
        wp.launch(
            _selection_active,
            self.task_active.shape,
            [self.owner.placement, self._ids, self.task_active],
            [self._active],
            device=self.owner.device,
        )
        return wp.to_torch(self._active)

    def active_counts(self) -> torch.Tensor:
        """Current participating scalar/body count per policy actor (GPU int64)."""
        return self.dense_active().sum(dim=1)

    def joint_types(self) -> torch.Tensor:
        if self.joint_ids is None:
            raise ValueError("Joint types require a coordinate or DOF selection.")
        out = wp.empty(self.task_active.shape, dtype=int, device=self.owner.device)
        wp.launch(_joint_types, out.shape, [self.owner.placement, self._joint_types], [out], device=self.owner.device)
        return wp.to_torch(out)

    def scalar_field(self, source: Literal["state", "model", "control"], attribute: str) -> NativeScalarField:
        """Borrow selected native scalar storage without flattening or copying physics."""
        key = source, attribute
        if key in self._fields:
            return self._fields[key]
        if source not in ("state", "model", "control") or self.frequency == BODY:
            raise ValueError("Scalar fields require a joint selection and state/model/control source.")
        columns, offsets = self._columns.copy(), np.zeros(self._columns.shape, np.float32)
        descriptors = []
        for prototype, (part, ids, group, mapping) in enumerate(
            zip(self.parts, self._indices, self.owner.runtime.prototypes, self.owner.mappings, strict=True)
        ):
            descriptor = _NativeScalarSource()
            if source == "model":
                values = getattr(part.owner.model, attribute)
                if values.dtype != wp.float32 or values.ndim != 1:
                    raise TypeError("Selected model properties must be scalar float32 arrays.")
                values = values.reshape((1, values.shape[0]))
                descriptor.broadcast_rows = 1
            else:
                descriptor.broadcast_rows = 0
                if source == "state" and attribute == "joint_q":
                    if self.frequency != JOINT_COORD:
                        raise ValueError("joint_q requires a coordinate selection.")
                    values, lookup = group.data.qpos, mapping.coord
                    offsets[prototype, : len(ids)] = [mapping.qref[int(index)] for index in ids]
                elif source == "state" and attribute == "joint_qd":
                    if self.frequency != JOINT_DOF:
                        raise ValueError("joint_qd requires a DOF selection.")
                    values, lookup = group.data.qvel, mapping.dof
                elif source == "control" and attribute in ("joint_target_q", "joint_target_qd"):
                    target = "position" if attribute == "joint_target_q" else "velocity"
                    values, lookup = group.data.ctrl, getattr(mapping, target)
                    if target == "position":
                        offsets[prototype, : len(ids)] = [mapping.target_ref[int(index)] for index in ids]
                elif source == "control" and attribute == "joint_f":
                    if self.frequency != JOINT_DOF:
                        raise ValueError("joint_f requires a DOF selection.")
                    values, lookup = group.data.qfrc_applied, mapping.dof
                else:
                    raise ValueError(f"Unsupported native scalar field {source}.{attribute}.")
                try:
                    columns[prototype, : len(ids)] = [lookup[int(index)] for index in ids]
                except KeyError as exc:
                    raise ValueError(f"Selected native {source}.{attribute} has no unique mapped target.") from exc
            if values.dtype != wp.float32 or values.ndim != 2:
                raise TypeError("Native scalar sources require two-dimensional float32 arrays.")
            if np.any(columns[prototype, : len(ids)] >= values.shape[1]):
                raise ValueError("A selected column exceeds its native source extent.")
            descriptor.values = values
            descriptors.append(descriptor)
        field = NativeScalarField()
        field.placement, field.task_active = self.owner.placement, self.task_active
        field.sources = wp.array(descriptors, dtype=_NativeScalarSource, device=self.owner.device)
        field.columns = wp.array(columns, dtype=int, device=self.owner.device)
        field.offsets = wp.array(offsets, dtype=float, device=self.owner.device)
        self._fields[key] = field
        return field

    def _read(self, source, attribute, fill):
        if self.frequency != BODY:
            out = wp.empty(self.task_active.shape, dtype=float, device=self.owner.device)
            wp.launch(
                _gather_scalars,
                out.shape,
                [self.scalar_field(source, attribute), fill],
                [out],
                device=self.owner.device,
            )
            return wp.to_torch(out)
        if attribute != "body_q" or source not in ("state", "model"):
            raise ValueError("Native body selections currently expose body_q only.")
        if source not in self._poses:
            descriptors, columns = [], self._columns.copy()
            for prototype, (part, ids, group, mapping) in enumerate(
                zip(self.parts, self._indices, self.owner.runtime.prototypes, self.owner.mappings, strict=True)
            ):
                descriptor = _NativePoseSource()
                if source == "state":
                    descriptor.position, descriptor.quaternion = group.data.xpos, group.data.xquat
                    descriptor.broadcast_rows, descriptor.wxyz = 0, 1
                    columns[prototype, : len(ids)] = [mapping.body[int(index)] for index in ids]
                else:
                    values = part.owner.model.body_q
                    shape, strides = (1, values.shape[0]), (values.capacity, values.strides[0])
                    descriptor.position = wp.array(
                        ptr=values.ptr, shape=shape, strides=strides, dtype=wp.vec3, device=values.device
                    )
                    descriptor.quaternion = wp.array(
                        ptr=values.ptr + 12, shape=shape, strides=strides, dtype=wp.quat, device=values.device
                    )
                    descriptor.broadcast_rows, descriptor.wxyz = 1, 0
                descriptors.append(descriptor)
            self._poses[source] = (
                wp.array(descriptors, dtype=_NativePoseSource, device=self.owner.device),
                wp.array(columns, dtype=int, device=self.owner.device),
            )
        sources, columns = self._poses[source]
        out = wp.empty(self.task_active.shape, dtype=wp.transform, device=self.owner.device)
        wp.launch(
            _gather_poses,
            out.shape,
            [self.owner.placement, sources, columns, self.task_active, fill],
            [out],
            device=self.owner.device,
        )
        return wp.to_torch(out)

    def read_state(self, attribute: str, fill: float = 0.0) -> torch.Tensor:
        """Gather native state into policy rows, using task coordinate conventions."""
        return self._read("state", attribute, fill)

    def read_model(self, attribute: str, fill: float = 0.0) -> torch.Tensor:
        """Gather authored immutable properties, broadcast explicitly from row zero."""
        return self._read("model", attribute, fill)

    def _write(self, source, attribute, values, env_ids):
        device = torch.device(str(self.owner.device))
        if env_ids is not None and (
            not isinstance(env_ids, torch.Tensor)
            or env_ids.ndim != 1
            or env_ids.dtype != torch.long
            or env_ids.device != device
        ):
            raise ValueError("Actor IDs must be a one-dimensional int64 tensor on the native device.")
        count = self.owner.num_envs if env_ids is None else len(env_ids)
        if (
            not isinstance(values, torch.Tensor)
            or values.shape != (count, self.width)
            or values.dtype != torch.float32
            or values.device != device
        ):
            raise ValueError("Native writes require float32 policy-width rows on the native device.")
        if env_ids is not None:
            torch._assert_async(
                ((env_ids >= 0) & (env_ids < self.owner.num_envs)).all(), "Actor IDs are outside the task."
            )
            seen = torch.zeros(self.owner.num_envs, dtype=torch.int32, device=device)
            seen.scatter_add_(0, env_ids, torch.ones_like(env_ids, dtype=torch.int32))
            torch._assert_async((seen <= 1).all(), "Actor IDs must be unique.")
        if count == 0:
            return
        actors = self.owner._actors if env_ids is None else wp.from_torch(env_ids.to(dtype=torch.int32).contiguous())
        wp.launch(
            _scatter,
            values.shape,
            [self.scalar_field(source, attribute), actors, wp.from_torch(values.contiguous(), dtype=wp.float32)],
            device=self.owner.device,
        )

    def write_state(self, attribute: str, values: torch.Tensor, env_ids=None) -> None:
        """Write native state [m/rad or m/s/rad/s] for unique, in-range actor IDs."""
        self._write("state", attribute, values, env_ids)

    def write_control(self, attribute: str, values: torch.Tensor, env_ids=None) -> None:
        """Write targets or applied force [N or Nm] for unique, in-range actor IDs."""
        self._write("control", attribute, values, env_ids)

    def prepare_contact_forces(self) -> None:
        """Prepare policy normal-force reductions before capturing native steps."""
        if self.frequency != BODY:
            raise ValueError("Contact force reductions require a body selection.")
        if self._contact_forces is not None:
            return
        self._contact_forces = wp.zeros(self.task_active.shape, dtype=wp.vec3, device=self.owner.device)
        self._contact_visible = wp.empty_like(self._contact_forces)
        self._contact_generations = wp.zeros(self.owner.num_envs, dtype=wp.uint64, device=self.owner.device)
        self._contact_maps = []
        for ids, group, mapping in zip(self._indices, self.owner.runtime.prototypes, self.owner.mappings, strict=True):
            slots = np.full(group.model.nbody, -1, np.int32)
            for selected, body in enumerate(ids):
                slots[mapping.body[int(body)]] = selected
            self._contact_maps.append(wp.array(slots, dtype=int, device=self.owner.device))

    def record_contact_forces(self, group, actor_for_id) -> None:
        """Record final-substep net normal forces [N], before the prototype leaves its branch."""
        if self._contact_forces is None:
            raise RuntimeError("Prepare contact reductions before graph capture.")
        prototype = group.index
        dimensions = group.world_capacity, self.width
        group.record_launch(
            _clear_contact_forces,
            dimensions,
            inputs=[self.owner.placement, prototype, actor_for_id, self._contact_forces, self._contact_generations],
            domain="world",
        )
        data, model, contact = group.data, group.model, group.data.contact
        group.record_launch(
            _accumulate_contact_forces,
            group.contact_capacity,
            inputs=[
                self.owner.placement,
                prototype,
                actor_for_id,
                group.world_count,
                group.contact_ready_count,
                data.nacon,
                model.geom_bodyid,
                self._contact_maps[prototype],
                contact.worldid,
                contact.geom,
                contact.frame,
                contact.friction,
                contact.dim,
                contact.efc_address,
                contact.adhesion,
                data.efc.force,
                data.njmax,
                int(model.opt.cone),
                self._contact_forces,
            ],
            domain="candidate",
        )

    def selected_net_normal_forces(self) -> torch.Tensor:
        """Read the last native step's net normal force [N]; reset lifetimes return zero."""
        if self._contact_forces is None:
            raise RuntimeError("Contact forces were not prepared in the native step program.")
        wp.launch(
            _visible_contact_forces,
            self.task_active.shape,
            [self.owner.placement, self._ids, self.task_active, self._contact_generations, self._contact_forces],
            [self._contact_visible],
            device=self.owner.device,
        )
        return wp.to_torch(self._contact_visible)


@dataclass(frozen=True)
class NativePrototypeMapping:
    """One immutable authored-Newton to native-MuJoCo topology conversion.

    Selections and reset payload compilation borrow these exact maps. No live
    state, actor placement or query expressions belong to this prepared object.
    """

    model: Model
    coord: MappingProxyType
    dof: MappingProxyType
    body: MappingProxyType
    qref: MappingProxyType
    position: MappingProxyType
    velocity: MappingProxyType
    target_ref: MappingProxyType
    coordinate_ids: np.ndarray
    dof_ids: np.ndarray
    coordinate_refs: np.ndarray
    root_body_ids: np.ndarray

    def __init__(self, owner: NewtonSelections, solver: SolverMuJoCo):
        if owner.model.world_count != 1 or solver.model is not owner.model:
            raise ValueError("Native mapping requires its exact authored one-world Newton model and solver.")
        cpu = solver.mj_model
        joints = solver.mjc_jnt_to_newton_jnt.numpy()[0]
        dofs = solver.mjc_dof_to_newton_dof.numpy()[0]
        model = owner.model
        qstarts, dstarts = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
        refs = getattr(getattr(model, "mujoco", None), "dof_ref", None)
        refs = np.zeros(model.joint_dof_count, np.float32) if refs is None else refs.numpy()
        mapping = {name: {} for name in ("coord", "dof", "body", "qref", "position", "velocity", "target_ref")}
        for native, joint in enumerate(joints):
            joint = int(joint)
            if joint < 0 or qstarts[joint + 1] - qstarts[joint] != 1 or dstarts[joint + 1] - dstarts[joint] != 1:
                raise ValueError("Native keyboard selections admit scalar joints only.")
            mapping["coord"][int(qstarts[joint])] = int(cpu.jnt_qposadr[native])
            mapping["qref"][int(qstarts[joint])] = float(refs[dstarts[joint]])
        for native, dof in enumerate(dofs):
            if dof < 0 or int(dof) in mapping["dof"]:
                raise ValueError("Native DOF mappings must be unique and complete.")
            mapping["dof"][int(dof)] = native
        for native, body in enumerate(solver.mjc_body_to_newton.numpy()[0]):
            if body >= 0:
                if int(body) in mapping["body"]:
                    raise ValueError("Native body mappings must be unique.")
                mapping["body"][int(body)] = native
        actuator_arrays = (
            solver.mjc_actuator_ctrl_source,
            solver.mjc_actuator_to_newton_idx,
            solver.mjc_actuator_to_newton_target_q_idx,
            solver.mjc_actuator_to_target_q_axis_idx,
            solver.mjc_actuator_to_newton_ball_jnt,
        )
        modes, encoded, targets, axes, balls = (
            np.empty(0, dtype=np.int32) if values is None else values.numpy() for values in actuator_arrays
        )
        for actuator, (mode, index, target, axis, ball) in enumerate(
            zip(modes, encoded, targets, axes, balls, strict=True)
        ):
            if mode != 0 or axis >= 0 or ball >= 0 or index == -1:
                raise ValueError("Native keyboard controls require mapped scalar joint-target actuators.")
            kind, key = ("position", int(target)) if index >= 0 else ("velocity", -int(index) - 2)
            if key < 0 or key in mapping[kind]:
                raise ValueError("Native controls require a unique actuator for each selected target.")
            mapping[kind][key] = actuator
            if index >= 0:
                mapping["target_ref"][key] = float(refs[index])
        if len(mapping["coord"]) != cpu.nq or len(mapping["dof"]) != cpu.nv:
            raise ValueError("Native scalar maps must cover every coordinate and DOF.")
        object.__setattr__(self, "model", model)
        for name, values in mapping.items():
            object.__setattr__(self, name, MappingProxyType(values))
        object.__setattr__(
            self, "coordinate_ids", np.array(sorted(mapping["coord"], key=mapping["coord"].get), dtype=np.int32)
        )
        object.__setattr__(self, "dof_ids", np.array(sorted(mapping["dof"], key=mapping["dof"].get), dtype=np.int32))
        object.__setattr__(
            self, "coordinate_refs", np.array([mapping["qref"][int(q)] for q in self.coordinate_ids], dtype=np.float32)
        )
        root_mapping = solver.mjc_mocap_to_newton_jnt
        root_joints = np.empty(0, dtype=np.int32) if root_mapping is None else root_mapping.numpy()[0]
        if np.any(root_joints < 0) or np.any(root_joints >= model.joint_count):
            raise ValueError("Native mocap roots must map to valid authored joints.")
        object.__setattr__(self, "root_body_ids", model.joint_child.numpy()[root_joints].copy())
        if not np.allclose(model.joint_X_c.numpy()[root_joints], [0, 0, 0, 0, 0, 0, 1], atol=1e-7, rtol=0):
            raise ValueError("Prepared fixed-root snapshot frames must match their child body frames.")
        for values in (self.coordinate_ids, self.dof_ids, self.coordinate_refs, self.root_body_ids):
            values.flags.writeable = False


class NativeSelections:
    """Bind authored selection metadata to native runtime storage and task handles."""

    def __init__(
        self,
        metadata: tuple[NewtonSelections, ...],
        mappings: tuple[NativePrototypeMapping, ...],
        runtime,
        actor_ids,
        actor_generations,
        *,
        num_envs: int,
        device,
        actor_active=None,
    ):
        self.metadata, self.mappings, self.runtime = tuple(metadata), tuple(mappings), runtime
        self.num_envs, self.device = num_envs, wp.get_device(device)
        if not metadata or len(metadata) != len(mappings) or len(metadata) != len(runtime.prototypes):
            raise ValueError("Native selections require one metadata/mapping source per prepared prototype.")
        if any(mapping.model is not owner.model for owner, mapping in zip(metadata, mappings, strict=True)):
            raise ValueError("Native mappings must belong to the exact prepared metadata model.")
        for array, dtype in ((actor_ids, wp.int32), (actor_generations, wp.uint64)):
            if array.shape != (num_envs,) or array.dtype != dtype or array.device != self.device:
                raise ValueError("Actor handles must have matching extent, dtype and native device.")
        self.actor_active = wp.ones(num_envs, dtype=bool, device=self.device) if actor_active is None else actor_active
        if (
            self.actor_active.shape != (num_envs,)
            or self.actor_active.dtype != wp.bool
            or self.actor_active.device != self.device
        ):
            raise ValueError("Actor participation must be a matching boolean device array.")
        self._actors = wp.array(np.arange(num_envs), dtype=int, device=self.device)
        self._bindings = {}
        self._retired = False
        capacities = []
        for group in runtime.prototypes:
            capacity = _NativeCapacity()
            capacity.ready_rows = group.world_ready_count
            capacities.append(capacity)
        placement = NativePlacement()
        placement.actor_ids, placement.actor_generations = actor_ids, actor_generations
        placement.actor_active = self.actor_active
        placement.directory = runtime.directory.data
        placement.capacities = wp.array(capacities, dtype=_NativeCapacity, device=self.device)
        self.placement = placement

    def bind(
        self, frequency: str, ids_by_prototype: Sequence[Sequence[int] | np.ndarray], *, policy_width: int | None = None
    ) -> NativeSelection:
        """Bind integer frequency IDs for every authored prototype, without paths."""
        if self._retired:
            raise RuntimeError("Cannot bind selections from a retired owner.")
        if policy_width is not None and (
            isinstance(policy_width, bool) or not isinstance(policy_width, int) or policy_width < 0
        ):
            raise ValueError("Policy width must be a nonnegative integer.")
        if len(ids_by_prototype) != len(self.metadata):
            raise ValueError("Provide one ordered numeric ID sequence per prototype.")
        parts = tuple(owner.bind(frequency, ids) for owner, ids in zip(self.metadata, ids_by_prototype, strict=True))
        key = (frequency, tuple(tuple(map(int, ids)) for ids in ids_by_prototype), policy_width)
        if key not in self._bindings:
            self._bindings[key] = NativeSelection(self, frequency, parts, policy_width=policy_width)
        return self._bindings[key]

    def retire(self) -> None:
        """Release cached bindings; existing selections retain their borrowed owners."""
        self._retired = True
        self._bindings.clear()


@wp.func
def _actor_at_row(placement: NativePlacement, prototype: int, row: int, actor_for_id: wp.array[int]) -> int:
    identity, generation, valid = world_handle_at(placement.directory, prototype, row)
    if not valid or identity >= actor_for_id.shape[0]:
        return -1
    actor = actor_for_id[identity]
    if actor < 0 or actor >= placement.actor_ids.shape[0] or placement.actor_ids[actor] != identity:
        return -1
    live_prototype, live_row = _native_location(placement, actor)
    if live_prototype != prototype or live_row != row:
        return -1
    return actor


@wp.kernel
def _clear_contact_forces(
    placement: NativePlacement,
    prototype: int,
    actor_for_id: wp.array[int],
    forces: wp.array2d[wp.vec3],
    generations: wp.array[wp.uint64],
):
    row, selected = wp.tid()
    actor = _actor_at_row(placement, prototype, row, actor_for_id)
    if actor >= 0:
        forces[actor, selected] = wp.vec3(0.0)
        if selected == 0:
            generations[actor] = placement.actor_generations[actor]


@wp.kernel
def _accumulate_contact_forces(
    placement: NativePlacement,
    prototype: int,
    actor_for_id: wp.array[int],
    live_worlds: wp.array[int],
    ready_contacts: wp.array[int],
    nacon: wp.array[int],
    geom_body: wp.array[int],
    body_slot: wp.array[int],
    contact_world: wp.array[int],
    contact_geom: wp.array[wp.vec2i],
    frame: wp.array[wp.mat33],
    friction: wp.array[vec5],
    dimensions: wp.array[int],
    addresses: wp.array2d[int],
    adhesion: wp.array[float],
    efc_force: wp.array2d[float],
    njmax: int,
    cone: int,
    out: wp.array2d[wp.vec3],
):
    contact = wp.tid()
    if contact >= wp.min(nacon[0], ready_contacts[0]):
        return
    row = contact_world[contact]
    if row < 0 or row >= live_worlds[0]:
        return
    actor = _actor_at_row(placement, prototype, row, actor_for_id)
    if actor < 0:
        return
    force = contact_force_fn(
        cone, frame, friction, dimensions, addresses, adhesion, efc_force, njmax, nacon, row, contact, False
    )
    normal = frame[contact][0] * force[0]
    geoms = contact_geom[contact]
    for side in range(2):
        geom = geoms[side]
        if geom >= 0 and geom < geom_body.shape[0]:
            body = geom_body[geom]
            if body >= 0 and body < body_slot.shape[0]:
                selected = body_slot[body]
                if selected >= 0:
                    sign = -1.0
                    if side == 1:
                        sign = 1.0
                    wp.atomic_add(out, actor, selected, sign * normal)


@wp.kernel
def _visible_contact_forces(
    placement: NativePlacement,
    columns: wp.array2d[int],
    task_active: wp.array2d[bool],
    generations: wp.array[wp.uint64],
    forces: wp.array2d[wp.vec3],
    out: wp.array2d[wp.vec3],
):
    actor, selected = wp.tid()
    value = wp.vec3(0.0)
    prototype, row = _native_location(placement, actor)
    if prototype >= 0 and task_active[actor, selected] and columns[prototype, selected] >= 0:
        if generations[actor] == placement.actor_generations[actor]:
            value = forces[actor, selected]
    out[actor, selected] = value
