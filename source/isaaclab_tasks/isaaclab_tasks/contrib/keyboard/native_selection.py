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

from typing import Literal

import numpy as np
import torch
import warp as wp
from mujoco_warp._src.support import contact_force_fn
from mujoco_warp._src.types import vec5

from .newton_selection import (
    BODY,
    JOINT_COORD,
    JOINT_DOF,
    NewtonSelections,
    NewtonSelectorCfg,
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
    prototype: wp.array[int]
    slot: wp.array[int]
    generation: wp.array[wp.uint64]
    starts: wp.array[int]
    slot_ids: wp.array[int]
    capacities: wp.array[_NativeCapacity]


@wp.func
def _native_location(placement: NativePlacement, actor: int):
    prototype, row = int(-1), int(-1)
    # An actor index can arrive through a caller-authored kernel, independently
    # of the checked Python write API. Reject it before any descriptor access.
    if actor < 0 or actor >= placement.actor_ids.shape[0]:
        return prototype, row
    identity = placement.actor_ids[actor]
    if placement.actor_active[actor] and identity >= 0 and identity < placement.prototype.shape[0]:
        if placement.generation[identity] == placement.actor_generations[actor]:
            candidate = placement.prototype[identity]
            if candidate >= 0 and candidate < placement.capacities.shape[0]:
                local_row = placement.slot[identity]
                slot = placement.starts[candidate] + local_row
                capacity = placement.capacities[candidate]
                if local_row >= 0 and local_row < capacity.ready_rows[0]:
                    if slot < placement.starts[candidate + 1] and placement.slot_ids[slot] == identity:
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
    """One declarative selection with stable handles and prototype-local columns."""

    # Manager config serialization sees only the declarative selector fields.
    __slots__ = (
        "__dict__",
        "owner",
        "parts",
        "width",
        "counts",
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

    def __init__(self, owner, cfg, parts):
        self.owner, self.parts = owner, tuple(parts)
        self.frequency, self.path = cfg.frequency, cfg.path
        self.count_per_world, self.dense_width = cfg.count_per_world, cfg.dense_width
        self.width = cfg.dense_width or cfg.count_per_world or max(part.width for part in parts)
        if any(part.width > self.width for part in parts):
            raise ValueError("Native selection exceeds the configured policy width.")
        self.counts = (self.width,) * owner.num_envs
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
            zip(self.parts, self._indices, self.owner.runtime.prototypes, self.owner._maps, strict=True)
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
                    values, lookup = group.data.qpos, mapping["coord"]
                    offsets[prototype, : len(ids)] = [mapping["qref"][int(index)] for index in ids]
                elif source == "state" and attribute == "joint_qd":
                    if self.frequency != JOINT_DOF:
                        raise ValueError("joint_qd requires a DOF selection.")
                    values, lookup = group.data.qvel, mapping["dof"]
                elif source == "control" and attribute in ("joint_target_q", "joint_target_qd"):
                    target = "position" if attribute == "joint_target_q" else "velocity"
                    values, lookup = group.data.ctrl, mapping[target]
                    if target == "position":
                        offsets[prototype, : len(ids)] = [mapping["target_ref"][int(index)] for index in ids]
                elif source == "control" and attribute == "joint_f":
                    if self.frequency != JOINT_DOF:
                        raise ValueError("joint_f requires a DOF selection.")
                    values, lookup = group.data.qfrc_applied, mapping["dof"]
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
                zip(self.parts, self._indices, self.owner.runtime.prototypes, self.owner._maps, strict=True)
            ):
                descriptor = _NativePoseSource()
                if source == "state":
                    descriptor.position, descriptor.quaternion = group.data.xpos, group.data.xquat
                    descriptor.broadcast_rows, descriptor.wxyz = 0, 1
                    columns[prototype, : len(ids)] = [mapping["body"][int(index)] for index in ids]
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
        for ids, group, mapping in zip(self._indices, self.owner.runtime.prototypes, self.owner._maps, strict=True):
            slots = np.full(group.model.nbody, -1, np.int32)
            for selected, body in enumerate(ids):
                slots[mapping["body"][int(body)]] = selected
            self._contact_maps.append(wp.array(slots, dtype=int, device=self.owner.device))

    def record_contact_forces(self, group, actor_for_id) -> None:
        """Record final-substep net normal forces [N], before the prototype leaves its branch."""
        if self._contact_forces is None:
            raise RuntimeError("Prepare contact reductions before graph capture.")
        prototype = next(i for i, candidate in enumerate(self.owner.runtime.prototypes) if candidate is group)
        dimensions = group.rows.capacity, self.width
        wp.launch(
            _clear_contact_forces,
            dimensions,
            [self.owner.placement, prototype, actor_for_id, self._contact_forces, self._contact_generations],
            device=self.owner.device,
        )
        group.observe_launch(_clear_contact_forces, dimensions, "world")
        data, model, contact = group.data, group.model, group.data.contact
        wp.launch(
            _accumulate_contact_forces,
            group.contacts.capacity,
            [
                self.owner.placement,
                prototype,
                actor_for_id,
                group.rows.count,
                group.contacts.ready_count,
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
            device=self.owner.device,
        )
        group.observe_launch(_accumulate_contact_forces, group.contacts.capacity, "candidate")

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


class NativeSelections:
    """Bind authored selection metadata to native runtime storage and task handles."""

    def __init__(
        self,
        metadata: tuple[NewtonSelections, ...],
        solvers,
        runtime,
        actor_ids,
        actor_generations,
        *,
        num_envs: int,
        device,
        actor_active=None,
    ):
        self.metadata, self.solvers, self.runtime = tuple(metadata), tuple(solvers), runtime
        self.num_envs, self.device = num_envs, wp.get_device(device)
        if not metadata or len(metadata) != len(solvers) or len(metadata) != len(runtime.prototypes):
            raise ValueError("Native selections require one metadata/solver source per prepared prototype.")
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
        self._maps, self._bindings = [], {}
        self._retired = False
        capacities = []
        for owner, solver, group in zip(metadata, solvers, runtime.prototypes, strict=True):
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
            modes = solver.mjc_actuator_ctrl_source.numpy()
            encoded = solver.mjc_actuator_to_newton_idx.numpy()
            targets = solver.mjc_actuator_to_newton_target_q_idx.numpy()
            axes = solver.mjc_actuator_to_target_q_axis_idx.numpy()
            balls = solver.mjc_actuator_to_newton_ball_jnt.numpy()
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
            self._maps.append(mapping)
            capacity = _NativeCapacity()
            capacity.ready_rows = group.rows.ready_count
            capacities.append(capacity)
        directory = runtime.directory.d
        placement = NativePlacement()
        placement.actor_ids, placement.actor_generations = actor_ids, actor_generations
        placement.actor_active = self.actor_active
        placement.prototype, placement.slot, placement.generation = (
            directory.prototype,
            directory.slot,
            directory.generation,
        )
        placement.starts, placement.slot_ids = directory.starts, directory.slot_id
        placement.capacities = wp.array(capacities, dtype=_NativeCapacity, device=self.device)
        self.placement = placement

    def resolve(self, cfg: NewtonSelectorCfg) -> NativeSelection:
        """Resolve labels once; reset and relocation only update borrowed handles."""
        if self._retired:
            raise RuntimeError("Cannot resolve selections from a retired owner.")
        patterns = (cfg.path,) if isinstance(cfg.path, str) else tuple(cfg.path)
        key = cfg.frequency, patterns, cfg.count_per_world, cfg.dense_width
        if key not in self._bindings:
            self._bindings[key] = NativeSelection(self, cfg, [owner.resolve(cfg) for owner in self.metadata])
        return self._bindings[key]

    def retire(self) -> None:
        """Release cached bindings; existing selections retain their borrowed owners."""
        self._retired = True
        self._bindings.clear()


@wp.func
def _actor_at_row(placement: NativePlacement, prototype: int, row: int, actor_for_id: wp.array[int]) -> int:
    capacity = placement.capacities[prototype]
    if row < 0 or row >= capacity.ready_rows[0]:
        return -1
    identity = placement.slot_ids[placement.starts[prototype] + row]
    if identity < 0 or identity >= actor_for_id.shape[0]:
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
