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
from typing import Literal

import numpy as np
import torch
import warp as wp
from gpu_components import directory as instance_directory
from gpu_components.directory_data import InstanceDirectoryData
from mujoco_warp import contact_force_fn, vec5
from newton import Model
from newton.solvers import MuJoCoModelMapping, mujoco_world_population_validate, mujoco_worlds_validate

from .newton_selection import (
    NewtonSelections,
    _validate_field_domain,
    _validate_write_env_indices,
    pose_field_active,
    pose_field_read,
    scalar_field_active,
    scalar_field_read,
    scalar_field_write,
)


@wp.struct
class _WorldReadiness:
    ready_world_count: wp.array[int]


@wp.struct
class EnvWorldBindings:
    """Task environment handles and participation, borrowing runtime placement/readiness.

    The directory alone owns world placement. This relation maps environment
    indices to generation-checked handles and certifies readable prototype rows.
    """

    world_id_by_env: wp.array[int]
    world_generation_by_env: wp.array[wp.uint64]
    env_participating: wp.array[bool]
    directory: InstanceDirectoryData
    world_readiness_by_prototype: wp.array[_WorldReadiness]


@wp.func
def _env_world_location(placement: EnvWorldBindings, env_index: int):
    prototype, row = int(-1), int(-1)
    # An environment index can arrive through a caller-authored kernel, independently
    # of the checked Python write API. Reject it before any descriptor access.
    if env_index < 0 or env_index >= placement.world_id_by_env.shape[0]:
        return prototype, row
    if placement.env_participating[env_index]:
        candidate, local_row, valid = instance_directory.location(
            placement.directory, placement.world_id_by_env[env_index], placement.world_generation_by_env[env_index]
        )
        if valid:
            capacity = placement.world_readiness_by_prototype[candidate]
            if local_row < capacity.ready_world_count[0]:
                prototype, row = candidate, local_row
    return prototype, row


@wp.struct
class _ScalarSource:
    values: wp.array2d[float]
    broadcast_rows: int


@wp.struct
class MuJoCoScalarField:
    """Borrow native scalar storage; task coordinates subtract ``offsets`` on read.

    Valid only while its selection owner and runtime are open. Join GPU users and
    drop captured borrowers before retirement; a Python reference cannot preserve
    a virtual range after its physical owner explicitly closes it.
    """

    env_world_bindings: EnvWorldBindings
    sources: wp.array[_ScalarSource]
    columns: wp.array2d[int]
    offsets: wp.array2d[float]
    element_participating: wp.array2d[bool]


@wp.func(module=scalar_field_active.module, name=scalar_field_active.key)
def scalar_field_active(field: MuJoCoScalarField, world: int, slot: int) -> bool:
    prototype, row = _env_world_location(field.env_world_bindings, world)
    if prototype < 0:
        return False
    if slot < 0 or slot >= field.element_participating.shape[1]:
        return False
    if not field.element_participating[world, slot]:
        return False
    return field.columns[prototype, slot] >= 0


@wp.func(module=scalar_field_read.module, name=scalar_field_read.key)
def scalar_field_read(field: MuJoCoScalarField, world: int, slot: int) -> float:
    prototype, row = _env_world_location(field.env_world_bindings, world)
    if prototype < 0:
        return 0.0
    if slot < 0 or slot >= field.element_participating.shape[1]:
        return 0.0
    if not field.element_participating[world, slot]:
        return 0.0
    column = field.columns[prototype, slot]
    if column < 0:
        return 0.0
    source = field.sources[prototype]
    if source.broadcast_rows != 0:
        row = 0
    return source.values[row, column] - field.offsets[prototype, slot]


@wp.func(module=scalar_field_write.module, name=scalar_field_write.key)
def scalar_field_write(field: MuJoCoScalarField, world: int, slot: int, value: float):
    prototype, row = _env_world_location(field.env_world_bindings, world)
    if prototype < 0:
        return
    if slot < 0 or slot >= field.element_participating.shape[1]:
        return
    if not field.element_participating[world, slot]:
        return
    column = field.columns[prototype, slot]
    if column >= 0:
        source = field.sources[prototype]
        # Immutable model fields have no writable physical row.
        if source.broadcast_rows == 0:
            source.values[row, column] = value + field.offsets[prototype, slot]


@wp.struct
class _PoseSource:
    position: wp.array2d[wp.vec3]
    quaternion: wp.array2d[wp.quat]
    broadcast_rows: int
    wxyz: int


@wp.struct
class MuJoCoPoseField:
    """Borrow read-only body poses [m, xyzw] through generation-checked placement.

    This derived observation shares the scalar field's owner and retirement law.
    The runtime owns all pose storage; this descriptor owns only index relations.
    """

    env_world_bindings: EnvWorldBindings
    sources: wp.array[_PoseSource]
    columns: wp.array2d[int]
    element_participating: wp.array2d[bool]


@wp.func(module=pose_field_active.module, name=pose_field_active.key)
def pose_field_active(field: MuJoCoPoseField, world: int, slot: int) -> bool:
    prototype, row = _env_world_location(field.env_world_bindings, world)
    if prototype < 0 or slot < 0 or slot >= field.element_participating.shape[1]:
        return False
    return field.element_participating[world, slot] and field.columns[prototype, slot] >= 0


@wp.func(module=pose_field_read.module, name=pose_field_read.key)
def pose_field_read(field: MuJoCoPoseField, world: int, slot: int) -> wp.transform:
    """Read one pose; excluded slots have zero position and zero quaternion."""
    value = wp.transform(wp.vec3(0.0), wp.quat(0.0, 0.0, 0.0, 0.0))
    prototype, row = _env_world_location(field.env_world_bindings, world)
    if prototype < 0 or slot < 0 or slot >= field.element_participating.shape[1]:
        return value
    if field.element_participating[world, slot]:
        column = field.columns[prototype, slot]
        if column >= 0:
            source = field.sources[prototype]
            if source.broadcast_rows != 0:
                row = 0
            rotation = source.quaternion[row, column]
            if source.wxyz != 0:
                rotation = wp.quat(rotation[1], rotation[2], rotation[3], rotation[0])
            value = wp.transform(source.position[row, column], rotation)
    return value


@wp.kernel
def _gather_scalars(field: MuJoCoScalarField, fill: float, out: wp.array2d[float]):
    env_index, selected = wp.tid()
    value = fill
    if scalar_field_active(field, env_index, selected):
        value = scalar_field_read(field, env_index, selected)
    out[env_index, selected] = value


@wp.kernel
def _gather_poses(
    field: MuJoCoPoseField,
    fill: float,
    out: wp.array2d[wp.transform],
):
    env_index, selected = wp.tid()
    value = wp.transform(wp.vec3(fill), wp.quat(fill, fill, fill, fill))
    if pose_field_active(field, env_index, selected):
        value = pose_field_read(field, env_index, selected)
    out[env_index, selected] = value


@wp.kernel
def _selection_active(
    placement: EnvWorldBindings,
    columns: wp.array2d[int],
    element_participating: wp.array2d[bool],
    out: wp.array2d[bool],
):
    env_index, selected = wp.tid()
    prototype, row = _env_world_location(placement, env_index)
    active = bool(False)
    if prototype >= 0 and element_participating[env_index, selected]:
        active = columns[prototype, selected] >= 0
    out[env_index, selected] = active


@wp.kernel
def _joint_types(placement: EnvWorldBindings, values: wp.array2d[int], out: wp.array2d[int]):
    env_index, selected = wp.tid()
    prototype, row = _env_world_location(placement, env_index)
    value = int(0)
    if prototype >= 0:
        value = values[prototype, selected]
    out[env_index, selected] = value


@wp.kernel
def _scatter(field: MuJoCoScalarField, env_indices: wp.array[int], values: wp.array2d[float]):
    row, selected = wp.tid()
    scalar_field_write(field, env_indices[row], selected, values[row, selected])


class MuJoCoSelection:
    """Numeric prototype-local columns placed through world ID and generation handles."""

    # Manager config serialization excludes runtime storage and retains no query.
    __slots__ = (
        "__dict__",
        "owner",
        "parts",
        "width",
        "prototype_selection_counts",
        "index_domain",
        "_indices",
        "_columns",
        "_ids",
        "element_participating",
        "_active",
        "_fields",
        "_poses",
        "_joint_types",
        "_contact_forces",
        "_contact_visible",
        "_contact_generations",
        "_contact_maps",
    )

    def __init__(self, owner, index_domain: Model.AttributeFrequency, parts, *, policy_width: int | None = None):
        self.owner, self.parts = owner, tuple(parts)
        self.index_domain = index_domain
        self.width = max(part.width for part in parts) if policy_width is None else policy_width
        if isinstance(self.width, bool) or not isinstance(self.width, int) or self.width < 0:
            raise ValueError("Policy width must be a nonnegative integer.")
        if any(part.width > self.width for part in parts):
            raise ValueError("Native selection exceeds the configured policy width.")
        # Static topology counts are per prototype; width is the padded policy extent.
        self.prototype_selection_counts = tuple(part.width for part in parts)
        self._indices = [part.ids.numpy() for part in parts]
        self._columns = np.full((len(parts), self.width), -1, np.int32)
        for prototype, ids in enumerate(self._indices):
            self._columns[prototype, : len(ids)] = ids
        self._ids = wp.array(self._columns, dtype=int, device=owner.device)
        self.element_participating = wp.ones((owner.num_envs, self.width), dtype=bool, device=owner.device)
        self._active = wp.empty_like(self.element_participating)
        self._fields, self._poses = {}, {}
        self._contact_forces = None
        if self.index_domain != Model.AttributeFrequency.BODY:
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
        """Participation by policy environment and slot; ordinary solver sleep is unrelated."""
        self.owner._check_active()
        wp.launch(
            _selection_active,
            self.element_participating.shape,
            [self.owner.env_world_bindings, self._ids, self.element_participating],
            [self._active],
            device=self.owner.device,
        )
        return wp.to_torch(self._active)

    @property
    def dense_shape(self) -> tuple[int, int]:
        """Logical environment and slot extents at the dense policy boundary."""
        return self.owner.num_envs, self.width

    def active_counts(self) -> torch.Tensor:
        """Current participating scalar/body count per policy environment (GPU int64)."""
        return self.dense_active().sum(dim=1)

    def joint_types(self) -> torch.Tensor:
        self.owner._check_active()
        if self.index_domain == Model.AttributeFrequency.BODY:
            raise ValueError("Joint types require a coordinate or DOF selection.")
        out = wp.empty(self.element_participating.shape, dtype=int, device=self.owner.device)
        wp.launch(
            _joint_types, out.shape, [self.owner.env_world_bindings, self._joint_types], [out], device=self.owner.device
        )
        return wp.to_torch(out)

    def scalar_field(self, source: Literal["state", "model", "control"], attribute: str) -> MuJoCoScalarField:
        """Borrow selected native scalar storage while the owner and runtime remain open."""
        self.owner._check_active()
        key = source, attribute
        if key in self._fields:
            return self._fields[key]
        if source not in ("state", "model", "control") or self.index_domain == Model.AttributeFrequency.BODY:
            raise ValueError("Scalar fields require a joint selection and state/model/control source.")
        populations = self.owner._borrow_populations()
        columns, offsets = self._columns.copy(), np.zeros(self._columns.shape, np.float32)
        descriptors = []
        for prototype, (part, ids, group, mapping) in enumerate(
            zip(self.parts, self._indices, populations, self.owner.mappings, strict=True)
        ):
            _validate_field_domain(part.owner.model, self.index_domain, attribute)
            descriptor = _ScalarSource()
            if source == "model":
                values = getattr(part.owner.model, attribute)
                if values.dtype != wp.float32 or values.ndim != 1:
                    raise TypeError("Selected model properties must be scalar float32 arrays.")
                values = values.reshape((1, values.shape[0]))
                descriptor.broadcast_rows = 1
            else:
                descriptor.broadcast_rows = 0
                if source == "state" and attribute == "joint_q":
                    values, native_ids = group.data.qpos, mapping.newton_coord_by_mujoco_qpos[0]
                elif source == "state" and attribute == "joint_qd":
                    values, native_ids = group.data.qvel, mapping.newton_dof_by_mujoco_dof[0]
                elif source == "control" and attribute in ("joint_target_q", "joint_target_qd"):
                    native_ids = (
                        mapping.newton_target_by_position_actuator
                        if attribute == "joint_target_q"
                        else mapping.newton_dof_by_velocity_actuator
                    )
                    values = group.data.ctrl
                elif source == "control" and attribute == "joint_f":
                    values, native_ids = group.data.qfrc_applied, mapping.newton_dof_by_mujoco_dof[0]
                else:
                    raise ValueError(f"Unsupported native scalar field {source}.{attribute}.")
                selected_columns = _native_columns(native_ids, ids)
                columns[prototype, : len(ids)] = selected_columns
                if source == "state" and attribute == "joint_q":
                    offsets[prototype, : len(ids)] = mapping.qpos_references[0, selected_columns]
                elif source == "control" and attribute == "joint_target_q":
                    offsets[prototype, : len(ids)] = mapping.position_references[0, selected_columns]
            if values.dtype != wp.float32 or values.ndim != 2:
                raise TypeError("Native scalar sources require two-dimensional float32 arrays.")
            if np.any(columns[prototype, : len(ids)] >= values.shape[1]):
                raise ValueError("A selected column exceeds its native source extent.")
            descriptor.values = values
            descriptors.append(descriptor)
        field = MuJoCoScalarField()
        field.env_world_bindings, field.element_participating = (
            self.owner.env_world_bindings,
            self.element_participating,
        )
        field.sources = wp.array(descriptors, dtype=_ScalarSource, device=self.owner.device)
        field.columns = wp.array(columns, dtype=int, device=self.owner.device)
        field.offsets = wp.array(offsets, dtype=float, device=self.owner.device)
        self._fields[key] = field
        return field

    def _read(self, source, attribute, fill):
        if self.index_domain != Model.AttributeFrequency.BODY:
            field = self.scalar_field(source, attribute)
            out = wp.empty(self.element_participating.shape, dtype=float, device=self.owner.device)
            wp.launch(
                _gather_scalars,
                out.shape,
                [field, fill],
                [out],
                device=self.owner.device,
            )
            return wp.to_torch(out)
        field = self.pose_field(source, attribute)
        out = wp.empty(self.dense_shape, dtype=wp.transform, device=self.owner.device)
        wp.launch(_gather_poses, out.shape, [field, fill], [out], device=self.owner.device)
        return wp.to_torch(out)

    def pose_field(self, source: Literal["state", "model"], attribute: str) -> MuJoCoPoseField:
        """Borrow poses while the selection owner and runtime remain open."""
        self.owner._check_active()
        if (
            self.index_domain != Model.AttributeFrequency.BODY
            or attribute != "body_q"
            or source not in ("state", "model")
        ):
            raise ValueError("Native pose fields require a body selection and state/model body_q.")
        if source not in self._poses:
            populations = self.owner._borrow_populations()
            descriptors, columns = [], self._columns.copy()
            for prototype, (part, ids, group, mapping) in enumerate(
                zip(self.parts, self._indices, populations, self.owner.mappings, strict=True)
            ):
                descriptor = _PoseSource()
                if source == "state":
                    descriptor.position, descriptor.quaternion = group.data.xpos, group.data.xquat
                    descriptor.broadcast_rows, descriptor.wxyz = 0, 1
                    columns[prototype, : len(ids)] = _native_columns(mapping.newton_body_by_mujoco_body[0], ids)
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
            field = MuJoCoPoseField()
            field.env_world_bindings = self.owner.env_world_bindings
            field.sources = wp.array(descriptors, dtype=_PoseSource, device=self.owner.device)
            field.columns = wp.array(columns, dtype=int, device=self.owner.device)
            field.element_participating = self.element_participating
            self._poses[source] = field
        return self._poses[source]

    def read_state(self, attribute: str, fill: float = 0.0) -> torch.Tensor:
        """Gather native state into policy rows, using task coordinate conventions."""
        return self._read("state", attribute, fill)

    def read_model(self, attribute: str, fill: float = 0.0) -> torch.Tensor:
        """Gather authored immutable properties, broadcast explicitly from row zero."""
        return self._read("model", attribute, fill)

    def _write(self, source, attribute, values, env_ids):
        field = self.scalar_field(source, attribute)
        device = torch.device(str(self.owner.device))
        count = _validate_write_env_indices(env_ids, self.owner.num_envs, device)
        if (
            not isinstance(values, torch.Tensor)
            or values.shape != (count, self.width)
            or values.dtype != torch.float32
            or values.device != device
        ):
            raise ValueError("Native writes require float32 policy-width rows on the native device.")
        if count == 0:
            return
        env_indices = (
            self.owner._env_indices if env_ids is None else wp.from_torch(env_ids.to(dtype=torch.int32).contiguous())
        )
        wp.launch(
            _scatter,
            values.shape,
            [field, env_indices, wp.from_torch(values.contiguous(), dtype=wp.float32)],
            device=self.owner.device,
        )

    def write_state(self, attribute: str, values: torch.Tensor, env_ids=None) -> None:
        """Write native state [m/rad or m/s/rad/s] for unique, in-range environment indices."""
        self._write("state", attribute, values, env_ids)

    def write_control(self, attribute: str, values: torch.Tensor, env_ids=None) -> None:
        """Write targets or applied force [N or Nm] for unique, in-range environment indices."""
        self._write("control", attribute, values, env_ids)

    def prepare_contact_forces(self) -> None:
        """Prepare policy normal-force reductions before capturing native steps."""
        self.owner._check_active()
        if self.index_domain != Model.AttributeFrequency.BODY:
            raise ValueError("Contact force reductions require a body selection.")
        if self._contact_forces is not None:
            return
        populations = self.owner._borrow_populations()
        self._contact_forces = wp.zeros(self.element_participating.shape, dtype=wp.vec3, device=self.owner.device)
        self._contact_visible = wp.empty_like(self._contact_forces)
        self._contact_generations = wp.zeros(self.owner.num_envs, dtype=wp.uint64, device=self.owner.device)
        self._contact_maps = []
        for ids, group, mapping in zip(self._indices, populations, self.owner.mappings, strict=True):
            slots = np.full(group.model.nbody, -1, np.int32)
            slots[_native_columns(mapping.newton_body_by_mujoco_body[0], ids)] = np.arange(len(ids))
            self._contact_maps.append(wp.array(slots, dtype=int, device=self.owner.device))

    def record_contact_forces(self, group, env_index_by_world_id) -> None:
        """Record final-substep net normal forces [N], before the prototype leaves its branch."""
        populations = self.owner._borrow_populations()
        if not 0 <= group.prototype_index < len(populations) or populations[group.prototype_index] is not group:
            raise ValueError("Contact reductions require a population from the selection's exact runtime.")
        if self._contact_forces is None:
            raise RuntimeError("Prepare contact reductions before graph capture.")
        prototype = group.prototype_index
        dimensions = group.world_capacity, self.width
        wp.launch(
            _clear_contact_forces,
            dimensions,
            inputs=[
                self.owner.env_world_bindings,
                prototype,
                env_index_by_world_id,
                self._contact_forces,
                self._contact_generations,
            ],
            device=self.owner.device,
        )
        data, model, contact = group.data, group.model, group.data.contact
        wp.launch(
            _accumulate_contact_forces,
            group.contact_capacity,
            inputs=[
                self.owner.env_world_bindings,
                prototype,
                env_index_by_world_id,
                group.world_live_count,
                group.contact_storage_ready_count,
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

    def selected_net_normal_forces(self) -> torch.Tensor:
        """Read the last native step's net normal force [N]; reset lifetimes return zero."""
        self.owner._check_active()
        if self._contact_forces is None:
            raise RuntimeError("Contact forces were not prepared in the native step program.")
        wp.launch(
            _visible_contact_forces,
            self.element_participating.shape,
            [
                self.owner.env_world_bindings,
                self._ids,
                self.element_participating,
                self._contact_generations,
                self._contact_forces,
            ],
            [self._contact_visible],
            device=self.owner.device,
        )
        return wp.to_torch(self._contact_visible)


def validate_native_mapping(owner: NewtonSelections, mapping: MuJoCoModelMapping) -> None:
    """Admit the task's scalar controls and fixed-root snapshots against producer metadata."""
    model = owner.model
    if model.world_count != 1 or mapping.newton_model is not model:
        raise ValueError("Native mapping requires its exact authored one-world Newton model.")
    coordinates, dofs = mapping.newton_coord_by_mujoco_qpos[0], mapping.newton_dof_by_mujoco_dof[0]
    if (
        len(coordinates) != mapping.mujoco_model.nq
        or len(dofs) != mapping.mujoco_model.nv
        or np.any(coordinates < 0)
        or np.any(coordinates >= model.joint_coord_count)
        or len(np.unique(coordinates)) != len(coordinates)
        or np.any(dofs < 0)
        or np.any(dofs >= model.joint_dof_count)
        or len(np.unique(dofs)) != len(dofs)
    ):
        raise ValueError("Native keyboard selections require complete unique scalar joint mappings.")
    qstarts, dstarts = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
    joints = np.searchsorted(qstarts[1:], coordinates, side="right")
    if np.any(np.diff(qstarts)[joints] != 1) or np.any(np.diff(dstarts)[joints] != 1):
        raise ValueError("Native keyboard selections admit scalar joints only.")
    bodies = mapping.newton_body_by_mujoco_body[0]
    bodies = bodies[bodies >= 0]
    if len(np.unique(bodies)) != len(bodies):
        raise ValueError("MuJoCo body mappings must be unique.")
    position, velocity = mapping.newton_target_by_position_actuator, mapping.newton_dof_by_velocity_actuator
    if (
        np.any(mapping.newton_control_by_direct_actuator >= 0)
        or np.any(mapping.axis_by_actuator >= 0)
        or np.any(mapping.newton_joint_by_ball_actuator >= 0)
        or np.any((position >= 0) == (velocity >= 0))
    ):
        raise ValueError("Native keyboard controls require mapped scalar joint-target actuators.")
    for indices in (position, velocity):
        indices = indices[indices >= 0]
        if len(np.unique(indices)) != len(indices):
            raise ValueError("Native controls require a unique actuator for each selected target.")
    roots = mapping.newton_joint_by_mujoco_mocap[0]
    if np.any(roots < 0) or np.any(roots >= model.joint_count):
        raise ValueError("Native mocap roots must map to valid authored joints.")
    if not np.allclose(model.joint_X_c.numpy()[roots], [0, 0, 0, 0, 0, 0, 1], atol=1e-7, rtol=0):
        raise ValueError("Prepared fixed-root snapshot frames must match their child body frames.")


def _native_columns(native_ids: np.ndarray, selected_ids: np.ndarray) -> np.ndarray:
    """Project an admitted producer relation into the task's selected-column order."""
    inverse = {int(index): column for column, index in enumerate(native_ids) if index >= 0}
    try:
        return np.asarray([inverse[int(index)] for index in selected_ids], dtype=np.int32)
    except KeyError as error:
        raise ValueError("Selected native field has no unique mapped target.") from error


class MuJoCoSelections:
    """Bind authored selection metadata to native runtime storage and task handles."""

    def __init__(
        self,
        metadata: tuple[NewtonSelections, ...],
        mappings: tuple[MuJoCoModelMapping, ...],
        runtime,
        world_id_by_env,
        world_generation_by_env,
        *,
        num_envs: int,
        device,
        env_participating=None,
    ):
        self.metadata, self.mappings, self.runtime = tuple(metadata), tuple(mappings), runtime
        self.num_envs, self.device = num_envs, wp.get_device(device)
        mujoco_worlds_validate(runtime)
        for population in runtime.populations:
            mujoco_world_population_validate(population)
        if runtime.device != self.device or any(owner.model.device != self.device for owner in metadata):
            raise ValueError("Selection metadata, runtime storage and handles must use the same device.")
        if not metadata or len(metadata) != len(mappings) or len(metadata) != len(runtime.populations):
            raise ValueError("Native selections require one metadata/mapping source per prepared prototype.")
        if any(mapping.newton_model is not owner.model for owner, mapping in zip(metadata, mappings, strict=True)):
            raise ValueError("Native mappings must belong to the exact prepared metadata model.")
        for owner, mapping in zip(metadata, mappings, strict=True):
            validate_native_mapping(owner, mapping)
        if any(
            mapping.mujoco_model is not population.model
            for mapping, population in zip(mappings, runtime.populations, strict=True)
        ):
            raise ValueError("MuJoCo mappings must belong to the exact runtime population model.")
        for array, dtype in ((world_id_by_env, wp.int32), (world_generation_by_env, wp.uint64)):
            if array.shape != (num_envs,) or array.dtype != dtype or array.device != self.device:
                raise ValueError("Environment world handles must have matching extent, dtype and native device.")
        self.env_participating = (
            wp.ones(num_envs, dtype=bool, device=self.device) if env_participating is None else env_participating
        )
        if (
            self.env_participating.shape != (num_envs,)
            or self.env_participating.dtype != wp.bool
            or self.env_participating.device != self.device
        ):
            raise ValueError("Environment participation must be a matching boolean device array.")
        self._env_indices = wp.array(np.arange(num_envs), dtype=int, device=self.device)
        self._bindings = {}
        self._retired = False
        capacities = []
        for group in runtime.populations:
            capacity = _WorldReadiness()
            capacity.ready_world_count = group.world_storage_ready_count
            capacities.append(capacity)
        placement = EnvWorldBindings()
        placement.world_id_by_env, placement.world_generation_by_env = world_id_by_env, world_generation_by_env
        placement.env_participating = self.env_participating
        placement.directory = runtime.directory
        placement.world_readiness_by_prototype = wp.array(capacities, dtype=_WorldReadiness, device=self.device)
        self.env_world_bindings = placement

    def bind(
        self,
        index_domain: Model.AttributeFrequency,
        indices_by_prototype: Sequence[Sequence[int] | np.ndarray],
        *,
        policy_width: int | None = None,
    ) -> MuJoCoSelection:
        """Bind integer indices for every authored prototype, without paths."""
        self._borrow_populations()
        if policy_width is not None and (
            isinstance(policy_width, bool) or not isinstance(policy_width, int) or policy_width < 0
        ):
            raise ValueError("Policy width must be a nonnegative integer.")
        if len(indices_by_prototype) != len(self.metadata):
            raise ValueError("Provide one ordered numeric ID sequence per prototype.")
        parts = tuple(
            owner.bind(index_domain, ids) for owner, ids in zip(self.metadata, indices_by_prototype, strict=True)
        )
        key = (index_domain, tuple(tuple(map(int, ids)) for ids in indices_by_prototype), policy_width)
        if key not in self._bindings:
            self._bindings[key] = MuJoCoSelection(self, index_domain, parts, policy_width=policy_width)
        return self._bindings[key]

    def _check_active(self):
        """Check prepared consumers' lifetime without revalidating fixed native descriptors."""
        if self._retired:
            raise RuntimeError("Cannot access selections from a retired owner.")
        mujoco_worlds_validate(self.runtime)

    def _borrow_populations(self):
        """Validate native descriptors before preparing a new consumer; reuse checks lifetime only."""
        self._check_active()
        for population in self.runtime.populations:
            mujoco_world_population_validate(population)
        return self.runtime.populations

    def retire(self) -> None:
        """Invalidate selection access after all GPU borrowers have joined and retired."""
        self._retired = True
        self._bindings.clear()


@wp.func
def _env_at_row(placement: EnvWorldBindings, prototype: int, row: int, env_index_by_world_id: wp.array[int]) -> int:
    identity, generation, valid = instance_directory.handle_at(placement.directory, prototype, row)
    if not valid or identity >= env_index_by_world_id.shape[0]:
        return -1
    env_index = env_index_by_world_id[identity]
    if (
        env_index < 0
        or env_index >= placement.world_id_by_env.shape[0]
        or placement.world_id_by_env[env_index] != identity
    ):
        return -1
    live_prototype, live_row = _env_world_location(placement, env_index)
    if live_prototype != prototype or live_row != row:
        return -1
    return env_index


@wp.kernel
def _clear_contact_forces(
    placement: EnvWorldBindings,
    prototype: int,
    env_index_by_world_id: wp.array[int],
    forces: wp.array2d[wp.vec3],
    generations: wp.array[wp.uint64],
):
    row, selected = wp.tid()
    env_index = _env_at_row(placement, prototype, row, env_index_by_world_id)
    if env_index >= 0:
        forces[env_index, selected] = wp.vec3(0.0)
        if selected == 0:
            generations[env_index] = placement.world_generation_by_env[env_index]


@wp.kernel
def _accumulate_contact_forces(
    placement: EnvWorldBindings,
    prototype: int,
    env_index_by_world_id: wp.array[int],
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
    env_index = _env_at_row(placement, prototype, row, env_index_by_world_id)
    if env_index < 0:
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
                    wp.atomic_add(out, env_index, selected, sign * normal)


@wp.kernel
def _visible_contact_forces(
    placement: EnvWorldBindings,
    columns: wp.array2d[int],
    element_participating: wp.array2d[bool],
    generations: wp.array[wp.uint64],
    forces: wp.array2d[wp.vec3],
    out: wp.array2d[wp.vec3],
):
    env_index, selected = wp.tid()
    value = wp.vec3(0.0)
    prototype, row = _env_world_location(placement, env_index)
    if prototype >= 0 and element_participating[env_index, selected] and columns[prototype, selected] >= 0:
        if generations[env_index] == placement.world_generation_by_env[env_index]:
            value = forces[env_index, selected]
    out[env_index, selected] = value
