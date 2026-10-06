# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard prototype preparation and atomic task reset payloads for native worlds.

Newton owns identities, placement, backing, native state and graph execution.
This module owns policy-row handles, requested keyboard variants and snapshots.
"""

from __future__ import annotations

import time

import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from gpu_components import directory as instance_directory
from gpu_components.directory_data import (
    InstanceCommands,
    InstanceDirectoryData,
    InstanceOperation,
    InstanceResults,
    InstanceStatus,
)
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.physics.worlds import NewtonWorldsBackendCfg
from newton.solvers import (
    mujoco_world_population_ready_capacity,
    mujoco_worlds_grow_backing,
    mujoco_worlds_resize_backing,
)

from isaaclab.utils import replace

from .keyboard_populations import prepare_keyboard_prototype
from .keyboards.keyboard_geometry import generate_keyboard, quat_rotate
from .mdp.reset import ResetKinematics, reset_root_state_uniform
from .mujoco_selection import (
    MuJoCoSelections,
    _accumulate_contact_forces,
    _clear_contact_forces,
    validate_native_mapping,
)
from .selection_paths import bind_selectors, query_selection_indices, resolve_selection, selector_key


@wp.kernel
def _begin_batch(commands: InstanceCommands, count: int):
    commands.sequence[0] += wp.uint64(1)
    commands.count[0] = count


@wp.kernel
def _reset_commands(
    commands: InstanceCommands,
    env_indices: wp.array[int],
    variants: wp.array[wp.int64],
    handles: wp.array[int],
    generations: wp.array[wp.uint64],
    create: int,
):
    request = wp.tid()
    env_index = env_indices[request]
    commands.operation[request] = int(InstanceOperation.REPLACE)
    commands.instance_id[request] = handles[env_index]
    commands.generation[request] = generations[env_index]
    commands.prototype[request] = int(variants[request])
    if create != 0:
        commands.operation[request] = int(InstanceOperation.CREATE)


@wp.kernel
def _validate_snapshot(
    commands: InstanceCommands,
    request_status: wp.array[int],
    consumed: wp.array[int],
    enabled: wp.array[int],
    payload: wp.array2d[float],
):
    request, column = wp.tid()
    if consumed[0] != 0 and enabled[0] != 0 and request < commands.count[0]:
        if request_status[request] == int(InstanceStatus.OK) and not wp.isfinite(payload[request, column]):
            wp.atomic_max(request_status, request, int(InstanceStatus.INVALID))


@wp.kernel
def _initialize_snapshot(
    requests: wp.array[int],
    destinations: wp.array[int],
    count: wp.array[int],
    status: wp.array[int],
    request_stride: int,
    payload_enabled: wp.array[int],
    payload: wp.array2d[float],
    q_columns: wp.array[int],
    qd_columns: wp.array[int],
    roots: wp.array[int],
    references: wp.array[float],
    q_offset: int,
    qd_offset: int,
    q: wp.array2d[float],
    qd: wp.array2d[float],
    mocap_pos: wp.array2d[wp.vec3],
    mocap_quat: wp.array2d[wp.quat],
):
    ordinal, column = wp.tid()
    if status[0] != 0 or payload_enabled[0] == 0:
        return
    for row in range(ordinal, count[0], request_stride):
        request, destination = requests[row], destinations[row]
        if column < q_columns.shape[0]:
            q[destination, column] = payload[request, q_offset + q_columns[column]] + references[column]
        if column < qd_columns.shape[0]:
            qd[destination, column] = payload[request, qd_offset + qd_columns[column]]
        if column < roots.shape[0]:
            begin = 7 * roots[column]
            mocap_pos[destination, column] = wp.vec3(
                payload[request, begin], payload[request, begin + 1], payload[request, begin + 2]
            )
            mocap_quat[destination, column] = wp.quat(
                payload[request, begin + 6],
                payload[request, begin + 3],
                payload[request, begin + 4],
                payload[request, begin + 5],
            )


@wp.kernel
def _acknowledge_snapshot(
    requests: wp.array[int],
    count: wp.array[int],
    status: wp.array[int],
    sequence: wp.array[wp.uint64],
    initialized_sequence: wp.array[wp.uint64],
):
    ordinal = wp.tid()
    # The preceding default copy and every snapshot column must finish first.
    if ordinal < count[0] and status[0] == 0:
        initialized_sequence[requests[ordinal]] = sequence[0]


@wp.kernel
def _publish_handles(
    results: InstanceResults,
    batch_status: wp.array[int],
    env_indices: wp.array[int],
    requested: wp.array[wp.int64],
    handles: wp.array[int],
    generations: wp.array[wp.uint64],
    inverse: wp.array[int],
    variants: wp.array[wp.int64],
    failed: wp.array[int],
):
    request = wp.tid()
    if batch_status[0] != 0 or results.status[request] != int(InstanceStatus.OK):
        wp.atomic_max(failed, 0, 1)
        return
    env_index, identity = env_indices[request], results.instance_id[request]
    handles[env_index] = identity
    generations[env_index] = results.generation[request]
    inverse[identity] = env_index
    variants[env_index] = requested[request]


@wp.kernel
def _backing_demand(commands: InstanceCommands, directory: InstanceDirectoryData, demand: wp.array2d[int]):
    prototype = wp.tid()
    before = directory.live_count[prototype]
    after = before
    for request in range(commands.count[0]):
        if commands.prototype[request] == prototype:
            before += 1
            after += 1
        if commands.operation[request] == int(InstanceOperation.REPLACE):
            source, row, valid = instance_directory.location(
                directory, commands.instance_id[request], commands.generation[request]
            )
            if valid and source == prototype:
                after -= 1
    demand[prototype, 0] = before
    demand[prototype, 1] = after


@wp.kernel
def _record_overflow(values: wp.array[int], prototype: int, sticky: wp.array[int]):
    if values[wp.tid()] != 0:
        wp.atomic_or(sticky, prototype, values[wp.tid()])


class KeyboardWorlds:
    """Prepare immutable keyboards and submit complete snapshots before publication.

    Replay and normal reset payloads are completed before native publication.
    Unchanged worlds retain their native state, controls and solver history.
    ``variant_ids`` records each successful publication; desired IDs are future
    requests and staged IDs are the next reset plan. A partially rejected batch
    retains its actual successes, invalidates the task, and requires closing it.
    Task/MDP state is not rolled back across already published physics lifetimes.
    """

    capacity_overflow_mask = ~int(mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS)

    def __init__(self, env):
        self.env = env
        self._sources, self._solvers = [], []
        self._selection_owner = None
        try:
            spare_bytes = env.cfg.worlds_spare_memory_budget_bytes
            if spare_bytes is not None and (type(spare_bytes) is not int or spare_bytes < 0):
                raise ValueError("Native spare backing budget must be a nonnegative integer or None.")
            physics = env.cfg.sim.physics.prototype_physics
            if not isinstance(physics, NewtonCfg) or not isinstance(physics.solver_cfg, MJWarpSolverCfg):
                raise ValueError("Native keyboard worlds require prepared Newton MJWarp physics.")
            if not physics.solver_cfg.use_mujoco_contacts or physics.collision_cfg is not None:
                raise ValueError("Native keyboard worlds require native contacts and no collision_cfg.")
            reset = env.cfg.commands.typing.reset
            if reset.replay_only and (not reset.enabled or reset.ik is None):
                raise ValueError("Native replay-only resets require an enabled typing curriculum with reset IK.")
            configs = env.cfg.keyboard_variants or (env.cfg.scene.keyboard.spawn,)
            robot_counts = env.cfg.robot_counts
            if (
                not robot_counts
                or any(type(count) is not int or count not in (1, 2) for count in robot_counts)
                or len(set(robot_counts)) != len(robot_counts)
            ):
                raise ValueError("Native keyboard robot_counts must contain unique choices of one or two arms.")
            self.configs = tuple(
                replace(cfg, topology_mode="exact", partition_mode="fixed_dof") for cfg in configs for _ in robot_counts
            )
            self.robot_counts = tuple(count for _ in configs for count in robot_counts)
            if reset.replay_only and reset.bank_path is None and reset.buffer_size < len(self.configs):
                raise ValueError("Replay-only online buffers require at least one snapshot per keyboard prototype.")
            if not reset.replay_only or reset.bank_path is None:
                if (
                    reset.ik is None
                    or reset.pre_solve_reset is None
                    or reset.pre_solve_reset.func is not reset_root_state_uniform
                ):
                    raise ValueError("Native normal resets require compact IK and reset_root_state_uniform pre-solve.")
                for event in vars(env.cfg.events).values():
                    if getattr(event, "mode", None) == "reset" and event.min_step_count_between_reset != 0:
                        raise ValueError("Native normal resets require unconditional reset events.")
                    if getattr(event, "mode", None) == "reset" and event.func is not reset_root_state_uniform:
                        raise ValueError("Native normal resets admit only fixed keyboard root reset events.")
                    if getattr(event, "mode", None) == "reset" and selector_key(event.params["roots"]) != selector_key(
                        reset.pre_solve_reset.params["roots"]
                    ):
                        raise ValueError("Native reset events must select the same keyboard roots as pre-solve.")
                for term in (reset.pre_solve_reset, *vars(env.cfg.events).values()):
                    if getattr(term, "func", None) is reset_root_state_uniform:
                        if any(
                            any(value != 0 for value in bounds) for bounds in term.params["velocity_range"].values()
                        ):
                            raise ValueError("Native fixed-root resets require zero velocity ranges.")
            if (
                env.cfg.sim.physics.deterministic
                or physics.deterministic
                or physics.deterministic_mode != "not_guaranteed"
            ):
                raise ValueError("Deterministic native world execution is not yet supported.")
            self.layouts = tuple(generate_keyboard(cfg) for cfg in self.configs)
            if any(layout.active_key_count % 6 or not 6 <= layout.active_key_count <= 108 for layout in self.layouts):
                raise ValueError("Keyboard prototypes require 6..108 keys in multiples of six.")
            self.key_counts = torch.tensor([layout.active_key_count for layout in self.layouts], device=env.device)
            self.backspace_slots = torch.tensor(
                [
                    next(key.slot for key in layout.active_keys if key.label.lower() in ("backspace", "bksp"))
                    for layout in self.layouts
                ],
                device=env.device,
            )
            self.labels = tuple(
                tuple(key.label for key in layout.keys) + ("",) * (108 - layout.slot_count) for layout in self.layouts
            )
            # Last successful task publication; the directory owns lifetime and placement.
            self.variant_ids = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
            self.desired_variant_ids = env.all_env_ids.remainder(len(self.configs)).long()
            self._staged_variant_ids = self.desired_variant_ids.clone()
            self.world_id_by_env = wp.full(env.num_envs, -1, dtype=wp.int32, device=env.device)
            self.world_generation_by_env = wp.zeros(env.num_envs, dtype=wp.uint64, device=env.device)
            self.env_index_by_world_id = wp.full(env.num_envs, -1, dtype=wp.int32, device=env.device)
            self.commands = instance_directory.allocate_commands(env.num_envs, device=env.device)
            self.results = instance_directory.allocate_results(env.num_envs, device=env.device)
            self._request_env_indices = wp.empty(env.num_envs, dtype=wp.int32, device=env.device)
            self._request_variants = torch.empty(env.num_envs, dtype=torch.long, device=env.device)
            self._payload_enabled = wp.zeros(1, dtype=wp.int32, device=env.device)
            self._failed = wp.zeros(1, dtype=wp.int32, device=env.device)
            self._demand = wp.empty((len(self.configs), 2), dtype=wp.int32, device=env.device)
            self.overflow = wp.zeros(len(self.configs), dtype=wp.int32, device=env.device)
            self._selector_cfgs = {}

            def remember(cfg):
                self._selector_cfgs[selector_key(cfg)] = cfg
                return cfg

            for cfg in self._manager_configs():
                bind_selectors(cfg, remember)
            command = env.cfg.commands.typing
            root_width = command.reset_roots.policy_width or command.reset_roots.count_per_world
            coord_width = command.reset_coords.policy_width or command.reset_coords.count_per_world
            dof_width = command.reset_dofs.policy_width or command.reset_dofs.count_per_world
            self._q_offset, self._qd_offset = 7 * root_width, 7 * root_width + coord_width
            self._payload = torch.empty((env.num_envs, self._qd_offset + dof_width), device=env.device)
            self._payload_wp = wp.from_torch(self._payload)
            prepared = self._prepare_prototypes(physics, command)
            self.reset_kinematics = None
            if not reset.replay_only or reset.bank_path is None:
                self._prepare_reset_staging(command)
            initial_counts = [
                env.num_envs // len(self.configs) + int(i < env.num_envs % len(self.configs))
                for i in range(len(self.configs))
            ]
            # Reserve addresses for the largest simultaneous old + replacement batch,
            # while backing only initial populations and bounded reset headroom.
            capacities = (2 * env.num_envs,) * len(self.configs)
            ready = tuple(min(2 * env.num_envs, n + max(8, n // 2)) for n in initial_counts)
            self.backend = env.sim.get_or_create_backend(
                NewtonWorldsBackendCfg(
                    prototypes=tuple(prepared),
                    world_capacities=capacities,
                    world_id_capacity=env.num_envs,
                    command_capacity=env.num_envs,
                    initial_world_ready_capacities=ready,
                    dt=env.physics_dt / physics.num_substeps,
                    substeps=physics.num_substeps,
                    memory_budget_bytes=env.cfg.worlds_memory_budget_bytes,
                )
            )
            runtime = self.backend.runtime
            self._selection_owner = MuJoCoSelections(
                tuple(self._sources),
                tuple(self._prototype_maps),
                runtime,
                self.world_id_by_env,
                self.world_generation_by_env,
                num_envs=env.num_envs,
                device=env.device,
            )
            contact_cfg = env.cfg.terminations.excessive_contact
            self._contact_selection = None
            if contact_cfg is not None:
                self._contact_selection = self.resolve(contact_cfg.params["bodies"])
                self._contact_selection.prepare_contact_forces()

            def initialize(group, requests, destinations, count, status, initialized_sequence, sequence):
                qmap, qdmap, roots, refs = self._snapshot_maps[group.prototype_index]
                request_stride = min(env.num_envs, 32)
                wp.launch(
                    _initialize_snapshot,
                    (request_stride, max(qmap.size, qdmap.size, roots.size)),
                    [
                        requests,
                        destinations,
                        count,
                        status,
                        request_stride,
                        self._payload_enabled,
                        self._payload_wp,
                        qmap,
                        qdmap,
                        roots,
                        refs,
                        self._q_offset,
                        self._qd_offset,
                        group.data.qpos,
                        group.data.qvel,
                        group.data.mocap_pos,
                        group.data.mocap_quat,
                    ],
                    device=env.device,
                )
                wp.launch(
                    _acknowledge_snapshot,
                    env.num_envs,
                    [requests, count, status, sequence, initialized_sequence],
                    device=env.device,
                )

            def validate(commands, request_status, consumed):
                wp.capture_if(
                    consumed,
                    on_true=lambda: wp.launch(
                        _validate_snapshot,
                        self._payload.shape,
                        [commands, request_status, consumed, self._payload_enabled, self._payload_wp],
                        device=env.device,
                    ),
                )

            def after_substep(group):
                wp.launch(
                    _record_overflow,
                    group.world_capacity,
                    inputs=[group.data.overflow, group.prototype_index, self.overflow],
                    device=env.device,
                )
                if self._contact_selection is not None:
                    self._contact_selection.record_contact_forces(group, self.env_index_by_world_id)

            def application_bindings(group, launches):
                counts = {
                    _record_overflow: group.world_live_count,
                    _clear_contact_forces: group.world_live_count,
                    _accumulate_contact_forces: group.contact_storage_ready_count,
                }
                return tuple((index, 0, counts[record.kernel]) for index, record in enumerate(launches)), (), ()

            wp.load_module(module=__name__, device=env.device)
            wp.load_module(module="isaaclab_tasks.contrib.keyboard.mujoco_selection", device=env.device)
            self.backend.prepare(
                self.commands,
                self.results,
                validate=validate,
                initialize=initialize,
                after_substep=after_substep,
                application_bindings=application_bindings,
                # Selection owns all contact descriptors and environment handles;
                # retain it directly rather than the task/backend/graph root.
                retain=(
                    self._payload_wp,
                    self._payload_enabled,
                    self._snapshot_maps,
                    self.overflow,
                    self._contact_selection,
                    self.env_index_by_world_id,
                ),
            )
            env.sim.physics_manager.install(self.backend)
            self._submit(env.all_env_ids, self.desired_variant_ids, create=True)
            self.reset_publication_count = 0
            self.last_changed_worlds = 0
            self.last_reset_publication_ms = 0.0
        except BaseException as error:
            try:
                try:
                    if self._sources:
                        wp.synchronize_device(env.device)
                finally:
                    self.close()
            except BaseException as cleanup_error:
                raise error from cleanup_error
            raise

    def _manager_configs(self):
        cfg = self.env.cfg
        return cfg.commands, cfg.actions, cfg.observations, cfg.rewards, cfg.terminations, cfg.events

    def _prepare_prototypes(self, physics, command):
        """Compile exact arm/keyboard models and their native reset and half-key mappings."""
        spacing = self.env.cfg.robot_base_min_spacing
        if type(spacing) not in (int, float) or not np.isfinite(spacing) or spacing < 0:
            raise ValueError("Robot base minimum spacing must be a finite nonnegative number.")
        half_spacing = spacing * 0.5
        key_arms = np.full((len(self.configs), 108), -1, dtype=np.int64)
        for prototype, (layout, count) in enumerate(zip(self.layouts, self.robot_counts, strict=True)):
            key_arms[prototype, : layout.active_key_count] = 0
            if count == 2:
                xs = np.array([key.center[0] for key in layout.active_keys])
                right = xs >= np.median(xs)
                if right.all() or not right.any():
                    right[:] = False
                    right[np.argsort(xs, kind="stable")[len(xs) // 2 :]] = True
                key_arms[prototype, : layout.active_key_count] = right
        self.key_arm_ids = torch.tensor(key_arms, device=self.env.device)
        self._snapshot_maps, self._prototype_maps, prepared = [], [], []
        for prototype, (cfg, layout, count) in enumerate(
            zip(self.configs, self.layouts, self.robot_counts, strict=True)
        ):
            robot_positions = (self.env.cfg.scene.robot.init_state.pos,)
            if count == 2:
                positions = []
                for arm in range(count):
                    center = np.mean(
                        [key.center[0] for key in layout.active_keys if key_arms[prototype, key.slot] == arm]
                    )
                    center = min(center, -half_spacing) if arm == 0 else max(center, half_spacing)
                    shift = quat_rotate(self.env.cfg.scene.keyboard.init_state.rot, (float(center), 0.0, 0.0))
                    positions.append(tuple(a + b for a, b in zip(self.env.cfg.scene.robot.init_state.pos, shift)))
                robot_positions = tuple(positions)
            solver, source = prepare_keyboard_prototype(
                self.env,
                cfg,
                physics,
                self._selector_cfgs.values(),
                contact_capacity=96 * count + 8 * layout.active_key_count,
                robot_positions=robot_positions,
            )
            self._solvers.append(solver)
            self._sources.append(source)
            solver.mjw_model.opt.timestep.fill_(self.env.physics_dt / physics.num_substeps)
            defaults = mjw.replicate_data(solver.mjw_data, 1)
            mjw.forward(solver.mjw_model, defaults)
            warm = mjw.replicate_data(defaults, 1)
            mjw.step(solver.mjw_model, warm)
            prepared.append((solver.mjw_model, defaults))
            mapping = solver.model_mapping
            validate_native_mapping(source, mapping)
            self._prototype_maps.append(mapping)
            self._snapshot_maps.append(self._snapshot_columns(mapping, source, command))
        self._dof_counts = torch.tensor(
            [solver.model.joint_dof_count for solver in self._solvers], device=self.env.device
        )
        return prepared

    def _snapshot_columns(self, mapping, source, command):
        qids = query_selection_indices(source.model, command.reset_coords)
        qdids = query_selection_indices(source.model, command.reset_dofs)
        roots = query_selection_indices(source.model, command.reset_roots)
        qcolumns, qdcolumns = {int(v): i for i, v in enumerate(qids)}, {int(v): i for i, v in enumerate(qdids)}
        if set(mapping.newton_coord_by_mujoco_qpos[0].tolist()) != set(qcolumns) or set(
            mapping.newton_dof_by_mujoco_dof[0].tolist()
        ) != set(qdcolumns):
            raise ValueError("Reset snapshots must cover every native scalar coordinate and velocity.")
        root_columns = {int(v): i for i, v in enumerate(roots)}
        qmap = np.array([qcolumns[int(v)] for v in mapping.newton_coord_by_mujoco_qpos[0]], dtype=np.int32)
        qdmap = np.array([qdcolumns[int(v)] for v in mapping.newton_dof_by_mujoco_dof[0]], dtype=np.int32)
        rootmap = np.array(
            [root_columns[int(v)] for v in source.model.joint_child.numpy()[mapping.newton_joint_by_mujoco_mocap[0]]],
            dtype=np.int32,
        )
        return tuple(
            wp.array(values, dtype=dtype, device=self.env.device)
            for values, dtype in (
                (qmap, wp.int32),
                (qdmap, wp.int32),
                (rootmap, wp.int32),
                (mapping.qpos_references[0], wp.float32),
            )
        )

    def _prepare_reset_staging(self, command):
        """Compile prototype-specific snapshot tables and one workspace shared by all present arms."""
        root_width = self._q_offset // 7
        defaults, key_roots, key_local, root_active, keyboard_columns = [], [], [], [], []
        robot_roots, robot_columns, robot_limits, finger_columns = [], [], [], []
        max_arms = max(self.robot_counts)
        num_tips = len(command.reset.ik.tip_offsets)
        self.reset_kinematics = None
        for prototype, (solver, source) in enumerate(zip(self._solvers, self._sources, strict=True)):
            model = solver.model
            roots, coords, dofs = (
                resolve_selection(source, cfg)
                for cfg in (command.reset_roots, command.reset_coords, command.reset_dofs)
            )
            poses = roots.read_model("body_q")[0]
            q, qd = coords.read_model("joint_q")[0], dofs.read_model("joint_qd")[0]
            snapshot_width = self._qd_offset + (command.reset_dofs.policy_width or command.reset_dofs.count_per_world)
            snapshot = torch.zeros(snapshot_width, device=self.env.device)
            snapshot[: poses.numel()] = poses.flatten()
            snapshot[self._q_offset : self._q_offset + len(q)] = q
            snapshot[self._qd_offset : self._qd_offset + len(qd)] = qd
            defaults.append(snapshot)
            active = torch.zeros(root_width, dtype=torch.bool, device=self.env.device)
            active[: len(poses)] = roots.dense_active()[0]
            root_active.append(active)
            root_ids = resolve_selection(source, replace(command.reset_roots, policy_width=None)).ids.numpy()
            root_index = {int(body): column for column, body in enumerate(root_ids)}
            parent, child = model.joint_parent.numpy(), model.joint_child.numpy()
            body_parent = {int(body): int(parent[joint]) for joint, body in enumerate(child)}
            key_ids = resolve_selection(source, replace(command.key_bodies, policy_width=None)).ids.numpy()
            local = np.zeros(
                (command.key_bodies.policy_width or command.key_bodies.count_per_world, 3), dtype=np.float32
            )
            columns = np.full(len(local), -1, dtype=np.int64)
            body_poses = model.body_q.numpy()
            for slot, body in enumerate(key_ids):
                root = int(body)
                while root not in root_index:
                    root = body_parent[root]
                    if root < 0:
                        raise ValueError("Every reset key must descend from a selected fixed root.")
                columns[slot] = root_index[root]
                local[slot] = np.array(
                    wp.transform_point(
                        wp.transform_inverse(wp.transform(*body_poses[root])), wp.vec3(*body_poses[body, :3])
                    )
                )
            key_roots.append(columns)
            key_local.append(local)
            selected_roots = resolve_selection(
                source, replace(command.reset.pre_solve_reset.params["roots"], policy_width=None)
            ).ids.numpy()
            root_mask = torch.zeros(root_width, dtype=torch.bool, device=self.env.device)
            root_mask[[root_index[int(body)] for body in selected_roots]] = True
            keyboard_columns.append(root_mask)
            robot = resolve_selection(source, command.robot_joints).ids.numpy()
            robot_dofs = resolve_selection(source, command.robot_dofs).ids.numpy()
            ik_dofs = resolve_selection(source, command.reset.ik.dofs).ids.numpy()
            ik_coords = resolve_selection(source, command.reset.ik.joints).ids.numpy()
            tips = resolve_selection(source, command.reset.ik.bodies).ids.numpy()
            if len(tips) != self.robot_counts[prototype] * num_tips:
                raise ValueError("Reset IK must select every configured fingertip for each authored arm.")
            body_roots = np.empty(model.body_count, dtype=np.int64)
            for body in range(model.body_count):
                root = body
                while root not in root_index:
                    root = body_parent[root]
                    if root < 0:
                        raise ValueError("Every reset body must descend from a selected fixed root.")
                body_roots[body] = root
            qs, ds = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
            coord_roots = body_roots[child[np.searchsorted(qs[1:], robot, side="right")]]
            dof_roots = body_roots[child[np.searchsorted(ds[1:], robot_dofs, side="right")]]
            ik_coord_roots = body_roots[child[np.searchsorted(qs[1:], ik_coords, side="right")]]
            ik_dof_roots = body_roots[child[np.searchsorted(ds[1:], ik_dofs, side="right")]]
            qids = resolve_selection(source, replace(command.reset_coords, policy_width=None)).ids.numpy()
            qcolumn = {int(coord): column for column, coord in enumerate(qids)}
            roots_for_arms = np.full(max_arms, -1, dtype=np.int64)
            fingers_for_arms = np.full((max_arms, num_tips), -1, dtype=np.int64)
            columns_for_arms = limits_for_arms = None
            lower, upper = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
            for arm, root in enumerate(dict.fromkeys(body_roots[tips])):
                finger_ids = np.flatnonzero(body_roots[tips] == root)
                if len(finger_ids) != num_tips:
                    raise ValueError("Every reset arm requires the same configured fingertip order.")
                arm_tips = tips[finger_ids]
                arm_coords, arm_dofs = robot[coord_roots == root], robot_dofs[dof_roots == root]
                arm_ik_coords, arm_ik_dofs = ik_coords[ik_coord_roots == root], ik_dofs[ik_dof_roots == root]
                if len(arm_coords) != len(arm_dofs):
                    raise ValueError("Reset arms require matching scalar robot coordinates and DOFs.")
                robot_column = {int(coord): column for column, coord in enumerate(arm_coords)}
                ik_columns = np.array([robot_column[int(coord)] for coord in arm_ik_coords], dtype=np.int64)
                if self.reset_kinematics is None:
                    self.reset_ik_columns = torch.tensor(ik_columns, device=self.env.device)
                    self.reset_kinematics = ResetKinematics(
                        model,
                        arm_coords,
                        arm_ik_dofs,
                        arm_tips,
                        self.env.num_envs * max_arms,
                        command.reset.ik.tip_offsets,
                    )
                else:
                    if not np.array_equal(ik_columns, self.reset_ik_columns.cpu().numpy()):
                        raise ValueError("Keyboard arms must share local IK coordinate ordering.")
                    self.reset_kinematics.validate_model(model, arm_coords, arm_ik_dofs, arm_tips)
                if columns_for_arms is None:
                    columns_for_arms = np.full((max_arms, len(arm_coords)), -1, dtype=np.int64)
                    limits_for_arms = np.zeros((max_arms, len(arm_coords), 2), dtype=np.float32)
                roots_for_arms[arm] = root_index[root]
                fingers_for_arms[arm] = finger_ids
                columns_for_arms[arm] = [qcolumn[int(coord)] for coord in arm_coords]
                center = (lower[arm_dofs] + upper[arm_dofs]) * 0.5
                half = (upper[arm_dofs] - lower[arm_dofs]) * (0.5 * command.soft_joint_pos_limit_factor)
                limits_for_arms[arm] = np.stack((center - half, center + half), axis=-1)
            robot_roots.append(roots_for_arms)
            robot_columns.append(columns_for_arms)
            robot_limits.append(limits_for_arms)
            finger_columns.append(fingers_for_arms)
        self.reset_defaults = torch.stack(defaults)
        self.reset_root_active = torch.stack(root_active)
        self.reset_keyboard_roots = torch.stack(keyboard_columns)
        self.reset_key_root = torch.tensor(np.stack(key_roots), device=self.env.device)
        self.reset_key_local = torch.tensor(np.stack(key_local), device=self.env.device)
        self.reset_robot_roots = torch.tensor(np.stack(robot_roots), device=self.env.device)
        self.reset_robot_columns = torch.tensor(np.stack(robot_columns), device=self.env.device)
        self.reset_robot_limits = torch.tensor(np.stack(robot_limits), device=self.env.device)
        self.reset_finger_columns = torch.tensor(np.stack(finger_columns), device=self.env.device)

    def resolve(self, cfg):
        """Compile task paths once before returning a numeric runtime binding."""
        ids = tuple(query_selection_indices(source.model, cfg) for source in self._sources)
        return self._selection_owner.bind(cfg.index_domain, ids, policy_width=cfg.policy_width)

    def staged_variant_ids(self, env_ids):
        """Return requested prototypes staged for this episode boundary."""
        return self._staged_variant_ids[env_ids]

    def request_variants(self, env_ids, variant_ids=None):
        """Update desired keyboard assignments without changing a live episode."""
        self._validate_request(env_ids, variant_ids)
        if variant_ids is None:
            offsets = torch.randint(1, max(2, len(self.layouts)), env_ids.shape, device=self.env.device)
            variant_ids = (self.variant_ids[env_ids] + offsets) % len(self.layouts)
        self.desired_variant_ids[env_ids] = variant_ids

    def stage_variant_changes(self, eligible_ids):
        """Stage eligible switches; publication waits for the complete snapshot."""
        self._validate_request(eligible_ids)
        changed = eligible_ids[self.desired_variant_ids[eligible_ids] != self.variant_ids[eligible_ids]]
        self._staged_variant_ids[eligible_ids] = self.desired_variant_ids[eligible_ids]
        return changed

    def reset_from_snapshot(self, env_ids, variant_ids, snapshot):
        """Initialize and publish full native lifetimes from completed task snapshots."""
        self._validate_request(env_ids, variant_ids)
        if (
            snapshot.shape != (len(env_ids), self._payload.shape[1])
            or snapshot.dtype != torch.float32
            or snapshot.device != self._payload.device
        ):
            raise ValueError("Reset payload must contain complete float32 root, coordinate and velocity rows.")
        started = time.perf_counter()
        self._payload[: len(env_ids)].copy_(snapshot)
        self._payload_enabled.fill_(1)
        changed = (self.variant_ids[env_ids] != variant_ids).sum()
        self._submit(env_ids, variant_ids)
        self._staged_variant_ids[env_ids] = variant_ids
        self.last_changed_worlds = changed
        self.reset_publication_count += 1
        self.last_reset_publication_ms = (time.perf_counter() - started) * 1000.0

    def _validate_request(self, env_ids, variant_ids=None):
        self.env._check_active()
        if (
            not isinstance(env_ids, torch.Tensor)
            or env_ids.ndim != 1
            or env_ids.dtype != torch.long
            or env_ids.device != self.variant_ids.device
        ):
            raise ValueError("Environment indices must be a one-dimensional int64 tensor on the task device.")
        torch._assert_async(
            ((env_ids >= 0) & (env_ids < self.env.num_envs)).all(), "Environment indices are outside the task."
        )
        seen = torch.zeros_like(self.variant_ids, dtype=torch.int32)
        seen.scatter_add_(0, env_ids, torch.ones_like(env_ids, dtype=torch.int32))
        torch._assert_async((seen <= 1).all(), "Environment indices must be unique.")
        if variant_ids is not None:
            if (
                not isinstance(variant_ids, torch.Tensor)
                or variant_ids.dtype != torch.long
                or variant_ids.shape != env_ids.shape
                or variant_ids.device != env_ids.device
            ):
                raise ValueError("Variant IDs must be an equally sized int64 tensor on the task device.")
            torch._assert_async(
                ((variant_ids >= 0) & (variant_ids < len(self.layouts))).all(),
                "Variant IDs are outside the prepared keyboard set.",
            )

    def _submit(self, env_ids, variant_ids, *, create=False):
        count = len(env_ids)
        if not count:
            return
        wp.to_torch(self._request_env_indices)[:count].copy_(env_ids)
        self._request_variants[:count].copy_(variant_ids)
        try:
            wp.launch(_begin_batch, 1, [self.commands, count], device=self.env.device)
            wp.launch(
                _reset_commands,
                count,
                [
                    self.commands,
                    self._request_env_indices,
                    wp.from_torch(self._request_variants),
                    self.world_id_by_env,
                    self.world_generation_by_env,
                    int(create),
                ],
                device=self.env.device,
            )
            runtime = self.backend.runtime
            wp.launch(
                _backing_demand,
                len(self.layouts),
                [self.commands, runtime.directory, self._demand],
                device=self.env.device,
            )
            demand = self._demand.numpy()
            streams = (wp.get_stream(self.env.device),)
            targets, ready_capacities = [], []
            for group, (required, _) in zip(runtime.populations, demand, strict=True):
                required = int(required)
                ready = mujoco_world_population_ready_capacity(group)
                ready_capacities.append(ready)
                if required > ready or ready > 2 * required + 16 or required == 0:
                    ready = min(group.world_capacity, required + max(8, required // 2)) if required else 0
                targets.append(ready)
            if targets != ready_capacities:
                try:
                    if all(n >= ready for n, ready in zip(targets, ready_capacities, strict=True)):
                        mujoco_worlds_grow_backing(runtime, tuple(targets), streams=streams)
                    else:
                        mujoco_worlds_resize_backing(
                            runtime,
                            tuple(targets),
                            streams=streams,
                            spare_bytes=self.env.cfg.worlds_spare_memory_budget_bytes,
                        )
                except MemoryError:
                    # Headroom is optional; demand includes every live source and incoming lifetime.
                    required_capacities = tuple(int(value) for value in demand[:, 0])
                    if required_capacities == tuple(targets):
                        raise
                    mujoco_worlds_resize_backing(
                        runtime,
                        required_capacities,
                        streams=streams,
                        spare_bytes=self.env.cfg.worlds_spare_memory_budget_bytes,
                    )
            self.backend.forward()
            self._failed.zero_()
            wp.launch(
                _publish_handles,
                count,
                [
                    self.results,
                    runtime.batch_result.status,
                    self._request_env_indices,
                    wp.from_torch(self._request_variants),
                    self.world_id_by_env,
                    self.world_generation_by_env,
                    self.env_index_by_world_id,
                    wp.from_torch(self.variant_ids),
                    self._failed,
                ],
                device=self.env.device,
            )
            if self._failed.numpy()[0]:
                raise RuntimeError(
                    "Native snapshot publication failed; the environment is stopped before further MDP reads."
                )
            # Compaction made every live population a prefix. Return empty groups
            # immediately and bound past peaks, while retaining headroom for small resets.
            targets, ready_capacities = [], []
            for group, (_, live) in zip(runtime.populations, demand, strict=True):
                live = int(live)
                ready = mujoco_world_population_ready_capacity(group)
                ready_capacities.append(ready)
                if not live or ready > 2 * live + 16:
                    ready = min(group.world_capacity, live + max(8, live // 2)) if live else 0
                targets.append(ready)
            if targets != ready_capacities:
                mujoco_worlds_resize_backing(
                    runtime, tuple(targets), streams=streams, spare_bytes=self.env.cfg.worlds_spare_memory_budget_bytes
                )
        except BaseException:
            self.env._population_bindings_valid = False
            raise

    @property
    def populated_prototype_count(self):
        return (wp.to_torch(self.backend.runtime.directory.live_count) > 0).sum()

    @property
    def live_dof_count(self):
        counts = wp.to_torch(self.backend.runtime.directory.live_count)
        return (counts * self._dof_counts).sum()

    def reconcile_state(self, dirty_worlds, flags):
        if flags:
            raise ValueError("Native keyboard prototypes are immutable; reset instance roots through a snapshot.")
        self.backend.forward()

    def close(self):
        """Release task-owned descriptor caches after the simulation has joined readers."""
        if self._selection_owner is not None:
            self._selection_owner.retire()
        for source in self._sources:
            source.retire()
        self._sources.clear()
        self._solvers.clear()
