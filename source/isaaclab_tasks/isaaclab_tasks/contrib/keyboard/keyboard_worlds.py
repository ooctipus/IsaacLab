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
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.physics.worlds import NewtonWorldsBackendCfg
from newton.worlds import (
    WorldCommands,
    WorldDirectoryData,
    WorldOperation,
    WorldPhase,
    WorldResults,
    WorldStatus,
    WorldTransaction,
    create_world_commands,
    create_world_results,
    world_location,
)

from .keyboard_populations import prepare_keyboard_prototype
from .keyboards.keyboard_geometry import generate_keyboard
from .mdp.reset import ResetKinematics, reset_root_state_uniform
from .native_selection import NativePrototypeMapping, NativeSelections
from .selection_paths import bind_selectors, query_selection_ids, resolve_selection, selector_key


@wp.kernel
def _begin_batch(commands: WorldCommands, count: int):
    commands.sequence[0] += wp.uint64(1)
    commands.count[0] = count


@wp.kernel
def _reset_commands(
    commands: WorldCommands,
    actors: wp.array[int],
    variants: wp.array[wp.int64],
    handles: wp.array[int],
    generations: wp.array[wp.uint64],
    create: int,
):
    request = wp.tid()
    actor = actors[request]
    commands.op[request] = int(WorldOperation.RESET)
    commands.id[request] = handles[actor]
    commands.generation[request] = generations[actor]
    commands.prototype[request] = int(variants[request])
    if create != 0:
        commands.op[request] = int(WorldOperation.CREATE)


@wp.kernel
def _validate_snapshot(
    commands: WorldCommands,
    transaction: WorldTransaction,
    enabled: wp.array[int],
    payload: wp.array2d[float],
):
    request, column = wp.tid()
    if enabled[0] != 0 and transaction.phase[0] == int(WorldPhase.VALIDATED) and request < commands.count[0]:
        if not wp.isfinite(payload[request, column]):
            wp.atomic_max(transaction.status, request, int(WorldStatus.INVALID))


@wp.kernel
def _initialize_snapshot(
    requests: wp.array[int],
    destinations: wp.array[int],
    count: wp.array[int],
    status: wp.array[int],
    transaction: WorldTransaction,
    sequence: wp.array[wp.uint64],
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
    ordinal = wp.tid()
    if ordinal >= count[0] or status[0] != 0 or transaction.phase[0] != int(WorldPhase.ADMITTED):
        return
    request = requests[ordinal]
    if transaction.status[request] != int(WorldStatus.OK):
        return
    destination = destinations[ordinal]
    if payload_enabled[0] != 0:
        for joint in range(q_columns.shape[0]):
            q[destination, joint] = payload[request, q_offset + q_columns[joint]] + references[joint]
        for dof in range(qd_columns.shape[0]):
            qd[destination, dof] = payload[request, qd_offset + qd_columns[dof]]
        for root in range(roots.shape[0]):
            begin = 7 * roots[root]
            mocap_pos[destination, root] = wp.vec3(
                payload[request, begin], payload[request, begin + 1], payload[request, begin + 2]
            )
            mocap_quat[destination, root] = wp.quat(
                payload[request, begin + 6],
                payload[request, begin + 3],
                payload[request, begin + 4],
                payload[request, begin + 5],
            )
    # Defaults copied the entire native row first, including solver history and controls.
    transaction.initialized[request] = sequence[0]


@wp.kernel
def _publish_handles(
    results: WorldResults,
    batch_status: wp.array[int],
    actors: wp.array[int],
    requested: wp.array[wp.int64],
    handles: wp.array[int],
    generations: wp.array[wp.uint64],
    inverse: wp.array[int],
    variants: wp.array[wp.int64],
    failed: wp.array[int],
):
    request = wp.tid()
    if batch_status[0] != 0 or results.status[request] != int(WorldStatus.OK):
        wp.atomic_max(failed, 0, 1)
        return
    actor, identity = actors[request], results.id[request]
    handles[actor] = identity
    generations[actor] = results.generation[request]
    inverse[identity] = actor
    variants[actor] = requested[request]


@wp.kernel
def _backing_demand(commands: WorldCommands, directory: WorldDirectoryData, demand: wp.array2d[int]):
    prototype = wp.tid()
    before = directory.active_count[prototype]
    after = before
    for request in range(commands.count[0]):
        if commands.prototype[request] == prototype:
            before += 1
            after += 1
        if commands.op[request] == int(WorldOperation.RESET):
            source, row, valid = world_location(directory, commands.id[request], commands.generation[request])
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
            if reset.replay_only and reset.bank_path is None and reset.buffer_size < len(configs):
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
            configs = tuple(cfg.replace(topology_mode="exact", partition_mode="fixed_dof") for cfg in configs)
            self.layouts = tuple(generate_keyboard(cfg) for cfg in configs)
            if any(layout.active_key_count % 6 or not 6 <= layout.active_key_count <= 108 for layout in self.layouts):
                raise ValueError("Keyboard prototypes require 6..108 keys in multiples of six.")
            self.counts = torch.tensor([layout.active_key_count for layout in self.layouts], device=env.device)
            self.backspaces = torch.tensor(
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
            self.desired_variant_ids = env.all_env_ids.remainder(len(configs)).long()
            self._reset_variants = self.desired_variant_ids.clone()
            self.actor_ids = wp.full(env.num_envs, -1, dtype=wp.int32, device=env.device)
            self.actor_generations = wp.zeros(env.num_envs, dtype=wp.uint64, device=env.device)
            self.actor_for_id = wp.full(env.num_envs, -1, dtype=wp.int32, device=env.device)
            self.commands = create_world_commands(env.num_envs, device=env.device)
            self.results = create_world_results(env.num_envs, device=env.device)
            self._request_actors = wp.empty(env.num_envs, dtype=wp.int32, device=env.device)
            self._request_variants = torch.empty(env.num_envs, dtype=torch.long, device=env.device)
            self._payload_enabled = wp.zeros(1, dtype=wp.int32, device=env.device)
            self._failed = wp.zeros(1, dtype=wp.int32, device=env.device)
            self._demand = wp.empty((len(configs), 2), dtype=wp.int32, device=env.device)
            self.overflow = wp.zeros(len(configs), dtype=wp.int32, device=env.device)
            self._selector_cfgs = {}

            def remember(cfg):
                self._selector_cfgs[selector_key(cfg)] = cfg
                return cfg

            for cfg in self._manager_configs():
                bind_selectors(cfg, remember)
            command = env.cfg.commands.typing
            root_width = command.reset_roots.dense_width or command.reset_roots.count_per_world
            coord_width = command.reset_coords.dense_width or command.reset_coords.count_per_world
            dof_width = command.reset_dofs.dense_width or command.reset_dofs.count_per_world
            self._q_offset, self._qd_offset = 7 * root_width, 7 * root_width + coord_width
            self._payload = torch.empty((env.num_envs, self._qd_offset + dof_width), device=env.device)
            self._payload_wp = wp.from_torch(self._payload)
            self._snapshot_maps, self._prototype_maps, prepared = [], [], []
            for cfg, layout in zip(configs, self.layouts, strict=True):
                solver, source = prepare_keyboard_prototype(
                    env, cfg, physics, self._selector_cfgs.values(), contact_capacity=96 + 8 * layout.active_key_count
                )
                self._solvers.append(solver)
                self._sources.append(source)
                solver.mjw_model.opt.timestep.fill_(env.physics_dt / physics.num_substeps)
                defaults = mjw.replicate_data(solver.mjw_data, 1)
                mjw.forward(solver.mjw_model, defaults)
                warm = mjw.replicate_data(defaults, 1)
                workspace = mjw.make_step_workspace(solver.mjw_model, warm)
                mjw.step(solver.mjw_model, warm, workspace=workspace)
                prepared.append((solver.mjw_model, defaults))
                mapping = NativePrototypeMapping(source, solver)
                self._prototype_maps.append(mapping)
                self._snapshot_maps.append(self._snapshot_columns(mapping, source, command))
            self.reset_kinematics = None
            if not reset.replay_only or reset.bank_path is None:
                self._prepare_reset_staging(command)
            initial_counts = [
                env.num_envs // len(configs) + int(i < env.num_envs % len(configs)) for i in range(len(configs))
            ]
            # Reserve addresses for the largest simultaneous old + replacement batch,
            # while backing only initial populations and bounded reset headroom.
            capacities = (2 * env.num_envs,) * len(configs)
            ready = tuple(min(2 * env.num_envs, n + max(8, n // 2)) for n in initial_counts)
            self.backend = env.sim.get_or_create_backend(
                NewtonWorldsBackendCfg(
                    prototypes=tuple(prepared),
                    capacities=capacities,
                    id_capacity=env.num_envs,
                    command_capacity=env.num_envs,
                    initial_rows=ready,
                    dt=env.physics_dt / physics.num_substeps,
                    substeps=physics.num_substeps,
                    memory_budget_bytes=env.cfg.worlds_memory_budget_bytes,
                )
            )
            runtime = self.backend.runtime
            self._selection_owner = NativeSelections(
                tuple(self._sources),
                tuple(self._prototype_maps),
                runtime,
                self.actor_ids,
                self.actor_generations,
                num_envs=env.num_envs,
                device=env.device,
            )
            contact_cfg = env.cfg.terminations.excessive_contact
            self._contact_selection = None
            if contact_cfg is not None:
                self._contact_selection = self.resolve(contact_cfg.params["bodies"])
                self._contact_selection.prepare_contact_forces()

            def initialize(group, requests, destinations, count, status, transaction, sequence):
                qmap, qdmap, roots, refs = self._snapshot_maps[group.index]
                wp.launch(
                    _initialize_snapshot,
                    env.num_envs,
                    [
                        requests,
                        destinations,
                        count,
                        status,
                        transaction,
                        sequence,
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

            def validate(commands, transaction):
                wp.capture_if(
                    transaction.phase,
                    on_true=lambda: wp.launch(
                        _validate_snapshot,
                        self._payload.shape,
                        [commands, transaction, self._payload_enabled, self._payload_wp],
                        device=env.device,
                    ),
                )

            def after_substep(group):
                group.record_launch(
                    _record_overflow,
                    group.world_capacity,
                    inputs=[group.data.overflow, group.index, self.overflow],
                    domain="world",
                )
                if self._contact_selection is not None:
                    self._contact_selection.record_contact_forces(group, self.actor_for_id)

            wp.load_module(module=__name__, device=env.device)
            wp.load_module(module="isaaclab_tasks.contrib.keyboard.native_selection", device=env.device)
            self.backend.prepare(
                self.commands,
                self.results,
                validate=validate,
                initialize=initialize,
                after_substep=after_substep,
                retain=(self._payload_wp, self._payload_enabled, self._snapshot_maps, self.overflow),
            )
            env.sim.physics_manager.install(self.backend)
            self._submit(env.all_env_ids, self.desired_variant_ids, create=True)
            self.redistribution_count = 0
            self.last_changed_worlds = 0
            self.last_redistribution_ms = 0.0
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

    def _snapshot_columns(self, mapping, source, command):
        qids = query_selection_ids(source.model, command.reset_coords)
        qdids = query_selection_ids(source.model, command.reset_dofs)
        roots = query_selection_ids(source.model, command.reset_roots)
        qcolumns, qdcolumns = {int(v): i for i, v in enumerate(qids)}, {int(v): i for i, v in enumerate(qdids)}
        if set(mapping.coordinate_ids.tolist()) != set(qcolumns) or set(mapping.dof_ids.tolist()) != set(qdcolumns):
            raise ValueError("Reset snapshots must cover every native scalar coordinate and velocity.")
        root_columns = {int(v): i for i, v in enumerate(roots)}
        qmap = np.array([qcolumns[int(v)] for v in mapping.coordinate_ids], dtype=np.int32)
        qdmap = np.array([qdcolumns[int(v)] for v in mapping.dof_ids], dtype=np.int32)
        rootmap = np.array([root_columns[int(v)] for v in mapping.root_body_ids], dtype=np.int32)
        return tuple(
            wp.array(values, dtype=dtype, device=self.env.device)
            for values, dtype in (
                (qmap, wp.int32),
                (qdmap, wp.int32),
                (rootmap, wp.int32),
                (mapping.coordinate_refs, wp.float32),
            )
        )

    def _prepare_reset_staging(self, command):
        """Compile immutable snapshot/key tables and one compact robot reset workspace."""
        root_width = self._q_offset // 7
        defaults, key_roots, key_local, root_active, keyboard_columns = [], [], [], [], []
        self.reset_kinematics = None
        for prototype, (solver, source) in enumerate(zip(self._solvers, self._sources, strict=True)):
            model = solver.model
            roots, coords, dofs = (
                resolve_selection(source, cfg)
                for cfg in (command.reset_roots, command.reset_coords, command.reset_dofs)
            )
            poses = roots.read_model("body_q")[0]
            q, qd = coords.read_model("joint_q")[0], dofs.read_model("joint_qd")[0]
            snapshot_width = self._qd_offset + (command.reset_dofs.dense_width or command.reset_dofs.count_per_world)
            snapshot = torch.zeros(snapshot_width, device=self.env.device)
            snapshot[: poses.numel()] = poses.flatten()
            snapshot[self._q_offset : self._q_offset + len(q)] = q
            snapshot[self._qd_offset : self._qd_offset + len(qd)] = qd
            defaults.append(snapshot)
            active = torch.zeros(root_width, dtype=torch.bool, device=self.env.device)
            active[: len(poses)] = roots.dense_active()[0]
            root_active.append(active)
            root_ids = resolve_selection(source, command.reset_roots.replace(dense_width=None)).ids.numpy()
            root_index = {int(body): column for column, body in enumerate(root_ids)}
            parent, child = model.joint_parent.numpy(), model.joint_child.numpy()
            body_parent = {int(body): int(parent[joint]) for joint, body in enumerate(child)}
            key_ids = resolve_selection(source, command.key_bodies.replace(dense_width=None)).ids.numpy()
            local = np.zeros(
                (command.key_bodies.dense_width or command.key_bodies.count_per_world, 3), dtype=np.float32
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
                source, command.reset.pre_solve_reset.params["roots"].replace(dense_width=None)
            ).ids.numpy()
            root_mask = torch.zeros(root_width, dtype=torch.bool, device=self.env.device)
            root_mask[[root_index[int(body)] for body in selected_roots]] = True
            keyboard_columns.append(root_mask)
            robot = resolve_selection(source, command.robot_joints).ids.numpy()
            ik_dofs = resolve_selection(source, command.reset.ik.dofs).ids.numpy()
            ik_coords = resolve_selection(source, command.reset.ik.joints).ids.numpy()
            tip = int(resolve_selection(source, command.reset.ik.body).ids.numpy()[0])
            root = tip
            while root not in root_index:
                root = body_parent[root]
            qids = resolve_selection(source, command.reset_coords.replace(dense_width=None)).ids.numpy()
            qcolumn = {int(coord): column for column, coord in enumerate(qids)}
            robot_columns = np.array([qcolumn[int(coord)] for coord in robot], dtype=np.int64)
            robot_column = {int(coord): column for column, coord in enumerate(robot)}
            ik_columns = np.array([robot_column[int(coord)] for coord in ik_coords], dtype=np.int64)
            if prototype == 0:
                self.reset_robot_columns = torch.tensor(robot_columns, device=self.env.device)
                self.reset_ik_columns = torch.tensor(ik_columns, device=self.env.device)
                self.reset_robot_root = root_index[root]
                self.reset_kinematics = ResetKinematics(
                    model, robot, ik_dofs, tip, self.env.num_envs, command.reset.ik.tip_offset
                )
            else:
                if (
                    root_index[root] != self.reset_robot_root
                    or not np.array_equal(robot_columns, self.reset_robot_columns.cpu().numpy())
                    or not np.array_equal(ik_columns, self.reset_ik_columns.cpu().numpy())
                ):
                    raise ValueError("Keyboard snapshots must share the robot coordinate and root columns.")
                self.reset_kinematics.validate_model(model, robot, ik_dofs, tip)
        self.reset_defaults = torch.stack(defaults)
        self.reset_root_active = torch.stack(root_active)
        self.reset_keyboard_roots = torch.stack(keyboard_columns)
        self.reset_key_root = torch.tensor(np.stack(key_roots), device=self.env.device)
        self.reset_key_local = torch.tensor(np.stack(key_local), device=self.env.device)

    def resolve(self, cfg):
        """Compile task paths once before returning a numeric runtime binding."""
        ids = tuple(query_selection_ids(source.model, cfg) for source in self._sources)
        return self._selection_owner.bind(cfg.frequency, ids, policy_width=cfg.dense_width)

    def reset_variants(self, env_ids):
        """Return requested prototypes staged for this episode boundary."""
        return self._reset_variants[env_ids]

    def request(self, env_ids, variant_ids=None):
        """Update desired keyboard assignments without changing a live episode."""
        self._validate_request(env_ids, variant_ids)
        if variant_ids is None:
            offsets = torch.randint(1, max(2, len(self.layouts)), env_ids.shape, device=self.env.device)
            variant_ids = (self.variant_ids[env_ids] + offsets) % len(self.layouts)
        self.desired_variant_ids[env_ids] = variant_ids

    def redistribute(self, eligible_ids):
        """Stage eligible switches; publication waits for the complete snapshot."""
        self._validate_request(eligible_ids)
        changed = eligible_ids[self.desired_variant_ids[eligible_ids] != self.variant_ids[eligible_ids]]
        self._reset_variants[eligible_ids] = self.desired_variant_ids[eligible_ids]
        return changed

    def reset_snapshot(self, env_ids, variant_ids, snapshot):
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
        self.last_changed_worlds = changed
        self.redistribution_count += 1
        self.last_redistribution_ms = (time.perf_counter() - started) * 1000.0

    def _validate_request(self, env_ids, variant_ids=None):
        self.env._check_active()
        if (
            not isinstance(env_ids, torch.Tensor)
            or env_ids.ndim != 1
            or env_ids.dtype != torch.long
            or env_ids.device != self.variant_ids.device
        ):
            raise ValueError("Actor IDs must be a one-dimensional int64 tensor on the task device.")
        torch._assert_async(((env_ids >= 0) & (env_ids < self.env.num_envs)).all(), "Actor IDs are outside the task.")
        seen = torch.zeros_like(self.variant_ids, dtype=torch.int32)
        seen.scatter_add_(0, env_ids, torch.ones_like(env_ids, dtype=torch.int32))
        torch._assert_async((seen <= 1).all(), "Actor IDs must be unique.")
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
        wp.to_torch(self._request_actors)[:count].copy_(env_ids)
        self._request_variants[:count].copy_(variant_ids)
        try:
            wp.launch(_begin_batch, 1, [self.commands, count], device=self.env.device)
            wp.launch(
                _reset_commands,
                count,
                [
                    self.commands,
                    self._request_actors,
                    wp.from_torch(self._request_variants),
                    self.actor_ids,
                    self.actor_generations,
                    int(create),
                ],
                device=self.env.device,
            )
            runtime = self.backend.runtime
            wp.launch(
                _backing_demand,
                len(self.layouts),
                [self.commands, runtime.directory.data, self._demand],
                device=self.env.device,
            )
            demand = self._demand.numpy()
            streams = (wp.get_stream(self.env.device),)
            targets = []
            for group, (required, _) in zip(runtime.prototypes, demand, strict=True):
                required = int(required)
                ready = group.ready_worlds
                if required > ready or ready > 2 * required + 16 or required == 0:
                    ready = min(group.world_capacity, required + max(8, required // 2)) if required else 0
                targets.append(ready)
            if any(n != group.ready_worlds for group, n in zip(runtime.prototypes, targets, strict=True)):
                runtime.resize_backing(
                    tuple(targets), streams=streams, spare_bytes=self.env.cfg.worlds_spare_memory_budget_bytes
                )
            self.backend.forward()
            self._failed.zero_()
            wp.launch(
                _publish_handles,
                count,
                [
                    self.results,
                    runtime.directory.batch.status,
                    self._request_actors,
                    wp.from_torch(self._request_variants),
                    self.actor_ids,
                    self.actor_generations,
                    self.actor_for_id,
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
            targets = []
            for group, (_, live) in zip(runtime.prototypes, demand, strict=True):
                live = int(live)
                ready = group.ready_worlds
                if not live or ready > 2 * live + 16:
                    ready = min(group.world_capacity, live + max(8, live // 2)) if live else 0
                targets.append(ready)
            if any(n != group.ready_worlds for group, n in zip(runtime.prototypes, targets, strict=True)):
                runtime.resize_backing(
                    tuple(targets), streams=streams, spare_bytes=self.env.cfg.worlds_spare_memory_budget_bytes
                )
        except BaseException:
            self.env._population_bindings_valid = False
            raise

    @property
    def active_prototypes(self):
        return (wp.to_torch(self.backend.runtime.directory.data.active_count) > 0).sum()

    @property
    def native_dofs(self):
        counts = wp.to_torch(self.backend.runtime.directory.data.active_count)
        return (counts * (self.counts + 6)).sum()

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
