# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard prototypes, stable actor assignments, and exact native population bindings."""

from __future__ import annotations

import inspect
import time

import newton
import numpy as np
import torch
import warp as wp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.physics.population import NewtonPopulationBackendCfg
from newton.sensors import SensorContact
from newton.solvers import SolverMuJoCo
from newton.usd import SchemaResolverMjc, SchemaResolverNewton, SchemaResolverPhysx

from pxr import Usd, UsdGeom

from isaaclab.sim import use_stage
from isaaclab.utils import replace, to_dict

from .keyboards.keyboard_geometry import generate_keyboard
from .newton_selection import NewtonSelectionGroup, NewtonSelections
from .selection_paths import bind_selectors, resolve_selection, selector_key


def prepare_keyboard_prototype(
    env, keyboard_cfg, physics, selector_cfgs, *, contact_capacity=None, robot_positions=None
):
    """Author and finalize one immutable source; replication never repeats this work."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    scene = env.cfg.scene
    with use_stage(stage):
        positions = (scene.robot.init_state.pos,) if robot_positions is None else robot_positions
        for index, position in enumerate(positions):
            scene.robot.spawn.func(
                "/World/Robot" if index == 0 else f"/World/Robot_{index}",
                scene.robot.spawn,
                translation=position,
                orientation=scene.robot.init_state.rot,
                stage=stage,
            )
        for name, asset, spawn in (
            ("Keyboard", scene.keyboard, keyboard_cfg),
            ("GroundPlane", scene.plane, scene.plane.spawn),
        ):
            spawn.func(
                f"/World/{name}",
                spawn,
                translation=asset.init_state.pos,
                orientation=asset.init_state.rot,
                stage=stage,
            )
    builder = env.sim.physics_manager.create_builder(up_axis="Z")
    builder.begin_world()
    builder.add_usd(
        stage,
        root_path="/World",
        load_visual_shapes=bool(physics.load_visual_shapes),
        hide_collision_shapes=True,
        schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx(), SchemaResolverMjc()],
    )
    builder.end_world()
    model = builder.finalize(env.device)
    source = NewtonSelections(model)
    try:
        for cfg in selector_cfgs:
            resolve_selection(source, replace(cfg, policy_width=None))
        robot_q = resolve_selection(source, replace(env.cfg.actions.action.joints, policy_width=None))
        robot_qd = resolve_selection(source, replace(env.cfg.actions.action.dofs, policy_width=None))
        wp.to_torch(model.joint_q)[robot_q.dense_ids()] = 0.0
        wp.to_torch(model.joint_qd).zero_()
        wp.to_torch(model.joint_target_mode)[robot_qd.dense_ids()] = int(newton.JointTargetMode.POSITION_VELOCITY)
        roots = resolve_selection(source, replace(env.cfg.commands.typing.reset_roots, policy_width=None))
        root_joints = source.root_joint_ids[roots.dense_ids()]
        if torch.any(root_joints < 0) or torch.any(
            wp.to_torch(model.joint_type)[root_joints] != int(newton.JointType.FIXED)
        ):
            raise ValueError("Keyboard population reset snapshots require fixed articulation roots.")
        parameters = inspect.signature(SolverMuJoCo.__init__).parameters
        kwargs = {key: value for key, value in to_dict(physics.solver_cfg).items() if key in parameters}
        if contact_capacity is not None:
            kwargs["nconmax"] = contact_capacity
        solver = SolverMuJoCo(model, **kwargs)
        return solver, source
    except BaseException as error:
        try:
            try:
                wp.synchronize_device(model.device)
            finally:
                source.retire()
        except BaseException as cleanup_error:
            raise error from cleanup_error
        raise


class KeyboardPopulations:
    """Own keyboard assignments and task bindings; the simulation owns native resources.

    The desired assignment changes at reset. Native world counts change only at
    explicit redistribution boundaries. Logical actor IDs survive native row
    remapping, and unchanged counts retain their model, solver and CUDA graphs.
    """

    def __init__(self, env):
        self.env = env
        physics = env.cfg.sim.physics.prototype_physics
        if not isinstance(physics, NewtonCfg) or not isinstance(physics.solver_cfg, MJWarpSolverCfg):
            raise ValueError("Keyboard populations require Newton MJWarp prototype physics.")
        if not physics.solver_cfg.use_mujoco_contacts or physics.collision_cfg is not None:
            raise ValueError("Keyboard populations require native MJWarp contacts and no collision_cfg.")
        if env.cfg.sim.physics.deterministic or physics.deterministic or physics.deterministic_mode != "not_guaranteed":
            raise ValueError("Deterministic population execution is not yet supported.")
        configs = env.cfg.keyboard_variants or (env.cfg.scene.keyboard.spawn,)
        configs = tuple(replace(cfg, topology_mode="exact", partition_mode="fixed_dof") for cfg in configs)
        self.layouts = tuple(generate_keyboard(cfg) for cfg in configs)
        if any(layout.active_key_count % 6 or layout.active_key_count > 108 for layout in self.layouts):
            raise ValueError("Keyboard populations require 6..108 keys in multiples of six.")
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
        self.variant_ids = env.all_env_ids.remainder(len(configs)).long()
        self.desired_variant_ids = self.variant_ids.clone()
        self._actors = tuple(
            np.arange(index, env.num_envs, len(configs), dtype=np.int64) for index in range(len(configs))
        )
        self.worlds = tuple(torch.as_tensor(actors, device=env.device) for actors in self._actors)
        self._selector_cfgs = {}
        self._bindings = {}

        def prepare(cfg):
            self._selector_cfgs[selector_key(cfg)] = cfg
            return cfg

        for cfg in self._manager_configs():
            bind_selectors(cfg, prepare)
        self._sources = []
        self._contact_sources = []
        self._owners = [None] * len(configs)
        self._reset_masks = [None] * len(configs)
        try:
            prototypes = []
            for cfg in configs:
                solver, selections = prepare_keyboard_prototype(env, cfg, physics, self._selector_cfgs.values())
                prototypes.append(solver)
                self._sources.append(selections)
                contact_term = env.cfg.terminations.excessive_contact
                sensor = None
                if contact_term is not None:
                    bodies = resolve_selection(selections, replace(contact_term.params["bodies"], policy_width=None))
                    sensor = SensorContact(
                        solver.model, sensing_bodies=bodies.ids.numpy().tolist(), request_contact_attributes=False
                    )
                self._contact_sources.append(sensor)
            self.backend = env.sim.get_or_create_backend(
                NewtonPopulationBackendCfg(
                    prototypes=tuple(prototypes),
                    counts=tuple(map(len, self._actors)),
                    dt=env.physics_dt / physics.num_substeps,
                    substeps=physics.num_substeps,
                    use_cuda_graph=physics.use_cuda_graph,
                    stream_count=env.cfg.population_stream_count,
                )
            )
            env.sim.physics_manager.install(self.backend)
            self._bind_native()
            self.redistribution_count = 0
            self.last_changed_worlds = 0
            self.last_redistribution_ms = 0.0
        except BaseException as error:
            try:
                try:
                    wp.synchronize_device(env.device)
                finally:
                    self.close()
            except BaseException as cleanup_error:
                raise error from cleanup_error
            raise

    def _manager_configs(self):
        cfg = self.env.cfg
        return cfg.commands, cfg.actions, cfg.observations, cfg.rewards, cfg.terminations, cfg.events

    def committed_variant_ids(self, env_ids):
        """Return the committed prototypes used by an upcoming snapshot reset."""
        return self.variant_ids[env_ids]

    def reset_from_snapshot(self, env_ids, variant_ids, snapshot):
        """Restore a complete task snapshot into the already assigned exact populations."""
        from .mdp.reset import restore_reset_state

        torch._assert_async((variant_ids == self.variant_ids[env_ids]).all(), "Snapshot prototype is not committed.")
        command = self.env.command_manager.get_term("typing").cfg
        restore_reset_state(self.env, snapshot, env_ids, command.reset_roots, command.reset_coords, command.reset_dofs)

    @property
    def populated_prototype_count(self):
        return sum(count > 0 for count in self.backend.counts)

    @property
    def live_dof_count(self):
        return sum(p.model.joint_dof_count for p in self.backend.populations if p is not None)

    def _bind_native(self):
        models = {population.model for population in self.backend.populations if population is not None}
        for model in tuple(self.env.native_contacts):
            if model not in models:
                del self.env.native_contacts[model]
        for index, population in enumerate(self.backend.populations):
            owner = self._owners[index]
            if owner is not None and (population is None or owner.model is not population.model):
                # Backend replacement has fenced old consumers. Retire before any
                # allocation can fail or overwrite the last task-owned reference.
                owner.retire()
            if population is None:
                self._owners[index] = self._reset_masks[index] = None
            elif owner is None or owner.model is not population.model:
                self._owners[index] = NewtonSelections(
                    population.model,
                    state=population.state_0,
                    control=population.control,
                    solver=population.solver,
                    source=self._sources[index],
                )
                self._reset_masks[index] = wp.zeros(
                    population.model.world_count + 1, dtype=wp.bool, device=population.model.device
                )
                if self._contact_sources[index] is not None:
                    sensor = self._contact_sources[index].replicate(population.model)
                    contacts = newton.Contacts(
                        population.solver.mjw_data.naconmax,
                        0,
                        device=population.model.device,
                        requested_attributes={"force"},
                    )
                    self.env.native_contacts[population.model] = sensor, contacts

    def close(self):
        """Release task binding caches after the simulation fences native consumers."""
        for owner in (*self._owners, *self._sources):
            if owner is not None:
                owner.retire()
        self._bindings.clear()
        self._owners.clear()
        self._sources.clear()

    def _parts(self, cfg):
        native_cfg = replace(cfg, policy_width=None)
        return tuple(
            (resolve_selection(owner, native_cfg), worlds)
            for owner, worlds in zip(self._owners, self.worlds)
            if owner is not None
        )

    def resolve(self, cfg):
        key = selector_key(cfg)
        if key not in self._bindings:
            self._bindings[key] = NewtonSelectionGroup(
                cfg.index_domain, self._parts(cfg), self.env.num_envs, policy_width=cfg.policy_width
            )
        return self._bindings[key]

    def _validate_actor_request(self, env_ids, variant_ids=None):
        self.env._check_active()
        if not isinstance(env_ids, torch.Tensor) or env_ids.ndim != 1 or env_ids.dtype != torch.long:
            raise ValueError("Actor IDs must be a one-dimensional int64 tensor.")
        if env_ids.device != self.variant_ids.device:
            raise ValueError("Actor IDs must use the assignment tensor's device.")
        torch._assert_async(
            ((env_ids >= 0) & (env_ids < self.env.num_envs)).all(), "Actor IDs are outside the logical world range."
        )
        seen = torch.zeros_like(self.variant_ids, dtype=torch.int32)
        seen.scatter_add_(0, env_ids, torch.ones_like(env_ids, dtype=torch.int32))
        torch._assert_async((seen <= 1).all(), "Actor IDs must be unique.")
        if variant_ids is not None:
            if (
                not isinstance(variant_ids, torch.Tensor)
                or variant_ids.dtype != torch.long
                or variant_ids.shape != env_ids.shape
                or variant_ids.device != self.variant_ids.device
            ):
                raise ValueError("Variant IDs must be an equally sized int64 tensor on the assignment device.")
            torch._assert_async(
                ((variant_ids >= 0) & (variant_ids < len(self.layouts))).all(),
                "Variant IDs are outside the registered keyboard range.",
            )

    def request_variants(self, env_ids, variant_ids=None):
        """Queue reset requests on the GPU without resizing native models."""
        self._validate_actor_request(env_ids, variant_ids)
        if variant_ids is None:
            offsets = torch.randint(1, max(len(self.layouts), 2), env_ids.shape, device=self.env.device)
            variant_ids = (self.variant_ids[env_ids] + offsets) % len(self.layouts)
        self.desired_variant_ids[env_ids] = variant_ids

    def apply_pending_variants(self, eligible_ids):
        """Apply pending requests for explicitly eligible episode boundaries."""
        self._validate_actor_request(eligible_ids)
        assignments = self.variant_ids.clone()
        assignments[eligible_ids] = self.desired_variant_ids[eligible_ids]
        return self._replace(assignments)

    def apply(self, env_ids, variant_ids):
        """Apply an explicit assignment during initialization or a requested full reset."""
        if variant_ids is None:
            raise ValueError("Applying an assignment requires explicit variant IDs.")
        self._validate_actor_request(env_ids, variant_ids)
        assignments = self.variant_ids.clone()
        assignments[env_ids] = variant_ids
        changed = self._replace(assignments)
        self.desired_variant_ids[env_ids] = variant_ids
        return changed

    def _replace(self, assignments):
        self.env._check_active()
        if (
            not isinstance(assignments, torch.Tensor)
            or assignments.dtype != torch.long
            or assignments.shape != self.variant_ids.shape
            or assignments.device != self.variant_ids.device
        ):
            raise ValueError("A population plan requires one int64 variant per actor on the assignment device.")
        # Only the small actor plan crosses to the host, once per redistribution.
        # Model/state/control/history/contact payloads remain on the GPU.
        requested = assignments.cpu().numpy()
        if np.any(requested < 0) or np.any(requested >= len(self.layouts)):
            raise ValueError("The population plan contains an unregistered keyboard variant.")
        old_assignment = np.empty(self.env.num_envs, dtype=np.int64)
        for index, actors in enumerate(self._actors):
            old_assignment[actors] = index
        changed = np.flatnonzero(requested != old_assignment)
        if not len(changed):
            return torch.empty(0, dtype=torch.long, device=self.env.device)
        started = time.perf_counter()
        actors, survivors = [], []
        for index, previous in enumerate(self._actors):
            wanted = np.flatnonzero(requested == index)
            # Keep surviving actors in their old rows whenever those rows still
            # exist. Unchanged counts therefore require no state permutation.
            target = np.full(len(wanted), -1, dtype=np.int64)
            source_rows = np.flatnonzero(requested[previous] == index)
            keep = source_rows[source_rows < len(target)]
            target[keep] = previous[keep]
            remaining = wanted[~np.isin(wanted, target[keep])]
            target[target < 0] = remaining
            new_rows = {int(actor): row for row, actor in enumerate(target)}
            survivors.append((source_rows.tolist(), [new_rows[int(previous[row])] for row in source_rows]))
            actors.append(target)
        self.backend.replace(tuple(map(len, actors)), survivors=tuple(survivors))
        try:
            self._actors = tuple(actors)
            self.worlds = tuple(torch.as_tensor(ids, device=self.env.device) for ids in actors)
            self.variant_ids.copy_(assignments)
            self._bind_native()
            for key, selection in self._bindings.items():
                selection.rebind(self._parts(self._selector_cfgs[key]))
            self.redistribution_count += 1
            self.last_changed_worlds = len(changed)
            self.last_redistribution_ms = (time.perf_counter() - started) * 1e3
            return torch.as_tensor(changed, dtype=torch.long, device=self.env.device)
        except BaseException:
            # Native publication has completed. Keep the original error and leave
            # retirement to explicit close; partially rebound task state cannot run.
            self.env._population_bindings_valid = False
            raise

    def reconcile_state(self, dirty_worlds, flags):
        """Apply task-authored properties and state before the next native read."""
        self.env._check_active()
        for population, worlds, mask in zip(self.backend.populations, self.worlds, self._reset_masks):
            if population is not None:
                wp.to_torch(mask)[:-1] = dirty_worlds[worlds]
        self.backend.reconcile_state(self._reset_masks, flags)
