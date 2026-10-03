# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One global typing MDP composed with multiple exact Newton populations."""

from __future__ import annotations

import copy
import math
import operator

import gymnasium as gym
import numpy as np
import torch
import warp as wp
from isaaclab_newton.physics.population import NewtonPopulationCfg
from isaaclab_newton.physics.worlds import NewtonWorldsCfg

from isaaclab.managers import (
    ActionManager,
    CommandManager,
    CurriculumManager,
    EventManager,
    ObservationManager,
    RewardManager,
    TerminationManager,
)
from isaaclab.sim import SimulationContext
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.warp.utils import warp_on_torch_stream

from .keyboard_populations import KeyboardPopulations
from .keyboard_worlds import KeyboardWorlds
from .selection_paths import bind_selectors


class SO101KeyboardPopulationEnv(gym.Env):
    """Headless SAME_STEP vector environment with stable logical policy rows.

    Managers own global task state. The simulation owns native populations. This
    root controls the episode boundary between them, including explicit state
    reconciliation and delayed desired keyboard assignments.
    """

    metadata = {"render_modes": [], "autoreset_mode": gym.vector.AutoresetMode.SAME_STEP}

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__()
        self.cfg = cfg.copy()
        self.cfg.validate()
        if not isinstance(self.cfg.sim.physics, (NewtonPopulationCfg, NewtonWorldsCfg)):
            raise ValueError("The population task requires NewtonPopulationCfg or NewtonWorldsCfg.")
        if render_mode is not None or self.cfg.video_recorders:
            raise ValueError("Keyboard populations currently support headless training without video recording.")
        try:
            interval = operator.index(self.cfg.redistribution_interval)
        except TypeError as error:
            raise ValueError("redistribution_interval must be a positive integer.") from error
        if isinstance(self.cfg.redistribution_interval, bool) or interval < 1:
            raise ValueError("redistribution_interval must be a positive integer.")
        if self.cfg.redistribution_mode not in ("episode_boundary", "truncate_pending"):
            raise ValueError("Specify a supported redistribution mode.")
        self.cfg.redistribution_interval = interval
        if self.cfg.redistribution_mode == "truncate_pending" and not self.cfg.compute_final_obs:
            raise ValueError("Administrative redistribution requires compute_final_obs for timeout bootstrapping.")
        if self.cfg.redistribution_mode == "truncate_pending" and self.cfg.is_finite_horizon:
            raise ValueError("Administrative redistribution currently requires is_finite_horizon=False.")
        self.render_mode = render_mode
        self.num_envs = self.cfg.scene.num_envs
        self.device = self.cfg.sim.device
        if torch.device(self.device).type != "cuda":
            raise ValueError("The keyboard population task requires a CUDA device for MJWarp.")
        self.physics_dt = self.cfg.sim.dt
        self.step_dt = self.physics_dt * self.cfg.decimation
        self.max_episode_length_s = self.cfg.episode_length_s
        self.max_episode_length = math.ceil(self.max_episode_length_s / self.step_dt)
        self.cfg.seed = self.seed(self.cfg.seed if self.cfg.seed is not None else -1)
        self.all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        # Physical native worlds overlap geometrically and are isolated by world
        # IDs. No visualization grid offset enters reset snapshots or observations.
        self.env_origins = torch.zeros((self.num_envs, 3), device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_interrupted = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.common_step_counter = self._sim_step_counter = 0
        self.extras, self.obs_buf = {}, {}
        self._dirty_worlds = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._dirty = False
        self._dirty_flags = 0
        self._has_reset = False
        self._is_closed = False
        self._population_bindings_valid = True
        self.native_contacts = {}
        self.sim = SimulationContext(self.cfg.sim)
        try:
            with warp_on_torch_stream(self.device):
                if isinstance(self.cfg.sim.physics, NewtonWorldsCfg):
                    self.keyboard_variants = KeyboardWorlds(self)
                else:
                    self.keyboard_variants = KeyboardPopulations(self)
                self.sim.reset()
                self.event_manager = EventManager(
                    bind_selectors(copy.deepcopy(self.cfg.events), self.keyboard_variants.resolve), self
                )
                self.command_manager = CommandManager(
                    bind_selectors(copy.deepcopy(self.cfg.commands), self.keyboard_variants.resolve), self
                )
                self.action_manager = ActionManager(
                    bind_selectors(copy.deepcopy(self.cfg.actions), self.keyboard_variants.resolve), self
                )
                self.observation_manager = ObservationManager(
                    bind_selectors(copy.deepcopy(self.cfg.observations), self.keyboard_variants.resolve), self
                )
                self.termination_manager = TerminationManager(
                    bind_selectors(copy.deepcopy(self.cfg.terminations), self.keyboard_variants.resolve), self
                )
                self.reward_manager = RewardManager(
                    bind_selectors(copy.deepcopy(self.cfg.rewards), self.keyboard_variants.resolve), self
                )
                self.curriculum_manager = CurriculumManager(
                    bind_selectors(copy.deepcopy(self.cfg.curriculum), self.keyboard_variants.resolve), self
                )
                if "startup" in self.event_manager.available_modes:
                    self.event_manager.apply(mode="startup")
                self.single_observation_space = gym.spaces.Dict(
                    {
                        name: gym.spaces.Box(-np.inf, np.inf, shape=tuple(dim), dtype=np.float32)
                        for name, dim in self.observation_manager.group_obs_dim.items()
                    }
                )
                self.single_action_space = gym.spaces.Box(
                    -np.inf, np.inf, shape=(self.action_manager.total_action_dim,), dtype=np.float32
                )
                self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
                self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                raise error from cleanup_error
            raise

    @staticmethod
    def seed(seed=-1):
        """Seed the same task-level random generators as the reference environment."""
        return configure_seed(seed)

    def _check_active(self):
        if self._is_closed:
            raise RuntimeError("The population environment is closed.")
        if not self._population_bindings_valid:
            raise RuntimeError("Population bindings publication failed; close this environment and create another.")

    def invalidate_fk(self, env_ids):
        """Record task-authored state edits for reconciliation before native reads."""
        self._dirty_worlds[env_ids] = True
        self._dirty = True

    def notify_model_changed(self, flags, env_ids, *, root_poses_only=False):
        """Accumulate physical property edits at the next forward boundary."""
        self._dirty_flags |= int(flags)
        self.invalidate_fk(env_ids)

    def forward(self):
        """Reconcile reset coordinates/properties and native solver state exactly once."""
        self._check_active()
        with warp_on_torch_stream(self.device):
            if self._dirty:
                self.keyboard_variants.reconcile_state(self._dirty_worlds, self._dirty_flags)
                self._dirty_worlds.zero_()
                self._dirty_flags = 0
                self._dirty = False

    def curriculum_worlds(self, variant):
        """Use an existing homogeneous cohort as scratch worlds during initial IK sampling."""
        if self._has_reset:
            raise RuntimeError("Curriculum snapshots must be prepared before episodes start.")
        if isinstance(self.cfg.sim.physics, NewtonWorldsCfg):
            self.keyboard_variants.request_variants(self.all_env_ids, torch.full_like(self.all_env_ids, variant))
            self._apply_variant_requests(self.all_env_ids)
            return self.all_env_ids
        worlds = self.keyboard_variants.worlds[variant]
        if not len(worlds):
            # Tiny test/play batches can have fewer environments than prototypes. This
            # one-time scratch assignment is restored before any rollout starts.
            self.keyboard_variants.apply(self.all_env_ids, torch.full_like(self.all_env_ids, variant))
            worlds = self.keyboard_variants.worlds[variant]
        return worlds

    def finish_curriculum(self, original_variants):
        """Restore initial assignments after temporary scratch cohorts, if any."""
        if isinstance(self.cfg.sim.physics, NewtonWorldsCfg):
            self.keyboard_variants.request_variants(self.all_env_ids, original_variants)
            self._apply_variant_requests(self.all_env_ids)
        else:
            self.keyboard_variants.apply(self.all_env_ids, original_variants)

    def _reset_idx(self, env_ids):
        try:
            self.curriculum_manager.compute(env_ids=env_ids)
            replay_only = self.cfg.commands.typing.reset.replay_only
            native_worlds = isinstance(self.cfg.sim.physics, NewtonWorldsCfg)
            if self.cfg.keyboard_variants and not replay_only and not native_worlds:
                command = self.command_manager.get_term("typing").cfg
                zeros = torch.zeros((len(env_ids), command.keys.width), device=self.device)
                command.keys.write_state("joint_q", zeros, env_ids)
                command.key_dofs.write_state("joint_qd", zeros, env_ids)
                targets = command.keys if command.keys.use_coord_layout_targets else command.key_dofs
                targets.write_control("joint_target_q", zeros, env_ids)
                command.key_dofs.write_control("joint_target_qd", zeros, env_ids)
                command.key_dofs.write_control("joint_f", zeros, env_ids)
                self.invalidate_fk(env_ids)
            if "reset" in self.event_manager.available_modes and not replay_only:
                if native_worlds:
                    from .mdp.reset import sample_root_poses

                    # The admitted keyboard event is overwritten by snapshot/IK reset.
                    # Preserve its random draws without mutating the ending lifetime.
                    bank = self.keyboard_variants
                    roots = self.command_manager.get_term("typing").cfg.reset_roots.width
                    defaults = bank.reset_defaults[bank.staged_variant_ids(env_ids), : roots * 7]
                    for event in vars(self.cfg.events).values():
                        if getattr(event, "mode", None) == "reset":
                            sample_root_poses(
                                defaults.reshape(-1, roots, 7),
                                event.params["pose_range"],
                                event.params["velocity_range"],
                            )
                else:
                    self.event_manager.apply(
                        mode="reset", env_ids=env_ids, global_env_step_count=self.common_step_counter
                    )
            self.extras["log"] = {}
            for manager in (
                self.observation_manager,
                self.action_manager,
                self.reward_manager,
                self.curriculum_manager,
                self.command_manager,
                self.event_manager,
                self.termination_manager,
            ):
                self.extras["log"].update(manager.reset(env_ids))
            self.episode_length_buf[env_ids] = 0
            self.forward()
        except BaseException:
            self._population_bindings_valid = False
            raise

    def _apply_variant_requests(self, env_ids):
        bank = self.keyboard_variants
        if isinstance(bank, KeyboardWorlds):
            return bank.stage_variant_changes(env_ids)
        return bank.apply_pending_variants(env_ids)

    def reset_variant_ids(self, env_ids):
        """Query committed or staged prototypes for the current reset transaction."""
        bank = self.keyboard_variants
        return (
            bank.staged_variant_ids(env_ids)
            if isinstance(bank, KeyboardWorlds)
            else bank.committed_variant_ids(env_ids)
        )

    def reset_keyboard(self, env_ids, variant_ids):
        """Reset selected episodes immediately to explicitly requested keyboard prototypes."""
        self._check_active()
        with warp_on_torch_stream(self.device):
            self.keyboard_variants.request_variants(env_ids, variant_ids)
            self._apply_variant_requests(env_ids)
            self.episode_interrupted[env_ids] = False
            self._reset_idx(env_ids)

    def restore_reset_snapshot(self, env_ids, variant_ids, snapshot):
        """Publish complete physical reset state before the command's typing state."""
        self.keyboard_variants.reset_from_snapshot(env_ids, variant_ids, snapshot)

    def reset(self, env_ids=slice(None), seed=None, options=None):
        """Reset requested logical episodes; the first reset builds the shared curriculum."""
        self._check_active()
        if seed is not None:
            self.seed(seed)
        ids = self.all_env_ids[slice(None) if env_ids is None else env_ids]
        with warp_on_torch_stream(self.device):
            self.episode_interrupted[ids] = False
            if self._has_reset:
                self.keyboard_variants.request_variants(ids)
                self._apply_variant_requests(ids)
            self._reset_idx(ids)
            self.obs_buf = self.observation_manager.compute(update_history=True)
            self._has_reset = True
        return self.obs_buf, self.extras

    def step(self, action):
        """Advance eight native substeps with relative PD refreshed every physics frame."""
        self._check_active()
        with warp_on_torch_stream(self.device):
            self.episode_interrupted.zero_()
            self.extras.pop("final_obs", None)
            self.action_manager.process_action(action.to(self.device))
            for _ in range(self.cfg.decimation):
                self._sim_step_counter += 1
                self.action_manager.apply_action()
                self.sim.step(render=False)
            if isinstance(self.cfg.sim.physics, NewtonWorldsCfg):
                bank = self.keyboard_variants
                torch._assert_async(
                    ((wp.to_torch(bank.overflow) & bank.capacity_overflow_mask) == 0).all(),
                    "Native keyboard contact workspace overflow.",
                )
                torch._assert_async(
                    wp.to_torch(bank.backend.runtime.batch_result.status)[0] == 0,
                    "Native keyboard directory transaction failed.",
                )
            self.episode_length_buf += 1
            self.common_step_counter += 1
            self.reset_buf = self.termination_manager.compute()
            terminated = self.termination_manager.terminated
            truncated = self.termination_manager.time_outs & ~terminated
            self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
            reset_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            natural_resets = len(reset_ids)
            redistribute = self.common_step_counter % self.cfg.redistribution_interval == 0
            if self.cfg.compute_final_obs and (
                len(reset_ids) or (redistribute and self.cfg.redistribution_mode == "truncate_pending")
            ):
                # A continuing environment would update typing and record this sample before observing it.
                # Preview that successor before replacing its model, without advancing live MDP state or RNG.
                with (
                    torch.random.fork_rng(devices=[torch.device(self.device)]),
                    self.command_manager.get_term("typing").preview_step(self.step_dt),
                ):
                    self.extras["final_obs"] = self.observation_manager.preview()
            if len(reset_ids):
                self.keyboard_variants.request_variants(reset_ids)
            if redistribute:
                eligible = reset_ids if self.cfg.redistribution_mode == "episode_boundary" else self.all_env_ids
                if len(eligible):
                    changed = self._apply_variant_requests(eligible)
                    if self.cfg.redistribution_mode == "truncate_pending" and len(changed):
                        self.episode_interrupted[changed] = ~self.reset_buf[changed]
                        truncated[changed] = ~terminated[changed]
                        reset_ids = torch.cat((reset_ids, changed)).unique()
            self.reset_buf = terminated | truncated
            if len(reset_ids):
                self._reset_idx(reset_ids)
            self.command_manager.compute(dt=self.step_dt)
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)
            self.forward()
            self.obs_buf = self.observation_manager.compute(update_history=True)
            bank = self.keyboard_variants
            self.extras.setdefault("log", {}).update(
                {
                    "Populations/pending_fraction": (bank.desired_variant_ids != bank.variant_ids).float().mean(),
                    "Populations/natural_resets": natural_resets,
                    "Populations/interrupted_worlds": self.episode_interrupted.sum(),
                    "Populations/publications": (
                        bank.reset_publication_count if isinstance(bank, KeyboardWorlds) else bank.redistribution_count
                    ),
                    "Populations/last_changed_worlds": bank.last_changed_worlds,
                    "Populations/last_publication_host_ms": (
                        bank.last_reset_publication_ms
                        if isinstance(bank, KeyboardWorlds)
                        else bank.last_redistribution_ms
                    ),
                    "Populations/populated_prototype_count": bank.populated_prototype_count,
                    "Populations/live_dof_count": bank.live_dof_count,
                }
            )
            if isinstance(self.cfg.sim.physics, NewtonWorldsCfg):
                self.extras["log"]["Populations/prototypes_ever_at_iteration_limit"] = (
                    (wp.to_torch(bank.overflow) & ~bank.capacity_overflow_mask) != 0
                ).sum()
        return self.obs_buf, self.reward_buf, terminated, truncated, self.extras

    def close(self):
        """Retire native resources after outstanding GPU work and clear task references."""
        if not self._is_closed:
            try:
                try:
                    self.sim.stop()
                finally:
                    self.sim.clear_instance()
            finally:
                # SimulationContext already attempts every native cleanup before
                # reporting errors. Release task bindings even when teardown fails.
                try:
                    if hasattr(self, "keyboard_variants"):
                        self.keyboard_variants.close()
                finally:
                    for name in (
                        "command_manager",
                        "action_manager",
                        "observation_manager",
                        "termination_manager",
                        "reward_manager",
                        "curriculum_manager",
                        "event_manager",
                        "keyboard_variants",
                        "cfg",  # Bound selectors retain native resources; the caller's config is separate.
                    ):
                        if hasattr(self, name):
                            delattr(self, name)
                    self.obs_buf.clear()
                    self.native_contacts.clear()
                    self._is_closed = True
