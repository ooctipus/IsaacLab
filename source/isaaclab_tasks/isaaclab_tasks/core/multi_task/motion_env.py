# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime boundary for the shared motion-imitation environment."""

from __future__ import annotations

import copy
import random
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, ObservationManager
from isaaclab.utils.configclass import resolve_cfg_presets

from .motion.mdp.commands import MotionStatePayload
from .motion.mdp.runtime import MotionRuntime
from .motion_env_cfg import MotionImitationEnvCfg

if TYPE_CHECKING:
    from isaaclab.envs.common import VecEnvObs


@runtime_checkable
class _EvaluationStateTerm(Protocol):
    """Class-based manager term whose cross-call state participates in evaluation isolation."""

    def evaluation_state_dict(self) -> Mapping[str, torch.Tensor | float]:
        """Return detached state that evaluation may mutate."""

    def load_evaluation_state_dict(self, state: Mapping[str, torch.Tensor | float]) -> None:
        """Restore state returned by :meth:`evaluation_state_dict`."""


def _observation_manager_supports_selected_reset(manager: ObservationManager) -> bool:
    """Return whether global recomputation has no manager-owned hidden state."""
    return (
        not any(manager._group_obs_class_term_cfgs.values())
        and not manager._group_obs_class_instances
        and all(
            term_cfg.history_length == 0 and term_cfg.noise is None and not term_cfg.modifiers
            for term_cfgs in manager._group_obs_term_cfgs.values()
            for term_cfg in term_cfgs
        )
    )


class MotionImitationEnv(ManagerBasedRLEnv):
    """One simulator lifecycle for both native motion profiles."""

    cfg: MotionImitationEnvCfg

    def __init__(self, cfg: MotionImitationEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        resolve_cfg_presets(cfg)
        self._evaluation_clip_indices: torch.Tensor | None = None
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        self._evaluation_all_env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)

    def load_managers(self) -> None:
        """Attach transition state after the command manager builds its trajectory table."""
        super().load_managers()
        command = self.command_manager.get_term("motion")
        payload = command.payload
        if not isinstance(payload, MotionStatePayload):
            raise TypeError("The motion command must own MotionStatePayload.")
        runtime_factory = self.cfg.commands.motion.payload.transition_state_factory
        if not callable(runtime_factory):
            raise TypeError("The resolved motion profile must provide a callable transition-state factory.")
        runtime = runtime_factory(self, payload)
        for name in ("capture_current", "measure", "reset"):
            if not callable(getattr(runtime, name, None)):
                raise TypeError(f"Motion transition runtime must expose callable {name}().")
        expected_tensors = {
            "action_applied": ((self.num_envs,), torch.bool),
            "environment_reward": ((self.num_envs,), None),
            "auxiliary_evidence": ((self.num_envs, len(payload.auxiliary_evidence_names)), None),
        }
        for name, (shape, dtype) in expected_tensors.items():
            value = getattr(runtime, name, None)
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != shape
                or value.device != torch.device(self.device)
                or dtype is not None
                and value.dtype is not dtype
                or dtype is None
                and not value.is_floating_point()
            ):
                raise TypeError(f"Motion transition runtime {name} has a wrong tensor contract.")
        self._motion_runtime = cast(MotionRuntime, runtime)
        payload.attach_transition_state(self._motion_runtime)
        self._transition_episode_steps = torch.empty(
            self.num_envs,
            dtype=torch.int64,
            device=self.device,
        )
        self._final_observation_valid = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self._evaluation_task_rows = torch.empty(
            self.num_envs,
            dtype=torch.int64,
            device=self.device,
        )
        self._evaluation_selected = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self._selected_reset_observations_are_stateless = _observation_manager_supports_selected_reset(
            self.observation_manager
        )

    def _validate_motion_clip_reset(
        self,
        clip_indices: torch.Tensor,
        env_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        """Validate paired exact-reset vectors and return the selected environment rows."""
        if env_ids is None:
            env_ids = self._evaluation_all_env_ids
        if (
            env_ids.ndim != 1
            or env_ids.numel() < 1
            or clip_indices.shape != env_ids.shape
            or env_ids.dtype is not torch.int64
            or clip_indices.dtype is not torch.int64
            or env_ids.device != torch.device(self.device)
            or clip_indices.device != torch.device(self.device)
        ):
            raise ValueError("Evaluation environment and clip indices must be paired int64 device vectors.")
        torch._assert_async(
            torch.all((env_ids >= 0) & (env_ids < self.num_envs)),
            "Evaluation environment indices must be in range.",
        )
        torch._assert_async(
            torch.all(env_ids[1:] > env_ids[:-1]),
            "Evaluation environment indices must be strictly increasing.",
        )
        return env_ids

    def _bind_motion_clip_starts(self, env_ids: torch.Tensor, clip_indices: torch.Tensor) -> None:
        """Bind selected command and payload rows to exact source-frame-zero states."""
        command = self.command_manager.get_term("motion")
        payload = command.payload
        if not isinstance(payload, MotionStatePayload):
            raise TypeError("The motion command must own MotionStatePayload.")
        task_rows = self._evaluation_task_rows[: clip_indices.shape[0]]
        torch.index_select(command.table.clip_start_rows, 0, clip_indices, out=task_rows)
        torch._assert_async(
            torch.all(task_rows >= 0),
            "Every evaluation clip must have a motion task row.",
        )
        command.cmd_indices.index_copy_(0, env_ids, task_rows)
        payload.bind_clip_start(env_ids, clip_indices)

    def reset_motion_clips(
        self,
        clip_indices: torch.Tensor,
        *,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[VecEnvObs, dict]:
        """Reset selected environments to the start of explicit motion clips.

        Args:
            clip_indices: Motion clip index paired with each selected environment.
            env_ids: Selected environment indices. ``None`` selects every environment.

        Returns:
            Environment observations and extras after the selected reset.
        """
        env_ids = self._validate_motion_clip_reset(clip_indices, env_ids)
        if self._evaluation_clip_indices is not None:
            raise RuntimeError("An exact motion reset is already active.")
        self._evaluation_clip_indices = clip_indices
        try:
            return self.reset(env_ids=env_ids)
        finally:
            self._evaluation_clip_indices = None

    def reset_motion_clips_selected(
        self,
        clip_indices: torch.Tensor,
        *,
        env_ids: torch.Tensor,
    ) -> tuple[VecEnvObs, dict]:
        """Reset exact motion clips in a stateless observation profile.

        This evaluation-only lifecycle clears selected scene, observation, action, payload,
        and episode state. It intentionally excludes training curriculum, logging, recorder,
        command-resampling, and reset-randomization transactions. Observation-manager history,
        corruption, modifiers, and class terms are rejected because a global recomputation could
        otherwise mutate hidden state for unselected lanes.

        Args:
            clip_indices: Motion clip index paired with each selected environment.
            env_ids: Strictly increasing selected environment indices.

        Returns:
            Full environment observations with unselected rows preserved, plus extras.
        """
        env_ids = self._validate_motion_clip_reset(clip_indices, env_ids)
        if not self._selected_reset_observations_are_stateless:
            raise RuntimeError("Selected motion reset requires history-free, corruption-free stateless observations.")
        self.scene.reset(env_ids)
        self.observation_manager.reset(env_ids)
        self.action_manager.reset(env_ids)
        self.episode_length_buf.index_fill_(0, env_ids, 0)
        self._bind_motion_clip_starts(env_ids, clip_indices)

        self.scene.write_data_to_sim()
        self.sim.forward()
        if self.has_rtx_sensors and self.cfg.num_rerenders_on_reset > 0:
            for _ in range(self.cfg.num_rerenders_on_reset):
                self.sim.render()

        previous = self.obs_buf
        observations = self.observation_manager.compute(update_history=False)
        self._evaluation_selected.zero_()
        self._evaluation_selected.index_fill_(0, env_ids, True)
        for name, value in observations.items():
            prior = previous.get(name)
            if (
                not isinstance(value, torch.Tensor)
                or not isinstance(prior, torch.Tensor)
                or value.shape != prior.shape
                or value.shape[0] != self.num_envs
            ):
                raise TypeError("Selected motion reset requires flat tensor observation groups with stable shapes.")
            selected = self._evaluation_selected.view(self.num_envs, *((1,) * (value.ndim - 1)))
            torch.where(selected, value, prior, out=value)
        self.obs_buf = observations
        return self.obs_buf, self.extras

    def _reset_idx(self, env_ids) -> None:
        """Apply an exact evaluation binding after normal manager reset state is cleared."""
        super()._reset_idx(env_ids)
        if self._evaluation_clip_indices is None:
            return
        env_ids = torch.as_tensor(env_ids, dtype=torch.int64, device=self.device)
        self._bind_motion_clip_starts(env_ids, self._evaluation_clip_indices)

    def step(self, action: torch.Tensor):
        """Capture current history sources and publish the completed logical edge."""
        self._motion_runtime.capture_current(self.obs_buf)
        self._transition_episode_steps.copy_(self.episode_length_buf).add_(1)
        observations, rewards, terminated, truncated, extras = super().step(action)
        done = terminated | truncated
        extras["episode_steps"] = self._transition_episode_steps
        extras["action_applied"] = self._motion_runtime.action_applied
        extras["auxiliary_reward_evidence"] = self._motion_runtime.auxiliary_evidence
        self._final_observation_valid.zero_()
        if self.cfg.compute_final_obs and "final_obs" in extras:
            self._final_observation_valid.copy_(done)
        extras["final_obs_valid"] = self._final_observation_valid
        return observations, rewards, terminated, truncated, extras

    def _evaluation_state_terms(self) -> tuple[tuple[str, _EvaluationStateTerm], ...]:
        """Return every class-based manager term that evaluation can advance."""
        terms: list[tuple[str, _EvaluationStateTerm]] = []
        seen: set[int] = set()
        for mode in ("reset", "interval"):
            for name in self.event_manager.active_terms.get(mode, ()):
                term = self.event_manager.get_term_cfg(name).func
                if isinstance(term, type) and issubclass(term, ManagerTermBase):
                    raise RuntimeError(f"Class-based event term {mode}.{name} is not initialized.")
                if not isinstance(term, ManagerTermBase) or id(term) in seen:
                    continue
                if not isinstance(term, _EvaluationStateTerm):
                    raise TypeError(
                        f"Class-based event term {mode}.{name} must expose evaluation state before tracking."
                    )
                seen.add(id(term))
                terms.append((f"event.{mode}.{name}", term))
        for name in self.curriculum_manager.active_terms:
            term = self.curriculum_manager.get_term(name)
            if isinstance(term, type) and issubclass(term, ManagerTermBase):
                raise RuntimeError(f"Class-based curriculum term {name} is not initialized.")
            if not isinstance(term, ManagerTermBase) or id(term) in seen:
                continue
            if not isinstance(term, _EvaluationStateTerm):
                raise TypeError(f"Class-based curriculum term {name} must expose evaluation state before tracking.")
            seen.add(id(term))
            terms.append((f"curriculum.{name}", term))
        return tuple(terms)

    @contextmanager
    def evaluation_transaction(self, seed: int) -> Iterator[None]:
        """Isolate evaluator RNG and every persistent non-physical environment clock."""
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        torch_cpu_rng = torch.get_rng_state()
        torch_cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        motion_table = self.command_manager.get_term("motion").table
        motion_table_rng = motion_table.generator.get_state().clone()
        common_step_counter = self.common_step_counter
        simulation_step_counter = self._sim_step_counter
        episode_length = self.episode_length_buf.clone()
        command_clocks = tuple(
            (
                self.command_manager.get_term(name).time_left.clone(),
                self.command_manager.get_term(name).command_counter.clone(),
            )
            for name in self.command_manager.active_terms
        )
        event_clocks = tuple(value.clone() for value in self.event_manager._interval_term_time_left)
        event_reset_steps = tuple(value.clone() for value in self.event_manager._reset_term_last_triggered_step_id)
        event_reset_once = tuple(value.clone() for value in self.event_manager._reset_term_last_triggered_once)
        curriculum_state = copy.deepcopy(self.curriculum_manager._curriculum_state)
        state_terms = self._evaluation_state_terms()
        term_states = tuple((name, term, copy.deepcopy(term.evaluation_state_dict())) for name, term in state_terms)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        motion_table.generator.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            yield
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(torch_cpu_rng)
            if torch_cuda_rng is not None:
                torch.cuda.set_rng_state_all(torch_cuda_rng)
            motion_table.generator.set_state(motion_table_rng)
            self.common_step_counter = common_step_counter
            self._sim_step_counter = simulation_step_counter
            self.episode_length_buf.copy_(episode_length)
            for name, (time_left, command_counter) in zip(
                self.command_manager.active_terms,
                command_clocks,
                strict=True,
            ):
                term = self.command_manager.get_term(name)
                term.time_left.copy_(time_left)
                term.command_counter.copy_(command_counter)
            if len(event_clocks) != len(self.event_manager._interval_term_time_left):
                raise RuntimeError("Evaluation changed the configured interval-event clocks.")
            for current, saved in zip(self.event_manager._interval_term_time_left, event_clocks, strict=True):
                current.copy_(saved)
            if len(event_reset_steps) != len(self.event_manager._reset_term_last_triggered_step_id) or len(
                event_reset_once
            ) != len(self.event_manager._reset_term_last_triggered_once):
                raise RuntimeError("Evaluation changed the configured reset-event clocks.")
            for current, saved in zip(
                self.event_manager._reset_term_last_triggered_step_id,
                event_reset_steps,
                strict=True,
            ):
                current.copy_(saved)
            for current, saved in zip(
                self.event_manager._reset_term_last_triggered_once,
                event_reset_once,
                strict=True,
            ):
                current.copy_(saved)
            self.curriculum_manager._curriculum_state.clear()
            self.curriculum_manager._curriculum_state.update(copy.deepcopy(curriculum_state))
            current_terms = self._evaluation_state_terms()
            if tuple(name for name, _term in current_terms) != tuple(name for name, _term, _state in term_states):
                raise RuntimeError("Evaluation changed the configured stateful manager terms.")
            for (_name, current), (_saved_name, saved_term, state) in zip(
                current_terms,
                term_states,
                strict=True,
            ):
                if current is not saved_term:
                    raise RuntimeError("Evaluation replaced a stateful manager term.")
                current.load_evaluation_state_dict(state)


__all__ = ["MotionImitationEnv"]
