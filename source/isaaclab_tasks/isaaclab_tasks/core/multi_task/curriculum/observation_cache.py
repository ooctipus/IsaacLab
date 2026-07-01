# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cold construction of expression-bound observation caches."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

from .reset_state import temporary_seed

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@torch.no_grad()
def _materialize_observations(
    env: ManagerBasedRLEnv,
    num_rows: int,
    bind_rows: Callable[[torch.Tensor, torch.Tensor], None],
    refresh: Callable[[], None],
) -> TensorDict:
    """Evaluate every row in env-sized batches before the first reset."""
    if hasattr(env, "obs_buf"):
        raise RuntimeError("Observation caches must be materialized before the first environment reset.")
    if num_rows < 1:
        raise ValueError("Observation caches require at least one row.")

    cache = TensorDict({}, batch_size=[num_rows], device=env.device)
    all_env_ids = torch.arange(env.num_envs, device=env.device)
    with temporary_seed(0):
        for start in range(0, num_rows, env.num_envs):
            task_rows = torch.arange(start, min(start + env.num_envs, num_rows), device=env.device)
            env_ids = all_env_ids[: task_rows.numel()]
            bind_rows(env_ids, task_rows)
            env.sim.forward()
            env.scene.update(dt=0.0)
            for sensor in env.scene.sensors.values():
                sensor.update(dt=0.0, force_recompute=True)
            refresh()

            observations = TensorDict(
                env.observation_manager.compute(update_history=False),
                batch_size=[env.num_envs],
            )
            if cache.is_empty():
                for key, value in observations.items(include_nested=True, leaves_only=True):
                    cache.set(
                        key,
                        torch.empty((num_rows, *value.shape[1:]), dtype=value.dtype, device=value.device),
                    )
            for key, value in observations.items(include_nested=True, leaves_only=True):
                cache.get(key).index_copy_(0, task_rows, value[env_ids])
    return cache.lock_()


def materialize_state_command_observations(env: ManagerBasedRLEnv, command_name: str) -> TensorDict:
    """Materialize observations after each row's normal spawn binding."""
    command = env.command_manager.get_term(command_name)
    return _materialize_observations(
        env,
        command.table.num_tasks,
        command.bind_rows,
        lambda: command.payload.update(0.0, command.command, command.error),
    )


def materialize_state_command_target_observations(env: ManagerBasedRLEnv, command_name: str) -> TensorDict:
    """Materialize observations with physics and command bound to each target row."""
    command = env.command_manager.get_term(command_name)

    def bind_target(env_ids: torch.Tensor, task_rows: torch.Tensor) -> None:
        command.cmd_indices[env_ids] = task_rows
        command.payload.bind_target(env_ids, task_rows)

    return _materialize_observations(
        env,
        command.table.num_tasks,
        bind_target,
        lambda: command.payload.update(0.0, command.command, command.error),
    )


def evaluate_observation_cache_bind(
    expression: str,
    env: ManagerBasedRLEnv,
    bindings: Mapping[str, object] | None = None,
) -> TensorDict:
    """Evaluate a cache expression against sampler bindings plus cold materializers."""
    namespace = dict(bindings or {})
    namespace.update(
        env=env,
        materialize_state_command_observations=materialize_state_command_observations,
        materialize_state_command_target_observations=materialize_state_command_target_observations,
    )
    observations = eval(expression, namespace)  # noqa: S307
    if not isinstance(observations, TensorDict) or len(observations.batch_size) == 0 or observations.batch_size[0] < 1:
        raise TypeError("Observation-cache expressions must resolve to a non-empty TensorDict.")
    return observations if observations.is_locked else observations.lock_()


class ObservationCache(ManagerTermBase):
    """Own one immutable observation cache built from a configured expression."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self.observations: TensorDict = evaluate_observation_cache_bind(cfg.params["observations_bind"], env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        observations_bind: str,
    ) -> None:
        """Retain the construction-time cache without per-reset work."""
        del env, env_ids, observations_bind
