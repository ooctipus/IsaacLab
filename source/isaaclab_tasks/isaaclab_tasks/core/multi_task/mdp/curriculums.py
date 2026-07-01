# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase

from isaaclab_tasks.core.multi_task.curriculum import SamplerCfg, StateLayoutCfg, SuccessMonitorCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from isaaclab_tasks.core.multi_task.curriculum import ValueShiftSamplingStrategy


class success_rate_sampler(ManagerTermBase):
    """Update item success rates and sample the next item indices."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._log_counter = -1
        self._monitor_initialized = False

        self.sample_indices: torch.Tensor = eval(cfg.params["sample_indices_bind"])  # noqa: S307
        self.success: torch.Tensor = eval(cfg.params["success_bind"])  # noqa: S307
        layout = cfg.params["layout"].build(env)
        self.success_rates = torch.zeros(layout.num_items, device=self.sample_indices.device)

        monitor_cfg: SuccessMonitorCfg = cfg.params["success_monitor_cfg"]
        monitor_cfg.num_monitored_data = layout.num_items
        monitor_cfg.device = env.device
        monitor_cfg.max_updates = env.num_envs if monitor_cfg.max_updates is None else monitor_cfg.max_updates
        self.success_monitor = monitor_cfg.class_type(monitor_cfg, self.success_rates)

        self._sampling_cfg: SamplerCfg = cfg.params["sampling"]
        if self._sampling_cfg.max_samples is None:
            self._sampling_cfg.max_samples = env.num_envs
        self._sampler = self._sampling_cfg.class_type(
            self._sampling_cfg, layout, env=env, success_rates=self.success_rates
        )

    @property
    def value_shift(self) -> ValueShiftSamplingStrategy:
        """Return the selected value-shift strategy and its learner buffers."""
        return self._sampler.value_shift

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        sample_indices_bind: str,
        success_bind: str,
        layout: StateLayoutCfg,
        sampling: SamplerCfg,
        success_monitor_cfg: SuccessMonitorCfg,
        sampler_visual_logger: Callable[..., None] | None = None,
        sampler_visual_log_period: int = 1000,
    ) -> dict[str, torch.Tensor]:
        if env_ids.numel() == 0:
            return {"success": self.success_rates.mean()}

        if self._monitor_initialized:
            prev_idx = self.sample_indices[env_ids]
            self.success_monitor.success_update(prev_idx, self.success[env_ids])
        else:
            self._monitor_initialized = True

        num_samples = self._sampling_cfg.max_samples if self._sampling_cfg.warp else len(env_ids)
        probs, choices = self._sampler.probabilities_and_sample(num_samples)
        self.sample_indices[env_ids] = choices[: len(env_ids)]

        if sampler_visual_logger is not None:
            # The RL wrapper resets the env once before the runner starts logging.
            # Skip that deterministic pre-run call, then emit on counter 0 and every period after.
            if self._log_counter >= 0 and self._log_counter % sampler_visual_log_period == 0:
                sampler_visual_logger(env, self._sampler, self.success_rates, probs)
            self._log_counter += 1

        return {"success": self.success_rates.mean()}
