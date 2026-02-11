# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Factory to wrap controllers for per-env use: slice buffers to assigned_envs and filter env_ids in reset/reset_idx."""

from __future__ import annotations

from typing import Any

import torch

from .per_env_mixin import PerEnvMixin


class PerEnvControllerWrapper(PerEnvMixin):
    """Wraps a controller so buffers are sliced to assigned_envs and reset/reset_idx filter env_ids.

    Takes only ``assigned_envs`` (global env indices) and the controller instance; no cfg.
    """

    def __init__(self, assigned_envs: tuple[int, ...] | list[int], controller: Any) -> None:
        self._assigned_envs = tuple(assigned_envs)
        self._assigned_env_to_local = {env_idx: idx for idx, env_idx in enumerate(self._assigned_envs)}
        self._controller = controller
        self._slice_controller_buffers(self._controller, self._assigned_envs)

    def _initialize_impl(self) -> None:
        """No-op; wrapper is not part of asset/sensor init flow."""
        pass

    def _slice_controller_buffers(
        self,
        controller: Any,
        assigned_envs: tuple[int, ...],
    ) -> None:
        """In-place slice any tensor attribute with shape[0] == num_envs/num_robots to assigned_envs."""
        n = getattr(controller, "num_envs", None) or getattr(controller, "num_robots", None)
        if n is None:
            return
        device = getattr(controller, "_device", "cuda:0")
        env_idx = torch.tensor(assigned_envs, device=device)
        for attr in list(vars(controller).keys()):
            val = getattr(controller, attr, None)
            if isinstance(val, torch.Tensor) and val.ndim >= 1 and val.shape[0] == n:
                setattr(controller, attr, val[env_idx].clone())
        if hasattr(controller, "num_envs"):
            controller.num_envs = len(assigned_envs)
        if hasattr(controller, "num_robots"):
            controller.num_robots = len(assigned_envs)

    @property
    def device(self):
        """Delegate to wrapped controller for PerEnvMixin._filter_env_ids."""
        return getattr(self._controller, "device", None) or getattr(self._controller, "_device", "cuda:0")

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Filter env_ids to assigned subset then call controller.reset."""
        resolved = self._filter_env_ids(env_ids)
        if resolved.numel() == 0:
            return
        try:
            self._controller.reset(resolved)
        except TypeError:
            self._controller.reset()

    def reset_idx(self, robot_ids: torch.Tensor | None = None) -> None:
        """Filter robot_ids to assigned subset then call controller.reset_idx."""
        resolved = self._filter_env_ids(robot_ids)
        if resolved.numel() == 0:
            return
        self._controller.reset_idx(resolved)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._controller, name)


def make_per_env_controller_wrapper(
    assigned_envs: tuple[int, ...] | list[int],
    controller: Any,
) -> PerEnvControllerWrapper:
    """Wrap a controller for per-env use: slice buffers and filter env_ids in reset/reset_idx.

    Args:
        assigned_envs: Global environment (or robot) indices this controller should manage.
        controller: The controller instance to wrap (e.g. DifferentialIKController, OperationalSpaceController).

    Returns:
        A wrapper that delegates to the controller but slices buffers to assigned_envs
        and filters env_ids/robot_ids in reset() and reset_idx().
    """
    if not assigned_envs:
        raise ValueError("make_per_env_controller_wrapper: assigned_envs must be non-empty.")
    if not all(isinstance(x, int) for x in assigned_envs):
        raise TypeError("make_per_env_controller_wrapper: assigned_envs must be a sequence of int.")
    return PerEnvControllerWrapper(tuple(assigned_envs), controller)
