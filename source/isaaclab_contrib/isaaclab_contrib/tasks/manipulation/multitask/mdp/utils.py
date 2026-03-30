# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared config types and utilities for multitask MDP terms."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ManagerTermBaseCfg

ScatterResult = tuple[torch.Tensor | slice, torch.Tensor]
"""Return type for ``@scatterable`` functions: ``(env_ids, group_local_result)``."""


@configclass
class PoseCommandRanges:
    """Uniform sampling ranges for a pose command [m, rad]."""

    pos_x: tuple[float, float] = (0.0, 0.0)
    """Min/max for X position [m]."""
    pos_y: tuple[float, float] = (0.0, 0.0)
    """Min/max for Y position [m]."""
    pos_z: tuple[float, float] = (0.0, 0.0)
    """Min/max for Z position [m]."""
    roll: tuple[float, float] = (0.0, 0.0)
    """Min/max for roll angle [rad]."""
    pitch: tuple[float, float] = (0.0, 0.0)
    """Min/max for pitch angle [rad]."""
    yaw: tuple[float, float] = (0.0, 0.0)
    """Min/max for yaw angle [rad]."""


def scatterable(func):
    """Decorator for group-aware MDP terms that produce partial-env results.

    The wrapped function returns ``(env_ids, result)`` where ``result``
    is group-local with shape ``(group_size, ...)``.  The decorator
    scatters into a full ``(num_envs, ...)`` buffer and returns a
    :class:`torch.Tensor`.

    Standalone calls reuse a persistent buffer (zero allocation after
    first call).  When called by :class:`scatter_term` with
    ``_out=buf``, the result is scattered into the provided buffer.
    """

    @functools.wraps(func)
    def wrapper(env, *args, _out=None, **kwargs):
        env_ids, result = func(env, *args, **kwargs)
        if _out is None:
            if not hasattr(wrapper, "_buf"):
                wrapper._buf = torch.zeros(env.num_envs, *result.shape[1:], dtype=result.dtype, device=env.device)
            _out = wrapper._buf
            _out.zero_()
        _out[env_ids] = result
        return _out

    return wrapper


class scatter_term(ManagerTermBase):
    """Collects multiple ``@scatterable`` children into one output buffer.

    Pre-allocates a single ``(num_envs, D)`` buffer in ``__init__``.
    Each step: zeros the buffer, calls each child with ``_out=buf``
    so they scatter directly into it, returns the buffer.

    Children are :class:`ManagerTermBaseCfg` instances (just ``func`` +
    ``params``).  The outer term type (``ObsTerm``, ``RewTerm``, etc.)
    carries weights, noise, and other manager-specific fields.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._terms: list[ManagerTermBaseCfg] = cfg.params["terms"]
        sample = self._terms[0].func(env, **self._terms[0].params)
        self._buf = torch.zeros_like(sample)

    def __call__(self, env: ManagerBasedRLEnv, terms: list | None = None) -> torch.Tensor:
        self._buf.zero_()
        for term_cfg in self._terms:
            term_cfg.func(env, **term_cfg.params, _out=self._buf)
        return self._buf
