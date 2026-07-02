# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated compatibility boundary for the task-local CRL runner."""

from __future__ import annotations

import warnings

from rsl_rl.env import VecEnv

from .crl_runner import CrlRunner


class OffPolicyRunner(CrlRunner):
    """Deprecated task-local name for :class:`CrlRunner`.

    .. deprecated:: 8.0.2
        Use :class:`CrlRunner`. Generic replay algorithms should use
        :class:`rsl_rl.runners.OffPolicyRunner`.
    """

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        """Warn and construct the CRL runner through its canonical implementation."""
        warnings.warn(
            "OffPolicyRunner is deprecated; use CrlRunner.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)


__all__ = ["OffPolicyRunner"]
