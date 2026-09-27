# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility stand-in for the removed ``AppLauncher``."""

from __future__ import annotations

import argparse
import logging

from .sim_launcher import add_launcher_args

logger = logging.getLogger(__name__)


class _NoOpApp:
    def is_running(self) -> bool:
        # nothing runs, so legacy ``while app.is_running():`` loops end instead of spinning forever
        return False

    def update(self) -> None:
        pass

    def close(self, *args, **kwargs) -> None:
        pass


class AppLauncher:
    """Does nothing; use :func:`~isaaclab.app.launch_simulation` to start the simulation runtime.

    :func:`~isaaclab.app.launch_simulation` starts the runtime a config needs (for example Kit for
    Isaac Sim PhysX), so this class only accepts the old arguments to keep existing scripts importable.
    """

    def __init__(self, *args, **kwargs):
        logger.warning("AppLauncher no longer starts the simulation runtime; use isaaclab.app.launch_simulation.")
        self.app = _NoOpApp()

    @staticmethod
    def add_app_launcher_args(parser: argparse.ArgumentParser) -> None:
        """Add the launcher arguments, see :func:`~isaaclab.app.add_launcher_args`."""
        add_launcher_args(parser)
