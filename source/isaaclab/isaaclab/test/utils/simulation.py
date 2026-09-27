# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Start the simulation runtime for a test module."""

from __future__ import annotations

import contextlib

_RUNTIME = contextlib.ExitStack()


def launch_test_simulation(**launcher_args):
    """Start Isaac Sim / Kit for the rest of the test process through :func:`~isaaclab.app.launch_simulation`.

    Call it at module level, before importing modules that need Kit. The runtime stays up until the
    process exits, when the Kit launcher closes it with the process's exit status.

    Args:
        **launcher_args: Launcher arguments, for example ``device`` or ``enable_cameras``.

    Returns:
        The running Kit application, for tests that pump it with ``update()``.
    """
    from isaaclab.app import launch_simulation

    _RUNTIME.enter_context(launch_simulation(None, {"require_kit": True, "headless": True, **launcher_args}))
    import omni.kit.app

    return omni.kit.app.get_app()
