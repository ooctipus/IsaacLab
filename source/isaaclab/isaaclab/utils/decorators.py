# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generic decorators used across Isaac Lab."""

from __future__ import annotations

import inspect
from functools import wraps


def filter_env_ids_arg(method):
    """Decorator for methods that take an ``env_ids`` argument.

    When ``self.is_heterogeneous`` is true, replaces ``env_ids`` with
    ``self._filter_env_ids(env_ids)`` before calling the method. Use on
    Articulation/RigidObject methods that accept ``env_ids`` and need
    global_to_local filtering in heterogeneous multi-env setups.
    """
    sig = inspect.signature(method)

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, "is_heterogeneous", False):
            return method(self, *args, **kwargs)
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        if "env_ids" in bound.arguments:
            bound.arguments["env_ids"] = self._filter_env_ids(bound.arguments["env_ids"])
        params = list(sig.parameters.keys())
        kwargs_out = {p: bound.arguments[p] for p in params if p != "self"}
        return method(self, **kwargs_out)

    return wrapper
