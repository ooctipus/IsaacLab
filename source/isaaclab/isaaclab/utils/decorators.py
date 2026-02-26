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
from enum import Enum
from functools import wraps


class FilterEnvIdsSkipMethodNames(Enum):
    """Method names to skip for :func:`filter_env_ids_arg`.

    These delegate to other methods that are wrapped; wrapping these too
    would double-filter env_ids. Only the leaf write_*_to_sim methods are wrapped.
    """

    WRITE_ROOT_STATE_TO_SIM = "write_root_state_to_sim"
    WRITE_ROOT_COM_STATE_TO_SIM = "write_root_com_state_to_sim"
    WRITE_ROOT_LINK_STATE_TO_SIM = "write_root_link_state_to_sim"
    WRITE_ROOT_POSE_TO_SIM = "write_root_pose_to_sim"
    WRITE_ROOT_VELOCITY_TO_SIM = "write_root_velocity_to_sim"
    WRITE_JOINT_STATE_TO_SIM = "write_joint_state_to_sim"

    @classmethod
    def skip_set(cls) -> frozenset[str]:
        """Frozenset of method names to skip for filter_env_ids_arg."""
        return frozenset(m.value for m in cls)


def filter_env_ids_arg(method):
    """Decorator for methods that take an ``env_ids`` argument.

    When ``self.is_heterogeneous`` is true, replaces ``env_ids`` with
    ``self._filter_env_ids(env_ids)`` before calling the method. Use on
    Articulation/RigidObject methods that accept ``env_ids`` and need
    global_to_local filtering in heterogeneous multi-env setups.
    """
    sig = inspect.signature(method)
    has_env_ids = "env_ids" in sig.parameters

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, "is_heterogeneous", False):
            return method(self, *args, **kwargs)
        if not has_env_ids:
            return method(self, *args, **kwargs)
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        bound.arguments["env_ids"] = self._filter_env_ids(bound.arguments["env_ids"])
        params = list(sig.parameters.keys())
        kwargs_out = {p: bound.arguments[p] for p in params if p != "self"}
        return method(self, **kwargs_out)

    return wrapper
