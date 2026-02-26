# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OpenXR teleoperation devices (legacy).

.. deprecated::
    This package has moved to :mod:`isaaclab_teleop.deprecated.openxr`.
    Please migrate to :mod:`isaaclab_teleop` which provides the
    :class:`~isaaclab_teleop.IsaacTeleopDevice` as a replacement.

    Imports from this package will continue to work for backwards
    compatibility.  Individual class constructors emit
    :class:`DeprecationWarning` at instantiation time.
"""

_OPENXR_ATTRS = {
    "ManusVive", "ManusViveCfg",
    "OpenXRDevice", "OpenXRDeviceCfg",
    "XrAnchorRotationMode", "XrCfg", "remove_camera_configs",
}


def __getattr__(name: str):
    if name in _OPENXR_ATTRS:
        try:
            import isaaclab_teleop.deprecated.openxr as _openxr

            return getattr(_openxr, name)
        except ImportError:
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}. "
                "isaaclab_teleop is not installed."
            )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    try:
        import isaaclab_teleop.deprecated.openxr as _openxr

        return list(_OPENXR_ATTRS) + dir(_openxr)
    except ImportError:
        return list(_OPENXR_ATTRS)
