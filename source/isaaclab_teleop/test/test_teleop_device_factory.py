# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from typing import cast

import pytest
from isaaclab_teleop.deprecated.teleop_device_factory import create_teleop_device

from isaaclab.devices import DeviceBase, DeviceCfg


class _TestDevice(DeviceBase):
    def __init__(self, cfg: DeviceCfg):
        super().__init__()
        self.cfg = cfg

    def reset(self):
        pass

    def add_callback(self, key, func):
        pass


def test_create_teleop_device():
    """The legacy factory still constructs a cfg-selected non-XR device."""
    cfg = DeviceCfg(class_type=_TestDevice)
    with pytest.deprecated_call():
        device = create_teleop_device("test", {"test": cfg})

    assert isinstance(device, _TestDevice)
    assert device.cfg is cfg


def test_create_teleop_device_rejects_invalid_selection():
    """The legacy factory reports missing and data-only configurations directly."""
    with pytest.deprecated_call(), pytest.raises(ValueError, match="Device 'gamepad' not found"):
        create_teleop_device("gamepad", {"test": DeviceCfg(class_type=_TestDevice)})

    class UnsupportedCfg:
        pass

    cfgs = cast(dict[str, DeviceCfg], {"unsupported": UnsupportedCfg()})
    with pytest.deprecated_call(), pytest.raises(ValueError, match="does not declare class_type"):
        create_teleop_device("unsupported", cfgs)


def test_deprecated_openxr_devices_are_absent():
    """No deprecated device may reconstruct an XR anchor outside the clone plan."""
    import isaaclab_teleop.deprecated.openxr as openxr

    import isaaclab.devices as devices

    source_root = Path(__file__).resolve().parents[2]
    package_roots = (
        source_root / "isaaclab/isaaclab/devices/openxr",
        source_root / "isaaclab_teleop/isaaclab_teleop/deprecated/openxr",
    )
    forbidden_files = {
        "common.py",
        "manus_vive.py",
        "manus_vive_utils.py",
        "openxr_device.py",
        "xr_anchor_utils.py",
        "xr_cfg.py",
    }

    assert not any((root / name).exists() for root in package_roots for name in forbidden_files)
    forbidden_symbols = ("ManusVive", "ManusViveCfg", "OpenXRDevice", "OpenXRDeviceCfg", "XrCfg")
    assert not any(hasattr(module, name) for module in (devices, openxr) for name in forbidden_symbols)
