# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SMPL simulator/source coordinate-frame ownership."""

from __future__ import annotations

import re


def smpl_live_joint_source_names(live_joint_names: tuple[str, ...]) -> tuple[str, ...]:
    """Resolve simulator three-axis joint labels to native HumEnv coordinates."""
    source_names: list[str] = []
    for name in live_joint_names:
        joint_name, separator, component = name.rpartition(":")
        match = re.fullmatch(r"(.+)_x_\1_y_\1_z", joint_name)
        if not separator or match is None or component not in ("0", "1", "2"):
            raise ValueError("SMPL live joints must use native Body_x_Body_y_Body_z:0/1/2 coordinate labels.")
        source_names.append(f"{match.group(1)}_{'xyz'[int(component)]}")
    resolved = tuple(source_names)
    if len(set(resolved)) != len(resolved):
        raise ValueError("SMPL live joints do not resolve to unique source coordinates.")
    return resolved
