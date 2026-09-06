# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared self-contained articulated fixtures for terrain retarget tests."""

from pathlib import Path

import pytest

_TOPOLOGY_MJCF = """<mujoco model="canonical_topology_test">
  <compiler angle="radian" inertiafromgeom="false"/>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <body name="base" pos="0 0 0.6">
      <freejoint name="root"/>
      <inertial pos="0.18 0.02 0.12" mass="8" diaginertia="0.2 0.2 0.2"/>
      <geom type="box" size="0.2 0.12 0.08"/>
      <body name="sensor_mount" pos="0 0 0.2">
        <inertial pos="0.02 0 0.03" mass="0.5" diaginertia="0.01 0.01 0.01"/>
        <geom type="sphere" size="0.03"/>
      </body>
      <body name="front_left_FOOT" pos="0.45 0.3 -0.5">
        <joint name="front_left_joint" type="hinge" axis="0 1 0" range="-2 2"/>
        <inertial pos="0.08 0 0.04" mass="1" diaginertia="0.01 0.01 0.01"/>
        <geom type="sphere" size="0.04"/>
      </body>
      <body name="front_right_FOOT" pos="0.45 -0.3 -0.5">
        <joint name="front_right_joint" type="hinge" axis="0 1 0" range="-2 2"/>
        <inertial pos="0.08 0 0.04" mass="1" diaginertia="0.01 0.01 0.01"/>
        <geom type="sphere" size="0.04"/>
      </body>
      <body name="rear_left_FOOT" pos="-0.45 0.3 -0.5">
        <joint name="rear_left_joint" type="hinge" axis="0 1 0" range="-2 2"/>
        <inertial pos="-0.04 0 0.03" mass="1" diaginertia="0.01 0.01 0.01"/>
        <geom type="sphere" size="0.04"/>
      </body>
      <body name="rear_right_FOOT" pos="-0.45 -0.3 -0.5">
        <joint name="rear_right_joint" type="hinge" axis="0 1 0" range="-2 2"/>
        <inertial pos="-0.04 0 0.03" mass="1" diaginertia="0.01 0.01 0.01"/>
        <geom type="sphere" size="0.04"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def canonical_topology_mjcf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write one fixed-plus-revolute MJCF with nonzero body COM offsets."""
    path = tmp_path_factory.mktemp("canonical_topology") / "canonical_topology_test.xml"
    path.write_text(_TOPOLOGY_MJCF)
    return path
