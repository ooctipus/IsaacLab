# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify state writes invalidate cached properties derived from changed state."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

from types import SimpleNamespace

import pytest
from isaaclab_newton.assets.articulation.articulation_data import ArticulationData
from isaaclab_newton.assets.rigid_object.rigid_object_data import RigidObjectData
from isaaclab_newton.assets.rigid_object_collection.rigid_object_collection_data import RigidObjectCollectionData
from isaaclab_newton.physics import NewtonManager as SimulationManager

_POSE_DERIVED = (
    "projected_gravity_b",
    "heading_w",
    "root_link_lin_vel_b",
    "root_link_ang_vel_b",
    "root_com_lin_vel_b",
    "root_com_ang_vel_b",
)
_VELOCITY_DERIVED = _POSE_DERIVED[2:]
_ALL_FIELDS = _POSE_DERIVED + (
    "root_com_pose_w",
    "body_com_pose_w",
    "body_link_pose_w",
    "root_state_w",
    "root_link_state_w",
    "root_com_state_w",
    "body_state_w",
    "body_link_state_w",
    "body_com_state_w",
    "body_com_jacobian_w",
    "gravity_compensation_forces",
    "mass_matrix",
    "root_link_vel_w",
    "body_com_vel_w",
    "body_link_vel_w",
)


def _primed(data_cls):
    data = object.__new__(data_cls)
    for name in _ALL_FIELDS:
        setattr(data, f"_{name}", SimpleNamespace(timestamp=1.0))
    data._fk_timestamp = 1.0
    data._root_view = SimpleNamespace(articulation_ids=None)
    return data


@pytest.mark.parametrize(
    ("data_cls", "derived"),
    (
        (ArticulationData, _POSE_DERIVED),
        (RigidObjectData, _POSE_DERIVED),
        (RigidObjectCollectionData, _POSE_DERIVED[:2]),
    ),
)
def test_pose_reset_invalidates_cached_derived_properties(data_cls, derived, monkeypatch) -> None:
    monkeypatch.setattr(SimulationManager, "invalidate_fk", lambda **_: None)
    data = _primed(data_cls)
    data._reset_pose()
    assert all(getattr(data, f"_{name}").timestamp == -1.0 for name in derived)


@pytest.mark.parametrize("data_cls", (ArticulationData, RigidObjectData))
def test_velocity_reset_invalidates_cached_body_frame_velocities(data_cls, monkeypatch) -> None:
    monkeypatch.setattr(SimulationManager, "invalidate_fk", lambda **_: None)
    data = _primed(data_cls)
    data._reset_velocity()
    assert all(getattr(data, f"_{name}").timestamp == -1.0 for name in _VELOCITY_DERIVED)
