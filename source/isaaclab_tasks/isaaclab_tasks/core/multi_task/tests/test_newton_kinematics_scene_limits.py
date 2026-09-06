# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for scene-owned Newton kinematic limits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from isaaclab_newton.sim import NewtonMjcfFileCfg
from newton import JointType

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from isaaclab_tasks.core.multi_task.kinematics import NewtonKinematics, NewtonKinematicsBuildCfg, NewtonKinematicsCfg

_TWO_JOINT_MJCF = """
<mujoco model="scene_limits">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base">
      <freejoint name="root"/>
      <geom name="base_geom" type="sphere" size="0.05" mass="1.0"/>
      <body name="link_a" pos="0 0 0.1">
        <joint name="joint_a" type="hinge" axis="0 1 0" range="-1 1"/>
        <geom name="link_a_geom" type="capsule" size="0.02 0.05" mass="0.5"/>
        <body name="link_b" pos="0 0 0.1">
          <joint name="joint_b" type="hinge" axis="1 0 0" range="-2 2"/>
          <geom name="link_b_geom" type="capsule" size="0.02 0.05" mass="0.25"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _write_mjcf(tmp_path: Path) -> Path:
    path = tmp_path / "scene_limits.xml"
    path.write_text(_TWO_JOINT_MJCF)
    return path


def _articulation(path: Path, actuator: ImplicitActuatorCfg, *, soft_limit_factor: float = 0.8) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="/World/Robot",
        spawn=NewtonMjcfFileCfg(asset_path=str(path)),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
        actuators={"legs": actuator},
        soft_joint_pos_limit_factor=soft_limit_factor,
    )


def test_from_articulation_applies_exact_and_regex_actuator_limit_values(tmp_path: Path) -> None:
    """Scene actuator dictionaries must replace parsed velocity and effort limits by DoF name."""
    articulation = _articulation(
        _write_mjcf(tmp_path),
        ImplicitActuatorCfg(
            joint_names_expr=["joint_.*"],
            velocity_limit_sim={"joint_a": 3.0, "joint_b": 4.0},
            effort_limit_sim={"joint_a": 10.0, "joint_b": 20.0},
            stiffness={"joint_a": 100.0, "joint_b": 200.0},
            damping={"joint_a": 5.0, "joint_b": 6.0},
            armature={"joint_a": 0.1, "joint_b": 0.2},
            friction={"joint_a": 0.3, "joint_b": 0.4},
        ),
    )

    kinematics = NewtonKinematics.from_articulation(
        NewtonKinematicsBuildCfg(collapse_fixed_joints=False), articulation, "cpu"
    )
    indices = {name: index for index, name in enumerate(kinematics.joint_qd_names)}
    joint_indices = [indices["joint_a"], indices["joint_b"]]

    np.testing.assert_array_equal(kinematics.model.joint_velocity_limit.numpy()[joint_indices], (3.0, 4.0))
    np.testing.assert_array_equal(kinematics.model.joint_effort_limit.numpy()[joint_indices], (10.0, 20.0))
    np.testing.assert_array_equal(kinematics.model.joint_target_ke.numpy()[joint_indices], (100.0, 200.0))
    np.testing.assert_array_equal(kinematics.model.joint_target_kd.numpy()[joint_indices], (5.0, 6.0))
    np.testing.assert_allclose(kinematics.model.joint_armature.numpy()[joint_indices], (0.1, 0.2))
    np.testing.assert_allclose(kinematics.model.joint_friction.numpy()[joint_indices], (0.3, 0.4))
    np.testing.assert_array_equal(kinematics.topology.joint_velocity_lower[joint_indices], (-3.0, -4.0))
    np.testing.assert_array_equal(kinematics.topology.joint_velocity_upper[joint_indices], (3.0, 4.0))
    free_dofs = kinematics.topology.joint_type[kinematics.topology.dof_joint] == int(JointType.FREE)
    assert np.count_nonzero(free_dofs) == 6
    np.testing.assert_array_equal(kinematics.topology.joint_velocity_lower[free_dofs], -np.inf)
    np.testing.assert_array_equal(kinematics.topology.joint_velocity_upper[free_dofs], np.inf)
    np.testing.assert_array_equal(kinematics.topology.joint_effort_lower[joint_indices], (-10.0, -20.0))
    np.testing.assert_array_equal(kinematics.topology.joint_effort_upper[joint_indices], (10.0, 20.0))
    assert kinematics.topology.soft_joint_position_limit_factor == 0.8
    np.testing.assert_allclose(kinematics.topology.joint_limit_soft_lower[joint_indices], (-0.8, -1.6))
    np.testing.assert_allclose(kinematics.topology.joint_limit_soft_upper[joint_indices], (0.8, 1.6))


def test_free_joint_velocity_limits_are_unbounded(tmp_path: Path) -> None:
    """FREE-joint coordinates must not inherit Newton's finite sentinel velocity limits."""
    kinematics = NewtonKinematics(NewtonKinematicsCfg(mjcf_path=str(_write_mjcf(tmp_path)), device="cpu"))
    free_dofs = kinematics.topology.joint_type[kinematics.topology.dof_joint] == int(JointType.FREE)

    assert np.count_nonzero(free_dofs) == 6
    np.testing.assert_array_equal(kinematics.topology.joint_velocity_lower[free_dofs], -np.inf)
    np.testing.assert_array_equal(kinematics.topology.joint_velocity_upper[free_dofs], np.inf)


def test_from_articulation_matches_legacy_implicit_limits_and_group_write_order(tmp_path: Path) -> None:
    """Legacy implicit effort aliases and later actuator groups must match runtime writes."""
    path = _write_mjcf(tmp_path)
    parsed = NewtonKinematics(NewtonKinematicsCfg(mjcf_path=str(path), device="cpu"))
    articulation = _articulation(
        path,
        ImplicitActuatorCfg(
            joint_names_expr=["joint_.*"],
            effort_limit=7.0,
            effort_limit_sim=None,
            velocity_limit=2.0,
            velocity_limit_sim=None,
            stiffness=0.0,
            damping=0.0,
        ),
    )
    articulation.actuators["joint_a_override"] = ImplicitActuatorCfg(
        joint_names_expr=["joint_a"],
        effort_limit_sim=11.0,
        velocity_limit_sim=5.0,
        stiffness=0.0,
        damping=0.0,
    )

    resolved = NewtonKinematics.from_articulation(
        NewtonKinematicsBuildCfg(collapse_fixed_joints=False), articulation, "cpu"
    )
    indices = {name: index for index, name in enumerate(resolved.joint_qd_names)}
    joint_a = indices["joint_a"]
    joint_b = indices["joint_b"]

    assert resolved.model.joint_effort_limit.numpy()[joint_a] == 11.0
    assert resolved.model.joint_effort_limit.numpy()[joint_b] == 7.0
    assert resolved.model.joint_velocity_limit.numpy()[joint_a] == 5.0
    assert resolved.model.joint_velocity_limit.numpy()[joint_b] == parsed.model.joint_velocity_limit.numpy()[joint_b]


def test_mechanics_identity_covers_scene_overrides_and_collider_content(tmp_path: Path) -> None:
    """The canonical mechanics digest must change with resolved limits or parsed geometry."""
    path = _write_mjcf(tmp_path)

    def build(velocity_b: float, asset_path: Path = path) -> NewtonKinematics:
        articulation = _articulation(
            asset_path,
            ImplicitActuatorCfg(
                joint_names_expr=["joint_.*"],
                velocity_limit_sim={"joint_a": 3.0, "joint_b": velocity_b},
                effort_limit_sim={"joint_a": 10.0, "joint_b": 20.0},
                stiffness=0.0,
                damping=0.0,
            ),
        )
        return NewtonKinematics.from_articulation(
            NewtonKinematicsBuildCfg(collapse_fixed_joints=False), articulation, "cpu"
        )

    baseline = build(4.0)
    repeated = build(4.0)
    changed_limit = build(5.0)
    changed_path = tmp_path / "scene_limits_changed.xml"
    changed_path.write_text(_TWO_JOINT_MJCF.replace('size="0.02 0.05"', 'size="0.03 0.05"', 1))
    changed_collider = build(4.0, changed_path)

    assert len(baseline.mechanics_identity_sha256) == 64
    assert baseline.mechanics_identity_sha256 == repeated.mechanics_identity_sha256
    assert baseline.mechanics_identity_sha256 != changed_limit.mechanics_identity_sha256
    assert baseline.mechanics_identity_sha256 != changed_collider.mechanics_identity_sha256


def test_from_articulation_keeps_parsed_limits_when_actuator_limits_are_unset(tmp_path: Path) -> None:
    """Unset implicit-actuator limits must retain the parsed simulator values."""
    path = _write_mjcf(tmp_path)
    parsed = NewtonKinematics(NewtonKinematicsCfg(mjcf_path=str(path), device="cpu"))
    articulation = _articulation(
        path,
        ImplicitActuatorCfg(
            joint_names_expr=["joint_.*"],
            velocity_limit_sim=None,
            effort_limit_sim=None,
            stiffness=0.0,
            damping=0.0,
        ),
    )

    resolved = NewtonKinematics.from_articulation(
        NewtonKinematicsBuildCfg(collapse_fixed_joints=False), articulation, "cpu"
    )

    np.testing.assert_array_equal(
        resolved.model.joint_velocity_limit.numpy(), parsed.model.joint_velocity_limit.numpy()
    )
    np.testing.assert_array_equal(resolved.model.joint_effort_limit.numpy(), parsed.model.joint_effort_limit.numpy())
    expected_velocity = np.abs(parsed.model.joint_velocity_limit.numpy())
    free_dofs = resolved.topology.joint_type[resolved.topology.dof_joint] == int(JointType.FREE)
    expected_velocity[free_dofs] = np.inf
    np.testing.assert_array_equal(resolved.topology.joint_velocity_lower, -expected_velocity)
    np.testing.assert_array_equal(resolved.topology.joint_velocity_upper, expected_velocity)
