# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for task-level FeatherPGS presets."""

import pytest
from isaaclab_newton.physics import FeatherPGSSolverCfg, NewtonCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.core.lift.lift_env_cfg import PhysicsCfg as LiftPhysicsCfg
from isaaclab_tasks.core.reach.reach_env_cfg import ReachPhysicsCfg
from isaaclab_tasks.core.velocity.velocity_env_cfg import RoughPhysicsCfg
from isaaclab_tasks.utils.hydra import collect_presets, resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_LOCOMOTION_TASKS = [
    "IsaacContrib-Velocity-Flat-AnymalB",
    "IsaacContrib-Velocity-Rough-AnymalB",
    "IsaacContrib-Velocity-Flat-AnymalC",
    "IsaacContrib-Velocity-Rough-AnymalC",
    "Isaac-Velocity-Flat-AnymalD",
    "Isaac-Velocity-Rough-AnymalD",
    "IsaacContrib-Velocity-Flat-UnitreeA1",
    "IsaacContrib-Velocity-Rough-UnitreeA1",
    "IsaacContrib-Velocity-Flat-UnitreeGo1",
    "IsaacContrib-Velocity-Rough-UnitreeGo1",
    "Isaac-Velocity-Flat-UnitreeGo2",
    "Isaac-Velocity-Rough-UnitreeGo2",
    "Isaac-Velocity-Flat-Cassie",
    "Isaac-Velocity-Rough-Cassie",
    "Isaac-Velocity-Flat-G1",
    "Isaac-Velocity-Rough-G1",
    "Isaac-Velocity-Flat-H1",
    "Isaac-Velocity-Rough-H1",
]

_OTHER_TASKS = [
    "Isaac-Reach-Franka",
    "Isaac-Reach-UR10",
    "Isaac-Cartpole",
    "Isaac-Cartpole-Direct",
    "Isaac-Ant",
    "Isaac-Ant-Direct",
    "Isaac-Reorient-Cube-Allegro-Direct",
    "Isaac-Reorient-Cube-Shadow-Direct",
    "Isaac-Humanoid",
    "Isaac-Humanoid-Direct",
    "Isaac-Open-Drawer-Franka",
    "Isaac-Reorient-KukaAllegro",
    "Isaac-Lift-KukaAllegro",
]


@pytest.mark.parametrize("task_name", _LOCOMOTION_TASKS + _OTHER_TASKS)
def test_feather_pgs_task_preset_resolves_to_solver(task_name: str) -> None:
    """Resolve every supported task to the FeatherPGS manager configuration."""
    cfg = resolve_presets(load_cfg_from_registry(task_name, "env_cfg_entry_point"), {"feather_pgs"})

    assert collect_presets(cfg) == {}
    assert isinstance(cfg.sim.physics, NewtonCfg)
    assert isinstance(cfg.sim.physics.solver_cfg, FeatherPGSSolverCfg)


@pytest.mark.parametrize("task_name", _LOCOMOTION_TASKS)
def test_locomotion_tasks_use_shared_feather_pgs_preset(task_name: str) -> None:
    """Keep locomotion solver ownership at the shared velocity composition root."""
    raw_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    assert isinstance(raw_cfg.sim.physics, RoughPhysicsCfg)

    cfg = resolve_presets(raw_cfg, {"feather_pgs"})
    solver = cfg.sim.physics.solver_cfg
    assert solver.pgs_iterations == 8
    assert solver.dense_max_constraints == 64
    assert cfg.sim.physics.num_substeps == 2
    assert cfg.sim.physics.default_shape_cfg.margin == pytest.approx(0.01)


@pytest.mark.parametrize(
    ("task_name", "physics_cfg_type"),
    [
        ("Isaac-Reach-Franka", ReachPhysicsCfg),
        ("Isaac-Reach-UR10", ReachPhysicsCfg),
        ("Isaac-Reorient-KukaAllegro", LiftPhysicsCfg),
        ("Isaac-Lift-KukaAllegro", LiftPhysicsCfg),
    ],
)
def test_robot_variants_reuse_family_physics_preset(task_name: str, physics_cfg_type: type) -> None:
    """Keep robot variants on their task family's existing physics preset class."""
    cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")

    assert isinstance(cfg.sim.physics, physics_cfg_type)
