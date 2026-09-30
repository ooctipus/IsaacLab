# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for ``NewtonActuator`` authoring with pure USD.

These run on an in-memory USD stage and intentionally do NOT launch Isaac Sim / Kit:
:func:`~isaaclab.sim.schemas.define_actuator_properties` authors prototypes before any
replication, so the clone plan is the only thing that knows which prims it has to reach.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.cloner import ClonePlan
from isaaclab.sim.schemas.schemas_actuators import define_actuator_properties

ROBOT_EXPR = "/World/envs/env_[^/]+/Robot"
"""Prim path expression an articulation cfg carries when it is scoped to the env namespace."""


def _make_articulation(stage: Usd.Stage, path: str) -> None:
    """Author an articulation root with one revolute joint at ``path``."""
    UsdPhysics.ArticulationRootAPI.Apply(UsdGeom.Xform.Define(stage, path).GetPrim())
    UsdPhysics.RevoluteJoint.Define(stage, f"{path}/joint_a")


@pytest.fixture
def stage_with_two_prototypes() -> Usd.Stage:
    """A stage holding the two prototypes a two-variant heterogeneous articulation spawns."""
    stage = Usd.Stage.CreateInMemory()
    _make_articulation(stage, "/World/envs/env_0/Robot")
    _make_articulation(stage, "/World/envs/env_1/Robot")
    return stage


def _author(monkeypatch: pytest.MonkeyPatch, stage: Usd.Stage, plan: ClonePlan | None) -> None:
    """Run the authoring step against ``stage`` with ``plan`` published on the simulation."""
    import isaaclab.sim as sim_module

    context = SimpleNamespace(cfg=SimpleNamespace(use_newton_actuators=True), get_clone_plan=lambda: plan)
    monkeypatch.setattr(sim_module, "SimulationContext", SimpleNamespace(instance=lambda: context))
    cfg = IdealPDActuatorCfg(joint_names_expr=["joint_a"], stiffness=10.0, damping=1.0, effort_limit=5.0)
    define_actuator_properties(ROBOT_EXPR, {"legs": cfg}, stage)


def _actuator_names(stage: Usd.Stage, root: str) -> list[str]:
    """Names of the ``NewtonActuator`` prims authored directly under ``root``."""
    return [
        child.GetName() for child in stage.GetPrimAtPath(root).GetChildren() if child.GetTypeName() == "NewtonActuator"
    ]


def test_authors_every_prototype_the_plan_names(stage_with_two_prototypes, monkeypatch):
    """A heterogeneous articulation has one prototype per variant, and all of them get authored.

    Resolving the expression off the stage instead picks whichever prototype comes first, so the
    environments cloned from the other variants silently run without the Lab-configured actuators.
    """
    # The third row is a variant no env drew, so its prototype was never spawned: visiting it
    # would raise on a prim that is not there.
    plan = ClonePlan(
        sources=("/World/envs/env_0/Robot", "/World/envs/env_1/Robot", "/World/envs/env_2/Robot"),
        destinations=("/World/envs/env_{}/Robot",) * 3,
        clone_mask=np.asarray([[True, False], [False, True], [False, False]], dtype=np.bool_),
        env_ids=np.asarray([0, 1], dtype=np.int64),
    )

    _author(monkeypatch, stage_with_two_prototypes, plan)

    assert _actuator_names(stage_with_two_prototypes, "/World/envs/env_0/Robot") == ["legs_joint_a_actuator"]
    assert _actuator_names(stage_with_two_prototypes, "/World/envs/env_1/Robot") == ["legs_joint_a_actuator"]


def test_rejects_authoring_without_a_clone_plan(stage_with_two_prototypes, monkeypatch):
    """Actuator authoring cannot recover prototype ownership from stage traversal order."""
    with pytest.raises(RuntimeError, match="requires an active clone plan"):
        _author(monkeypatch, stage_with_two_prototypes, None)

    assert _actuator_names(stage_with_two_prototypes, "/World/envs/env_0/Robot") == []
    assert _actuator_names(stage_with_two_prototypes, "/World/envs/env_1/Robot") == []
