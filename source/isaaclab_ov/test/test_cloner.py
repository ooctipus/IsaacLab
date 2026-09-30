# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for OvPhysX cloning."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from isaaclab_ov.cloner import OvReplicateContext, ovphysx_replicate
from isaaclab_ov.cloner.replicate import _expand_for_ovstage

from pxr import Gf, Sdf, Usd, UsdGeom

from isaaclab.sim import SimulationContext


@pytest.fixture(autouse=True)
def active_simulation(monkeypatch):
    """Provide the simulation-scoped resource consumed by the public raw clone function."""
    simulation = SimpleNamespace(stage=Usd.Stage.CreateInMemory())
    context = OvReplicateContext(simulation)
    simulation.get_or_create_backend = lambda backend_type, *_args: context
    monkeypatch.setattr(SimulationContext, "instance", staticmethod(lambda: simulation))
    return context._direct_physics_rows


def _pose_matrix(position: tuple[float, float, float], quaternion: tuple[float, float, float, float]) -> Gf.Matrix4d:
    """Build a USD pose matrix from an xyzw quaternion."""
    matrix = Gf.Matrix4d(1.0)
    matrix.SetTranslateOnly(Gf.Vec3d(*position))
    matrix.SetRotateOnly(Gf.Quatd(quaternion[3], Gf.Vec3d(*quaternion[:3])))
    return matrix


def test_nested_clone_uses_final_target_pose(active_simulation):
    """Nested clone rows keep their source-local pose under the target environment."""
    half_sqrt_two = math.sqrt(0.5)
    source_half_angle_sin = 0.5
    source_half_angle_cos = math.sqrt(0.75)
    target_half_angle_sin = math.sin(math.pi / 8.0)
    target_half_angle_cos = math.cos(math.pi / 8.0)
    stage = Usd.Stage.CreateInMemory()

    source_env = UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    source_env.AddTransformOp().Set(_pose_matrix((4.0, 5.0, 6.0), (0.0, half_sqrt_two, 0.0, half_sqrt_two)))
    source_row = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot")
    source_row.AddTransformOp().Set(
        _pose_matrix((0.0, 1.0, 2.0), (source_half_angle_sin, 0.0, 0.0, source_half_angle_cos))
    )

    ovphysx_replicate(
        stage,
        sources=["/World/envs/env_0/Robot", "/World/envs/env_9/Inactive"],
        destinations=["/World/envs/env_{}/Robot", "/World/envs/env_{}/Inactive"],
        env_ids=np.array([0, 1], dtype=np.int64),
        mapping=np.array([[True, True], [False, False]], dtype=np.bool_),
        positions=np.array([[4.0, 5.0, 6.0], [10.0, 20.0, 30.0]], dtype=np.float32),
        quaternions=np.array(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, target_half_angle_sin, target_half_angle_cos]],
            dtype=np.float32,
        ),
    )

    expected_orientation = np.array(
        [
            target_half_angle_cos * source_half_angle_sin,
            target_half_angle_sin * source_half_angle_sin,
            target_half_angle_sin * source_half_angle_cos,
            target_half_angle_cos * source_half_angle_cos,
        ],
        dtype=np.float32,
    )
    expected_transform = (10.0 - half_sqrt_two, 20.0 + half_sqrt_two, 32.0, *expected_orientation.tolist())

    assert len(active_simulation) == 1
    pending_source, pending_targets, pending_transforms = active_simulation[0]
    assert pending_source == "/World/envs/env_0/Robot"
    assert pending_targets == ("/World/envs/env_1/Robot",)
    assert len(pending_transforms) == 1
    assert pending_transforms[0][:3] == pytest.approx(expected_transform[:3])
    orientation = np.asarray(pending_transforms[0][3:], dtype=np.float32)
    if np.dot(orientation, expected_orientation) < 0.0:
        orientation = -orientation
    assert orientation.tolist() == pytest.approx(expected_orientation.tolist())


def test_raw_replicate_rejects_invalid_source_prim():
    """Active clone rows require a valid source prim."""
    stage = Usd.Stage.CreateInMemory()
    with pytest.raises(ValueError, match="/World/envs/env_0/Robot"):
        ovphysx_replicate(
            stage,
            sources=["/World/envs/env_0/Robot"],
            destinations=["/World/envs/env_{}/Robot"],
            env_ids=np.array([0, 1], dtype=np.int64),
            mapping=np.array([[True, True]], dtype=np.bool_),
        )


def test_raw_replicate_rejects_invalid_source_anchor():
    """Active nested clone rows require a valid source-environment anchor."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot")

    class StageWithoutAnchor:
        def GetPrimAtPath(self, path):
            if str(path) == "/World/envs/env_0":
                return Usd.Prim()
            return stage.GetPrimAtPath(path)

    with pytest.raises(ValueError, match="/World/envs/env_0"):
        ovphysx_replicate(
            StageWithoutAnchor(),
            sources=["/World/envs/env_0/Robot"],
            destinations=["/World/envs/env_{}/Robot"],
            env_ids=np.array([0, 1], dtype=np.int64),
            mapping=np.array([[True, True]], dtype=np.bool_),
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [("positions", np.zeros((1, 3), dtype=np.float32)), ("quaternions", np.zeros((1, 4), dtype=np.float32))],
)
def test_raw_replicate_rejects_pose_array_missing_selected_environment(name, value):
    """Provided pose arrays include every selected environment."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot")
    with pytest.raises(ValueError, match=name):
        ovphysx_replicate(
            stage,
            sources=["/World/envs/env_0/Robot"],
            destinations=["/World/envs/env_{}/Robot"],
            env_ids=np.array([0, 1], dtype=np.int64),
            mapping=np.array([[True, True]], dtype=np.bool_),
            **{name: value},
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [("positions", np.zeros((2, 2), dtype=np.float32)), ("quaternions", np.zeros((2, 3), dtype=np.float32))],
)
def test_raw_replicate_rejects_malformed_pose_array(name, value):
    """Provided pose arrays use the documented component counts."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot")
    with pytest.raises(ValueError, match=rf"{name} must have shape"):
        ovphysx_replicate(
            stage,
            sources=["/World/envs/env_0/Robot"],
            destinations=["/World/envs/env_{}/Robot"],
            env_ids=np.array([0, 1], dtype=np.int64),
            mapping=np.array([[True, True]], dtype=np.bool_),
            **{name: value},
        )


_OVSTAGE_SNAPSHOT = """#usda 1.0
def Xform "World"
{
    def Xform "envs"
    {
        def Xform "env_0"
        {
            def Xform "Robot"
            {
                def Material "glass"
                {
                    token outputs:mdl:surface.connect = </World/envs/env_0/Robot/glass/Shader.outputs:out>
                    def Shader "Shader"
                    {
                        token outputs:out
                    }
                }
            }
        }
        def Xform "env_1"
        {
            def Xform "Robot" (instanceable = true)
            {
                rel material:binding = </World/envs/env_1/Robot/glass>
            }
        }
    }
}
"""


def test_ovstage_snapshot_expands_clone_rows_and_instancing():
    """Cloned materials must own their shaders and instanced prims must be expanded."""
    layer = Sdf.Layer.CreateAnonymous(".usda")
    layer.ImportFromString(
        _expand_for_ovstage(_OVSTAGE_SNAPSHOT, [("/World/envs/env_0/Robot/glass", ["/World/envs/env_1/Robot/glass"])])
    )

    source = layer.GetAttributeAtPath("/World/envs/env_0/Robot/glass.outputs:mdl:surface")
    target = layer.GetAttributeAtPath("/World/envs/env_1/Robot/glass.outputs:mdl:surface")
    assert list(source.connectionPathList.explicitItems) == [
        Sdf.Path("/World/envs/env_0/Robot/glass/Shader.outputs:out")
    ]
    assert list(target.connectionPathList.explicitItems) == [
        Sdf.Path("/World/envs/env_1/Robot/glass/Shader.outputs:out")
    ]
    assert layer.GetPrimAtPath("/World/envs/env_1/Robot/glass/Shader") is not None
    assert not layer.GetPrimAtPath("/World/envs/env_1/Robot").HasInfo("instanceable")
