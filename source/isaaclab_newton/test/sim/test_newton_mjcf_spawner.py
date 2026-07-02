# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Newton-native MJCF stage contract."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import torch
from isaaclab_newton.cloner.newton_clone_utils import (
    build_source_builders,
    rename_builder_labels,
    replicate_builder_mapping,
)
from isaaclab_newton.physics import NewtonManager, NewtonMJWarpManager
from isaaclab_newton.physics import newton_manager as newton_manager_module
from isaaclab_newton.physics.visualization_builder import build_visualization_builder_from_stage_envs
from isaaclab_newton.sim import NewtonMjcfFileCfg
from isaaclab_newton.sim.spawners.mjcf import (
    NEWTON_MJCF_ASSET_PATH_ATTR,
    NEWTON_MJCF_SELF_COLLISION_ATTR,
    spawn_newton_mjcf,
)
from newton import ModelBuilder
from newton.selection import ArticulationView
from newton.solvers import SolverMuJoCo

from pxr import Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.cloner import ClonePlan
from isaaclab.sim.spawners.from_files.from_files import spawn_from_mjcf


def test_core_and_newton_mjcf_configs_have_distinct_owners(tmp_path: Path):
    """Keep core conversion and Newton-native loading as explicit boundaries."""
    source = tmp_path / "robot.xml"
    source.write_text("<mujoco model='empty'><worldbody/></mujoco>")

    core_cfg = sim_utils.MjcfFileCfg(asset_path=str(source))
    native_cfg = NewtonMjcfFileCfg(asset_path=str(source), self_collision=False)
    assert str(core_cfg.func).endswith(":spawn_from_mjcf")
    assert core_cfg.func._resolve() is spawn_from_mjcf
    assert str(native_cfg.func).endswith(":spawn_newton_mjcf")
    assert native_cfg.func._resolve() is spawn_newton_mjcf

    stage = Usd.Stage.CreateInMemory()
    with sim_utils.use_stage(stage):
        prim = native_cfg.func(
            "/World/Robot",
            native_cfg,
            translation=(1.0, 2.0, 3.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
        )

    assert prim.GetTypeName() == "Xform"
    assert prim.GetAttribute(NEWTON_MJCF_ASSET_PATH_ATTR).Get().path == str(source)
    assert prim.GetAttribute(NEWTON_MJCF_SELF_COLLISION_ATTR).Get() is False
    assert not prim.GetChildren()


def test_native_mjcf_source_labels_bind_two_newton_worlds(tmp_path: Path):
    """Prefix native labels before replication so every world binds independently."""
    source = tmp_path / "walker.xml"
    source.write_text(
        """<mujoco model="walker">
  <worldbody>
    <body name="root">
      <freejoint/>
      <geom name="root_geom" type="sphere" size="0.1"/>
      <body name="link" pos="0 0 0.2">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom name="link_geom" type="capsule" size="0.05 0.1"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""
    )
    source_path = "/World/envs/env_0/Robot"
    stage = Usd.Stage.CreateInMemory()
    with sim_utils.use_stage(stage):
        cfg = NewtonMjcfFileCfg(asset_path=str(source))
        cfg.func(source_path, cfg, translation=(0.0, 0.0, 0.5))

    source_builders = build_source_builders(
        stage,
        (source_path,),
        lambda: ModelBuilder(up_axis="Z"),
        NewtonManager._import_stage,
        simplify_meshes=False,
    )
    source_builder = source_builders[source_path]
    assert source_builder.articulation_label == [f"{source_path}/walker"]
    assert source_builder.body_label == [
        f"{source_path}/walker/worldbody/root",
        f"{source_path}/walker/worldbody/root/link",
    ]
    assert tuple(source_builder.body_q[0].p) == (0.0, 0.0, 0.5)

    builder = ModelBuilder(up_axis="Z")
    mapping = torch.ones((1, 2), dtype=torch.bool)
    positions = torch.tensor(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)), dtype=torch.float32)
    orientations = torch.tensor(((0.0, 0.0, 0.0, 1.0),) * 2, dtype=torch.float32)
    replicate_builder_mapping(builder, (source_path,), mapping, positions, orientations, source_builders)
    rename_builder_labels(
        builder,
        (source_path,),
        ("/World/envs/env_{}/Robot",),
        torch.tensor((0, 1), dtype=torch.int64),
        mapping,
    )

    assert builder.articulation_label == [
        "/World/envs/env_0/Robot/walker",
        "/World/envs/env_1/Robot/walker",
    ]
    model = builder.finalize(device="cpu")
    view = ArticulationView(model, "/World/envs/env_*/Robot/walker")
    assert view.world_count == 2
    assert view.count == 2
    assert view.count_per_world == 1


def test_native_mjcf_import_uses_solver_owned_equality_policy():
    """Keep equality conversion in the active Newton solver boundary."""
    stage_info = object()
    with (
        mock.patch.object(newton_manager_module, "add_usd_with_scoped_custom_frequencies", return_value=stage_info),
        mock.patch.object(newton_manager_module, "add_native_mjcf_from_stage") as native_import,
    ):
        assert NewtonManager._import_stage(object(), object()) is stage_info
        assert native_import.call_args.kwargs["convert_mjc_equality_constraints"] is True

        native_import.reset_mock()
        assert NewtonMJWarpManager._import_stage(object(), object()) is stage_info
        assert native_import.call_args.kwargs["convert_mjc_equality_constraints"] is False


def test_native_mjcf_weld_has_one_solver_owned_constraint(tmp_path: Path):
    """Import one weld exactly once for Newton, MJWarp, and visualization."""
    source = tmp_path / "welded.xml"
    source.write_text(
        """<mujoco model="welded">
  <worldbody>
    <body name="first">
      <freejoint/>
      <geom type="sphere" size="0.1"/>
    </body>
    <body name="second" pos="0 0 0.2">
      <freejoint/>
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
  <equality>
    <weld name="pair" body1="first" body2="second"/>
  </equality>
</mujoco>"""
    )

    robot_path = "/World/envs/env_0/Robot"
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/envs")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    with sim_utils.use_stage(stage):
        cfg = NewtonMjcfFileCfg(asset_path=str(source))
        cfg.func(robot_path, cfg)

    def import_with(manager_type: type[NewtonManager]) -> ModelBuilder:
        builder = ModelBuilder(up_axis="Z")
        manager_type._register_builder_attributes(builder)
        manager_type._import_stage(builder, stage, root_path=robot_path)
        return builder

    equality_label = f"{robot_path}/welded/pair"
    ordinary = import_with(NewtonManager)
    ordinary_native_labels = ordinary.custom_attributes["mujoco:equality_constraint_label"].values
    ordinary_targets = ordinary.custom_attributes["mujoco:equality_constraint_target"].values
    assert ordinary_native_labels == [equality_label]
    assert [ordinary.joint_label[index] for index in ordinary_targets] == [equality_label]
    assert ordinary.joint_label.count(equality_label) == 1

    mjwarp = import_with(NewtonMJWarpManager)
    assert mjwarp.custom_attributes["mujoco:equality_constraint_label"].values == [equality_label]
    assert mjwarp.custom_attributes["mujoco:equality_constraint_target"].values == []
    assert equality_label not in mjwarp.joint_label

    def mjwarp_builder_factory() -> ModelBuilder:
        builder = ModelBuilder(up_axis="Z")
        NewtonMJWarpManager._register_builder_attributes(builder)
        return builder

    source_builders = build_source_builders(
        stage,
        (robot_path,),
        mjwarp_builder_factory,
        NewtonMJWarpManager._import_stage,
        simplify_meshes=False,
    )
    replicated = mjwarp_builder_factory()
    mapping = torch.ones((1, 2), dtype=torch.bool)
    positions = torch.tensor(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)), dtype=torch.float32)
    orientations = torch.tensor(((0.0, 0.0, 0.0, 1.0),) * 2, dtype=torch.float32)
    replicate_builder_mapping(
        replicated,
        (robot_path,),
        mapping,
        positions,
        orientations,
        source_builders,
    )
    rename_builder_labels(
        replicated,
        (robot_path,),
        ("/World/envs/env_{}/Robot",),
        torch.tensor((0, 1), dtype=torch.int64),
        mapping,
    )
    replicated_labels = replicated.custom_attributes["mujoco:equality_constraint_label"].values
    assert replicated_labels == [
        "/World/envs/env_0/Robot/welded/pair",
        "/World/envs/env_1/Robot/welded/pair",
    ]
    assert replicated.custom_attributes["mujoco:equality_constraint_target"].values == []
    assert all(label not in replicated.joint_label for label in replicated_labels)

    model = replicated.finalize(device="cpu")
    assert model.world_count == 2
    assert model.mujoco.equality_constraint_count == 2
    solver = SolverMuJoCo(model)
    # Separate-world MJWarp compiles one template constraint and maps one copy
    # to each finalized Newton world.
    assert solver.mj_model.neq == 1
    assert solver.mjw_model.neq == 1
    assert solver.mjc_eq_to_newton_eq.shape == (2, 1)
    assert solver.mjc_eq_to_newton_eq.numpy().tolist() == [[0], [1]]

    clone_plan = ClonePlan(
        sources=(robot_path,),
        destinations=("/World/envs/env_{}/Robot",),
        clone_mask=torch.tensor([[True]], dtype=torch.bool),
        env_ids=torch.tensor([0], dtype=torch.int64),
    )
    visualization = build_visualization_builder_from_stage_envs(
        stage,
        [(0, "/World/envs/env_0")],
        clone_plan,
    )
    assert visualization.custom_attributes["mujoco:equality_constraint_label"].values == []
    assert visualization.custom_attributes["mujoco:equality_constraint_target"].values == []
    assert equality_label not in visualization.joint_label
