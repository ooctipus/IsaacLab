# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulator-free Factory reset-state bank tests."""

from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg

from isaaclab_tasks.core.multi_task.factory.mdp.reset_state_command_cfg import FactoryResetStateTableCfg
from isaaclab_tasks.core.multi_task.factory.mdp.reset_state_task_table import (
    _build_reset_state_bank,
    _reset_state_layout,
    _task_table_view,
    _validate_reset_asset_owners,
    build_factory_reset_state_task_table,
    factory_family_quotas,
)
from isaaclab_tasks.core.multi_task.factory.retarget.cfg import (
    FactoryFamilyCfg,
    FactoryGeometryCfg,
    FactoryGraspTargetGenerateCfg,
)
from isaaclab_tasks.core.multi_task.factory.retarget.task_table_builder import _factory_joints_within_limit
from isaaclab_tasks.core.multi_task.factory_env_cfg import FactoryResetAssetsCfg
from isaaclab_tasks.core.multi_task.mdp.commands.state_command import (
    ResetStateBank,
    ResetStateLayout,
    TaskTableQuality,
)


def _pose(x: float) -> torch.Tensor:
    return torch.tensor([x, x + 0.1, x + 0.2, 0.0, 0.0, 0.0, 1.0])


def test_factory_joint_limits_map_dofs_to_generalized_coordinates() -> None:
    """Factory limits index velocity DoFs while candidate values index generalized coordinates."""
    geometry = SimpleNamespace(
        device="cpu",
        arm_coords=[2, 4],
        arm_dofs=[0, 1],
        kinematics=SimpleNamespace(
            topology=SimpleNamespace(
                joint_limit_lower=np.asarray((-1.0, -2.0), dtype=np.float32),
                joint_limit_upper=np.asarray((1.0, 2.0), dtype=np.float32),
            )
        ),
    )
    joint_q = torch.zeros(3, 5)
    joint_q[1, 2] = 0.91
    joint_q[2, 4] = -1.81

    result = _factory_joints_within_limit(geometry, joint_q, limit_ratio=0.9)

    assert result.tolist() == [True, False, False]


def test_grasp_seed_selection_preserves_unconstrained_roll() -> None:
    """Nearest geometric seeds must span roll about the pad-pair axis."""
    from isaaclab_tasks.core.multi_task.factory.retarget.samplers import GraspPairSampler, pair_features

    sampler = object.__new__(GraspPairSampler)
    sampler.device = torch.device("cpu")
    sampler.cfg = SimpleNamespace(seed_axis_scale=0.3)
    target_a = torch.tensor(((0.0, 0.05, 0.0),))
    target_b = torch.tensor(((0.0, -0.05, 0.0),))
    target_features = pair_features(target_a, target_b, sampler.cfg.seed_axis_scale)
    sampler.tpl_feats = target_features.expand(32, -1).clone()
    sampler.tpl_feats[:, 0] += torch.arange(32) * 1.0e-4
    sampler.tpl_approach = torch.cat(
        (
            torch.tensor(((0.0, 0.0, 1.0),)).expand(8, -1),
            torch.tensor(((1.0, 0.0, 0.0),)).expand(8, -1),
            torch.tensor(((0.0, 0.0, -1.0),)).expand(8, -1),
            torch.tensor(((-1.0, 0.0, 0.0),)).expand(8, -1),
        )
    ).contiguous()
    sampler.tpl_arm = torch.arange(32, dtype=torch.float32).unsqueeze(-1)

    _, _, seed_arm, _ = sampler.seed_targets(target_a, target_b, ik_seeds_per_grasp=4)

    assert set((seed_arm[:, 0].long() // 8).tolist()) == {0, 1, 2, 3}


def test_gripper_collision_probes_include_exact_pad_points(monkeypatch) -> None:
    """Final collision certification must include both exact pad points."""
    from isaaclab_tasks.core.multi_task.factory.retarget import model as factory_model

    geometry = object.__new__(factory_model.FactoryGeometry)
    geometry.model = SimpleNamespace(
        shape_source_ptr=SimpleNamespace(numpy=lambda: np.array((101, 102, 103), dtype=np.uint64)),
        shape_transform=SimpleNamespace(
            numpy=lambda: np.array(((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),) * 3, dtype=np.float32)
        ),
    )
    geometry.gripper_bodies = [9, 10, 11]
    geometry.pad_bodies = [10, 11]
    geometry.pad_offsets = torch.tensor(((0.1, 0.2, 0.3), (-0.1, -0.2, -0.3)))
    shape_for_body = {9: 0, 10: 1, 11: 2}
    monkeypatch.setattr(
        factory_model,
        "model_collision_shape_indices",
        lambda _model, body: np.array((shape_for_body[body],), dtype=np.int32),
    )
    monkeypatch.setattr(
        factory_model,
        "model_shape_surface_probes",
        lambda _model, shape, count, _rng: np.full((count, 3), float(shape), dtype=np.float32),
    )

    geometry._setup_gripper_probes(6, np.random.default_rng(7))

    np.testing.assert_allclose(geometry.gripper_probes[-2:], geometry.pad_offsets.numpy())
    np.testing.assert_array_equal(geometry.gripper_probe_bodies[-2:], geometry.pad_bodies)


def test_factory_table_builds_declared_entity_layout_without_mutating_config(monkeypatch) -> None:
    """Pure construction must retain declared order and canonical endpoint states."""
    from isaaclab_tasks.core.multi_task.factory.mdp import reset_state_task_table
    from isaaclab_tasks.core.multi_task.factory.retarget import task_table_builder

    class _FakeModel:
        def __init__(self) -> None:
            self.model = object()
            self.arm_coords = [0]
            self.arm_joint_names = ["joint_a"]
            self.finger_coords = [1]
            self.finger_joint_names = ["finger"]
            self.kinematics = SimpleNamespace(
                default_joint_q=np.array([0.25, 0.02]),
                find_joint_scalar_coordinates=lambda _pattern: ([0, 1], [0, 1], ["joint_a", "finger"]),
            )
            self.device = "cpu"

    class _FakeBuilder:
        received_kinematics_cfg = None
        received_cfg = None
        received_scene = None
        received_device = None
        result = None

        def __init__(self, kinematics_cfg, cfg, scene_cfg, device, families, rng) -> None:
            type(self).received_kinematics_cfg = kinematics_cfg
            type(self).received_cfg = cfg
            type(self).received_scene = scene_cfg
            type(self).received_device = device
            assert tuple(family.name for family in families) == ("assembly",)
            assert rng.torch.device == torch.device("cpu")
            self.geometry = _FakeModel()

        def build_family_table(self, rows_per_board: int, families, rng):
            assert rows_per_board == 2
            assert tuple(family.name for family in families) == ("assembly",)
            assert rng.torch.device == torch.device("cpu")
            type(self).result = SimpleNamespace(
                joint_q=torch.tensor([[0.0, 0.01], [0.1, 0.02], [0.2, 0.03], [0.3, 0.04]]),
                held_pose=torch.stack([_pose(1.0), _pose(2.0), _pose(3.0), _pose(4.0)]),
                board_pose=torch.stack([_pose(10.0), _pose(10.0), _pose(20.0), _pose(20.0)]),
                board_asset_poses={"fixed_asset": torch.stack([_pose(11.0), _pose(11.0), _pose(21.0), _pose(21.0)])},
                board_index=torch.tensor([0, 0, 1, 1]),
                tag=torch.tensor([0, 1, 0, 1]),
                tag_names=["assembled", "free"],
                family=torch.zeros(4, dtype=torch.long),
                task_family=torch.zeros(4, dtype=torch.long),
                task_family_names=("assembly",),
                pad_targets=torch.zeros(4, 2, 3),
                quality_names=("family_id",),
                quality=torch.zeros(4, 1),
                is_grasped=torch.ones(4, dtype=torch.bool),
            )
            return type(self).result

    monkeypatch.setattr(task_table_builder, "FactoryTaskTableBuilder", _FakeBuilder)
    view = object()
    monkeypatch.setattr(reset_state_task_table, "_task_table_view", lambda *_args: view)

    robot = ArticulationCfg(
        prim_path="/Robot",
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.0),
            joint_pos={"joint_a": 0.25, "finger": 0.02},
        ),
        actuators={},
    )

    def rigid(path: str) -> RigidObjectCfg:
        return RigidObjectCfg(prim_path=path, spawn=None)

    scene_cfg = SimpleNamespace(
        robot=robot,
        nistboard=rigid("/Board"),
        fixed_asset=rigid("/Bolt"),
        held_asset=rigid("/Nut"),
    )
    geometry_cfg = SimpleNamespace(
        board=SimpleNamespace(
            num_boards=2,
            board_asset_cfg=SimpleNamespace(name="nistboard"),
            fixed_asset_cfg=SimpleNamespace(name="fixed_asset"),
            fixed_asset_map={"fixed_asset": "fixture"},
        ),
        held_asset_cfg=SimpleNamespace(name="held_asset"),
        robot=SimpleNamespace(asset_cfg=SimpleNamespace(name="robot")),
    )
    table_cfg = SimpleNamespace(
        kinematics=object(),
        geometry=geometry_cfg,
        families=(SimpleNamespace(name="assembly"),),
        seed=13,
        rows_per_board=2,
        targets_per_board=1,
        state_table_fps_features=None,
        allowed_tag_pairs=None,
    )
    command_cfg = SimpleNamespace(
        reset_assets=("held_asset", "robot", "nistboard", "fixed_asset"),
        task_table=table_cfg,
    )

    ambient_rng_state = torch.random.get_rng_state()
    table = build_factory_reset_state_task_table(command_cfg, scene_cfg, "cpu")

    torch.testing.assert_close(torch.random.get_rng_state(), ambient_rng_state)
    assert _FakeBuilder.received_kinematics_cfg is table_cfg.kinematics
    assert _FakeBuilder.received_cfg is geometry_cfg
    assert _FakeBuilder.received_scene is scene_cfg
    assert _FakeBuilder.received_device == "cpu"
    assert table.states.layout.names == command_cfg.reset_assets
    assert table.states.layout.kinds == ("rigid_object", "articulation", "rigid_object", "rigid_object")
    assert table.states.layout.joint_names == ((), ("joint_a", "finger"), (), ())
    assert table.states.layout.joint_offsets == (0, 0, 2, 2, 2)
    assert table.view is view
    assert not hasattr(table, "state_data")
    assert not hasattr(table, "endpoint_indices")
    assert table.num_states == 4
    assert table.num_tasks == 4
    assert table.states.layout.entity_count == len(FactoryResetAssetsCfg().default) == 4

    held_index = table.states.layout.entity_index("held_asset")
    robot_index = table.states.layout.entity_index("robot")
    torch.testing.assert_close(table.states.root_pose[:, held_index], _FakeBuilder.result.held_pose)
    torch.testing.assert_close(
        table.states.root_pose[:, robot_index, :3],
        torch.tensor([[0.5, 0.0, 0.0]]).expand(4, -1),
    )
    torch.testing.assert_close(
        table.states.joint_position,
        torch.tensor([[0.0, 0.01], [0.1, 0.02], [0.2, 0.03], [0.3, 0.04]]),
    )
    spawn_rows, target_rows = table.gather(torch.arange(table.num_tasks))
    torch.testing.assert_close(table.state_board_indices[spawn_rows], table.state_board_indices[target_rows])
    assert not hasattr(table, "slot_indices")
    assert not hasattr(table, "task_tag_indices")


@pytest.mark.parametrize("asset_count", (1, 3))
def test_board_attached_assets_follow_sampled_board_transform(asset_count: int, monkeypatch) -> None:
    """Every mapped asset pose must be composed from the same sampled board pose."""
    from isaaclab_tasks.core.multi_task.factory.retarget import samplers

    sampler = object.__new__(samplers.HeldAssetPlacementSampler)
    sampler.device = torch.device("cpu")
    sampler.generator = torch.Generator().manual_seed(17)
    sampler.cfg = SimpleNamespace(
        board=SimpleNamespace(
            pose_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "yaw": (1.5707963, 1.5707963)},
            clear_tol=0.0,
            oversample=1.0,
        )
    )
    sampler.model = SimpleNamespace(static_obstacles={})
    sampler._board_init_pos = torch.tensor(((1.0, 2.0, 3.0),))
    sampler._board_init_quat = torch.tensor(((0.0, 0.0, 0.0, 1.0),))
    sampler._board_probes = torch.zeros(1, 3)
    sampler._board_asset_offsets = {
        f"attached_{index}": (
            torch.tensor(((float(index + 1), 0.0, 0.0),)),
            torch.tensor(((0.0, 0.0, 0.0, 1.0),)),
        )
        for index in range(asset_count)
    }
    sampler.model.board_group_names = ("nistboard", *tuple(sampler._board_asset_offsets))
    sampler._board_group_probes = tuple(torch.zeros(1, 3) for _ in sampler.model.board_group_names)
    sampler._board_group_edge_bodies = None
    sampler._board_group_edge_p0 = None
    sampler._board_group_edge_p1 = None
    monkeypatch.setattr(
        samplers,
        "points_min_sd",
        lambda points, *_args: torch.ones(points.shape[0], dtype=torch.float32),
    )
    monkeypatch.setattr(
        samplers,
        "edges_vs_posed_mesh_hit",
        lambda body_q, *_args: torch.zeros(body_q.shape[0], dtype=torch.bool),
    )
    monkeypatch.setattr(
        samplers,
        "grid_bucket_downsample",
        lambda _features, count, generator: torch.arange(count),
    )
    monkeypatch.setattr(
        samplers.HeldAssetPlacementSampler,
        "_default_robot_clear",
        lambda _self, _poses, _points, body_q, _tol: torch.ones(body_q.shape[0], dtype=torch.bool),
    )

    board_pose, attached_poses = sampler._sample_board(1)
    assert len(attached_poses) == asset_count
    for name, (offset_position, offset_rotation) in sampler._board_asset_offsets.items():
        expected_position, expected_rotation = math_utils.combine_frame_transforms(
            board_pose[:, :3], board_pose[:, 3:7], offset_position, offset_rotation
        )
        torch.testing.assert_close(attached_poses[name], torch.cat((expected_position, expected_rotation), dim=-1))


def test_six_entity_factory_bank_materializes_every_board_attached_asset() -> None:
    """Gear variants must store all six reset entities without scene-init fallback."""
    robot = ArticulationCfg(
        prim_path="/Robot",
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(joint_pos={"joint_a": 0.25, "finger": 0.02}),
        actuators={},
    )

    def rigid(path: str) -> RigidObjectCfg:
        return RigidObjectCfg(prim_path=path, spawn=None)

    scene_cfg = SimpleNamespace(
        robot=robot,
        nistboard=rigid("/Board"),
        fixed_asset=rigid("/Fixed"),
        held_asset=rigid("/Held"),
        medium_gear=rigid("/Medium"),
        large_gear=rigid("/Large"),
    )
    reset_assets = FactoryResetAssetsCfg().gear_mesh_small
    layout = _reset_state_layout(scene_cfg, reset_assets, "robot", ("joint_a", "finger"))
    result = SimpleNamespace(
        joint_q=torch.tensor(((0.1, 0.02), (0.2, 0.03))),
        held_pose=torch.stack((_pose(1.0), _pose(2.0))),
        board_pose=torch.stack((_pose(10.0), _pose(20.0))),
        board_asset_poses={
            "fixed_asset": torch.stack((_pose(11.0), _pose(21.0))),
            "medium_gear": torch.stack((_pose(12.0), _pose(22.0))),
            "large_gear": torch.stack((_pose(13.0), _pose(23.0))),
        },
    )
    geometry_cfg = SimpleNamespace(
        held_asset_cfg=SimpleNamespace(name="held_asset"),
        board=SimpleNamespace(board_asset_cfg=SimpleNamespace(name="nistboard")),
        robot=SimpleNamespace(asset_cfg=SimpleNamespace(name="robot")),
    )

    bank = _build_reset_state_bank(geometry_cfg, scene_cfg, layout, result, (0, 1))

    assert bank.layout.entity_count == len(reset_assets) == 6
    for name, expected in result.board_asset_poses.items():
        torch.testing.assert_close(bank.root_pose[:, bank.layout.entity_index(name)], expected)


def test_six_entity_factory_view_retains_every_rigid_reset_asset() -> None:
    """The shared inspector view must map all five rigid entities in a gear task."""
    import newton

    robot_builder = newton.ModelBuilder()
    robot_body = robot_builder.add_body(label="robot")
    robot_builder.add_shape_box(robot_body, hx=0.1, hy=0.1, hz=0.1)
    triangle_vertices = np.array(((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0)), dtype=np.float32)
    triangle_faces = np.array(((0, 1, 2),), dtype=np.int32)
    rigid_names = ("nistboard", "fixed_asset", "held_asset", "medium_gear", "large_gear")
    layout = ResetStateLayout(
        names=("robot", *rigid_names),
        kinds=("articulation", *("rigid_object",) * len(rigid_names)),
        joint_names=((),) * 6,
        joint_offsets=(0,) * 7,
    )
    root_pose = torch.zeros(1, 6, 7)
    root_pose[..., 6] = 1.0
    states = ResetStateBank(layout, root_pose, torch.zeros(1, 6, 6), torch.empty(1, 0), torch.empty(1, 0))
    geometry = SimpleNamespace(
        obstacle_geom={},
        builder=robot_builder,
        nq=len(robot_builder.joint_q),
        model=object(),
        kinematics=SimpleNamespace(find_joint_scalar_coordinates=lambda _pattern: ([], [], [])),
        board_verts=triangle_vertices,
        board_faces=triangle_faces,
        held_verts=triangle_vertices,
        held_faces=triangle_faces,
        board_asset_geom={
            name: (triangle_vertices, triangle_faces) for name in ("fixed_asset", "medium_gear", "large_gear")
        },
    )
    table_builder = SimpleNamespace(
        geometry=geometry,
        cfg=SimpleNamespace(
            board=SimpleNamespace(board_asset_cfg=SimpleNamespace(name="nistboard")),
            held_asset_cfg=SimpleNamespace(name="held_asset"),
            robot=SimpleNamespace(asset_cfg=SimpleNamespace(name="robot")),
        ),
    )

    view = _task_table_view(
        table_builder,
        states,
        torch.tensor((0,), dtype=torch.int64),
        torch.tensor((0,), dtype=torch.int64),
        torch.zeros(1, 2, 3),
        TaskTableQuality(("fixture",), torch.zeros(1, 1), scope="state"),
    )

    assert view.kinematic_view.root_entity_names == rigid_names
    assert view.kinematic_view.root_state_indices.tolist() == [1, 2, 3, 4, 5]


def _quota_family(name: str, fraction: float, candidate_oversample: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(name=name, fraction=fraction, candidate_oversample=candidate_oversample)


def test_factory_family_quotas_use_stable_largest_remainder() -> None:
    """Integer quotas must preserve the exact row budget and declaration-order ties."""
    families = (
        _quota_family("assembly_grasp", 0.25),
        _quota_family("assembly_approach", 0.25),
        _quota_family("support_grasp", 0.1),
        _quota_family("support_approach", 0.1),
        _quota_family("free_grasp", 0.3),
    )
    assert factory_family_quotas(8, families) == (2, 2, 1, 1, 2)
    tied = (_quota_family("first", 0.5), _quota_family("second", 0.25), _quota_family("third", 0.25))
    assert factory_family_quotas(2, tied) == (1, 1, 0)


@pytest.mark.parametrize(
    ("rows_per_board", "families", "message"),
    (
        (0, (_quota_family("only", 1.0),), "positive integer"),
        (2, (), "at least one"),
        (2, (_quota_family("same", 0.5), _quota_family("same", 0.5)), "nonempty and unique"),
        (2, (_quota_family("left", 0.4), _quota_family("right", 0.4)), "sum to one"),
        (2, (_quota_family("only", 1.0, 0.5),), "at least one"),
    ),
)
def test_factory_family_quotas_reject_invalid_declarations(rows_per_board, families, message: str) -> None:
    """Quota policy must fail at the public task-table boundary."""
    with pytest.raises(ValueError, match=message):
        factory_family_quotas(rows_per_board, families)


def test_factory_architecture_rejects_removed_global_policy_and_facades() -> None:
    """Factory ownership must remain family-local with no compatibility facade."""
    factory_root = Path(__file__).parents[1] / "factory"
    assert not (factory_root / "retarget" / "pipeline.py").exists()
    assert not (factory_root / "retarget" / "objectives.py").exists()
    cfg_source = (factory_root / "retarget" / "cfg.py").read_text()
    builder_source = (factory_root / "retarget" / "task_table_builder.py").read_text()
    sampler_source = (factory_root / "retarget" / "samplers.py").read_text()
    assert ".pipeline:" not in cfg_source
    assert "FactoryReach" not in cfg_source + builder_source
    environment_source = (factory_root.parent / "factory_env_cfg.py").read_text()
    assert "factory_reach" not in cfg_source + builder_source
    assert 'tag_indices["on_table"]' not in builder_source
    assert 'tag_indices["in_air"]' not in builder_source
    assert "reaching_" not in builder_source
    assert {field.name for field in fields(FactoryGeometryCfg)}.isdisjoint({"scene", "device"})
    assert "finger_squeeze" not in {field.name for field in fields(FactoryGraspTargetGenerateCfg)}
    assert {field.name for field in fields(FactoryResetStateTableCfg)}.isdisjoint({"nut_bounds", "finger_squeeze"})
    assert "IKObjectiveMeshCollisionCfg" not in environment_source + builder_source
    assert "refine_iterations" not in environment_source + cfg_source + builder_source
    assert "gripper_probe_contact_slots" not in builder_source + (factory_root / "retarget" / "model.py").read_text()
    assert all(
        name not in sampler_source
        for name in ("board_stats", "candidate_pair_a", "candidate_pair_b", "surface_points", "pair_aperture")
    )
    assert "reach" not in {field.name for field in fields(FactoryFamilyCfg)}


def test_factory_solve_rejects_hidden_candidate_compaction(monkeypatch) -> None:
    """Generation fixes cardinality; criteria, not the solver, own acceptance."""
    from isaaclab_tasks.core.multi_task.factory.retarget import task_table_builder

    candidates = SimpleNamespace(t_plus=torch.zeros(4, 3), joint_q=None)

    def compact(values, _cfg):
        values.joint_q = torch.zeros(3, 2)
        return values

    monkeypatch.setattr(task_table_builder, "_factory_solve_family", compact)
    with pytest.raises(RuntimeError, match="preserve candidate count"):
        task_table_builder.factory_solve_ik(SimpleNamespace(), candidates)


def test_factory_solve_rejects_unknown_generated_target_bind() -> None:
    """A misspelled objective target must fail before Newton solver construction."""
    from isaaclab_tasks.core.multi_task.factory.retarget import task_table_builder
    from isaaclab_tasks.core.multi_task.kinematics.ik_objectives.cfg import BodyPointsCfg, IKObjectivePositionCfg
    from isaaclab_tasks.core.multi_task.kinematics.ik_objectives.context import IKObjectiveBuild

    def build(cfg, _context):
        return IKObjectiveBuild((SimpleNamespace(),), target_bind=cfg.target_bind)

    objective = IKObjectivePositionCfg(
        class_type=build,
        name="grasp",
        current=BodyPointsCfg(asset="robot", bodies=("left_finger", "right_finger")),
        target_bind="generated.grasp_point_typo",
    )
    kinematics = SimpleNamespace(body_names=("left_finger", "right_finger"))
    geometry = SimpleNamespace(
        cfg=SimpleNamespace(robot=SimpleNamespace(asset_cfg=SimpleNamespace(name="robot"))),
        device=torch.device("cpu"),
        pad_bodies=(0, 1),
        pad_offsets=torch.zeros(2, 3),
    )
    candidates = SimpleNamespace(
        kinematics=kinematics,
        geometry=geometry,
        t_plus=torch.zeros(1, 3),
        t_minus=torch.zeros(1, 3),
    )

    with pytest.raises(ValueError, match="Unknown Factory objective target binding"):
        task_table_builder._factory_solve_targets(
            candidates,
            SimpleNamespace(objectives=(objective,)),
            torch.empty(1, 0),
        )


def test_factory_reset_assets_require_exact_unique_owners() -> None:
    """Every reset entity must have exactly one visible Factory geometry owner."""
    geometry = SimpleNamespace(
        robot=SimpleNamespace(asset_cfg=SimpleNamespace(name="robot")),
        held_asset_cfg=SimpleNamespace(name="held_asset"),
        board=SimpleNamespace(
            board_asset_cfg=SimpleNamespace(name="nistboard"),
            fixed_asset_cfg=SimpleNamespace(name="fixed_asset"),
            fixed_asset_map={"fixed_asset": "fixture", "medium_gear": "gear"},
        ),
    )

    _validate_reset_asset_owners(("robot", "nistboard", "fixed_asset", "medium_gear", "held_asset"), geometry)
    with pytest.raises(ValueError, match="duplicate names"):
        _validate_reset_asset_owners(("robot", "nistboard", "fixed_asset", "held_asset", "held_asset"), geometry)
    with pytest.raises(ValueError, match="missing=.*medium_gear.*extra=.*table"):
        _validate_reset_asset_owners(("robot", "nistboard", "fixed_asset", "held_asset", "table"), geometry)


def test_board_group_collision_includes_attached_gear(monkeypatch) -> None:
    """Board qualification must reject an attached gear collision missed by the board shell."""
    from isaaclab_tasks.core.multi_task.factory.retarget import samplers

    sampler = object.__new__(samplers.HeldAssetPlacementSampler)
    sampler.device = torch.device("cpu")
    sampler.generator = torch.Generator().manual_seed(19)
    sampler.cfg = SimpleNamespace(
        board=SimpleNamespace(
            pose_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            clear_tol=0.0,
            oversample=1.0,
        )
    )
    sampler.model = SimpleNamespace(
        board_group_names=("nistboard", "medium_gear"),
        static_obstacles={"fixture": SimpleNamespace(id=7)},
    )
    sampler._board_init_pos = torch.zeros(1, 3)
    sampler._board_init_quat = torch.tensor(((0.0, 0.0, 0.0, 1.0),))
    sampler._board_asset_offsets = {
        "medium_gear": (torch.tensor(((10.0, 0.0, 0.0),)), sampler._board_init_quat.clone())
    }
    sampler._board_group_probes = (torch.zeros(1, 3), torch.zeros(1, 3))
    sampler._board_group_edge_bodies = None
    sampler._board_group_edge_p0 = None
    sampler._board_group_edge_p1 = None

    def group_distance(points, *_args):
        collides = points[..., 0].amax(dim=-1) > 5.0
        return torch.where(collides, -torch.ones(points.shape[0]), torch.ones(points.shape[0]))

    monkeypatch.setattr(samplers, "points_min_sd", group_distance)
    monkeypatch.setattr(
        samplers,
        "edges_vs_posed_mesh_hit",
        lambda body_q, *_args: torch.zeros(body_q.shape[0], dtype=torch.bool),
    )

    monkeypatch.setattr(
        samplers.HeldAssetPlacementSampler,
        "_default_robot_clear",
        lambda _self, _poses, _points, body_q, _tol: torch.ones(body_q.shape[0], dtype=torch.bool),
    )
    with pytest.raises(RuntimeError, match="board-group sampling"):
        sampler._sample_board(1)


def test_board_library_rejects_default_robot_collision_before_fps(monkeypatch) -> None:
    """Board selection must reject complete default-robot collisions before FPS."""
    from isaaclab_tasks.core.multi_task.factory.retarget import samplers

    sampler = object.__new__(samplers.HeldAssetPlacementSampler)
    sampler.device = torch.device("cpu")
    sampler.generator = torch.Generator().manual_seed(23)
    sampler.cfg = SimpleNamespace(
        board=SimpleNamespace(
            pose_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            clear_tol=0.0,
            oversample=1.0,
        )
    )
    sampler.model = SimpleNamespace(board_group_names=("nistboard",), static_obstacles={})
    sampler._board_init_pos = torch.zeros(1, 3)
    sampler._board_init_quat = torch.tensor(((0.0, 0.0, 0.0, 1.0),))
    sampler._board_asset_offsets = {}
    sampler._board_group_probes = (torch.zeros(1, 3),)
    sampler._board_group_edge_bodies = None
    sampler._board_group_edge_p0 = None
    sampler._board_group_edge_p1 = None
    monkeypatch.setattr(
        samplers.HeldAssetPlacementSampler,
        "_default_robot_clear",
        lambda _self, _poses, _points, body_q, _tol: torch.zeros(body_q.shape[0], dtype=torch.bool),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="board-group sampling"):
        sampler._sample_board(1)


def test_factory_collision_helpers_launch_on_declared_device() -> None:
    """Every Factory collision wrapper must launch on its declared device."""
    from isaaclab_tasks.core.multi_task.factory.retarget import criteria

    function_names = {
        "collision_min_sd",
        "posed_collision_min_sd",
        "points_min_sd",
        "points_vs_body_meshes_min_sd",
        "edges_vs_posed_mesh_hit",
        "posed_edges_vs_body_meshes_hit",
        "self_collision_min_sd",
    }
    tree = ast.parse(Path(criteria.__file__).read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert function_names <= functions.keys()
    for name in function_names:
        launches = [
            node
            for node in ast.walk(functions[name])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "wp"
            and node.func.attr == "launch"
        ]
        assert len(launches) == 1, name
        assert any(
            keyword.arg == "device" and isinstance(keyword.value, ast.Name) and keyword.value.id == "device"
            for keyword in launches[0].keywords
        ), name


def test_factory_selection_meets_per_board_quota_and_drops_partial_boards() -> None:
    """Selection requests a per-board quota and leaves cross-family qualification global."""

    from isaaclab_tasks.core.multi_task.factory.retarget.task_table_builder import factory_fps_selection
    from isaaclab_tasks.core.multi_task.mdp.commands.state_command import make_task_table_rng

    row_count = 8
    held_pose = torch.zeros(row_count, 7)
    held_pose[:, 0] = torch.arange(row_count, dtype=torch.float32)
    held_pose[:, 6] = 1.0
    ee_approach = torch.zeros(row_count, 3)
    ee_approach[:, 2] = 1.0
    candidates = SimpleNamespace(
        geometry=SimpleNamespace(device=torch.device("cpu"), ee_body=0),
        joint_q=torch.zeros(row_count, 1),
        board_library=(torch.zeros(2, 7), {}),
        board_index=torch.tensor((0, 0, 0, 0, 1, 1, 1, 1)),
        held_pose=held_pose,
        ee_approach=ee_approach,
        tag=torch.zeros(row_count, dtype=torch.long),
        tag_names=("fixture",),
    )
    cfg = SimpleNamespace(
        position_frame="world",
        position_axes=(0, 1, 2),
        position_weight=1.0,
        approach_weight=0.0,
        tag_weight=0.0,
    )

    selected = factory_fps_selection(
        cfg,
        candidates,
        torch.ones(row_count, dtype=torch.bool),
        2,
        make_task_table_rng(17, "cpu"),
    )

    assert selected.numel() == 4
    torch.testing.assert_close(torch.bincount(candidates.board_index[selected], minlength=2), torch.tensor((2, 2)))
    partial = torch.tensor((True, True, True, True, True, False, False, False))
    selected = factory_fps_selection(cfg, candidates, partial, 2, make_task_table_rng(17, "cpu"))
    assert selected.numel() == 2
    assert candidates.board_index[selected].eq(0).all()
