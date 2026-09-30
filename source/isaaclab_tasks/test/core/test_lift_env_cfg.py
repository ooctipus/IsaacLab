# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavioral tests for the unified dexterous Lift and Reorient tasks."""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isaaclab import cloner
from isaaclab.cloner import ClonePlan
from isaaclab.cloner.clone_plan import ArticulationLayout, FrameLayout, GeometryLayout, RigidBodyLayout
from isaaclab.managers import CommandTerm, ObservationTermCfg, SceneEntityCfg

from isaaclab_tasks.core.lift import mdp
from isaaclab_tasks.core.lift.config.kuka_allegro.kuka_allegro_env_cfg import KukaAllegroSceneCfg
from isaaclab_tasks.core.lift.mdp import utils as lift_utils
from isaaclab_tasks.core.lift.mdp.commands.pose_commands import (
    CableUniformPoseCommand,
    DeformableUniformPoseCommand,
    ObjectUniformPoseCommand,
)
from isaaclab_tasks.utils.hydra import resolve_presets

lift_events = importlib.import_module("isaaclab_tasks.core.lift.mdp.events")


class _MarkerSpy:
    def __init__(self, _cfg=None):
        self.calls: list[tuple[tuple, dict]] = []

    def set_visibility(self, _visible: bool) -> None:
        pass

    def visualize(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))


class _FakeScene(dict):
    def __init__(self, environment_ids: torch.Tensor, **assets):
        super().__init__(assets)
        self._ALL_INDICES = environment_ids
        self.env_origins = torch.zeros((len(environment_ids), 3))


def test_conditional_reset_reconciles_before_state_dependent_terms(monkeypatch: pytest.MonkeyPatch) -> None:
    """State-dependent terms read reconciled state without forcing another global forward."""
    calls = []
    scene = SimpleNamespace(clone_plan=SimpleNamespace(clone_mask=torch.ones((1, 2), dtype=torch.bool)))
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        scene=scene,
        sim=SimpleNamespace(forward=lambda: calls.append("forward")),
    )
    reset = object.__new__(lift_events.conditional_reset)
    reset._prefilled = False
    reset._buffer = torch.empty(0, 0)
    reset._descriptor = torch.empty(0, 0)
    reset._reset_assets = []
    reset._group = torch.empty(0, dtype=torch.long)
    reset._fill = torch.empty(0, dtype=torch.long)
    reset._monitor = None
    reset._success_term = None
    reset._playing_row = torch.full((env.num_envs,), -1, dtype=torch.long)

    def criterion(*_args, **_kwargs):
        calls.append("criterion")
        return torch.ones(env.num_envs, dtype=torch.bool)

    monkeypatch.setattr(
        lift_events,
        "get_reset_state",
        lambda _env, ids, *_args, **_kwargs: torch.zeros((len(ids), 1)),
    )
    monkeypatch.setattr(lift_events, "set_reset_state", lambda *_args, **_kwargs: calls.append("restore"))

    reset(
        env,
        torch.arange(env.num_envs),
        terms={"write": SimpleNamespace(func=lambda *_args, **_kwargs: calls.append("write"), params={})},
        state_dependent_terms={
            "dependent": SimpleNamespace(func=lambda *_args, **_kwargs: calls.append("dependent"), params={})
        },
        valid_criteria={"criterion": SimpleNamespace(func=criterion, params={})},
        buffer_size_per_group=1,
    )

    assert calls == ["write", "forward", "dependent", "criterion", "restore"]


def test_camera_normalization_is_stationary() -> None:
    """RGB and depth normalization must not depend on per-frame statistics."""
    rgb = torch.tensor([0.0, 127.5, 255.0])
    depth = torch.tensor([0.0, 2.0])

    assert torch.allclose(mdp.vision_camera._rgb_norm(None, rgb), torch.tensor([-0.5, 0.0, 0.5]))
    assert torch.allclose(mdp.vision_camera._depth_norm(None, depth), torch.tanh(depth / 2) - 0.5)


def test_kuka_scene_declares_lift_geometry_without_environment_cfg_discovery() -> None:
    """The flat Kuka scene manifest carries the geometry consumed by lift terms."""
    scene_cfg = resolve_presets(KukaAllegroSceneCfg())

    plan = cloner.make_clone_plan(
        (scene_cfg.robot, scene_cfg.object),
        num_clones=2,
        env_spacing=scene_cfg.env_spacing,
        geometry_prim_paths=scene_cfg.geometry_prim_paths,
        clone_strategy=scene_cfg.clone_cfg.clone_strategy,
        env_template=scene_cfg.clone_cfg.clone_template,
    )

    assert plan.geometry_requests == (
        r"/World/envs/env_[^/]+/Robot",
        r"/World/envs/env_[^/]+/Object",
    )


@pytest.mark.parametrize(
    "data_type,buffer_shape,normalize,expected_shape",
    [
        ("rgb", (2, 8, 16, 3), True, (3, 8, 16)),
        ("depth", (2, 8, 16, 1), True, (1, 8, 16)),
        ("albedo", (2, 8, 16, 4), False, (8, 16, 4)),
    ],
)
def test_vision_camera_declares_shape_without_reading_data(
    data_type: str, buffer_shape: tuple[int, ...], normalize: bool, expected_shape: tuple[int, ...]
) -> None:
    """Camera terms derive their shape from allocated metadata without triggering a render."""

    class FakeCamera:
        cfg = SimpleNamespace(data_types=[data_type])
        output_shapes = {data_type: buffer_shape}

        @property
        def data(self):
            raise AssertionError("Shape declaration must not read camera data.")

    env = SimpleNamespace(scene=SimpleNamespace(sensors={"camera": FakeCamera()}))
    cfg = ObservationTermCfg(
        func=mdp.vision_camera,
        params={"sensor_cfg": SceneEntityCfg("camera"), "normalize": normalize},
    )

    term = mdp.vision_camera(cfg, env)

    assert term._output_shape == expected_shape


def test_visualization_markers_are_scene_owned() -> None:
    from isaaclab_tasks.core.lift.config.franka.franka_env_cfg import FrankaLiftEnvCfg

    cfg = FrankaLiftEnvCfg()
    assert cfg.scene.command_goal_marker is cfg.commands.object_pose.goal_pose_visualizer_cfg
    assert cfg.scene.command_current_marker is cfg.commands.object_pose.curr_pose_visualizer_cfg
    assert cfg.scene.success_marker is cfg.commands.object_pose.success_marker_cfg
    assert (
        cfg.scene.object_point_cloud_marker is cfg.observations.perception.object_point_cloud.params["visualizer_cfg"]
    )


def test_lift_pose_markers_forward_environment_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every lift pose, goal, and success marker should retain its environment ownership."""
    num_envs = 3
    environment_ids = torch.arange(num_envs)
    identity_quat = torch.zeros((num_envs, 4))
    identity_quat[:, 0] = 1.0
    root_pos_w = torch.zeros((num_envs, 3))
    root_pose_w = torch.cat((root_pos_w, identity_quat), dim=-1)

    robot = SimpleNamespace(
        is_initialized=True,
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=root_pos_w),
            root_quat_w=SimpleNamespace(torch=identity_quat),
        ),
    )
    object_asset = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=root_pos_w),
            root_quat_w=SimpleNamespace(torch=identity_quat),
            root_link_pose_w=SimpleNamespace(torch=root_pose_w),
        )
    )
    success_asset = SimpleNamespace(data=SimpleNamespace(root_pos_w=SimpleNamespace(torch=root_pos_w)))
    scene = _FakeScene(environment_ids, robot=robot, object=object_asset, table=success_asset)
    env = SimpleNamespace(num_envs=num_envs, device="cpu", scene=scene)
    cfg = SimpleNamespace(
        asset_name="robot",
        object_name="object",
        success_vis_asset_name="table",
        success_marker_cfg=SimpleNamespace(class_type=_MarkerSpy),
        goal_pose_visualizer_cfg=SimpleNamespace(class_type=_MarkerSpy),
        curr_pose_visualizer_cfg=SimpleNamespace(class_type=_MarkerSpy),
        position_only=True,
        cmd_kind=None,
        element_names=None,
    )

    def _initialize_command_term(command, command_cfg, command_env) -> None:
        command.cfg = command_cfg
        command._env = command_env
        command.metrics = {}

    monkeypatch.setattr(CommandTerm, "__init__", _initialize_command_term)
    command = ObjectUniformPoseCommand(cfg, env)
    command._set_debug_vis_impl(True)
    command._debug_vis_callback(None)
    command.cfg.position_only = False
    command._debug_vis_callback(None)
    command._update_metrics()
    DeformableUniformPoseCommand._update_metrics(command)
    command._segment_position_w = lambda: root_pos_w
    CableUniformPoseCommand._update_metrics(command)
    CableUniformPoseCommand._debug_vis_callback(command, None)

    expected_call_counts = {
        command.success_visualizer: 4,
        command.goal_visualizer: 3,
        command.curr_visualizer: 3,
    }
    for visualizer, expected_count in expected_call_counts.items():
        assert len(visualizer.calls) == expected_count
        for _, kwargs in visualizer.calls:
            assert torch.equal(kwargs["environment_ids"], environment_ids)


def test_lift_point_cloud_markers_repeat_environment_ids_per_point() -> None:
    """Flattened point-cloud markers should retain env-major ownership."""
    num_envs = 3
    num_points = 4
    identity_quat = torch.zeros((num_envs, 4))
    identity_quat[:, 0] = 1.0
    root_pos_w = torch.zeros((num_envs, 3))
    points_local = torch.arange(num_envs * num_points * 3, dtype=torch.float32).view(num_envs, num_points, 3)

    term = object.__new__(mdp.object_point_cloud_b)
    term.object = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=root_pos_w),
            root_quat_w=SimpleNamespace(torch=identity_quat),
        )
    )
    term.ref_asset = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=root_pos_w),
            root_quat_w=SimpleNamespace(torch=identity_quat),
        )
    )
    term.points_local = points_local
    term.points_w = torch.zeros_like(points_local)
    term.visualizer = _MarkerSpy()
    env = SimpleNamespace(num_envs=num_envs)

    term(env, num_points=num_points, visualize=True)

    assert len(term.visualizer.calls) == 1
    _, kwargs = term.visualizer.calls[0]
    assert torch.equal(kwargs["translations"], term.points_w.view(-1, 3))
    assert torch.equal(kwargs["environment_ids"], torch.arange(num_envs).repeat_interleave(num_points))


def test_collision_meshes_are_assembled_from_planned_geometry(monkeypatch) -> None:
    """Identical clone rows share one body-local mesh without consulting USD."""
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    clone_mask = np.ones(2, dtype=np.bool_)
    root = "/World/envs/env_{}/Object"
    body = f"{root}/Body"
    mesh = f"{body}/mesh"
    root_frame = FrameLayout(root, "/World/envs/env_0/Object", "/World/envs/env_{}", 0, None, clone_mask=clone_mask)
    mesh_frame = FrameLayout(
        mesh,
        "/World/envs/env_0/Object/Body/mesh",
        body,
        0,
        None,
        body_path=body,
        pose=(0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        clone_mask=clone_mask,
    )
    body_prototype = RigidBodyLayout(
        body,
        "/World/envs/env_*/Object/Body",
        0,
        None,
        "Body",
        source_path="/World/envs/env_0/Object/Body",
        clone_mask=clone_mask,
    )
    geometry = GeometryLayout(
        mesh,
        "/World/envs/env_0/Object/Body/mesh",
        0,
        vertices,
        faces,
        mesh_frame,
        collision=True,
        clone_mask=clone_mask,
    )
    plan = ClonePlan(
        sources=(root.format(0),),
        destinations=(root,),
        clone_mask=torch.ones((1, 2), dtype=torch.bool),
        env_ids=torch.arange(2),
        is_complete=True,
        frame_prototypes=(root_frame, mesh_frame),
        rigid_body_prototypes=(body_prototype,),
        geometry_prototypes=(geometry,),
        _env_ids_cpu=(0, 1),
    )
    monkeypatch.setattr(lift_utils, "_clone_plan", lambda: plan)
    monkeypatch.setattr(
        ClonePlan,
        "_materialize_frame",
        lambda *_args: pytest.fail("Lift object geometry must remain prototype-sized."),
    )

    meshes, env_mesh = lift_utils.collect_rigid_object_collision_meshes(2, r"/World/envs/env_[^/]+/Object")

    assert len(meshes) == 1
    np.testing.assert_array_equal(env_mesh, [0, 0])
    np.testing.assert_allclose(meshes[0].vertices[0], [0.25, 0.0, 0.0])


def test_body_collision_meshes_skip_selected_bodies_without_colliders(monkeypatch) -> None:
    """A selected rigid body without collision geometry does not participate in clearance."""
    root = "/World/envs/env_{}/Robot"
    exact_root = root.format(0)
    mesh = f"{root}/colliding/mesh"
    clone_mask = np.ones(1, dtype=np.bool_)
    colliding = RigidBodyLayout(
        f"{root}/colliding",
        "/World/envs/env_*/Robot/colliding",
        0,
        None,
        "colliding",
        f"{exact_root}/colliding",
        clone_mask=clone_mask,
    )
    empty = RigidBodyLayout(
        f"{root}/empty",
        "/World/envs/env_*/Robot/empty",
        0,
        None,
        "empty",
        f"{exact_root}/empty",
        clone_mask=clone_mask,
    )
    root_frame = FrameLayout(root, exact_root, "/World/envs/env_{}", 0, None, clone_mask=clone_mask)
    body_frame = FrameLayout(
        f"{root}/colliding",
        colliding.source_path,
        root,
        0,
        None,
        body_path=f"{root}/colliding",
        body_view_path=colliding.view_path,
        clone_mask=clone_mask,
    )
    mesh_frame = FrameLayout(
        mesh,
        mesh.format(0),
        f"{root}/colliding",
        0,
        None,
        body_path=f"{root}/colliding",
        body_view_path=colliding.view_path,
        clone_mask=clone_mask,
    )
    geometry = GeometryLayout(
        mesh,
        mesh.format(0),
        0,
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        np.array([[0, 1, 2]], dtype=np.int32),
        mesh_frame,
        collision=True,
        clone_mask=clone_mask,
    )
    articulation = ArticulationLayout(root, "/World/envs/env_*/Robot", 0, (), (colliding, empty), clone_mask)
    plan = ClonePlan(
        sources=(exact_root,),
        destinations=(root,),
        clone_mask=torch.ones((1, 1), dtype=torch.bool),
        env_ids=torch.arange(1),
        is_complete=True,
        frame_prototypes=(root_frame, body_frame, mesh_frame),
        rigid_body_prototypes=(colliding, empty),
        geometry_prototypes=(geometry,),
        articulation_prototypes=(articulation,),
        _env_ids_cpu=(0,),
    )
    robot = SimpleNamespace(
        cfg=SimpleNamespace(prim_path=r"/World/envs/env_[^/]+/Robot"),
        find_bodies=lambda _names: ([0, 1], ["colliding", "empty"]),
    )
    monkeypatch.setattr(lift_utils, "_clone_plan", lambda: plan)

    meshes, names = lift_utils.collect_body_collision_meshes(robot, ".*")

    assert names == ["colliding", "empty"]
    assert set(meshes) == {0}
