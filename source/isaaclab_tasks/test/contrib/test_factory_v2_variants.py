# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the homogeneous Factory v2 assembly catalog."""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
import warp as wp
from tensordict import TensorDict

import isaaclab.utils.math as math_utils
from isaaclab import cloner
from isaaclab.managers import EventTermCfg, ManagerTermBase, ObservationTermCfg, SceneEntityCfg

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.contrib.nistv2.factory_presets as factory_presets
import isaaclab_tasks.contrib.nistv2.factory_scenes_cfg as factory_scenes_cfg
import isaaclab_tasks.contrib.nistv2.mdp.events as factory_events
import isaaclab_tasks.contrib.nistv2.mdp.observations as factory_observations
import isaaclab_tasks.contrib.nistv2.utils.collision_analyzer as factory_collision_analyzer
from isaaclab_tasks.contrib.nistv2.assembly_profile import AssemblyProfile
from isaaclab_tasks.contrib.nistv2.assembly_variants import ASSEMBLY_VARIANT_NAMES, ASSEMBLY_VARIANTS
from isaaclab_tasks.contrib.nistv2.config.agents.models import MLPEncoder, SimBaBlock, SimBaModel, SimBaNetwork
from isaaclab_tasks.contrib.nistv2.config.agents.rsl_rl_ppo_cfg import FactoryPPORunnerCfg, SimBaModelCfg
from isaaclab_tasks.contrib.nistv2.factory_env_cfg import FactoryObservationsCfg
from isaaclab_tasks.contrib.nistv2.factory_scenes_cfg import FactorySceneCfg, _paired_clone_strategy
from isaaclab_tasks.contrib.nistv2.mdp.assembly_variants import AssemblyVariantContext
from isaaclab_tasks.contrib.nistv2.mdp.observations import (
    _scene_point_cloud_in_root_frame,
    asset_link_velocity_in_root_asset_frame,
    scene_point_cloud_b,
    target_asset_pose_in_root_asset_frame,
)
from isaaclab_tasks.contrib.nistv2.reset_env_cfg import ACCUMULATOR_RESET
from isaaclab_tasks.contrib.nistv2.utils import reset_state
from isaaclab_tasks.contrib.nistv2.utils.event_combinators import ChainedResetTerms, TermChoice, reset_accumulator
from isaaclab_tasks.contrib.nistv2.utils.pose_offset import Offset
from isaaclab_tasks.contrib.nistv2.utils.sampling import SamplerCfg, UniformSamplingStrategyCfg
from isaaclab_tasks.core.lift.mdp.events_cfg import SuccessMonitorCfg


class _MeshAsset:
    def __init__(self, variant_ids: torch.Tensor):
        self.device = "cpu"
        self.num_instances = len(variant_ids)
        self.num_mesh_variants = len(ASSEMBLY_VARIANTS)
        self.mesh_variant_ids = SimpleNamespace(torch=variant_ids.clone())

    def write_mesh_variant_to_sim(self, variant_ids: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.mesh_variant_ids.torch[env_ids] = variant_ids


def _variant_context() -> tuple[AssemblyVariantContext, _MeshAsset, _MeshAsset]:
    ids = torch.arange(len(ASSEMBLY_VARIANTS), dtype=torch.int32)
    fixed, held = _MeshAsset(ids), _MeshAsset(ids)
    env = SimpleNamespace(
        device="cpu",
        num_envs=len(ids),
        scene={
            "fixed_asset": fixed,
            "held_asset": held,
        },
    )
    cfg = EventTermCfg(
        func=AssemblyVariantContext,
        mode="startup",
        params={
            "fixed_asset_cfg": SceneEntityCfg("fixed_asset"),
            "held_asset_cfg": SceneEntityCfg("held_asset"),
            "variant_names": ASSEMBLY_VARIANT_NAMES,
        },
    )
    return AssemblyVariantContext(cfg, env), fixed, held


def test_factory_v2_registers_its_own_environment() -> None:
    """Keep v2 registration independent from the v1 task."""
    spec = gym.spec("IsaacContrib-Factory-V2-Franka")
    assert spec.kwargs["env_cfg_entry_point"] == "isaaclab_tasks.contrib.nistv2.factory_env_cfg:FactoryBaseEnvCfg"


def test_scene_uses_one_ordered_pair_bank() -> None:
    """Build only the fixed and held assets from the same 20-entry catalog."""
    scene = FactorySceneCfg()
    assert len(ASSEMBLY_VARIANTS) == 20
    assert len(set(ASSEMBLY_VARIANT_NAMES)) == len(ASSEMBLY_VARIANTS)
    assert set(ASSEMBLY_VARIANT_NAMES).issuperset({"gear_mesh_small", "gear_mesh_medium", "gear_mesh_large"})
    assert scene.fixed_asset.mesh_variants_enabled
    assert scene.held_asset.mesh_variants_enabled
    assert scene.held_asset.mesh_variant_inertia_diagonal_offset == 1.0e-5

    fixed_paths = [cfg.usd_path for cfg in scene.fixed_asset.spawn.assets_cfg]
    held_paths = [cfg.usd_path for cfg in scene.held_asset.spawn.assets_cfg]
    assert fixed_paths == [variant.fixed_asset.spawn.usd_path for variant in ASSEMBLY_VARIANTS]
    assert held_paths == [variant.held_asset.spawn.usd_path for variant in ASSEMBLY_VARIANTS]


def test_v2_has_one_scene_type_and_no_v1_dependency() -> None:
    """Keep scene ownership in v2's single composition root."""
    scene_types = {
        name
        for name, value in vars(factory_scenes_cfg).items()
        if isinstance(value, type) and value.__module__ == factory_scenes_cfg.__name__
    }
    assert scene_types == {"FactorySceneCfg"}

    package_root = Path(factory_scenes_cfg.__file__).parent
    offenders = [
        path.relative_to(package_root)
        for pattern in ("*.py", "*.pyi")
        for path in package_root.rglob(pattern)
        if "isaaclab_tasks.contrib.nist." in path.read_text()
    ]
    assert offenders == []
    forbidden_symbols = ("spare_gear", "spare_assets", "spare_offsets", "spare_pose")
    spare_owners = [
        path.relative_to(package_root)
        for pattern in ("*.py", "*.pyi")
        for path in package_root.rglob(pattern)
        if any(symbol in path.read_text() for symbol in forbidden_symbols)
    ]
    assert spare_owners == []
    for obsolete in (
        "FixedAssetMapCfg",
        "HeldAssetTipCfg",
        "ResetAssetsCfg",
        "HeldAssetObstaclesCfg",
        "RobotObstaclesCfg",
    ):
        assert not hasattr(factory_presets, obsolete)


def test_clone_strategy_only_emits_matching_pair_indices() -> None:
    """Never seed Newton with a crossed fixed/held pair."""
    indices = torch.arange(len(ASSEMBLY_VARIANTS))
    paired = indices[:, None].expand(-1, 2)
    crossed = paired.clone()
    crossed[:, 1] = crossed[:, 1].roll(1)
    combinations = torch.cat((paired, crossed))
    chosen = _paired_clone_strategy(combinations, 4, "cpu")

    torch.testing.assert_close(chosen, chosen[:, :1].expand_as(chosen))
    torch.testing.assert_close(chosen[:, 0], indices[:4])


def test_four_world_plan_stages_complete_variant_bank() -> None:
    """Keep every variant source available without adding live worlds."""
    scene = FactorySceneCfg(num_envs=4)
    for asset in (scene.fixed_asset, scene.held_asset):
        asset.prim_path = cloner.expand_env_regex_ns(asset.prim_path, scene.clone_cfg.clone_template)
    plan = cloner.make_clone_plan(
        [scene.fixed_asset, scene.held_asset],
        num_clones=4,
        env_spacing=scene.env_spacing,
        device="cpu",
        clone_strategy=_paired_clone_strategy,
        env_template=scene.clone_cfg.clone_template,
    )

    assert plan.clone_mask.shape == (2 * len(ASSEMBLY_VARIANTS), 4)
    for paths in (scene.fixed_asset.spawn.spawn_paths, scene.held_asset.spawn.spawn_paths):
        assert paths is not None
        assert len(paths) == len(ASSEMBLY_VARIANTS)
        assert all(path is not None for path in paths)
        assert sum(path.startswith("/World/envs/") for path in paths) == 4


def test_context_gathers_all_variant_geometry_from_mesh_ids() -> None:
    """Use the Newton mesh index as the sole task-geometry index."""
    context, fixed, held = _variant_context()
    env_ids = torch.arange(len(ASSEMBLY_VARIANTS))
    zeros = torch.zeros((len(env_ids), 3))
    identity = math_utils.default_orientation(len(env_ids), "cpu")

    offsets = {
        "board": [variant.board_offset for variant in ASSEMBLY_VARIANTS],
        "fixed_tip": [variant.fixed_tip for variant in ASSEMBLY_VARIANTS],
        "held_align": [variant.held_align for variant in ASSEMBLY_VARIANTS],
        "held_grasp_point": [variant.held_grasp_point for variant in ASSEMBLY_VARIANTS],
        "held_grasp_middle": [variant.held_grasp_middle for variant in ASSEMBLY_VARIANTS],
        "assembled": [AssemblyProfile(variant.profile).assembled_offset for variant in ASSEMBLY_VARIANTS],
    }
    for name, expected in offsets.items():
        pos, quat = context.combine(name, zeros, identity, env_ids)
        torch.testing.assert_close(pos, torch.tensor([offset.pos for offset in expected]))
        torch.testing.assert_close(quat, torch.tensor([offset.quat for offset in expected]))

    torch.testing.assert_close(
        context.grasp_diameter(env_ids),
        torch.tensor([variant.held_grasp_diameter for variant in ASSEMBLY_VARIANTS]),
    )
    axes = ("x", "y", "z", "roll", "pitch", "yaw")
    expected_ranges = torch.tensor(
        [[variant.grasped_pose_range[axis] for axis in axes] for variant in ASSEMBLY_VARIANTS]
    )
    torch.testing.assert_close(context.pose_range("grasped", env_ids), expected_ranges)
    static_offset = wp.to_torch(context.offset_warp(Offset(pos=(0.25, 0.0, 0.0))))
    torch.testing.assert_close(static_offset[:, 0], torch.full((len(ASSEMBLY_VARIANTS),), 0.25))
    torch.manual_seed(7)
    profile_pos, profile_quat = context.sample_profile((0.0, 0.0), env_ids)
    torch.manual_seed(7)
    expected_profile = [AssemblyProfile(variant.profile).sample((0.0, 0.0), 1, "cpu") for variant in ASSEMBLY_VARIANTS]
    torch.testing.assert_close(profile_pos, torch.cat([sample[0] for sample in expected_profile]))
    torch.testing.assert_close(profile_quat, torch.cat([sample[1] for sample in expected_profile]))
    torch.testing.assert_close(context.one_hot(), torch.eye(len(ASSEMBLY_VARIANTS)))

    prepared = env_ids.flip(0)
    context.prepare(prepared)
    context.select(env_ids)
    torch.testing.assert_close(fixed.mesh_variant_ids.torch, prepared.to(torch.int32))
    torch.testing.assert_close(held.mesh_variant_ids.torch, prepared.to(torch.int32))


def test_context_defaults_selection_to_first_variant() -> None:
    """Keep startup selection valid before the accumulator prepares a batch."""
    context, fixed, held = _variant_context()
    env_ids = torch.arange(len(ASSEMBLY_VARIANTS))

    context.select(env_ids)

    expected = torch.zeros(len(env_ids), dtype=torch.int32)
    torch.testing.assert_close(fixed.mesh_variant_ids.torch, expected)
    torch.testing.assert_close(held.mesh_variant_ids.torch, expected)


def test_reset_partition_defaults_to_first_choice() -> None:
    """Keep the dispatcher valid before the accumulator prepares a batch."""
    selected = torch.full((4,), -1, dtype=torch.long)

    def select_first(_env, env_ids):
        selected[env_ids] = 0

    def select_second(_env, env_ids):
        selected[env_ids] = 1

    terms = {
        "first": EventTermCfg(func=select_first, mode="reset"),
        "second": EventTermCfg(func=select_second, mode="reset"),
    }
    cfg = EventTermCfg(func=TermChoice, mode="reset", params={"terms": terms})
    env = SimpleNamespace(num_envs=4, device="cpu")
    choice = TermChoice(cfg, env)
    env_ids = torch.arange(4)

    choice(env, env_ids, terms)
    torch.testing.assert_close(selected, torch.zeros(4, dtype=torch.long))

    choice.prepare(torch.ones(4, dtype=torch.long))
    choice(env, env_ids, terms)
    torch.testing.assert_close(selected, torch.ones(4, dtype=torch.long))


def test_pose_observation_caches_inputs_and_tracks_live_variant_ids() -> None:
    """Resolve invariant inputs once while following mesh changes."""
    target_pose = torch.tensor(
        [
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
        ]
    )
    root_pose = torch.tensor(
        [
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
            [[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
        ]
    )
    scene_reads: list[str] = []

    class Scene(dict):
        def __getitem__(self, name: str):
            scene_reads.append(name)
            return super().__getitem__(name)

    variant_reads: list[str] = []
    variant_ids = torch.tensor([0, 1], dtype=torch.int32)
    variant_offsets = wp.array(
        [
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=wp.transformf,
        device="cpu",
    )

    def offset_warp(offset: Offset | str) -> wp.array(dtype=wp.transformf):
        if isinstance(offset, str):
            return variant_offsets
        return wp.full(2, wp.transformf(*offset.pose), dtype=wp.transformf, device="cpu")

    variants = SimpleNamespace(variant_ids_warp=wp.from_torch(variant_ids), offset_warp=offset_warp)
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        scene=Scene(
            target=SimpleNamespace(
                data=SimpleNamespace(
                    body_link_pose_w=SimpleNamespace(
                        warp=wp.array(target_pose.numpy(), dtype=wp.transformf, device="cpu")
                    )
                )
            ),
            root=SimpleNamespace(
                data=SimpleNamespace(
                    body_link_pose_w=SimpleNamespace(
                        warp=wp.array(root_pose.numpy(), dtype=wp.transformf, device="cpu")
                    )
                )
            ),
        ),
        event_manager=SimpleNamespace(
            get_term_cfg=lambda name: variant_reads.append(name) or SimpleNamespace(func=variants)
        ),
    )
    cfg = ObservationTermCfg(
        func=target_asset_pose_in_root_asset_frame,
        params={
            "target_asset_cfg": SceneEntityCfg("target", body_ids=[1]),
            "root_asset_cfg": SceneEntityCfg("root", body_ids=[0]),
            "target_asset_offset": "target_frame",
            "root_asset_offset": Offset(pos=(0.25, 0.0, 0.0)),
        },
    )

    term = target_asset_pose_in_root_asset_frame(cfg, env)
    assert isinstance(term, ManagerTermBase)
    expected = torch.tensor([[1.25, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    torch.testing.assert_close(term(env, **cfg.params), expected)

    variant_ids.copy_(torch.tensor([1, 0], dtype=torch.int32))
    expected = torch.tensor([[1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.75, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    torch.testing.assert_close(term(env, **cfg.params), expected)

    assert scene_reads == ["target", "root"]
    assert variant_reads == ["assembly_variants"]


def test_pose_observation_uses_one_offset_parameter_per_frame() -> None:
    """Keep static and variant offsets in the same parameters."""
    parameters = inspect.signature(target_asset_pose_in_root_asset_frame.__call__).parameters
    assert "target_asset_offset" in parameters
    assert "root_asset_offset" in parameters
    assert isinstance(parameters["target_asset_offset"].default, Offset)
    assert isinstance(parameters["root_asset_offset"].default, Offset)
    assert "variant_context" not in parameters
    assert "target_variant_offset" not in parameters
    assert "root_variant_offset" not in parameters


def test_policy_observes_current_scene_geometry_without_repeating_its_history() -> None:
    """Replace explicit assembly identity and poses with one compact geometry owner."""
    observations = FactoryObservationsCfg()
    policy = observations.policy
    perception = observations.perception

    assert perception.scene_point_cloud.func is scene_point_cloud_b
    assert perception.scene_point_cloud.params == {
        "fixed_asset_cfg": SceneEntityCfg("fixed_asset"),
        "held_asset_cfg": SceneEntityCfg("held_asset"),
        "robot_asset_cfg": SceneEntityCfg("robot"),
        "fixed_num_points": 256,
        "held_num_points": 256,
        "robot_num_points": 256,
        "flatten": True,
    }
    assert perception.scene_point_cloud.history_length == 0
    assert policy.end_effector_vel_lin_ang_b.history_length == 5
    assert policy.joint_pos.history_length == 5
    assert policy.prev_action.history_length == 5
    assert policy.history_length is None
    assert not hasattr(policy, "end_effector_pose")
    assert not hasattr(policy, "held_asset_in_fixed_asset_frame")
    assert not hasattr(policy, "fixed_asset_in_end_effector_frame")
    assert not hasattr(policy, "assembly_variant")


def test_factory_agent_routes_perception_through_point_cloud_encoder() -> None:
    """Keep geometry out of the state MLP and route it through the point-cloud encoder."""
    observations = FactoryObservationsCfg()
    runner = FactoryPPORunnerCfg()

    assert not hasattr(observations.policy, "scene_point_cloud")
    assert observations.perception.scene_point_cloud.func is scene_point_cloud_b
    assert runner.obs_groups.default == {
        "actor": ["policy", "perception"],
        "critic": ["policy", "perception"],
    }
    assert runner.actor.class_name.endswith(":SimBaModel")
    assert runner.critic.class_name.endswith(":SimBaModel")
    assert runner.actor.hidden_dim == 256
    assert runner.actor.num_blocks == 2
    assert runner.actor.expansion_factor == 4
    assert runner.actor.activation == "swish"
    assert set(runner.actor.encoder_cfg) == {"perception"}
    encoder_cfg = runner.actor.encoder_cfg["perception"]
    assert isinstance(encoder_cfg, SimBaModelCfg.MLPEncoderCfg)
    assert encoder_cfg.hidden_dims == [256]
    assert encoder_cfg.output_dim == 128
    assert encoder_cfg.activation == "elu"
    assert encoder_cfg.last_activation == "elu"
    assert not hasattr(runner.actor, "point_cloud_group")
    serialized_encoder = runner.to_dict()["actor"]["encoder_cfg"]["perception"]
    assert serialized_encoder["class_name"].endswith(":MLPEncoder")
    assert runner.algorithm.default.num_mini_batches == 4


def test_simba_model_combines_flattened_scene_mlp_with_residual_head() -> None:
    """Encode ordered scene points once before the SimBa residual head."""
    batch_size, num_clouds, points_per_cloud = 4, 3, 5
    state = torch.randn(batch_size, 7)
    points = torch.randn(batch_size, num_clouds, points_per_cloud, 3)
    observations = TensorDict({"policy": state, "perception": points.flatten(1)}, batch_size=[batch_size])
    model = SimBaModel(
        observations,
        {"actor": ["policy", "perception"]},
        "actor",
        output_dim=2,
        hidden_dim=16,
        num_blocks=2,
        expansion_factor=4,
        activation="swish",
        obs_normalization=False,
        encoder_cfg={
            "perception": {
                "class_name": MLPEncoder,
                "hidden_dims": [16],
                "output_dim": 6,
                "activation": "elu",
                "last_activation": "elu",
            }
        },
    )

    encoder = model.encoders["perception"]
    assert isinstance(encoder, MLPEncoder)
    assert encoder.mlp[0].weight.shape == (16, num_clouds * points_per_cloud * 3)
    assert encoder.mlp[2].weight.shape == (6, 16)
    assert isinstance(model.mlp, SimBaNetwork)
    assert sum(isinstance(module, SimBaBlock) for module in model.mlp) == 2
    torch.testing.assert_close(torch.jit.script(model.as_jit())(state, observations["perception"]), model(observations))


def test_simba_model_accepts_custom_encoder() -> None:
    """Keep the SimBa head independent of the observation encoder implementation."""

    class CustomEncoder(torch.nn.Module):
        def __init__(self, input_shape: tuple[int, ...], output_dim: int) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(int(np.prod(input_shape)), output_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x.flatten(start_dim=1))

    batch_size = 4
    observations = TensorDict(
        {"policy": torch.randn(batch_size, 7), "perception": torch.randn(batch_size, 2, 3)},
        batch_size=[batch_size],
    )
    model = SimBaModel(
        observations,
        {"critic": ["policy", "perception"]},
        "critic",
        output_dim=1,
        hidden_dim=16,
        encoder_cfg={"perception": {"class_name": CustomEncoder, "output_dim": 5}},
    )

    assert isinstance(model.encoders["perception"], CustomEncoder)
    assert model(observations).shape == (batch_size, 1)


def test_scene_point_cloud_selects_live_variants_and_tracks_robot_links() -> None:
    """Transform all three ordered point segments into the robot root frame in one launch."""
    fixed_points = wp.array([[[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]], dtype=wp.vec3f, device="cpu")
    held_points = wp.array([[[0.0, 1.0, 0.0]], [[0.0, 2.0, 0.0]]], dtype=wp.vec3f, device="cpu")
    robot_points = wp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], dtype=wp.vec3f, device="cpu")
    robot_body_ids = wp.array([0, 1], dtype=wp.int32, device="cpu")
    variant_ids = wp.array([0, 1], dtype=wp.int32, device="cpu")
    fixed_poses = wp.array(
        [[[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], [[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]],
        dtype=wp.transformf,
        device="cpu",
    )
    held_poses = wp.array(
        [[[0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0]], [[0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0]]],
        dtype=wp.transformf,
        device="cpu",
    )
    robot_poses = wp.array(
        [
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
            [[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
        ],
        dtype=wp.transformf,
        device="cpu",
    )
    root_poses = wp.array(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
        dtype=wp.transformf,
        device="cpu",
    )
    output = wp.empty((2, 4), dtype=wp.vec3f, device="cpu")

    wp.launch(
        _scene_point_cloud_in_root_frame,
        dim=(2, 4),
        inputs=[
            fixed_points,
            held_points,
            robot_points,
            robot_body_ids,
            variant_ids,
            fixed_poses,
            held_poses,
            robot_poses,
            root_poses,
            wp.vec3i(1, 1, 2),
        ],
        outputs=[output],
        device="cpu",
    )

    expected = torch.tensor(
        [
            [[11.0, 0.0, 0.0], [0.0, 11.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 2.0]],
            [[11.0, -1.0, 0.0], [-1.0, 11.0, 0.0], [1.0, -1.0, 1.0], [-1.0, 1.0, 2.0]],
        ]
    )
    torch.testing.assert_close(wp.to_torch(output), expected)


def test_articulation_mesh_queries_resolve_clone_sources(monkeypatch) -> None:
    """Pass clone expressions through the source resolver before querying USD."""
    asset_path = "/World/envs/env_[^/]+/Robot"
    body_path = f"{asset_path}/Geometry/panda_link0"
    resolved_paths = []

    def resolve(path_expr, predicate, expected_num_matches):
        assert path_expr == asset_path
        assert expected_num_matches == 1
        return [(object(), body_path)]

    class ResolvedBody(RuntimeError):
        pass

    def stop_after_resolution(num_envs, prim_path_pattern, **kwargs):
        resolved_paths.append(prim_path_pattern)
        raise ResolvedBody

    for module in (factory_collision_analyzer, factory_observations):
        monkeypatch.setattr(module, "resolve_matching_prims_from_source", resolve, raising=False)
        monkeypatch.setattr(module, "RigidObjectHasher", stop_after_resolution)

    asset = SimpleNamespace(
        body_names=["panda_link0"],
        cfg=SimpleNamespace(prim_path=asset_path),
    )
    env = SimpleNamespace(num_envs=4, device="cpu", scene={"robot": asset})
    analyzer_cfg = SimpleNamespace(
        asset_cfg=SimpleNamespace(name="robot", body_names=None),
        obstacle_cfgs=(),
    )
    with pytest.raises(ResolvedBody):
        factory_collision_analyzer.CollisionAnalyzer(analyzer_cfg, env)

    asset_cfg = SimpleNamespace(body_ids=[0])
    with pytest.raises(ResolvedBody):
        factory_observations._sample_articulation_points(env, asset, asset_cfg, num_points=1)

    assert resolved_paths == [body_path, body_path]


def test_velocity_observation_caches_resolved_assets() -> None:
    """Resolve velocity observation inputs once without changing its output."""
    half_sqrt_two = 0.5**0.5
    target_pose = np.array(
        [
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, half_sqrt_two, half_sqrt_two],
            ],
        ],
        dtype=np.float32,
    )
    target_velocity = np.array(
        [
            [[0.0] * 6, [1.0, 2.0, 3.0, 0.0, 0.0, 2.0]],
            [[0.0] * 6, [1.0, 2.0, 3.0, 0.0, 0.0, 2.0]],
        ],
        dtype=np.float32,
    )
    root_pose = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, half_sqrt_two, half_sqrt_two],
        ],
        dtype=np.float32,
    )
    scene_reads: list[str] = []

    class Scene(dict):
        def __getitem__(self, name: str):
            scene_reads.append(name)
            return super().__getitem__(name)

    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        scene=Scene(
            target=SimpleNamespace(
                data=SimpleNamespace(
                    body_com_vel_w=SimpleNamespace(
                        warp=wp.array(target_velocity, dtype=wp.spatial_vectorf, device="cpu")
                    ),
                    body_link_pose_w=SimpleNamespace(warp=wp.array(target_pose, dtype=wp.transformf, device="cpu")),
                )
            ),
            root=SimpleNamespace(
                data=SimpleNamespace(
                    root_link_pose_w=SimpleNamespace(warp=wp.array(root_pose, dtype=wp.transformf, device="cpu"))
                )
            ),
        ),
    )
    cfg = ObservationTermCfg(
        func=asset_link_velocity_in_root_asset_frame,
        params={
            "target_asset_cfg": SceneEntityCfg("target", body_ids=[1]),
            "root_asset_cfg": SceneEntityCfg("root"),
            "target_asset_offset": Offset(pos=(0.5, 0.0, 0.0)),
        },
    )

    term = asset_link_velocity_in_root_asset_frame(cfg, env)
    assert isinstance(term, ManagerTermBase)
    expected = torch.tensor([[1.0, 3.0, 3.0, 0.0, 0.0, 2.0], [2.0, 0.0, 3.0, 0.0, 0.0, 2.0]])
    for _ in range(2):
        torch.testing.assert_close(term(env, **cfg.params), expected, atol=1.0e-6, rtol=1.0e-6)

    assert scene_reads == ["target", "root"]


class _StateAsset(_MeshAsset):
    def __init__(self):
        super().__init__(torch.tensor([2, 7], dtype=torch.int32))
        self.data = SimpleNamespace(root_state_w=wp.array(np.arange(26, dtype=np.float32).reshape(2, 13), device="cpu"))
        self.calls: list[str] = []

    def write_mesh_variant_to_sim(self, variant_ids: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.calls.append("mesh")
        super().write_mesh_variant_to_sim(variant_ids, env_ids)

    def write_root_state_to_sim(self, root_state: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.calls.append("root")
        self.written_root_state = root_state


def test_reset_state_restores_mesh_before_pose() -> None:
    """Carry mesh identity through the accumulator's flat state table."""
    asset = _StateAsset()
    scene = SimpleNamespace(
        _articulations={},
        _rigid_objects={"object": asset},
        env_origins=torch.zeros((2, 3)),
    )
    env = SimpleNamespace(scene=scene)
    env_ids = torch.arange(2)

    state = reset_state.get_reset_state(env, env_ids, ["object"])
    torch.testing.assert_close(state[:, 0], torch.tensor([2.0, 7.0]))
    state[:, 0] = torch.tensor([4.0, 9.0])
    reset_state.set_reset_state(env, state, env_ids, ["object"])

    assert asset.calls == ["mesh", "root"]
    torch.testing.assert_close(asset.mesh_variant_ids.torch, torch.tensor([4, 9], dtype=torch.int32))


def test_settled_pose_bank_steps_only_during_bootstrap(monkeypatch) -> None:
    """Replace live-reset stepping with a temporary per-variant pose bank."""
    poses = torch.zeros((4, 7))
    poses[:, 6] = 1.0
    velocities = torch.ones((4, 6))

    class Asset:
        num_mesh_variants = 2
        mesh_variant_ids = SimpleNamespace(torch=torch.zeros(4, dtype=torch.int32))
        data = SimpleNamespace(
            root_link_pose_w=SimpleNamespace(torch=poses),
            root_com_vel_w=SimpleNamespace(torch=velocities),
        )

        def write_mesh_variant_to_sim(self, variant_ids: torch.Tensor, env_ids: torch.Tensor) -> None:
            self.mesh_variant_ids.torch[env_ids] = variant_ids

        def write_root_link_pose_to_sim(self, pose: torch.Tensor, env_ids: torch.Tensor) -> None:
            poses[env_ids] = pose

        def write_root_com_velocity_to_sim(self, velocity: torch.Tensor, env_ids: torch.Tensor) -> None:
            velocities[env_ids] = velocity

    held_asset = Asset()

    def drop(_env, env_ids: torch.Tensor) -> None:
        ids = held_asset.mesh_variant_ids.torch[env_ids].float()
        poses[env_ids, 0] = _env.scene.env_origins[env_ids, 0] + ids
        poses[env_ids, 2] = 0.05

    class Scene(dict):
        _articulations = {}
        _rigid_objects = {"held_asset": held_asset}
        env_origins = torch.zeros((4, 3))
        env_origins[:, 0] = torch.arange(4, dtype=torch.float32)

        def update(self, dt: float) -> None:
            pass

    class Sim:
        physics_manager = SimpleNamespace(handles_decimation=lambda: True)
        steps = 0

        def step(self, render: bool) -> None:
            self.steps += 1
            poses[:, 2] -= 0.01

    env = SimpleNamespace(
        num_envs=4,
        device="cpu",
        scene=Scene(held_asset=held_asset),
        sim=Sim(),
        cfg=SimpleNamespace(decimation=4),
        step_dt=0.04,
    )
    cfg = EventTermCfg(
        func=factory_events.settle_held_asset,
        mode="reset",
        params={
            "held_asset_cfg": SceneEntityCfg("held_asset"),
            "drop_term": EventTermCfg(func=drop, mode="reset"),
            "scene_assets": ["held_asset"],
            "num_steps": 2,
        },
    )

    held_asset.mesh_variant_ids.torch[:] = torch.tensor([1, 0, 1, 0])
    original_poses = poses.clone()
    original_velocities = velocities.clone()
    original_variants = held_asset.mesh_variant_ids.torch.clone()
    monkeypatch.setattr(factory_events.reset_state, "get_reset_state", lambda *args, **kwargs: torch.empty(0))

    def restore(*args, **kwargs) -> None:
        poses.copy_(original_poses)
        velocities.copy_(original_velocities)
        held_asset.mesh_variant_ids.torch.copy_(original_variants)

    monkeypatch.setattr(factory_events.reset_state, "set_reset_state", restore)
    term = factory_events.settle_held_asset(cfg, env)
    term(env, torch.tensor([0, 2]), **cfg.params)

    assert env.sim.steps == 2
    torch.testing.assert_close(held_asset.mesh_variant_ids.torch, original_variants)
    torch.testing.assert_close(poses[[1, 3]], original_poses[[1, 3]])
    torch.testing.assert_close(poses[[0, 2], 0] - env.scene.env_origins[[0, 2], 0], torch.ones(2))
    torch.testing.assert_close(poses[[0, 2], 2], torch.full((2,), 0.03))
    torch.testing.assert_close(velocities[[0, 2]], torch.zeros((2, 6)))
    torch.testing.assert_close(velocities[[1, 3]], original_velocities[[1, 3]])

    held_asset.mesh_variant_ids.torch[:] = torch.tensor([0, 1, 0, 1])
    term(env, torch.arange(4), **cfg.params)
    assert env.sim.steps == 2
    torch.testing.assert_close(poses[:, 0] - env.scene.env_origins[:, 0], torch.tensor([0.0, 1.0, 0.0, 1.0]))
    torch.testing.assert_close(poses[:, 2], torch.full((4,), 0.03))
    torch.testing.assert_close(velocities, torch.zeros_like(velocities))


def test_reset_table_tiles_the_label_asset_grid() -> None:
    """Keep enough production states to cover every reset-label and asset cell."""
    reset_choice = ACCUMULATOR_RESET.params["reset_term"].params["terms"]["reset_strategies"]
    start_pick = reset_choice.params["terms"]["start_pick"]
    num_cells = len(reset_choice.params["terms"]) * len(ASSEMBLY_VARIANTS)
    assert ACCUMULATOR_RESET.params["state_table_size"] >= num_cells
    assert set(reset_choice.params) == {"terms"}
    assert "settling_term" not in ACCUMULATOR_RESET.params
    assert start_pick.params["terms"]["reset_held_asset"].func is factory_events.settle_held_asset
    assert not {
        "state_tag_names_bind",
        "state_tag_indices_bind",
        "state_tag_weight_bind",
    }.intersection(ACCUMULATOR_RESET.params)


def test_accumulator_starts_without_an_assigned_slot() -> None:
    """Do not record the first reset as a failure against slot zero."""
    cfg = SimpleNamespace(
        params={
            "acceptance_conditions": {},
            "reset_assets": [],
            "state_table_size": 1,
            "sampling": SamplerCfg(strategies=[UniformSamplingStrategyCfg()]),
            "success_monitor_cfg": SuccessMonitorCfg(),
        }
    )
    env = SimpleNamespace(
        num_envs=4,
        device="cpu",
        scene=SimpleNamespace(_articulations={}, _rigid_objects={}),
    )
    accumulator = reset_accumulator(cfg, env)
    assert torch.all(accumulator.sampled_slots == -1)


def test_accumulator_soft_balances_and_removes_precollection_terms(monkeypatch) -> None:
    """Bias collection toward sparse cells, then discard the one-shot term tree."""

    class Variants:
        variant_names = ("asset_a", "asset_b")
        variant_ids = torch.zeros(1, dtype=torch.int32)

        def prepare(self, variant_ids: torch.Tensor) -> None:
            self.variant_ids.copy_(variant_ids)

    class Reset:
        def __init__(self, choice: TermChoice):
            self.terms = {"reset_strategies": SimpleNamespace(func=choice)}
            self.is_valid = torch.ones(1, dtype=torch.bool)

        def __call__(self, env, env_ids: torch.Tensor) -> None:
            pass

    variants = Variants()
    choice = TermChoice.__new__(TermChoice)
    choice.term_samples = torch.zeros(1, dtype=torch.long)
    choice._next_samples = torch.zeros(1, dtype=torch.long)
    choice.term_partitions = {"start_pick": SimpleNamespace()}
    reset_term = SimpleNamespace(func=Reset(choice), params={})
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        event_manager=SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(func=variants)),
    )
    accumulator = reset_accumulator.__new__(reset_accumulator)
    accumulator._variant_context_name = "assembly_variants"
    accumulator._tag_term_name = "reset_strategies"
    accumulator._state_target_size = 3
    accumulator.cfg = SimpleNamespace(params={"reset_term": reset_term})
    accumulator.acceptance_conditions = {}
    accumulator.reset_assets = []
    accumulator.state_tag_names = []
    accumulator.variant_names = ()
    accumulator.state_data = torch.zeros((3, 1))
    accumulator.state_cell_indices = torch.full((3,), -1, dtype=torch.long)
    accumulator.precollecting_phase = True

    planned_cells = iter((0, 1, 0))
    sampling_weights = []

    def sample(weights: torch.Tensor, num_samples: int, replacement: bool) -> torch.Tensor:
        assert num_samples == 1
        assert replacement
        sampling_weights.append(weights.clone())
        return torch.tensor([next(planned_cells)])

    monkeypatch.setattr(torch, "multinomial", sample)
    monkeypatch.setattr(reset_state, "get_reset_state", lambda *args, **kwargs: torch.ones((1, 1)))
    accumulator._precollect(env, reset_term)

    torch.testing.assert_close(sampling_weights[0], torch.ones(2))
    assert 0.0 < sampling_weights[1][0] < sampling_weights[1][1]
    assert "reset_term" not in accumulator.cfg.params
    assert inspect.signature(reset_accumulator.__call__).parameters["reset_term"].default is None
    assert not hasattr(TermChoice, "release_temporary_state")
    assert not hasattr(ChainedResetTerms, "release_temporary_state")
    assert not hasattr(factory_events.settle_held_asset, "release_temporary_state")
    assert not accumulator.precollecting_phase


def test_accumulator_reports_adaptive_cell_probabilities() -> None:
    """Report the effective curriculum distribution instead of flattening every grid cell."""
    accumulator = reset_accumulator.__new__(reset_accumulator)
    accumulator.state_cell_indices = torch.arange(6).repeat_interleave(4)
    weights = torch.tensor([20.0, 1.0, 1.0, 1.0] * 6)
    weights[4:8] = 1.0
    probabilities = weights / weights.sum()
    sampled: dict[str, torch.Tensor] = {}

    def sample(probs: torch.Tensor, count: int) -> torch.Tensor:
        sampled["probabilities"] = probs.clone()
        return torch.multinomial(probs, count, replacement=True)

    accumulator._sampler = SimpleNamespace(
        probabilities=lambda: probabilities,
        sample=sample,
    )
    accumulator.precollecting_phase = False
    accumulator._requested_reset_assets = []
    accumulator.reset_assets = []
    accumulator.state_data = torch.empty((24, 0))
    accumulator.sampled_slots = torch.full((8,), -1, dtype=torch.long)
    accumulator._num_cells = 6
    accumulator._num_variants = 3
    accumulator.state_tag_names = ["reset_a", "reset_b"]
    accumulator.variant_names = ("asset_a", "asset_b", "asset_c")
    accumulator.cell_success_rate = torch.empty((2, 3))
    accumulator.cell_probabilities = torch.empty((2, 3))
    accumulator.success_monitor = SimpleNamespace(
        success_buf=torch.zeros((24, 1)),
        success_size=torch.zeros(24, dtype=torch.long),
        get_mean_success_rate=lambda: 0.0,
    )
    env = SimpleNamespace(
        num_envs=8,
        device="cpu",
        scene=SimpleNamespace(_articulations={}, _rigid_objects={}),
        termination_manager=SimpleNamespace(
            get_term_cfg=lambda _: SimpleNamespace(func=SimpleNamespace(is_success=torch.zeros(8, dtype=torch.bool)))
        ),
        extras={},
    )

    torch.manual_seed(7)
    accumulator(
        env,
        torch.arange(8),
        None,
        [],
        {},
        24,
        SuccessMonitorCfg(),
        SamplerCfg(),
        report=True,
    )

    expected = torch.zeros(6).scatter_add_(0, accumulator.state_cell_indices, sampled["probabilities"]).view(2, 3)
    actual = env.extras["heatmap"]["Metrics/ResetProbs"]["values"]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(sampled["probabilities"][0] / sampled["probabilities"][1], torch.tensor(20.0))
    assert not torch.allclose(actual, torch.full_like(actual, 1.0 / 6))
    torch.testing.assert_close(actual.sum(dim=1), torch.full((2,), 1.0 / 2), atol=2e-4, rtol=0.0)
    torch.testing.assert_close(actual.sum(dim=0), torch.full((3,), 1.0 / 3))


def test_accumulator_success_grid_pools_outcomes_by_label_and_asset() -> None:
    """Compute each grid cell from the episodes actually measured in that cell."""
    accumulator = reset_accumulator.__new__(reset_accumulator)
    accumulator.state_cell_indices = torch.tensor([0, 0, 1, 2, 3, 3])
    accumulator._num_cells = 4
    accumulator.cell_success_rate = torch.empty((2, 2))
    accumulator.success_monitor = SimpleNamespace(
        success_buf=torch.tensor(
            [
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        success_size=torch.tensor([3, 1, 1, 0, 2, 2]),
    )

    accumulator._update_cell_success_rate()

    torch.testing.assert_close(accumulator.cell_success_rate[0], torch.tensor([0.5, 1.0]))
    assert torch.isnan(accumulator.cell_success_rate[1, 0])
    torch.testing.assert_close(accumulator.cell_success_rate[1, 1], torch.tensor(0.25))


def test_accumulator_success_monitor_tracks_every_state_slot() -> None:
    """Give every stored pose an independent curriculum history."""

    class Monitor:
        def __init__(self, cfg, num_partitions: int, partition_size: int, device: str):
            self.partition_size = partition_size
            self.success_rate = torch.zeros(num_partitions * partition_size, device=device)

    accumulator = reset_accumulator.__new__(reset_accumulator)
    accumulator.precollecting_phase = False
    accumulator._requested_reset_assets = []
    accumulator.reset_assets = []
    accumulator.state_data = torch.empty((24, 0))
    accumulator.sampled_slots = torch.full((8,), -1, dtype=torch.long)
    accumulator.success_monitor = None
    accumulator._success_monitor_cfg = SimpleNamespace(class_type=Monitor)
    env = SimpleNamespace(
        device="cpu",
        termination_manager=SimpleNamespace(get_term_cfg=lambda _: SimpleNamespace(func=SimpleNamespace())),
    )

    accumulator(env, torch.empty(0, dtype=torch.long), [], {}, 24, SuccessMonitorCfg(), SamplerCfg())

    assert accumulator.success_monitor.partition_size == 24
    assert accumulator.monitor_success_rate is accumulator.success_monitor.success_rate
    assert "synchronize" not in reset_accumulator.__dict__


def test_accumulator_exposes_only_the_requested_metric_schema() -> None:
    """Keep reset reporting to two grids and one scalar curve."""
    source = inspect.getsource(reset_accumulator.__call__)
    metrics = set(re.findall(r"Metrics/[A-Za-z_/]+", source))
    assert metrics == {"Metrics/ResetSuccessRate", "Metrics/ResetProbs", "Metrics/success_rate"}
