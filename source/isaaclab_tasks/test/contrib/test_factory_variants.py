# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the homogeneous Factory assembly catalog."""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager
from rsl_rl.modules import commit_normalization
from tensordict import TensorDict

import isaaclab.utils.math as math_utils
from isaaclab import cloner
from isaaclab.envs.mdp.events import randomize_rigid_body_material
from isaaclab.managers import CurriculumTermCfg, EventTermCfg, ManagerTermBase, ObservationTermCfg, SceneEntityCfg

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.contrib.nist.mdp.observations as factory_observations
import isaaclab_tasks.contrib.nist.mdp.variant_events as factory_events
import isaaclab_tasks.contrib.nist.utils.collision_analyzer as factory_collision_analyzer
import isaaclab_tasks.contrib.nist.utils.rigid_object_hasher as factory_rigid_object_hasher
from isaaclab_tasks.contrib.nist.assembly_profile import AssemblyProfile
from isaaclab_tasks.contrib.nist.assembly_variants import ASSEMBLY_VARIANT_NAMES, ASSEMBLY_VARIANTS
from isaaclab_tasks.contrib.nist.config.agents.models import (
    MLPEncoder,
    SimBaBlock,
    SimBaModel,
    SimBaModelCfg,
    SimBaNetwork,
)
from isaaclab_tasks.contrib.nist.config.agents.rsl_rl_ppo_cfg import FactoryVariantPPORunnerCfg
from isaaclab_tasks.contrib.nist.factory_env_cfg import FactoryBaseEnvCfg as StaticFactoryEnvCfg
from isaaclab_tasks.contrib.nist.factory_variant_env_cfg import (
    FactoryVariantEnvCfg,
    FactoryVariantObservationsCfg,
)
from isaaclab_tasks.contrib.nist.factory_variant_scene_cfg import FactoryVariantSceneCfg, _paired_clone_strategy
from isaaclab_tasks.contrib.nist.mdp.assembly_variants import AssemblyVariantContext
from isaaclab_tasks.contrib.nist.mdp.observations import (
    _scene_point_cloud_in_root_frame,
    asset_link_velocity_in_root_asset_frame,
    scene_point_cloud_b,
    target_asset_pose_in_root_asset_frame,
)
from isaaclab_tasks.contrib.nist.utils import reset_state
from isaaclab_tasks.contrib.nist.utils.event_combinators import ChainedResetTerms
from isaaclab_tasks.contrib.nist.utils.pose_offset import Offset
from isaaclab_tasks.contrib.nist.utils.sampling import SamplerCfg, UniformSamplingStrategyCfg
from isaaclab_tasks.contrib.nist.utils.variant_event_combinators import (
    PreparedTermChoice,
    variant_reset_accumulator,
)
from isaaclab_tasks.contrib.nist.variant_reset_env_cfg import VARIANT_ACCUMULATOR_RESET
from isaaclab_tasks.contrib.nistv2.board_layout import (
    AssemblySetCfg,
    BoardLayout,
    board_layout,
)
from isaaclab_tasks.contrib.nistv2.factory_env_cfg import FactoryBoardSceneCfg
from isaaclab_tasks.contrib.nistv2.mdp.assembly_state import (
    _update_assembly_state,
    assembly_progress_context,
    assembly_success_reward,
)
from isaaclab_tasks.contrib.nistv2.mdp.events import randomize_rigid_body_materials
from isaaclab_tasks.contrib.nistv2.mdp.metrics import BoardMetrics
from isaaclab_tasks.contrib.nistv2.mdp.reset import (
    ASSEMBLED,
    FALLEN,
    PARTIAL_ASSEMBLY,
    RESET_LABELS,
    TARGET,
    BalancedResetPlanner,
    ResetPlan,
    board_reset,
    initial_unfinished_time_out,
)
from isaaclab_tasks.contrib.nistv2.newton_selection import NewtonBodySelectorCfg
from isaaclab_tasks.core.lift.mdp.events_cfg import SuccessMonitorCfg
from isaaclab_tasks.utils import SuccessMonitor

DEFAULT_BOARD_LAYOUT = board_layout(AssemblySetCfg())
NEAR_PREASSEMBLED_LABEL = RESET_LABELS.index("start_near_preassembled")
BOARD_ASSEMBLY_VARIANTS = DEFAULT_BOARD_LAYOUT.variants
NUM_VARIANTS = DEFAULT_BOARD_LAYOUT.num_variants
NUM_SLOTS = DEFAULT_BOARD_LAYOUT.num_slots
HELD_ASSET_NAMES = DEFAULT_BOARD_LAYOUT.held_asset_names
FIXED_ASSET_NAMES = DEFAULT_BOARD_LAYOUT.fixed_asset_names
FIXTURE_VARIANT_INDICES = DEFAULT_BOARD_LAYOUT.fixture_variant_indices


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


def test_factory_variant_uses_the_nist_composition_root() -> None:
    """Register the variant task without preserving the former v2 API."""
    spec = gym.spec("IsaacContrib-Factory-Variant-Franka")
    assert spec.kwargs["env_cfg_entry_point"] == (
        "isaaclab_tasks.contrib.nist.factory_variant_env_cfg:FactoryVariantEnvCfg"
    )
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith("rsl_rl_ppo_cfg:FactoryVariantPPORunnerCfg")
    assert not {
        "IsaacContrib-Factory-V2-Franka",
        "IsaacContrib-Factory-V2-Video-Franka",
    }.intersection(gym.registry)


def test_scene_uses_one_ordered_pair_bank() -> None:
    """Build only the fixed and held assets from the same 20-entry catalog."""
    scene = FactoryVariantSceneCfg()
    board_material = scene.nistboard.spawn.physics_material

    assert isinstance(board_material, list)
    assert board_material[0].static_friction == 0.75
    assert board_material[0].dynamic_friction == 0.75
    assert board_material[1].contact_stiffness == 1.6e5
    assert board_material[1].contact_damping == 800.0
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


def test_full_board_package_owns_only_its_composition_and_runtime() -> None:
    """Keep the assembly catalog and reusable Factory terms owned by NIST v1."""
    contrib_root = Path(__file__).parents[2] / "isaaclab_tasks" / "contrib"
    package_root = contrib_root / "nistv2"
    files = {
        path.relative_to(package_root).as_posix() for path in package_root.rglob("*") if path.suffix in {".py", ".pyi"}
    }
    assert files == {
        "__init__.py",
        "board_layout.py",
        "config/__init__.py",
        "config/agents/__init__.py",
        "config/agents/__init__.pyi",
        "config/agents/rsl_rl_ppo_cfg.py",
        "factory_env.py",
        "factory_env_cfg.py",
        "mdp/__init__.py",
        "mdp/__init__.pyi",
        "mdp/assembly_state.py",
        "mdp/events.py",
        "mdp/metrics.py",
        "mdp/observations.py",
        "mdp/reset.py",
        "newton_selection.py",
    }
    assert not {"assembly_variants.py", "assembly_profile.py", "factory_assets_cfg.py"}.intersection(files)
    assert [path.relative_to(contrib_root).as_posix() for path in contrib_root.rglob("factory_variant_env_cfg.py")] == [
        "nist/factory_variant_env_cfg.py"
    ]
    scene_module = inspect.getmodule(FactoryBoardSceneCfg)
    assert scene_module is not None
    scene_source = inspect.getsource(scene_module)
    assert "resolve_presets" not in scene_source
    assert "def _fixed_asset_cfg" not in scene_source
    assert "def _held_asset_cfg" not in scene_source
    for path in package_root.rglob("*.py"):
        source = path.read_text()
        if path.name != "board_layout.py":
            assert "contrib.nist.assembly_variants" not in source
        assert "num_assemblies" not in source
        assert "FactoryBoardAssemblySetCfg" not in source
        assert re.search(r"\bassets_\d+\b", source) is None
        assert "def compose(" not in source
        assert ".compose(" not in source


def test_full_board_default_layout_matches_variant_catalog_with_one_slot() -> None:
    """Use the Variant catalog with one homogeneous fixed/held pair by default."""
    layout = DEFAULT_BOARD_LAYOUT
    held_rows = torch.tensor(layout.held_variant_rows())

    assert isinstance(layout, BoardLayout)
    assert layout.variant_names == tuple(name for name in ASSEMBLY_VARIANT_NAMES if name != "nut_thread_m4")
    assert tuple(variant.name for variant in layout.variants) == layout.variant_names
    assert NUM_VARIANTS == 19
    assert NUM_SLOTS == 1
    assert layout.held_asset_names == tuple(f"held_{index:02d}" for index in range(NUM_SLOTS))
    assert layout.asset_labels == (
        "N8",
        "N12",
        "N16",
        "Gs",
        "Gm",
        "Gl",
        "R4",
        "R8",
        "R12",
        "R16",
        "P4",
        "P8",
        "P12",
        "P16",
        "USB",
        "WP",
        "BNC",
        "DS",
        "RJ",
    )
    assert held_rows.shape == (NUM_VARIANTS, NUM_SLOTS)
    assert held_rows.shape == (NUM_VARIANTS, 1)
    torch.testing.assert_close(held_rows[:, 0], torch.arange(NUM_VARIANTS))


@pytest.mark.parametrize("num_slots", [1, 2, 3])
def test_full_board_slots_keep_the_complete_variant_bank(num_slots: int) -> None:
    """Change physical held bodies without narrowing the eligible variants."""
    layout = board_layout(AssemblySetCfg(num_slots=num_slots))
    rows = torch.tensor(layout.held_variant_rows())

    assert layout.num_variants == NUM_VARIANTS
    assert layout.num_slots == num_slots
    assert rows.shape == (NUM_VARIANTS, num_slots)
    assert (rows[:, 0].sort().values == torch.arange(NUM_VARIANTS)).all()
    if num_slots > 1:
        assert (rows.sort(dim=1).values.diff(dim=1) > 0).all()


def test_full_board_layout_applies_regex_selection_in_catalog_order() -> None:
    """Filter the canonical catalog without making regex match order observable."""
    layout = board_layout(AssemblySetCfg(include=r"^rod_insert_", exclude=r"_4mm$"))

    assert layout.variant_names == ("rod_insert_8mm", "rod_insert_12mm", "rod_insert_16mm")
    assert layout.asset_labels == ("R8", "R12", "R16")
    assert layout.num_variants == 3
    assert layout.num_slots == 1


def test_full_board_layout_deduplicates_shared_gear_fixture() -> None:
    """Stage one fixed gear base when two held gear variants share it."""
    layout = board_layout(AssemblySetCfg(include=r"^gear_mesh_(small|medium)$", exclude=None))
    rows = torch.tensor(layout.held_variant_rows())

    assert layout.variant_names == ("gear_mesh_small", "gear_mesh_medium")
    assert layout.fixture_index_by_variant == (0, 0)
    assert layout.fixture_variant_indices == (0,)
    assert layout.fixed_asset_names == ("fixed_00",)
    assert not layout.fixed_assets_are_variant_banks
    assert rows.shape == (2, 1)


def test_full_board_layout_rejects_invalid_and_empty_selections() -> None:
    """Reject malformed filters, invalid counts, and selections without an asset pair."""
    with pytest.raises(ValueError, match="Invalid assembly selection regex"):
        board_layout(AssemblySetCfg(include="["))
    with pytest.raises(ValueError, match="at least one asset pair"):
        board_layout(AssemblySetCfg(include=r"^missing_asset$", exclude=None))
    with pytest.raises(ValueError, match="must be positive"):
        board_layout(AssemblySetCfg(num_slots=0))
    with pytest.raises(ValueError, match="exceeds"):
        board_layout(AssemblySetCfg(num_slots=21))
    with pytest.raises(ValueError, match="at least two assembly variants"):
        board_layout(AssemblySetCfg(include=r"^rod_insert_8mm$", exclude=None))
    with pytest.raises(ValueError, match="Unknown assembly name"):
        board_layout(("missing_asset",))


def test_asset_count_is_not_a_task_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep physical slot count on the assembly-set configuration."""
    from isaaclab_tasks.utils.hydra import resolve_task_config

    monkeypatch.setattr(sys, "argv", ["train.py", "presets=assets_3"])
    with pytest.raises(ValueError, match="Unknown preset.*assets_3"):
        resolve_task_config("IsaacContrib-Factory-Board-Reset-Franka", "rsl_rl_cfg_entry_point")


def test_one_slot_override_keeps_every_variant_eligible(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve one physical asset without narrowing its mesh bank."""
    from isaaclab_tasks.utils.hydra import resolve_task_config

    monkeypatch.setattr(sys, "argv", ["train.py", "env.assembly_set.num_slots=1"])
    cfg, _ = resolve_task_config("IsaacContrib-Factory-Board-Reset-Franka", "rsl_rl_cfg_entry_point")
    cfg = cfg.replace(scene=cfg.scene.copy(), events=cfg.events.copy())

    held = [name for name in vars(cfg.scene) if name.startswith("held_")]
    fixed = [name for name in vars(cfg.scene) if name.startswith("fixed_")]
    assert held == ["held_00"]
    assert fixed == ["fixed_00"]
    assert len(cfg.scene.held_00.spawn.assets_cfg) == NUM_VARIANTS
    assert len(cfg.scene.fixed_00.spawn.assets_cfg) == NUM_VARIANTS
    assert cfg.scene.fixed_00.mesh_variants_enabled
    assert cfg.scene.assembly_contact.filter_prim_paths_expr == ["{ENV_REGEX_NS}/fixed_00/.*"]
    assert cfg.events.reset_board.params["variant_names"] == DEFAULT_BOARD_LAYOUT.variant_names
    assert cfg.events.reset_board.params["num_slots"] == 1
    assert cfg.events.reset_board.params["unfinished_count"] is None
    assert cfg.episode_length_s == 14.0
    assert cfg.terminations.time_out.params["fixed_horizon_s"] is None


@pytest.mark.parametrize(("num_slots", "expected_replay_rows"), [(1, 0), (2, 128), (10, 640), (19, 1216)])
def test_board_replay_budget_tracks_slot_count(
    monkeypatch: pytest.MonkeyPatch, num_slots: int, expected_replay_rows: int
) -> None:
    """Allocate bounded replay rows only for multi-slot board workloads."""
    from isaaclab_tasks.utils.hydra import resolve_task_config

    monkeypatch.setattr(sys, "argv", ["train.py", f"env.assembly_set.num_slots={num_slots}"])
    cfg, _ = resolve_task_config("IsaacContrib-Factory-Board-Reset-Franka", "rsl_rl_cfg_entry_point")
    cfg = cfg.replace(scene=cfg.scene.copy(), events=cfg.events.copy())

    assert cfg.sim.physics.collision_cfg.sdf_contact_replay_max_per_world == expected_replay_rows
    if num_slots == 1:
        assert cfg.sim.physics.solver_cfg.sleep_tolerance is None
    else:
        assert cfg.sim.physics.solver_cfg.sleep_tolerance == pytest.approx(0.003)


@pytest.mark.parametrize(("num_slots", "horizon_s"), [(1, 8.0), (2, 15.0)])
def test_mixed_horizon_experiment_overrides_resolve(
    monkeypatch: pytest.MonkeyPatch, num_slots: int, horizon_s: float
) -> None:
    """Resolve the fixed and dynamic horizon populations together."""
    from isaaclab_tasks.utils.hydra import resolve_task_config

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            f"env.assembly_set.num_slots={num_slots}",
            f"env.terminations.time_out.params.fixed_horizon_s={horizon_s}",
            "env.terminations.time_out.params.dynamic_env_count=1024",
            "env.events.reset_board.params.success_monitor_env_count=1024",
        ],
    )
    cfg, _ = resolve_task_config("IsaacContrib-Factory-Board-Reset-Franka", "rsl_rl_cfg_entry_point")
    cfg = cfg.replace(scene=cfg.scene.copy(), events=cfg.events.copy())

    assert cfg.events.reset_board.params["num_slots"] == num_slots
    assert cfg.events.reset_board.params["success_monitor_env_count"] == 1024
    assert cfg.terminations.time_out.params["fixed_horizon_s"] == horizon_s
    assert cfg.terminations.time_out.params["dynamic_env_count"] == 1024


def test_all_sockets_preset_is_independent_of_slot_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stage every eligible fixture without changing held capacity or reset difficulty."""
    from isaaclab_tasks.contrib.nistv2.factory_env_cfg import FactoryBoardEnvCfg
    from isaaclab_tasks.utils.hydra import resolve_task_config

    monkeypatch.setattr(sys, "argv", ["train.py", "presets=all_sockets", "env.assembly_set.num_slots=2"])
    cfg, _ = resolve_task_config("IsaacContrib-Factory-Board-Reset-Franka", "rsl_rl_cfg_entry_point")
    cfg = cfg.replace(scene=cfg.scene.copy(), events=cfg.events.copy())
    layout = board_layout(cfg.assembly_set)

    assert isinstance(cfg, FactoryBoardEnvCfg)
    assert cfg.assembly_set.spawn_all_sockets is True
    assert layout.num_slots == 2
    assert layout.num_fixed_slots == layout.num_fixtures
    assert not layout.fixed_assets_are_variant_banks
    assert len([name for name in vars(cfg.scene) if name.startswith("held_")]) == 2
    assert len([name for name in vars(cfg.scene) if name.startswith("fixed_")]) == layout.num_fixtures
    assert cfg.events.reset_board.params["unfinished_count"] is None
    assert cfg.events.reset_board.params["spawn_all_sockets"] is True


@pytest.mark.parametrize("num_slots", [1, 2, 3])
def test_matching_socket_count_tracks_held_capacity(num_slots: int) -> None:
    """Use no more physical socket bodies than simultaneous held assets."""
    layout = board_layout(AssemblySetCfg(num_slots=num_slots))

    assert layout.num_fixed_slots == num_slots
    assert layout.fixed_assets_are_variant_banks
    assert len(layout.clone_rows()) == layout.num_variants
    assert all(len(row) == 2 * num_slots for row in layout.clone_rows())


@pytest.mark.parametrize("num_slots", [1, 3])
def test_shared_gear_variants_need_one_physical_socket(num_slots: int) -> None:
    """Map all gear variants to one base without overlapping duplicate bodies."""
    layout = board_layout(AssemblySetCfg(include=r"^gear_mesh_", exclude=None, num_slots=num_slots))

    assert layout.fixture_index_by_variant == (0, 0, 0)
    assert layout.fixture_variant_indices == (0,)
    assert layout.fixed_asset_names == ("fixed_00",)
    assert not layout.fixed_assets_are_variant_banks


def test_full_board_two_slot_override_recomposes_scene_events_and_horizon(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive every cardinality-dependent config from the slot-count override."""
    from isaaclab_tasks.utils.hydra import resolve_task_config

    monkeypatch.setattr(sys, "argv", ["train.py", "env.assembly_set.num_slots=2"])
    cfg, _ = resolve_task_config("IsaacContrib-Factory-Board-Reset-Franka", "rsl_rl_cfg_entry_point")
    cfg = cfg.replace(scene=cfg.scene.copy(), events=cfg.events.copy())
    layout = board_layout(AssemblySetCfg(num_slots=2))

    assert cfg.assembly_set.num_slots == 2
    assert layout.num_variants == 19
    assert layout.num_slots == 2
    selected_scene_assets = {name for name in vars(cfg.scene) if name.startswith("fixed_") or name.startswith("held_")}
    assert selected_scene_assets == set(layout.fixed_asset_names + layout.held_asset_names)
    expected_held_usds = tuple(variant.held_asset.spawn.usd_path for variant in layout.variants)
    for name in layout.held_asset_names:
        held_asset = getattr(cfg.scene, name)
        assert tuple(asset.usd_path for asset in held_asset.spawn.assets_cfg) == expected_held_usds
    assert cfg.scene.clone_cfg.valid_set == [[0, 0, 0, *row] for row in layout.clone_rows()]
    assert cfg.scene.assembly_contact.filter_prim_paths_expr == [
        f"{{ENV_REGEX_NS}}/{name}/.*" for name in layout.fixed_asset_names
    ]

    assert cfg.events.reset_board.params["variant_names"] == layout.variant_names
    assert cfg.events.reset_board.params["num_slots"] == 2
    assert cfg.events.reset_board.params["unfinished_count"] is None
    assert cfg.events.assembly_state.params["held_bodies"].path == tuple(
        rf".*/{name}(?:/.*)?" for name in layout.held_asset_names
    )
    assert cfg.events.assembly_state.params["fixed_bodies"].path == tuple(
        rf".*/{name}(?:/.*)?" for name in layout.fixed_asset_names
    )
    assert tuple(asset.name for asset in cfg.events.held_asset_material.params["asset_cfgs"]) == layout.held_asset_names
    fixed_material_assets = tuple(asset.name for asset in cfg.events.fixed_asset_material.params["asset_cfgs"])
    assert fixed_material_assets == layout.fixed_asset_names
    assert cfg.events.robot_material.params["asset_cfg"] == SceneEntityCfg("robot")
    assert cfg.episode_length_s == 14.0 * layout.num_slots
    assert cfg.sim.physics.solver_cfg.enable_sleeping is True
    assert cfg.sim.physics.solver_cfg.nvmax is None


def test_one_slot_scene_stages_paired_variant_banks() -> None:
    """Build paired canonical fixed and held banks for the Variant-compatible default."""
    from isaaclab_tasks.contrib.nistv2.factory_env_cfg import FactoryBoardEnvCfg

    cfg = FactoryBoardEnvCfg()
    cfg.scene.num_envs = 4
    scene = cfg.scene
    fixed = [getattr(scene, name) for name in FIXED_ASSET_NAMES]
    held = [getattr(scene, name) for name in HELD_ASSET_NAMES]
    assert scene.table.spawn.fix_root_link
    assert scene.nistboard.spawn.fix_root_link
    assert all(spawn.fix_root_link for asset in fixed for spawn in asset.spawn.assets_cfg)
    cfgs = [scene.table, scene.nistboard, scene.robot, *fixed, *held]
    for cfg in cfgs:
        cfg.prim_path = cloner.expand_env_regex_ns(cfg.prim_path, scene.clone_cfg.clone_template)
    plan = cloner.make_clone_plan(
        cfgs,
        num_clones=scene.num_envs,
        env_spacing=scene.env_spacing,
        device="cpu",
        clone_strategy=scene.clone_cfg.clone_strategy,
        valid_set=torch.tensor(scene.clone_cfg.valid_set),
        env_template=scene.clone_cfg.clone_template,
    )

    assert scene.clone_cfg.valid_set == [[0, 0, 0, *row] for row in DEFAULT_BOARD_LAYOUT.clone_rows()]
    assert FIXED_ASSET_NAMES == ("fixed_00",)
    assert HELD_ASSET_NAMES == ("held_00",)
    assert scene.assembly_contact.prim_path == "{ENV_REGEX_NS}/held_.*/.*"
    assert scene.assembly_contact.filter_prim_paths_expr == [
        f"{{ENV_REGEX_NS}}/{name}/.*" for name in FIXED_ASSET_NAMES
    ]
    assert scene.assembly_contact.history_length == 1
    assert plan.clone_mask.shape == (3 + (len(fixed) + len(held)) * NUM_VARIANTS, 4)
    torch.testing.assert_close(plan.clone_mask.sum(dim=0), torch.full((4,), len(cfgs)))
    for asset in (*fixed, *held):
        assert asset.mesh_variants_enabled
        assert len(asset.spawn.assets_cfg) == NUM_VARIANTS
    assert [cfg.usd_path for cfg in fixed[0].spawn.assets_cfg] == [
        variant.fixed_asset.spawn.usd_path for variant in BOARD_ASSEMBLY_VARIANTS
    ]
    assert [cfg.usd_path for cfg in held[0].spawn.assets_cfg] == [
        variant.held_asset.spawn.usd_path for variant in BOARD_ASSEMBLY_VARIANTS
    ]


def test_one_slot_rl_cfg_matches_the_variant_training_contract() -> None:
    """Keep K=1 observations, learner, physics, and horizon aligned with Variant."""
    from isaaclab_tasks.contrib.nistv2.config.agents.rsl_rl_ppo_cfg import FactoryBoardPPORunnerCfg
    from isaaclab_tasks.contrib.nistv2.factory_env_cfg import (
        FactoryBoardEnvCfg,
        FactoryBoardPhysicsCfg,
        FactoryBoardRewardsCfg,
        FactoryBoardTerminationsCfg,
    )
    from isaaclab_tasks.utils import PresetCfg, resolve_presets

    cfg = FactoryBoardEnvCfg()
    variant = FactoryVariantEnvCfg()
    runner = FactoryBoardPPORunnerCfg()
    variant_runner = FactoryVariantPPORunnerCfg()
    spec = gym.spec("IsaacContrib-Factory-Board-Reset-Franka")

    assert FactoryBoardRewardsCfg.__bases__ == (object,)
    assert FactoryBoardTerminationsCfg.__bases__ == (object,)
    assert FactoryBoardPhysicsCfg.__bases__ == (PresetCfg,)
    assert list(vars(cfg.rewards)) == [
        "action_l2",
        "action_rate_l2",
        "joint_effort",
        "early_termination",
        "success_reward",
        "solver_reset_reward",
    ]
    assert list(vars(cfg.terminations)) == [
        "time_out",
        "assembly_contact_force",
        "oob",
        "progress_context",
        "abnormal",
        "success",
        "solver_reset_required",
    ]
    assert spec.entry_point == "isaaclab_tasks.contrib.nistv2.factory_env:FactoryBoardEnv"
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith("rsl_rl_ppo_cfg:FactoryBoardPPORunnerCfg")
    assert spec.kwargs["rsl_rl_mlp_cfg_entry_point"].endswith("rsl_rl_ppo_cfg:FactoryPPORunnerCfg")
    assert runner.obs_groups.default == variant_runner.obs_groups.default
    assert runner.actor.to_dict() == variant_runner.actor.to_dict()
    assert runner.critic.to_dict() == variant_runner.critic.to_dict()
    assert runner.init_at_random_ep_len is False
    assert cfg.sim.physics.newton_mjwarp.to_dict() == variant.sim.physics.to_dict()
    assert cfg.sim.physics.newton_mjwarp.solver_cfg.nconmax == 600
    assert cfg.sim.physics.newton_mjwarp.solver_cfg.nvmax is None
    assert cfg.sim.physics.newton_mjwarp.solver_cfg.enable_sleeping is None
    assert cfg.scene.num_envs == variant.scene.num_envs == 4096
    assert list(vars(cfg.observations.policy))[5:] == list(vars(variant.observations.policy))[5:]
    assert cfg.observations.policy.held_asset_in_fixed_asset_frame.history_length == 5
    assert cfg.observations.policy.end_effector_vel_lin_ang_b.history_length == 5
    assert cfg.observations.policy.joint_pos.history_length == 5
    assert cfg.observations.policy.prev_action.history_length == 5
    assert not hasattr(cfg.observations.policy, "fixed_asset_in_end_effector_frame")
    assert not hasattr(cfg.observations.policy, "joint_vel")
    assert cfg.observations.perception.scene_point_cloud.params == {
        "fixed_num_points": 256,
        "held_num_points": 256,
        "robot_num_points": 256,
        "flatten": True,
    }
    assert cfg.observations.perception.scene_point_cloud.history_length == 0
    assert list(vars(cfg.curriculum)) == ["metrics"]
    assert cfg.curriculum.metrics.func is not None
    assert cfg.episode_length_s == variant.episode_length_s == 14.0
    assert cfg.terminations.assembly_contact_force.func is not None
    assert cfg.events.assembly_state.params["contact_sensor_cfg"] == SceneEntityCfg("assembly_contact")
    assert cfg.events.assembly_state.params["contact_force_threshold"] == 50.0

    reset_params = cfg.events.reset_board.params
    monitor_cfg = reset_params["success_monitor_cfg"]
    sampler_cfg = reset_params["sampling"]
    unfinished_count = reset_params["unfinished_count"]
    assert isinstance(unfinished_count, PresetCfg)
    assert unfinished_count.to_dict() == {"default": None, "unfinished_1": 1}
    assert resolve_presets(unfinished_count) is None
    assert resolve_presets(unfinished_count, selected=("unfinished_1",)) == 1
    progress_goal = reset_params["progress_goal"]
    assert isinstance(progress_goal, PresetCfg)
    assert progress_goal.to_dict() == {"default": False, "progress_goal": True}
    assert resolve_presets(progress_goal) is False
    assert resolve_presets(progress_goal, selected=("progress_goal",)) is True
    assert isinstance(monitor_cfg, PresetCfg)
    assert resolve_presets(monitor_cfg).monitored_history_len == 5
    assert resolve_presets(monitor_cfg, selected=("success_estimator",)) is None
    assert reset_params["success_monitor_env_count"] is None
    assert sampler_cfg.eps == 1.0e-4
    assert len(sampler_cfg.strategies) == 2
    assert sampler_cfg.strategies[0].target == 0.66
    assert sampler_cfg.strategies[0].kappa == 1.0
    assert sampler_cfg.strategies[0].success_rate_bind == "success_rates"
    assert sampler_cfg.strategies[1].value_shift_bind == "value_shifts"
    assert "build_state_features" not in reset_params
    assert reset_params["fixed_asset_pose_range"] == {
        "x": (0.075, 0.25),
        "y": (-0.25, 0.25),
        "yaw": (-3.14, 3.14),
    }


@pytest.mark.parametrize(
    ("value_shift_preset", "value_shift_weight"),
    [(None, 0.0), ("value_shift", 0.0005), ("value_shift_005", 0.005), ("value_shift_05", 0.05)],
)
@pytest.mark.parametrize("success_estimator", [False, True])
@pytest.mark.parametrize("fixed_timeout", [False, True])
def test_full_board_curriculum_presets_compose_independently(
    monkeypatch: pytest.MonkeyPatch,
    value_shift_preset: str | None,
    value_shift_weight: float,
    success_estimator: bool,
    fixed_timeout: bool,
) -> None:
    """Resolve every learner-signal and timeout combination without coupled presets."""
    from isaaclab_tasks.utils.hydra import resolve_task_config

    selected = [
        name
        for name, enabled in (("success_estimator", success_estimator), ("fixed_timeout", fixed_timeout))
        if enabled
    ]
    if value_shift_preset is not None:
        selected.append(value_shift_preset)
    argv = ["train.py"]
    if selected:
        argv.append(f"presets={','.join(selected)}")
    monkeypatch.setattr(sys, "argv", argv)

    env_cfg, agent_cfg = resolve_task_config("IsaacContrib-Factory-Board-Reset-Franka", "rsl_rl_cfg_entry_point")
    state_curriculum = agent_cfg.algorithm.state_curriculum_cfg
    strategies = env_cfg.events.reset_board.params["sampling"].strategies

    assert (state_curriculum.value_shift_cfg is not None) is (value_shift_preset is not None)
    assert (state_curriculum.success_estimator_cfg is not None) is success_estimator
    assert agent_cfg.algorithm.num_learning_epochs == 5
    assert agent_cfg.algorithm.num_mini_batches == 4
    assert strategies[0].weight == 1.0
    assert strategies[0].success_rate_bind == "success_rates"
    assert strategies[1].weight == value_shift_weight
    assert (env_cfg.events.reset_board.params["success_monitor_cfg"] is None) is success_estimator
    assert "build_state_features" not in env_cfg.events.reset_board.params
    if success_estimator:
        estimator_cfg = state_curriculum.success_estimator_cfg.to_dict()
        assert not {"num_batches", "batch_size", "evaluation_batch_size", "prior_count"} & estimator_cfg.keys()
    assert env_cfg.terminations.time_out.params["dynamic"] is not fixed_timeout
    assert env_cfg.terminations.time_out.params["fixed_horizon_s"] is None
    assert env_cfg.terminations.time_out.params["dynamic_env_count"] is None


@pytest.mark.parametrize(
    ("selected", "progress_goal", "gamma"),
    [
        ((), False, 0.995),
        (("gamma_0999",), False, 0.999),
        (("progress_goal", "gamma_0999"), True, 0.999),
        (("progress_goal", "gamma_09999"), True, 0.9999),
    ],
)
def test_full_board_progress_goal_and_discount_presets_are_independent(
    monkeypatch: pytest.MonkeyPatch, selected: tuple[str, ...], progress_goal: bool, gamma: float
) -> None:
    """Toggle hidden progress goals, infinite horizon, and discount independently."""
    from isaaclab_tasks.utils.hydra import resolve_task_config

    monkeypatch.setattr(sys, "argv", ["train.py"])
    baseline_env_cfg, baseline_agent_cfg = resolve_task_config(
        "IsaacContrib-Factory-Board-Reset-Franka", "rsl_rl_cfg_entry_point"
    )
    argv = ["train.py"]
    if selected:
        argv.append(f"presets={','.join(selected)}")
    monkeypatch.setattr(sys, "argv", argv)
    env_cfg, agent_cfg = resolve_task_config("IsaacContrib-Factory-Board-Reset-Franka", "rsl_rl_cfg_entry_point")

    assert env_cfg.events.reset_board.params["progress_goal"] is progress_goal
    assert env_cfg.terminations.time_out.params["enabled"] is not progress_goal
    assert agent_cfg.algorithm.gamma == pytest.approx(gamma)
    assert env_cfg.observations.to_dict() == baseline_env_cfg.observations.to_dict()
    assert agent_cfg.obs_groups == baseline_agent_cfg.obs_groups


def test_full_board_material_randomization_matches_v1() -> None:
    """Randomize every training asset with V1 friction ranges and no duplicate inertia event."""
    from isaaclab_tasks.contrib.nistv2.factory_env_cfg import FactoryBoardEnvCfg

    events = FactoryBoardEnvCfg().events
    assert events.held_asset_material.mode == events.fixed_asset_material.mode == "startup"
    assert events.held_asset_material.func is events.fixed_asset_material.func is randomize_rigid_body_materials
    material_params = {
        "static_friction_range": (0.4, 1.0),
        "dynamic_friction_range": (0.4, 1.0),
        "restitution_range": (0.0, 0.0),
        "num_buckets": 64,
    }
    assert events.held_asset_material.params == material_params | {
        "asset_cfgs": tuple(SceneEntityCfg(name) for name in HELD_ASSET_NAMES)
    }
    assert events.fixed_asset_material.params == material_params | {
        "asset_cfgs": tuple(SceneEntityCfg(name) for name in FIXED_ASSET_NAMES)
    }
    assert events.robot_material.func is randomize_rigid_body_material
    assert events.robot_material.params == {
        "static_friction_range": (0.75, 0.75),
        "dynamic_friction_range": (0.75, 0.75),
        "restitution_range": (0.0, 0.0),
        "num_buckets": 64,
        "asset_cfg": SceneEntityCfg("robot"),
    }
    assert all("inertia" not in name for name in vars(events))


def test_full_board_material_dispatcher_runs_each_asset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Apply the visible material configuration to every selected asset."""
    import isaaclab_tasks.contrib.nistv2.mdp.events as events_module

    calls = []

    class MaterialTerm:
        def __init__(self, cfg: EventTermCfg, env) -> None:
            self.asset_cfg = cfg.params["asset_cfg"]

        def __call__(self, env, env_ids, *params) -> None:
            calls.append((env, env_ids, self.asset_cfg, params))

    monkeypatch.setattr(events_module, "randomize_rigid_body_material", MaterialTerm)
    env = object()
    asset_cfgs = (SceneEntityCfg("held_00"), SceneEntityCfg("held_01"))
    params = {
        "asset_cfgs": asset_cfgs,
        "static_friction_range": (0.4, 1.0),
        "dynamic_friction_range": (0.4, 1.0),
        "restitution_range": (0.0, 0.0),
        "num_buckets": 64,
    }
    cfg = EventTermCfg(func=randomize_rigid_body_materials, mode="startup", params=params)
    term = randomize_rigid_body_materials(cfg, env)
    term(env, None, **params)

    assert [call[2] for call in calls] == list(asset_cfgs)
    assert all(call[:2] == (env, None) for call in calls)


def test_newton_body_selector_preserves_requested_order_and_aliases() -> None:
    """Resolve an ordered dense body table while allowing shared fixture bodies."""
    model = SimpleNamespace(
        world_count=2,
        body_label=[
            "/World/Ground",
            "/World/envs/env_0/held_00/mesh_a",
            "/World/envs/env_0/fixed_gear_base/base",
            "/World/envs/env_1/held_00/mesh_b",
            "/World/envs/env_1/fixed_gear_base/base",
        ],
        body_world=np.array([-1, 0, 0, 1, 1], dtype=np.int32),
    )
    selection = NewtonBodySelectorCfg(
        path=(r".*/held_00(?:/.*)?", r".*/fixed_gear_base(?:/.*)?", r".*/fixed_gear_base(?:/.*)?")
    ).resolve(model)

    assert selection.ids == ((1, 2, 2), (3, 4, 4))


def test_newton_body_selector_rejects_incomplete_worlds() -> None:
    """Fail before training if a requested body is absent from any world."""
    model = SimpleNamespace(
        world_count=2,
        body_label=["/World/envs/env_0/held_00/mesh"],
        body_world=np.array([0], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="matched no body in world 1"):
        NewtonBodySelectorCfg(path=r".*/held_00(?:/.*)?").resolve(model)


def test_one_slot_assembly_state_keeps_canonical_variant_channels() -> None:
    """Keep canonical frames and catch contact with a non-target fixture."""
    body_q = wp.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=wp.transformf,
        device="cpu",
    )
    held_ids = wp.array([[2]], dtype=wp.int32, device="cpu")
    board_ids = wp.array([[1]], dtype=wp.int32, device="cpu")
    root_ids = wp.array([[0]], dtype=wp.int32, device="cpu")
    variant_ids = wp.array([[2]], dtype=wp.uint8, device="cpu")
    identity = wp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]] * 3, dtype=wp.transformf, device="cpu")
    contact_forces = wp.zeros((1, 3), dtype=wp.vec3f, device="cpu")
    wp.to_torch(contact_forces)[0, 0, 0] = 2.0
    assembly_frames = wp.empty((1, 6), dtype=wp.transformf, device="cpu")
    variant_active = wp.empty((1, 3), dtype=wp.float32, device="cpu")
    asset_assembled = wp.empty((1, 3), dtype=wp.bool, device="cpu")
    all_success = wp.empty(1, dtype=wp.bool, device="cpu")
    task_success = wp.empty(1, dtype=wp.bool, device="cpu")
    contact_exceeded = wp.empty(1, dtype=wp.bool, device="cpu")
    out_of_bound = wp.empty(1, dtype=wp.bool, device="cpu")

    wp.launch(
        _update_assembly_state,
        dim=1,
        inputs=[
            body_q,
            held_ids,
            board_ids,
            root_ids,
            variant_ids,
            wp.array([1], dtype=wp.uint8, device="cpu"),
            wp.array([1], dtype=wp.uint8, device="cpu"),
            wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3f, device="cpu"),
            identity,
            identity,
            identity,
            3,
            1,
            3,
            contact_forces,
            1.0,
            0.001,
            wp.vec3f(-10.0, -10.0, -10.0),
            wp.vec3f(10.0, 10.0, 10.0),
        ],
        outputs=[
            assembly_frames,
            variant_active,
            asset_assembled,
            all_success,
            task_success,
            contact_exceeded,
            out_of_bound,
        ],
        device="cpu",
    )

    torch.testing.assert_close(wp.to_torch(variant_active), torch.tensor([[0.0, 0.0, 1.0]]))
    torch.testing.assert_close(wp.to_torch(asset_assembled), torch.tensor([[False, False, True]]))
    assert wp.to_torch(all_success).item()
    assert wp.to_torch(task_success).item()
    assert wp.to_torch(contact_exceeded).item()
    frames = wp.to_torch(assembly_frames)
    torch.testing.assert_close(frames[0, 0], torch.zeros(7))
    torch.testing.assert_close(frames[0, 2], torch.zeros(7))
    torch.testing.assert_close(frames[0, [1, 3, 4, 5], 0], torch.ones(4))


def test_assembly_state_uses_net_progress_from_the_preassembled_baseline() -> None:
    """Require net assembled-count gain without crediting removal and reinsertion."""
    num_envs = 5
    num_slots = 3
    body_q = wp.array(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
        dtype=wp.transformf,
        device="cpu",
    )
    held_ids = wp.array([[1, 1, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=wp.int32, device="cpu")
    root_ids = wp.zeros((num_envs, 1), dtype=wp.int32, device="cpu")
    variant_ids = wp.array([[0, 1, 2]] * num_envs, dtype=wp.uint8, device="cpu")
    identity = wp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]] * 3, dtype=wp.transformf, device="cpu")
    assembly_frames = wp.empty((num_envs, 6), dtype=wp.transformf, device="cpu")
    variant_active = wp.empty((num_envs, 3), dtype=wp.float32, device="cpu")
    asset_assembled = wp.empty((num_envs, 3), dtype=wp.bool, device="cpu")
    all_success = wp.empty(num_envs, dtype=wp.bool, device="cpu")
    task_success = wp.empty(num_envs, dtype=wp.bool, device="cpu")
    contact_exceeded = wp.empty(num_envs, dtype=wp.bool, device="cpu")
    out_of_bound = wp.empty(num_envs, dtype=wp.bool, device="cpu")

    wp.launch(
        _update_assembly_state,
        dim=num_envs,
        inputs=[
            body_q,
            held_ids,
            root_ids,
            root_ids,
            variant_ids,
            wp.array([2, 2, 3, 3, 3], dtype=wp.uint8, device="cpu"),
            wp.array([1, 1, 2, 2, 3], dtype=wp.uint8, device="cpu"),
            wp.zeros(num_envs, dtype=wp.vec3f, device="cpu"),
            identity,
            identity,
            identity,
            3,
            num_slots,
            1,
            wp.zeros((num_envs * num_slots, 1), dtype=wp.vec3f, device="cpu"),
            1.0,
            0.001,
            wp.vec3f(-10.0, -10.0, -10.0),
            wp.vec3f(10.0, 10.0, 10.0),
        ],
        outputs=[
            assembly_frames,
            variant_active,
            asset_assembled,
            all_success,
            task_success,
            contact_exceeded,
            out_of_bound,
        ],
        device="cpu",
    )

    torch.testing.assert_close(wp.to_torch(task_success), torch.tensor([False, False, False, True, True]))
    torch.testing.assert_close(wp.to_torch(all_success), torch.tensor([False, False, False, False, True]))


def test_full_board_reset_planner_balances_accepted_marginals() -> None:
    """Balance stored states softly while preserving all per-slot reset choices."""
    torch.manual_seed(7)
    num_slots = 3
    planner = BalancedResetPlanner(num_slots, "cpu", num_variants=NUM_VARIANTS)
    coarse_counts = torch.zeros(3, dtype=torch.long)
    for batch in range(128):
        variants = (torch.arange(512)[:, None] + torch.arange(num_slots)[None, :] + batch * num_slots) % NUM_VARIANTS
        unfinished_before = planner.unfinished_counts.clone()
        progress_goals_before = planner.progress_goal_counts.clone()
        focus_slots_before = planner.focus_slot_counts.clone()
        unfinished_labels_before = planner.unfinished_label_counts.clone()
        cells_before = planner.cell_counts.clone()
        plan = planner.sample(variants)
        torch.testing.assert_close(planner.unfinished_counts, unfinished_before)
        torch.testing.assert_close(planner.progress_goal_counts, progress_goals_before)
        torch.testing.assert_close(planner.focus_slot_counts, focus_slots_before)
        torch.testing.assert_close(planner.unfinished_label_counts, unfinished_labels_before)
        torch.testing.assert_close(planner.cell_counts, cells_before)
        rows = torch.arange(len(plan.focus_slot))
        near_preassembled = plan.label == NEAR_PREASSEMBLED_LABEL
        torch.testing.assert_close(plan.required_assembly_gain, plan.unfinished_count)
        assert plan.unfinished[rows[~near_preassembled], plan.focus_slot[~near_preassembled]].all()
        assert (~plan.unfinished[rows[near_preassembled], plan.focus_slot[near_preassembled]]).all()
        assert (plan.unfinished_count[near_preassembled] < num_slots).all()
        assert (plan.slot_state[~plan.unfinished] == ASSEMBLED).all()
        assert (plan.slot_state[rows[~near_preassembled], plan.focus_slot[~near_preassembled]] == TARGET).all()
        assert (plan.slot_state[rows[near_preassembled], plan.focus_slot[near_preassembled]] == ASSEMBLED).all()
        coarse = plan.slot_state[plan.slot_state < ASSEMBLED].long()
        coarse_counts += torch.bincount(coarse, minlength=3)
        planner.accept(plan, variants, rows)

    unfinished_label_counts = planner.unfinished_label_counts
    assert planner.unfinished_counts.sum() == 65536
    assert planner.progress_goal_counts.sum() == 65536
    assert torch.equal(planner.progress_goal_counts.diag(), planner.unfinished_counts)
    assert planner.focus_slot_counts.sum() == 65536
    assert unfinished_label_counts.sum() == 65536
    assert planner.cell_counts.sum() == 65536
    assert planner.unfinished_counts.max() < planner.unfinished_counts.min() * 1.1
    assert planner.focus_slot_counts.max() < planner.focus_slot_counts.min() * 1.1
    assert (planner.cell_counts.max(dim=1).values < planner.cell_counts.min(dim=1).values * 1.25).all()
    assert coarse_counts.max() < coarse_counts.min() * 1.05
    assert (unfinished_label_counts[:-1].max(dim=1).values < unfinished_label_counts[:-1].min(dim=1).values * 1.1).all()
    final_feasible = torch.cat(
        (
            unfinished_label_counts[-1, :NEAR_PREASSEMBLED_LABEL],
            unfinished_label_counts[-1, NEAR_PREASSEMBLED_LABEL + 1 :],
        )
    )
    assert final_feasible.max() < final_feasible.min() * 1.1
    assert (unfinished_label_counts[:-1, NEAR_PREASSEMBLED_LABEL] > 0).all()
    assert unfinished_label_counts[-1, NEAR_PREASSEMBLED_LABEL] == 0


def test_full_board_reset_planner_balances_intermediate_progress_goals() -> None:
    """Cover every reachable progress target without enforcing exact batch quotas."""
    torch.manual_seed(11)
    num_slots = 4
    planner = BalancedResetPlanner(num_slots, "cpu", num_variants=NUM_VARIANTS, progress_goal=True)
    for batch in range(64):
        variants = (torch.arange(512)[:, None] + torch.arange(num_slots)[None, :] + batch) % NUM_VARIANTS
        plan = planner.sample(variants)
        assert ((plan.required_assembly_gain >= 1) & (plan.required_assembly_gain <= plan.unfinished_count)).all()
        planner.accept(plan, variants, torch.arange(len(plan.unfinished_count)))

    for unfinished, counts in enumerate(planner.progress_goal_counts, 1):
        reachable = counts[:unfinished]
        assert (reachable > 0).all()
        assert reachable.max() < reachable.min() * 1.2
        assert (counts[unfinished:] == 0).all()


def test_full_board_reset_planner_can_fix_unfinished_count() -> None:
    """Keep all other reset choices active when a preset fixes the unfinished count."""
    num_slots = 3
    planner = BalancedResetPlanner(num_slots, "cpu", unfinished_count=1, num_variants=NUM_VARIANTS)
    variants = (torch.arange(512)[:, None] + torch.arange(num_slots)[None, :]) % NUM_VARIANTS
    plan = planner.sample(variants)
    rows = torch.arange(512)

    assert (plan.unfinished_count == 1).all()
    assert (plan.required_assembly_gain == 1).all()
    assert (plan.unfinished.sum(dim=1) == 1).all()
    near_preassembled = plan.label == NEAR_PREASSEMBLED_LABEL
    assert plan.unfinished[rows[~near_preassembled], plan.focus_slot[~near_preassembled]].all()
    assert (~plan.unfinished[rows[near_preassembled], plan.focus_slot[near_preassembled]]).all()
    planner.accept(plan, variants, rows)
    torch.testing.assert_close(planner.unfinished_counts, torch.tensor([512, 0, 0]))
    assert planner.unfinished_label_counts.sum() == 512
    with pytest.raises(ValueError, match="between 1 and 3"):
        BalancedResetPlanner(num_slots, "cpu", unfinished_count=4, num_variants=NUM_VARIANTS)


@pytest.mark.parametrize("num_slots", [1, 3])
def test_full_board_reset_excludes_near_preassembled_when_every_slot_is_unfinished(num_slots: int) -> None:
    """Do not reinterpret the sequencing reset when no assembled focus exists."""
    planner = BalancedResetPlanner(num_slots, "cpu", unfinished_count=num_slots, num_variants=NUM_VARIANTS)
    variants = (torch.arange(4096)[:, None] + torch.arange(num_slots)[None, :]) % NUM_VARIANTS
    plan = planner.sample(variants)
    rows = torch.arange(len(plan.focus_slot))

    assert (plan.label != NEAR_PREASSEMBLED_LABEL).all()
    assert plan.unfinished.all()
    assert (plan.slot_state[rows, plan.focus_slot] == TARGET).all()
    planner.accept(plan, variants, rows)
    assert planner.unfinished_label_counts[:, NEAR_PREASSEMBLED_LABEL].sum() == 0


def test_full_board_reset_samples_v1_focus_fixture_pose() -> None:
    """Sample the focus fixture first, then solve the board pose from its offset."""
    reset = board_reset.__new__(board_reset)
    reset._env = SimpleNamespace(device="cpu")
    reset.num_variants = NUM_VARIANTS
    reset._board_default_pose = torch.tensor([0.25, -0.1, 0.04, 0.0, 0.0, 0.0, 1.0])
    reset._board_offsets = torch.tensor([variant.board_offset.pose for variant in BOARD_ASSEMBLY_VARIANTS])
    offset_pos, offset_quat = reset._board_offsets[:, :3], reset._board_offsets[:, 3:]
    inverse_quat = math_utils.quat_inv(offset_quat)
    inverse_pos = -math_utils.quat_apply(inverse_quat, offset_pos)
    reset._inverse_board_offsets = torch.cat((inverse_pos, inverse_quat), dim=1)
    reset._fixed_asset_pose_range = torch.tensor(
        [[0.12, 0.12], [-0.08, -0.08], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.6, 0.6]]
    )
    focus_variants = torch.tensor([0, NUM_VARIANTS - 1])

    board_pose = reset._sample_board_pose(focus_variants)
    focus_offsets = reset._board_offsets[focus_variants]
    focus_pos, focus_quat = math_utils.combine_frame_transforms(
        board_pose[:, :3], board_pose[:, 3:], focus_offsets[:, :3], focus_offsets[:, 3:]
    )
    nominal_pos, nominal_quat = math_utils.combine_frame_transforms(
        reset._board_default_pose[:3].expand(2, -1),
        reset._board_default_pose[3:].expand(2, -1),
        focus_offsets[:, :3],
        focus_offsets[:, 3:],
    )
    yaw = torch.full((2,), 0.6)
    zero = torch.zeros_like(yaw)
    expected_quat = math_utils.quat_mul(nominal_quat, math_utils.quat_from_euler_xyz(zero, zero, yaw))

    torch.testing.assert_close(focus_pos, nominal_pos + torch.tensor([0.12, -0.08, 0.0]))
    torch.testing.assert_close(focus_quat, expected_quat)
    assert not torch.allclose(board_pose[0], board_pose[1])


def test_full_board_reset_labels_follow_curriculum_order() -> None:
    """Order reset categories from the broadest state to the easiest state."""
    assert RESET_LABELS == (
        "start_random",
        "start_near_preassembled",
        "start_near_grasped",
        "start_pick",
        "start_grasped",
        "grasped_near_goal",
        "start_near_assembled",
        "start_assembled",
    )


def test_full_board_reset_places_robot_near_preassembled_focus() -> None:
    """Reuse the assembled-focus IK path without moving the completed asset."""
    reset = board_reset.__new__(board_reset)
    reset._env = SimpleNamespace(device="cpu", scene=SimpleNamespace(env_origins=torch.tensor([[10.0, 0.0, 0.0]])))
    poses = torch.tensor([[[0.2, 0.1, 0.3, 0.0, 0.0, 0.0, 1.0], [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]]])
    variants = torch.tensor([[2, 5]])
    plan = ResetPlan(
        unfinished_count=torch.tensor([1]),
        required_assembly_gain=torch.tensor([1]),
        unfinished=torch.tensor([[True, False]]),
        focus_slot=torch.tensor([1]),
        label=torch.tensor([NEAR_PREASSEMBLED_LABEL]),
        slot_state=torch.tensor([[FALLEN, ASSEMBLED]], dtype=torch.uint8),
    )
    calls: dict[str, object] = {}

    def apply_offset(pose: torch.Tensor, variant_ids: torch.Tensor, offset_index: int):
        calls["pose"] = pose.clone()
        calls["variant_ids"] = variant_ids.clone()
        calls["offset_index"] = offset_index
        return torch.tensor([[4.0, 5.0, 6.0]]), torch.tensor([[0.0, 0.0, 0.0, 1.0]])

    def solve_end_effector(env_ids, reference_pos, reference_quat, ranges, upright, iterations, pose_tolerance):
        calls["solve"] = (env_ids, reference_pos, reference_quat, ranges, upright, iterations, pose_tolerance)
        return torch.ones(len(env_ids), dtype=torch.bool)

    reset._apply_offset = apply_offset
    reset._held_target_ranges = lambda count, label: torch.full((count, 6, 2), 7.0)
    reset._solve_end_effector = solve_end_effector
    reset._set_gripper = lambda env_ids, variant_ids, flexible: calls.update(
        gripper=(env_ids.clone(), variant_ids.clone(), flexible)
    )

    original_poses = poses.clone()
    valid = reset._reset_focus(torch.tensor([0]), poses, variants, plan)

    torch.testing.assert_close(valid, torch.tensor([True]))
    torch.testing.assert_close(poses, original_poses)
    torch.testing.assert_close(calls["pose"], torch.tensor([[11.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]]))
    torch.testing.assert_close(calls["variant_ids"], torch.tensor([5]))
    assert calls["offset_index"] == 2
    _, reference_pos, _, ranges, upright, iterations, tolerance = calls["solve"]
    torch.testing.assert_close(reference_pos, torch.tensor([[4.0, 5.0, 6.0]]))
    torch.testing.assert_close(ranges, torch.full((1, 6, 2), 7.0))
    assert not upright
    assert iterations == (25, 35)
    assert tolerance == (0.001, 0.05)
    _, gripper_variants, flexible = calls["gripper"]
    torch.testing.assert_close(gripper_variants, torch.tensor([5]))
    assert not flexible


def test_full_board_start_pick_uses_v1_drop_support() -> None:
    """Drop every held slot across the full V1 workspace before settling."""

    class Asset:
        def write_mesh_variant_to_sim(self, variant_ids: torch.Tensor, env_ids: torch.Tensor) -> None:
            self.variant_ids = variant_ids.clone()

        def write_root_link_pose_to_sim(self, pose: torch.Tensor, env_ids: torch.Tensor) -> None:
            self.pose = pose.clone()

        def write_root_com_velocity_to_sim(self, velocity: torch.Tensor, env_ids: torch.Tensor) -> None:
            self.velocity = velocity.clone()

    num_envs = 4096
    reset = board_reset.__new__(board_reset)
    reset.num_slots = 2
    reset._held = (Asset(), Asset())
    reset._env = SimpleNamespace(device="cpu", scene=SimpleNamespace(env_origins=torch.zeros((num_envs, 3))))
    reset._zero_root_velocity = torch.zeros((num_envs, 6))
    variants = torch.zeros((num_envs, reset.num_slots), dtype=torch.long)

    torch.manual_seed(7)
    reset._drop_held_assets(torch.arange(num_envs), variants)

    for asset in reset._held:
        positions = asset.pose[:, :3]
        assert ((positions[:, 0] >= 0.0) & (positions[:, 0] <= 0.5)).all()
        assert ((positions[:, 1] >= -0.5) & (positions[:, 1] <= 0.5)).all()
        assert ((positions[:, 2] >= 0.12) & (positions[:, 2] <= 0.14)).all()
        assert positions[:, 0].max() - positions[:, 0].min() > 0.45
        assert positions[:, 1].max() - positions[:, 1].min() > 0.9
        torch.testing.assert_close(asset.velocity, torch.zeros((num_envs, 6)))


def test_full_board_fallen_bank_keeps_board_and_stores_board_relative_poses() -> None:
    """Settle against the board and retain the resulting board-relative configuration."""

    class Asset:
        def __init__(self, pose: torch.Tensor) -> None:
            self.data = SimpleNamespace(
                root_link_pose_w=SimpleNamespace(torch=pose.clone()),
                root_pos_w=SimpleNamespace(torch=pose[:, :3].clone()),
                root_quat_w=SimpleNamespace(torch=pose[:, 3:].clone()),
            )

        def write_root_link_pose_to_sim_index(self, *, root_pose: torch.Tensor, env_ids: torch.Tensor) -> None:
            self.pose = root_pose.clone()
            self.data.root_link_pose_w.torch[env_ids] = root_pose
            self.data.root_pos_w.torch[env_ids] = root_pose[:, :3]
            self.data.root_quat_w.torch[env_ids] = root_pose[:, 3:]

        def write_root_com_velocity_to_sim(self, velocity: torch.Tensor, env_ids: torch.Tensor) -> None:
            self.velocity = velocity.clone()

    origin = torch.tensor([[2.0, 1.0, 0.0]])
    yaw = torch.tensor([torch.pi / 2])
    zero = torch.zeros_like(yaw)
    board_pose = torch.cat((torch.tensor([[2.5, 0.75, 0.1]]), math_utils.quat_from_euler_xyz(zero, zero, yaw)), dim=1)
    local_pose = torch.tensor([[0.2, 0.1, 0.2, 0.0, 0.0, 0.0, 1.0]])
    held_pos, held_quat = math_utils.combine_frame_transforms(
        board_pose[:, :3], board_pose[:, 3:], local_pose[:, :3], local_pose[:, 3:]
    )
    held_pose = torch.cat((held_pos, held_quat), dim=1)
    reset = board_reset.__new__(board_reset)
    reset.layout = board_layout(AssemblySetCfg(num_slots=1))
    reset.num_slots = 1
    reset.num_variants = reset.layout.num_variants
    reset._fallen_capacity = 1
    reset._settle_steps = 1
    reset._board_default_pose = board_pose[0].clone()
    reset._board_default_pose[:3] -= origin[0]
    reset._board = Asset(board_pose)
    reset._held = (Asset(held_pose),)
    reset._fixed = (Asset(board_pose),)
    reset._zero_root_velocity = torch.zeros((1, 6))
    reset.fixed_kind_by_slot = torch.zeros((1, 1), dtype=torch.int32)
    reset._env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        cfg=SimpleNamespace(decimation=1),
        step_dt=0.01,
        sim=SimpleNamespace(
            physics_manager=SimpleNamespace(handles_decimation=lambda: True),
            step=lambda render: None,
        ),
        scene=SimpleNamespace(env_origins=origin, update=lambda dt: None),
    )
    reset._park_robot = lambda env_ids: None
    reset._drop_held_assets = lambda env_ids, variants: None

    poses, variants = reset._precollect_fallen()

    torch.testing.assert_close(reset._board.pose, board_pose)
    fixed_parked = board_pose.clone()
    fixed_parked[:, 2] = 5.0
    torch.testing.assert_close(reset._fixed[0].pose, fixed_parked)
    torch.testing.assert_close(poses, local_pose[:, None])
    torch.testing.assert_close(variants, torch.zeros((1, 1), dtype=torch.uint8))
    torch.testing.assert_close(reset.fixed_kind_by_slot, torch.tensor([[-1]], dtype=torch.int32))


def test_full_board_fallen_poses_follow_the_sampled_board() -> None:
    """Move a pre-settled board-relative pose with the randomized board frame."""
    reset = board_reset.__new__(board_reset)
    reset.num_slots = 1
    reset._profiles = ()
    env_ids = torch.arange(2)
    variants = torch.zeros((2, 1), dtype=torch.long)
    plan = SimpleNamespace(slot_state=torch.full((2, 1), FALLEN, dtype=torch.uint8))
    poses = torch.tensor(
        [
            [[0.2, 0.1, 0.2, 0.0, 0.0, 0.0, 1.0]],
            [[0.2, 0.1, 0.2, 0.0, 0.0, 0.0, 1.0]],
        ]
    )
    yaw = torch.tensor([0.0, torch.pi / 2])
    zero = torch.zeros_like(yaw)
    board_pose = torch.cat(
        (torch.tensor([[0.5, -0.2, 0.1], [0.7, 0.3, 0.1]]), math_utils.quat_from_euler_xyz(zero, zero, yaw)),
        dim=1,
    )

    reset._compose_held_poses(env_ids, poses, variants, plan, board_pose)

    expected_pos, expected_quat = math_utils.combine_frame_transforms(
        board_pose[:, :3], board_pose[:, 3:], poses.new_tensor([[0.2, 0.1, 0.2]]).expand(2, -1)
    )
    torch.testing.assert_close(poses[:, 0], torch.cat((expected_pos, expected_quat), dim=1))


def test_full_board_reset_writes_only_matching_fixtures() -> None:
    """Switch sparse fixture slots, share the gear base, and park unused slots."""

    class Asset:
        def write_root_link_pose_to_sim_index(self, *, root_pose: torch.Tensor, env_ids: torch.Tensor) -> None:
            self.pose = root_pose.clone()
            self.env_ids = env_ids.clone()

        def write_mesh_variant_to_sim(self, variant_ids: torch.Tensor, env_ids: torch.Tensor) -> None:
            self.variant_ids = variant_ids.clone()

        def write_root_com_velocity_to_sim(self, velocity: torch.Tensor, env_ids: torch.Tensor) -> None:
            self.velocity = velocity.clone()

    reset = board_reset.__new__(board_reset)
    reset._env = SimpleNamespace(
        device="cpu", scene=SimpleNamespace(env_origins=torch.tensor([[0.0, 0.0, 0.0], [2.0, 1.0, 0.0]]))
    )
    reset.layout = board_layout(AssemblySetCfg(num_slots=2))
    reset._board = Asset()
    reset._fixed = tuple(Asset() for _ in reset.layout.fixed_asset_names)
    reset._board_offsets = torch.tensor([variant.board_offset.pose for variant in reset.layout.variants])
    reset._fixture_variant_indices = torch.tensor(reset.layout.fixture_variant_indices)
    reset._fixture_index_by_variant = torch.tensor(reset.layout.fixture_index_by_variant)
    reset.fixed_kind_by_slot = torch.full((2, reset.layout.num_fixed_slots), -1, dtype=torch.int32)
    reset._zero_root_velocity = torch.zeros((2, 6))
    yaw = torch.tensor([0.0, torch.pi / 2])
    zero = torch.zeros_like(yaw)
    board_pose = torch.cat(
        (
            torch.tensor([[0.2, 0.1, 0.04], [0.5, -0.2, 0.04]]),
            math_utils.quat_from_euler_xyz(zero, zero, yaw),
        ),
        dim=1,
    )
    env_ids = torch.tensor([0, 1])
    variants = torch.tensor([[7, 4], [0, 6]])

    reset._write_board_and_fixtures(env_ids, board_pose, variants)

    board_pose_w = board_pose.clone()
    board_pose_w[:, :3] += reset._env.scene.env_origins
    torch.testing.assert_close(reset._board.pose, board_pose_w)
    expected_kinds = reset._fixture_index_by_variant[variants].sort(dim=1).values.to(torch.int32)
    torch.testing.assert_close(reset.fixed_kind_by_slot, expected_kinds)
    for slot, asset in enumerate(reset._fixed):
        kinds = reset.fixed_kind_by_slot[:, slot]
        representatives = reset._fixture_variant_indices[kinds.clamp_min(0)]
        torch.testing.assert_close(asset.variant_ids, representatives.to(torch.int32))
        offset = reset._board_offsets[representatives]
        pos, quat = math_utils.combine_frame_transforms(
            board_pose_w[:, :3], board_pose_w[:, 3:], offset[:, :3], offset[:, 3:]
        )
        pos[:, 2] = torch.where(kinds >= 0, pos[:, 2], board_pose_w[:, 2] + 5.0 + 0.1 * slot)
        torch.testing.assert_close(asset.pose, torch.cat((pos, quat), dim=1))
        torch.testing.assert_close(asset.env_ids, env_ids)


def test_full_board_reset_writes_candidates_before_ordered_acceptance() -> None:
    """Screen written candidates against V1 bounds and every configured condition in order."""

    class Asset:
        def write_mesh_variant_to_sim(self, variant_ids: torch.Tensor, env_ids: torch.Tensor) -> None:
            calls.append("mesh")

        def write_root_link_pose_to_sim(self, pose: torch.Tensor, env_ids: torch.Tensor) -> None:
            self.pose = pose.clone()
            calls.append("pose")

        def write_root_com_velocity_to_sim(self, velocity: torch.Tensor, env_ids: torch.Tensor) -> None:
            calls.append("velocity")

    def first_condition(env, env_ids: torch.Tensor) -> torch.Tensor:
        assert all(hasattr(asset, "pose") for asset in reset._held)
        calls.append("first")
        return torch.tensor([True, True, False])

    def second_condition(env, env_ids: torch.Tensor) -> torch.Tensor:
        calls.append("second")
        return torch.tensor([True, False, True])

    calls: list[str] = []
    reset = board_reset.__new__(board_reset)
    reset._env = SimpleNamespace(device="cpu", scene=SimpleNamespace(env_origins=torch.zeros((3, 3))))
    reset._held = (Asset(), Asset())
    reset._zero_root_velocity = torch.zeros((3, 6))
    reset._held_asset_in_bound_range = torch.tensor([[0.05, 1.0], [-0.675, 0.675], [-0.05, 1.0]])
    reset._acceptance_conditions = {
        "first": first_condition,
        "second": SimpleNamespace(func=second_condition),
    }
    poses = torch.tensor(
        [
            [[0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0], [0.2, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]],
            [[0.01, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0], [0.2, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]],
            [[0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0], [0.2, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]],
        ]
    )

    valid = reset._validate_candidates(
        torch.arange(3), poses, torch.tensor([[0, 1], [2, 3], [4, 5]]), torch.ones(3, dtype=torch.bool)
    )

    torch.testing.assert_close(valid, torch.tensor([True, False, False]))
    assert calls[-2:] == ["first", "second"]
    assert calls.index("first") > max(index for index, call in enumerate(calls) if call == "velocity")


def test_full_board_reset_preserves_state_weights_under_marginal_balance() -> None:
    """Balance label and variant marginals while retaining ratios within each joint cell."""
    reset = board_reset.__new__(board_reset)
    reset._state_cell_indices = torch.arange(6).repeat_interleave(4)
    weights = torch.tensor([20.0, 1.0, 1.0, 1.0] * 6)
    weights[4:8] = 1.0
    base_probabilities = weights / weights.sum()
    sampled: dict[str, torch.Tensor] = {}

    def sample(probabilities: torch.Tensor, count: int) -> torch.Tensor:
        sampled["probabilities"] = probabilities.clone()
        return torch.multinomial(probabilities, count, replacement=True)

    reset._sampler = SimpleNamespace(probabilities=lambda: base_probabilities.clone(), sample=sample)
    reset.cell_probabilities = torch.empty((2, 3))
    reset.state_probabilities = torch.empty_like(base_probabilities)
    reset._raw_cell_probabilities = torch.empty(reset.cell_probabilities.numel())
    reset._cell_scale = torch.empty_like(reset._raw_cell_probabilities)

    reset._refresh_state_probabilities()
    probabilities, _ = reset._sample_marginally_balanced(8)

    torch.testing.assert_close(probabilities[0] / probabilities[1], torch.tensor(20.0))
    torch.testing.assert_close(reset.cell_probabilities.sum(dim=1), torch.full((2,), 0.5), atol=2e-4, rtol=0.0)
    torch.testing.assert_close(reset.cell_probabilities.sum(dim=0), torch.full((3,), 1.0 / 3))
    torch.testing.assert_close(probabilities, sampled["probabilities"])


def test_full_board_reset_handles_an_empty_marginal_row() -> None:
    """Keep probabilities finite when a reset label is structurally unavailable."""
    reset = board_reset.__new__(board_reset)
    reset._state_cell_indices = torch.tensor([0, 0, 1, 2, 3, 4])
    base_probabilities = torch.full((6,), 1.0 / 6.0)
    reset._sampler = SimpleNamespace(
        probabilities=lambda: base_probabilities.clone(),
        sample=lambda probabilities, count: torch.multinomial(probabilities, count, replacement=True),
    )
    reset.cell_probabilities = torch.empty((3, 3))
    reset.state_probabilities = torch.empty_like(base_probabilities)
    reset._raw_cell_probabilities = torch.empty(reset.cell_probabilities.numel())
    reset._cell_scale = torch.empty_like(reset._raw_cell_probabilities)

    reset._refresh_state_probabilities()
    probabilities, _ = reset._sample_marginally_balanced(8)

    assert probabilities.isfinite().all()
    torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0))
    assert (reset.cell_probabilities[2] == 0.0).all()


def test_full_board_reset_exposes_compact_state_curriculum_features() -> None:
    """Encode physical state and the learner-private absolute assembly target."""
    reset = board_reset.__new__(board_reset)
    reset._progress_goal = True
    reset._capacity = 2
    reset.num_slots = 2
    reset.num_variants = 3
    reset._robot = SimpleNamespace(num_joints=2)
    reset._board_pose = torch.arange(14, dtype=torch.float32).view(2, 7)
    reset._held_pose = torch.arange(28, dtype=torch.float32).view(2, 2, 7) / 10.0
    reset._robot_joint_pos = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    reset._variant_ids = torch.tensor([[0, 2], [1, 0]], dtype=torch.uint8)
    reset._slot_state = torch.tensor([[TARGET, ASSEMBLED], [FALLEN, PARTIAL_ASSEMBLY]], dtype=torch.uint8)
    reset._unfinished_count = torch.tensor([1, 2], dtype=torch.uint8)
    reset._required_assembly_gain = torch.tensor([1, 1], dtype=torch.uint8)
    reset._focus_slot = torch.tensor([0, 1], dtype=torch.uint8)
    reset._reset_label = torch.tensor([0, len(RESET_LABELS) - 1], dtype=torch.uint8)
    feature_dim = 7 + 14 + 2 + 2 * 3 + 1
    reset.state_features = torch.empty((reset._capacity, feature_dim))

    data_ptr = reset.state_features.data_ptr()
    reset._build_state_features()
    expected = torch.cat(
        (
            reset._board_pose,
            reset._held_pose.flatten(1),
            reset._robot_joint_pos,
            torch.eye(reset.num_variants)[reset._variant_ids.long()].flatten(1),
            torch.tensor([[2.0], [1.0]]),
        ),
        dim=1,
    )

    torch.testing.assert_close(reset.state_features, expected)
    reset._slot_state.zero_()
    reset._focus_slot.fill_(1)
    reset._reset_label.fill_(3)
    reset._unfinished_count.copy_(torch.tensor([2, 2], dtype=torch.uint8))
    reset._required_assembly_gain.copy_(torch.tensor([2, 1], dtype=torch.uint8))
    reset._build_state_features()
    torch.testing.assert_close(reset.state_features, expected)
    assert reset.state_features.data_ptr() == data_ptr
    reset._required_assembly_gain[0] = 1
    reset._build_state_features()
    expected[0, -1] = 1.0
    torch.testing.assert_close(reset.state_features, expected)

    reset._progress_goal = False
    reset.state_features = torch.empty((reset._capacity, feature_dim - 1))
    reset._build_state_features()
    torch.testing.assert_close(reset.state_features, expected[:, :-1])


def test_full_board_timeout_snapshots_initial_unfinished_count() -> None:
    """Give each initially unfinished asset an independent fixed time budget."""
    reset = board_reset.__new__(board_reset)
    reset.num_slots = 3
    reset.unfinished_count = torch.tensor([1, 3, 2], dtype=torch.uint8)
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        step_dt=0.25,
        episode_length_buf=torch.tensor([3, 11, 8]),
        event_manager=SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(func=reset)),
    )
    term = initial_unfinished_time_out(SimpleNamespace(params={"seconds_per_asset": 1.0}), env)

    term.reset()
    torch.testing.assert_close(term.episode_limit_steps, torch.tensor([4, 12, 8]))
    torch.testing.assert_close(term(env, seconds_per_asset=1.0), torch.tensor([False, False, True]))

    reset.unfinished_count[:] = 3
    torch.testing.assert_close(term.episode_limit_steps, torch.tensor([4, 12, 8]))
    reset.unfinished_count[0] = 2
    term.reset(torch.tensor([0]))
    torch.testing.assert_close(term.episode_limit_steps, torch.tensor([8, 12, 8]))

    fixed = initial_unfinished_time_out(SimpleNamespace(params={"seconds_per_asset": 1.0, "dynamic": False}), env)
    fixed.reset()
    torch.testing.assert_close(fixed.episode_limit_steps, torch.full((3,), 12))
    torch.testing.assert_close(fixed(env, seconds_per_asset=1.0, dynamic=False), torch.tensor([False, False, False]))

    horizon = initial_unfinished_time_out(
        SimpleNamespace(params={"seconds_per_asset": 1.0, "fixed_horizon_s": 1.5}), env
    )
    reset.unfinished_count.copy_(torch.tensor([1, 2, 3]))
    horizon.reset()
    torch.testing.assert_close(horizon.episode_limit_steps, torch.full((3,), 6))

    mixed = initial_unfinished_time_out(
        SimpleNamespace(params={"seconds_per_asset": 1.0, "fixed_horizon_s": 1.5, "dynamic_env_count": 1}), env
    )
    mixed.reset()
    torch.testing.assert_close(mixed.episode_limit_steps, torch.tensor([4, 6, 6]))

    disabled = initial_unfinished_time_out(SimpleNamespace(params={"enabled": False}), env)
    env.episode_length_buf.fill_(1_000_000)
    disabled.reset()
    torch.testing.assert_close(disabled(env, enabled=False), torch.zeros(3, dtype=torch.bool))


def test_full_board_progress_context_keeps_terminal_success_stable() -> None:
    """Do not let a post-reset assembly refresh overwrite terminal estimator labels."""
    source = torch.tensor([True, False, True])
    term = assembly_progress_context.__new__(assembly_progress_context)
    term._state = SimpleNamespace(task_success=source)
    term._dummy = torch.zeros(3, dtype=torch.bool)
    term._terminal_success = torch.empty_like(term._dummy)
    env = SimpleNamespace(extras={})

    result = term(env)
    source[:] = False

    torch.testing.assert_close(result, torch.zeros(3, dtype=torch.bool))
    torch.testing.assert_close(env.extras["successes"], torch.tensor([True, False, True]))
    assert env.extras["successes"] is term._terminal_success


def test_full_board_success_reward_matches_the_progress_goal() -> None:
    """Use the same success predicate for termination context and reward."""
    task_success = torch.tensor([True, False, True])
    env = SimpleNamespace(
        event_manager=SimpleNamespace(
            get_term_cfg=lambda name: SimpleNamespace(func=SimpleNamespace(task_success=task_success))
        )
    )

    torch.testing.assert_close(assembly_success_reward(env), torch.tensor([1.0, 0.0, 1.0]))


def test_full_board_reset_updates_and_samples_monitor_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route reset-buffer row outcomes through the active curriculum lifecycle."""

    class Sampler:
        count = 0

    reset = board_reset.__new__(board_reset)
    reset.layout = board_layout(AssemblySetCfg(num_slots=2))
    reset.num_variants = reset.layout.num_variants
    reset.num_slots = reset.layout.num_slots
    reset._ready = True
    reset._capacity = 4
    reset.state_features = None
    reset.success_monitor = SuccessMonitor(SuccessMonitorCfg(monitored_history_len=4), 1, 4, "cpu")
    reset._success_monitor_env_count = 1
    reset._sampler = Sampler()
    reset._sample_marginally_balanced = lambda count: (
        setattr(reset._sampler, "count", count) or torch.full((4,), 0.25),
        torch.tensor([3, 0, 2])[:count],
    )
    reset.sampled_state = torch.tensor([2, -1, 1])
    reset.unfinished_count = torch.zeros(3, dtype=torch.uint8)
    reset.required_assembly_gain = torch.zeros(3, dtype=torch.uint8)
    reset.focus_slot = torch.zeros(3, dtype=torch.uint8)
    reset.reset_label = torch.zeros(3, dtype=torch.uint8)
    reset.variant_ids = torch.zeros((3, reset.num_slots), dtype=torch.uint8)
    reset.slot_state = torch.zeros((3, reset.num_slots), dtype=torch.uint8)
    reset._slot_asleep = torch.zeros((3, reset.num_slots), dtype=torch.bool)
    reset._slot_asleep_warp = wp.from_torch(reset._slot_asleep)
    reset._sleep_assembled = True
    reset._held_body_ids = object()
    reset.fixed_kind_by_slot = torch.full((3, reset.layout.num_fixed_slots), -1, dtype=torch.int32)
    reset.revision = 0
    reset._unfinished_count = torch.tensor([1, 2, 1, 2], dtype=torch.uint8)
    reset._required_assembly_gain = torch.tensor([1, 2, 1, 1], dtype=torch.uint8)
    reset._focus_slot = torch.zeros(4, dtype=torch.uint8)
    reset._reset_label = torch.zeros(4, dtype=torch.uint8)
    reset._variant_ids = torch.tensor([[0, 1], [2, 3], [4, 5], [18, 7]], dtype=torch.uint8)
    reset._slot_state = torch.tensor(
        [[TARGET, ASSEMBLED], [TARGET, TARGET], [ASSEMBLED, TARGET], [TARGET, TARGET]], dtype=torch.uint8
    )
    refresh_count = 0

    def refresh_state_curriculum() -> None:
        nonlocal refresh_count
        refresh_count += 1

    reset.refresh_state_curriculum = refresh_state_curriculum
    written = {}
    reset._write_state = lambda env_ids, state_ids: written.update(env_ids=env_ids.clone(), state_ids=state_ids.clone())

    progress = SimpleNamespace(is_success=torch.tensor([True, True, False]))
    terminations = SimpleNamespace(
        terminated=torch.tensor([True, False, True]),
        time_outs=torch.zeros(3, dtype=torch.bool),
        get_term=lambda name: torch.zeros(3, dtype=torch.bool),
        get_term_cfg=lambda name: SimpleNamespace(func=progress),
    )
    env = SimpleNamespace(
        device="cpu",
        extras={},
        termination_manager=terminations,
    )
    reset._env = env
    env_ids = torch.arange(3)
    monkeypatch.setattr(NewtonManager, "set_body_sleep_state", lambda *args: None)
    reset(
        env,
        env_ids,
        None,
        None,
        None,
        DEFAULT_BOARD_LAYOUT.variant_names,
        reset.num_slots,
        4,
        unfinished_count=None,
        success_monitor_cfg=SuccessMonitorCfg(),
        success_monitor_env_count=1,
        sampling=SamplerCfg(),
        fixed_asset_pose_range={},
        held_asset_in_bound_range={},
        acceptance_conditions={},
    )

    torch.testing.assert_close(reset.success_monitor.success_size, torch.tensor([0, 0, 1, 0]))
    torch.testing.assert_close(reset.success_monitor.success_rate, torch.tensor([0.0, 0.0, 1.0, 0.0]))
    env.extras["log"] = {}
    reset.reset(env_ids)
    assert env.extras["log"]["Metrics/success_rate"] == pytest.approx(1.0)
    assert reset._sampler.count == 3
    assert refresh_count == 1
    torch.testing.assert_close(reset.sampled_state, torch.tensor([3, 0, 2]))
    torch.testing.assert_close(reset.unfinished_count, torch.tensor([2, 1, 1], dtype=torch.uint8))
    torch.testing.assert_close(reset.required_assembly_gain, torch.tensor([1, 1, 1], dtype=torch.uint8))
    torch.testing.assert_close(reset.variant_ids, torch.tensor([[18, 7], [0, 1], [4, 5]], dtype=torch.uint8))
    assert env.extras["diagnostics"]["factory_board_required_assembly_gain"] is reset.required_assembly_gain
    torch.testing.assert_close(written["state_ids"], torch.tensor([3, 0, 2]))

    reset.success_monitor = None
    reset.state_features = torch.empty((4, 1))
    reset.sampled_state.copy_(torch.tensor([0, 1, 2]))
    reset.unfinished_count.copy_(torch.tensor([1, 2, 1], dtype=torch.uint8))
    reset.required_assembly_gain.copy_(torch.tensor([1, 2, 1], dtype=torch.uint8))
    reset.outcome_state_ids = torch.full((3,), -1, dtype=torch.long)
    reset.outcome_hard_targets = torch.zeros(3)
    reset.outcome_grounded = torch.zeros(3, dtype=torch.bool)
    captured = torch.empty(0, dtype=torch.long)
    captured_targets = torch.empty(0, dtype=torch.uint8)

    def capture_outcome_features(env_ids: torch.Tensor) -> None:
        captured.resize_as_(env_ids).copy_(env_ids)
        targets = reset.num_slots - reset.unfinished_count[env_ids] + reset.required_assembly_gain[env_ids]
        captured_targets.resize_as_(targets).copy_(targets)

    reset._capture_outcome_features = capture_outcome_features
    progress.is_success.copy_(torch.tensor([True, False, False]))
    terminations.terminated.copy_(torch.tensor([True, False, True]))
    terminations.time_outs.copy_(torch.tensor([False, True, False]))
    terminations.get_term = lambda name: torch.tensor([False, False, True])

    reset(
        env,
        env_ids,
        None,
        None,
        None,
        DEFAULT_BOARD_LAYOUT.variant_names,
        reset.num_slots,
        4,
        unfinished_count=None,
        success_monitor_cfg=None,
        sampling=SamplerCfg(),
        fixed_asset_pose_range={},
        held_asset_in_bound_range={},
        acceptance_conditions={},
    )

    torch.testing.assert_close(reset.outcome_state_ids, torch.tensor([0, 1, -1]))
    torch.testing.assert_close(reset.outcome_hard_targets[:2], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(reset.outcome_grounded, torch.tensor([True, False, False]))
    torch.testing.assert_close(captured, torch.tensor([0, 1]))
    torch.testing.assert_close(captured_targets, torch.tensor([2, 2], dtype=torch.uint8))
    torch.testing.assert_close(reset.required_assembly_gain, torch.tensor([1, 1, 1], dtype=torch.uint8))


def test_full_board_reset_publishes_full_bank_success_estimate_without_history() -> None:
    """Expose learner-owned success scalars without rebuilding environment history."""
    reset = board_reset.__new__(board_reset)
    reset.success_monitor = None
    reset.mean_estimated_success_rate = torch.tensor(0.25)
    reset.success_target_grounded_fraction = torch.tensor(0.75)
    reset._env = SimpleNamespace(extras={})

    reset.reset()

    log = reset._env.extras["log"]
    assert log["Metrics/success_rate"] is reset.mean_estimated_success_rate
    assert log["Info/SuccessTargetGroundedFraction"] is reset.success_target_grounded_fraction
    reset.mean_estimated_success_rate.fill_(0.6)
    reset.success_target_grounded_fraction.fill_(0.5)
    assert log["Metrics/success_rate"].item() == pytest.approx(0.6)
    assert log["Info/SuccessTargetGroundedFraction"].item() == pytest.approx(0.5)
    assert not hasattr(board_reset, "record_success_targets")
    assert not hasattr(board_reset, "success_rate")
    assert not hasattr(board_reset, "success_size")


def test_full_board_metrics_are_full_bank_averages() -> None:
    """Aggregate the current estimator and sampler over every reset-bank row."""
    reset = board_reset.__new__(board_reset)
    reset.layout = board_layout(("nut_thread_m8", "nut_thread_m12", "nut_thread_m16"), num_slots=3)
    reset.num_slots = reset.layout.num_slots
    reset.num_variants = reset.layout.num_variants
    reset._capacity = 6
    reset._unfinished_count = torch.tensor([1, 2, 2, 3, 3, 3], dtype=torch.uint8)
    reset._variant_ids = torch.arange(3, dtype=torch.uint8).expand(6, -1).clone()
    initially_unfinished = torch.tensor(
        [[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=torch.bool
    )
    reset._slot_state = torch.where(initially_unfinished, TARGET, ASSEMBLED).to(torch.uint8)
    reset._state_cell_indices = torch.zeros(6, dtype=torch.long)
    reset._bank_unfinished_index = torch.empty(6, dtype=torch.long)
    reset.estimated_success_rate = torch.tensor([0.2, 0.4, 0.8, 0.1, 0.5, 0.9])
    reset.success_monitor = None
    probabilities = torch.tensor([0.1, 0.1, 0.2, 0.1, 0.2, 0.3])

    class Sampler:
        probability_calls = 0
        sample_calls = 0

        def probabilities(self) -> torch.Tensor:
            self.probability_calls += 1
            return probabilities

        def sample(self, values: torch.Tensor, count: int) -> torch.Tensor:
            assert values is reset.state_probabilities
            self.sample_calls += 1
            return torch.arange(count)

    reset._sampler = Sampler()
    reset.cell_probabilities = torch.empty((len(RESET_LABELS), 3))
    reset.state_probabilities = torch.empty(6)
    reset.reset_probability_mass = torch.empty(3)
    reset.reset_probability_total = torch.empty(())
    reset.reset_success_sum = torch.empty(3)
    reset.reset_state_count = torch.empty(3, dtype=torch.long)
    reset.asset_unassembled_sum = torch.empty((3, 3))
    reset.asset_unfinished_count = torch.empty((3, 3), dtype=torch.long)
    reset._raw_cell_probabilities = torch.empty(reset.cell_probabilities.numel())
    reset._cell_scale = torch.empty_like(reset._raw_cell_probabilities)
    reset._failure_probability = torch.empty(6)
    reset._initialize_bank_metrics()
    reset.refresh_state_curriculum()
    reset._sample_marginally_balanced(2)
    reset._sample_marginally_balanced(3)
    assert reset._sampler.probability_calls == 1
    assert reset._sampler.sample_calls == 2

    env = SimpleNamespace(
        device="cpu",
        num_envs=1,
        event_manager=SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(func=reset)),
        extras={},
    )

    metrics = BoardMetrics(CurriculumTermCfg(func=BoardMetrics), env)
    heatmaps = env.extras["heatmap"]

    assert set(heatmaps) == {
        "Metrics/ResetProbs",
        "Metrics/ResetSuccessRate",
        "Metrics/AssetUnassembledRate",
    }
    torch.testing.assert_close(reset.reset_probability_mass, torch.tensor([0.1, 0.3, 0.6]))
    torch.testing.assert_close(reset.reset_probability_total, torch.tensor(1.0))
    torch.testing.assert_close(reset.reset_success_sum, torch.tensor([0.2, 1.2, 1.5]))
    torch.testing.assert_close(reset.reset_state_count, torch.tensor([1, 2, 3]))
    torch.testing.assert_close(
        reset.asset_unassembled_sum, torch.tensor([[0.8, 0.0, 0.0], [0.8, 0.6, 0.2], [1.5, 1.5, 1.5]])
    )
    torch.testing.assert_close(reset.asset_unfinished_count, torch.tensor([[1, 0, 0], [2, 1, 1], [3, 3, 3]]))
    success_rates = reset.reset_success_sum / reset.reset_state_count
    torch.testing.assert_close(success_rates, torch.tensor([0.2, 0.6, 0.5]))
    torch.testing.assert_close(
        reset.asset_unassembled_sum / reset.asset_unfinished_count,
        torch.tensor([[0.8, float("nan"), float("nan")], [0.4, 0.6, 0.2], [0.5, 0.5, 0.5]]),
        equal_nan=True,
    )
    torch.testing.assert_close(
        (success_rates * reset.reset_state_count).sum() / reset.reset_state_count.sum(),
        reset.estimated_success_rate.mean(),
    )
    assert heatmaps["Metrics/ResetProbs"]["numerator"] is reset.reset_probability_mass
    assert heatmaps["Metrics/ResetSuccessRate"]["numerator"] is reset.reset_success_sum
    assert heatmaps["Metrics/ResetSuccessRate"]["color_label"] == "Estimated task success rate"
    assert heatmaps["Metrics/AssetUnassembledRate"]["numerator"] is reset.asset_unassembled_sum
    assert heatmaps["Metrics/AssetUnassembledRate"]["facet_labels"] == ("U=1", "U=2", "U=3")
    assert (
        tuple(label for row in heatmaps["Metrics/AssetUnassembledRate"]["cell_labels"] for label in row if label)[-1]
        == "N16"
    )
    assert not hasattr(metrics, "_episode_counts")


def test_static_and_variant_event_compositions_remain_distinct() -> None:
    """Preserve static inertia setup and initialize variants before dependent terms."""
    static = StaticFactoryEnvCfg()
    variant = FactoryVariantEnvCfg()

    assert tuple(static.events.to_dict()) == (
        "held_asset_material",
        "held_asset_inertia",
        "fixed_asset_material",
        "robot_material",
        "reset_strategies",
    )
    assert tuple(variant.events.to_dict()) == (
        "assembly_variants",
        "held_asset_material",
        "fixed_asset_material",
        "robot_material",
        "reset_strategies",
    )
    assert static.sim.render_interval == 1
    assert variant.sim.render_interval == variant.decimation


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
    scene = FactoryVariantSceneCfg(num_envs=4)
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
    cfg = EventTermCfg(func=PreparedTermChoice, mode="reset", params={"terms": terms})
    env = SimpleNamespace(num_envs=4, device="cpu")
    choice = PreparedTermChoice(cfg, env)
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


def test_pose_observation_supports_v1_static_offsets_without_variant_context() -> None:
    """Keep the canonical observation usable by the static V1 composition."""
    target_pose = wp.array([[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]], dtype=wp.transformf, device="cpu")
    root_pose = wp.array([[[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]], dtype=wp.transformf, device="cpu")
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        scene={
            "target": SimpleNamespace(data=SimpleNamespace(body_link_pose_w=SimpleNamespace(warp=target_pose))),
            "root": SimpleNamespace(data=SimpleNamespace(body_link_pose_w=SimpleNamespace(warp=root_pose))),
        },
    )
    cfg = ObservationTermCfg(
        func=target_asset_pose_in_root_asset_frame,
        params={
            "target_asset_cfg": SceneEntityCfg("target"),
            "root_asset_cfg": SceneEntityCfg("root"),
            "target_asset_offset": Offset(pos=(0.3, 0.0, 0.0)),
            "root_asset_offset": Offset(pos=(0.1, 0.0, 0.0)),
        },
    )

    output = target_asset_pose_in_root_asset_frame(cfg, env)(env, **cfg.params)
    torch.testing.assert_close(output, torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]))


def test_policy_observes_scene_geometry_and_directed_mating_pose() -> None:
    """Combine scene geometry with one directed mating-frame observation."""
    observations = FactoryVariantObservationsCfg()
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
    assert policy.held_asset_in_fixed_asset_frame.func is target_asset_pose_in_root_asset_frame
    assert policy.held_asset_in_fixed_asset_frame.params == {
        "target_asset_cfg": SceneEntityCfg("held_asset"),
        "root_asset_cfg": SceneEntityCfg("fixed_asset"),
        "target_asset_offset": "held_align",
        "root_asset_offset": "fixed_tip",
    }
    assert policy.held_asset_in_fixed_asset_frame.history_length == 5
    assert policy.end_effector_vel_lin_ang_b.history_length == 5
    assert policy.joint_pos.history_length == 5
    assert policy.prev_action.history_length == 5
    assert policy.history_length is None
    assert not hasattr(policy, "end_effector_pose")
    assert not hasattr(policy, "fixed_asset_in_end_effector_frame")
    assert not hasattr(policy, "assembly_variant")


def test_factory_agent_routes_perception_through_point_cloud_encoder() -> None:
    """Keep geometry out of the state MLP and route it through the point-cloud encoder."""
    observations = FactoryVariantObservationsCfg()
    runner = FactoryVariantPPORunnerCfg()

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


def test_simba_model_accumulates_all_normalized_groups() -> None:
    """Keep passthrough and encoded observations on one committed rollout frame."""
    observations = TensorDict(
        {"policy": torch.arange(12, dtype=torch.float32).view(4, 3), "perception": torch.ones(4, 6)},
        batch_size=[4],
    )
    model = SimBaModel(
        observations,
        {"actor": ["policy", "perception"]},
        "actor",
        output_dim=2,
        hidden_dim=8,
        obs_normalization=True,
        encoder_normalization=True,
        encoder_cfg={"perception": {"class_name": MLPEncoder, "hidden_dims": [8], "output_dim": 4}},
    )

    model.accumulate_normalization(observations)
    assert model.obs_normalizer.count == 0
    assert model.encoder_normalizers["perception"].count == 0

    commit_normalization((model,))
    torch.testing.assert_close(model.obs_normalizer.mean, observations["policy"].mean(0))
    torch.testing.assert_close(model.encoder_normalizers["perception"].mean, observations["perception"].mean(0))


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

    monkeypatch.setattr(factory_collision_analyzer, "resolve_matching_prims_from_source", resolve)
    monkeypatch.setattr(factory_collision_analyzer, "RigidObjectHasher", stop_after_resolution)
    monkeypatch.setattr(factory_observations, "resolve_matching_prims_from_source", resolve)
    monkeypatch.setattr(factory_rigid_object_hasher, "RigidObjectHasher", stop_after_resolution)

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
        func=factory_events.sample_settled_asset_pose,
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
    term = factory_events.sample_settled_asset_pose(cfg, env)
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
    reset_choice = VARIANT_ACCUMULATOR_RESET.params["reset_term"].params["terms"]["reset_strategies"]
    start_pick = reset_choice.params["terms"]["start_pick"]
    num_cells = len(reset_choice.params["terms"]) * len(ASSEMBLY_VARIANTS)
    assert VARIANT_ACCUMULATOR_RESET.params["state_table_size"] >= num_cells
    assert set(reset_choice.params) == {"terms"}
    assert "settling_term" not in VARIANT_ACCUMULATOR_RESET.params
    assert start_pick.params["terms"]["reset_held_asset"].func is factory_events.sample_settled_asset_pose
    assert not {
        "state_tag_names_bind",
        "state_tag_indices_bind",
        "state_tag_weight_bind",
    }.intersection(VARIANT_ACCUMULATOR_RESET.params)


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
    accumulator = variant_reset_accumulator(cfg, env)
    assert torch.all(accumulator.sampled_slots == -1)


def test_accumulator_soft_balances_and_removes_precollection_terms(monkeypatch) -> None:
    """Bias collection toward sparse cells, then discard the one-shot term tree."""

    class Variants:
        variant_names = ("asset_a", "asset_b")
        variant_ids = torch.zeros(1, dtype=torch.int32)

        def prepare(self, variant_ids: torch.Tensor) -> None:
            self.variant_ids.copy_(variant_ids)

    class Reset:
        def __init__(self, choice: PreparedTermChoice):
            self.terms = {"reset_strategies": SimpleNamespace(func=choice)}
            self.is_valid = torch.ones(1, dtype=torch.bool)

        def __call__(self, env, env_ids: torch.Tensor) -> None:
            pass

    variants = Variants()
    choice = PreparedTermChoice.__new__(PreparedTermChoice)
    choice.term_samples = torch.zeros(1, dtype=torch.long)
    choice._next_samples = torch.zeros(1, dtype=torch.long)
    choice.term_partitions = {"start_pick": SimpleNamespace()}
    reset_term = SimpleNamespace(func=Reset(choice), params={})
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        event_manager=SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(func=variants)),
    )
    accumulator = variant_reset_accumulator.__new__(variant_reset_accumulator)
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
    assert inspect.signature(variant_reset_accumulator.__call__).parameters["reset_term"].default is None
    assert not hasattr(PreparedTermChoice, "release_temporary_state")
    assert not hasattr(ChainedResetTerms, "release_temporary_state")
    assert not hasattr(factory_events.sample_settled_asset_pose, "release_temporary_state")
    assert not accumulator.precollecting_phase


def test_accumulator_reports_adaptive_cell_probabilities() -> None:
    """Report the effective curriculum distribution instead of flattening every grid cell."""
    accumulator = variant_reset_accumulator.__new__(variant_reset_accumulator)
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
    accumulator.sampled_cells = torch.full_like(accumulator.sampled_slots, -1)
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
    accumulator = variant_reset_accumulator.__new__(variant_reset_accumulator)
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

    accumulator = variant_reset_accumulator.__new__(variant_reset_accumulator)
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
    assert "synchronize" not in variant_reset_accumulator.__dict__


def test_accumulator_exposes_only_the_requested_metric_schema() -> None:
    """Keep reset reporting to two grids and one scalar curve."""
    source = inspect.getsource(variant_reset_accumulator.__call__)
    metrics = set(re.findall(r"Metrics/[A-Za-z_/]+", source))
    assert metrics == {"Metrics/ResetSuccessRate", "Metrics/ResetProbs", "Metrics/success_rate"}
