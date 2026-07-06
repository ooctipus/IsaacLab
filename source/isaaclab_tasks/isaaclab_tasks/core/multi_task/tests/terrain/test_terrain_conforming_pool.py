# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration test: terrain generation -> Position family -> task table.

No IsaacSim required -- uses TerrainGenerator + Newton + Warp directly.
"""

from __future__ import annotations

import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.terrain_generator import TerrainGenerator
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg

from isaaclab_tasks.core.multi_task.kinematics import NewtonKinematicsBuildCfg, NewtonKinematicsCfg
from isaaclab_tasks.core.multi_task.kinematics.ik_objectives.cfg import (
    BodyPointsCfg,
    EntityPositionCfg,
    EntityRotationCfg,
    IKObjectiveJointLimitCfg,
    IKObjectivePositionCfg,
    IKObjectiveRotationCfg,
)
from isaaclab_tasks.core.multi_task.mdp.commands.state_command.task_family import make_task_table_rng
from isaaclab_tasks.core.multi_task.terrain.mdp.commands.commands_cfg import (
    PositionIKSolveCfg,
    PositionTerrainStanceFamilyCfg,
    PositionTerrainStanceGenerateCfg,
    TaskTableCfg,
)
from isaaclab_tasks.core.multi_task.terrain.mdp.commands.task_table_builder import (
    _centered_sampling_bounds,
    _sampler_with_inner_sampling_bounds,
    _state_count_from_spacing,
    _synthesize_terrain_origins,
    _terrain_grid_bounds,
    build_task_table,
)
from isaaclab_tasks.core.multi_task.terrain.retarget.cfg import SamplerCfg


@pytest.fixture(scope="module", autouse=True)
def _init_warp():
    wp.init()


DEVICE = "cuda:0"
NUM_ROWS = 3
NUM_COLS = 4
CELL_SIZE = (8.0, 8.0)


@pytest.fixture(scope="module")
def kinematics_cfg():
    """Newton kinematics configuration for ANYmal-C."""
    from isaaclab.utils.assets import check_file_path, retrieve_file_path

    usd = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Robots/ANYbotics/anymal_c.usd"
    fs = check_file_path(usd)
    if fs == 2:
        usd = retrieve_file_path(usd, force_download=False)
    elif fs == 0:
        pytest.skip("ANYmal USD not available")

    return NewtonKinematicsCfg(
        usd_path=usd,
        device=DEVICE,
        default_pos=(0.0, 0.0, 0.6),
        default_joint_pos={
            ".*HAA": 0.0,
            ".*F_HFE": 0.4,
            ".*H_HFE": -0.4,
            ".*F_KFE": -0.8,
            ".*H_KFE": 0.8,
        },
    )


@pytest.fixture(scope="module")
def articulation_cfg(kinematics_cfg):
    """ANYmal-C scene declaration using the same local artifact as Newton."""
    from isaaclab_assets.robots.anymal import ANYMAL_C_CFG

    cfg = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.spawn.usd_path = kinematics_cfg.usd_path
    return cfg


@pytest.fixture(scope="module")
def simple_commands():
    """Minimal command dict for testing."""
    from isaaclab_tasks.core.multi_task.terrain.mdp.commands.commands_cfg import TerrainCommands

    return {
        "walk": TerrainCommands(
            match_base_pos=True,
            match_base_rot=False,
            duration=(2.0, 2.0),
        )
    }


def _table_cfg(_kinematics_cfg: NewtonKinematicsCfg, pool_spacing: float) -> TaskTableCfg:
    """Return the public Position family declaration used by integration tests."""
    return TaskTableCfg(
        kinematics=NewtonKinematicsBuildCfg(collapse_fixed_joints=False),
        pool_spacing=pool_spacing,
        families=(
            PositionTerrainStanceFamilyCfg(
                generate=(
                    PositionTerrainStanceGenerateCfg(
                        sampler=SamplerCfg(),
                        foot_body_names=("LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"),
                    ),
                ),
                solve=PositionIKSolveCfg(
                    objectives=(
                        IKObjectivePositionCfg(
                            name="foot_targets",
                            current=BodyPointsCfg(asset="robot", bodies=("LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT")),
                            target_bind="generated.foot_targets",
                            weight=1.0,
                        ),
                        IKObjectivePositionCfg(
                            name="base_position",
                            current=EntityPositionCfg(asset="robot"),
                            target_bind="generated.base_position",
                            weight=0.05,
                        ),
                        IKObjectiveRotationCfg(
                            name="base_rotation",
                            current=EntityRotationCfg(asset="robot"),
                            target_bind="generated.base_rotation",
                            weight=0.5,
                        ),
                        IKObjectiveJointLimitCfg(weight=10.0),
                    ),
                    max_iterations=200,
                ),
            ),
        ),
    )


class TestTaskTableSizingHelpers:
    """CPU-only checks for spacing-mode sizing and sampler bounds patching."""

    def test_spacing_pool_uses_inner_grid_area(self):
        origins = torch.zeros(5, 5, 3, dtype=torch.float32)
        xs = torch.arange(5, dtype=torch.float32) * 10.0 - 20.0
        ys = torch.arange(5, dtype=torch.float32) * 10.0 - 20.0
        origins[..., 0], origins[..., 1] = torch.meshgrid(xs, ys, indexing="ij")

        x_range, y_range = _terrain_grid_bounds(origins, (10.0, 10.0))

        assert len(x_range) == len(y_range) == 2
        assert x_range == pytest.approx((-25.0, 25.0))
        assert y_range == pytest.approx((-25.0, 25.0))
        assert _state_count_from_spacing(x_range, y_range, spacing=1.0, area_divisor=3.0) == 833

    def test_synthesized_origins_match_terrain_generator_layout(self):
        origins = _synthesize_terrain_origins(2, 3, (10.0, 20.0), device="cpu")

        torch.testing.assert_close(origins[..., 0], torch.tensor([[-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]]))
        torch.testing.assert_close(origins[..., 1], torch.tensor([[-20.0, 0.0, 20.0], [-20.0, 0.0, 20.0]]))
        torch.testing.assert_close(origins[..., 2], torch.zeros(2, 3))

    def test_centered_sampling_bounds_clip_inner_grid_area(self):
        x_range, y_range = _centered_sampling_bounds((-50.0, 50.0), (-100.0, 100.0), (10.0, 10.0))

        assert x_range == pytest.approx((-5.0, 5.0))
        assert y_range == pytest.approx((-5.0, 5.0))
        assert _state_count_from_spacing(x_range, y_range, spacing=0.1, area_divisor=3.0) == 3333

    def test_centered_sampling_bounds_clamp_to_grid(self):
        x_range, y_range = _centered_sampling_bounds((-4.0, 6.0), (-3.0, 3.0), (100.0, 4.0))

        assert x_range == pytest.approx((-4.0, 6.0))
        assert y_range == pytest.approx((-2.0, 2.0))

    def test_inner_sampling_bounds_do_not_mutate_input_cfg(self):
        cfg = SamplerCfg()

        patched = _sampler_with_inner_sampling_bounds(cfg, (-25.0, 25.0), (-25.0, 25.0))

        assert cfg.patch.x_range is None
        assert cfg.patch.y_range is None
        assert patched.patch.x_range == (-25.0, 25.0)
        assert patched.patch.y_range == (-25.0, 25.0)

    def test_command_preset_uses_deferred_kinematics_and_spacing(self):
        import importlib

        from isaaclab_tasks.core.multi_task.position_env_cfg import CommandsCfg
        from isaaclab_tasks.core.multi_task.terrain.mdp_presets import command_presets
        from isaaclab_tasks.utils import resolve_presets

        importlib.import_module("isaaclab_tasks.core.multi_task.terrain.mdp_presets.robots")
        cfg = resolve_presets(CommandsCfg(), {"anymal_c"})
        goal_cfg = cfg.goal_point
        table_cfg = goal_cfg.task_table

        assert not hasattr(table_cfg.kinematics, "usd_path")
        assert table_cfg.pool_spacing == pytest.approx(0.5)
        assert table_cfg.pool_spacing_area_divisor == pytest.approx(3.0)
        assert table_cfg.pool_sampling_size is None
        assert len(table_cfg.families) == 1
        family = table_cfg.families[0]
        assert family.name == "terrain_stance"
        assert family.generate[0].sampler.min_contacts == 3
        assert family.generate[0].sampler.terrain_snap_distance == pytest.approx(0.2)
        assert family.generate[0].sampler.fk_joint_range_overrides == {}
        assert family.generate[0].sampler.outward_snap_penalty == pytest.approx(1.0)
        assert "IKObjectiveJointRegularizeCfg" not in {
            type(objective).__name__ for objective in family.solve.objectives
        }
        assert not hasattr(table_cfg, "pipeline_cfg")
        assert table_cfg.pairing.max_spawns_per_cell == 20
        assert table_cfg.pairing.num_targets_per_cell == 20
        assert not hasattr(command_presets, "CommandsCfg")
        assert [criterion.name for criterion in family.criteria] == [
            "collision",
            "joint_limit",
            "lateral_hip_limit",
            "stability",
            "foot_err",
            "cost",
        ]

    def test_position_table_schema_and_sources_reject_removed_pipeline(self):
        from isaaclab_tasks.core.multi_task.terrain.mdp.commands.task_table_builder import RelativeStateTaskTable

        table_fields = {field.name for field in fields(RelativeStateTaskTable)}
        assert RelativeStateTaskTable.__dataclass_params__.frozen
        assert hasattr(RelativeStateTaskTable, "__slots__")
        assert {"states", "view", "kinematics", "contact_body_names", "contact_body_ids"} <= table_fields
        assert not {"target_fk_kin", "newton_joint_names", "newton_foot_body_ids"} & table_fields

        multi_task = Path(__file__).parents[2]
        builder_source = (multi_task / "terrain/mdp/commands/task_table_builder.py").read_text()
        assert "pipeline_cfg" not in builder_source
        assert "trace_span" not in builder_source
        assert "print(" not in builder_source
        assert not (multi_task / "terrain/retarget/pipeline.py").exists()
        assert not (multi_task / "kinematics/ik_objectives/terrain_collision.py").exists()
        assert not (multi_task / "kinematics/ik_objectives/terrain_contact.py").exists()


def _bin_xy(xy: torch.Tensor, origins: torch.Tensor, cell_size, num_rows, num_cols):
    """Return (tile_index, in_grid_mask) for world XY points.

    ``tile_index`` holds ``row * num_cols + col`` for points in-grid, else ``-1``.
    """
    cs = torch.tensor(cell_size, device=xy.device, dtype=xy.dtype)
    grid_origin = origins[0, 0, :2] - cs * 0.5
    row = ((xy[:, 0] - grid_origin[0]) / cs[0]).long()
    col = ((xy[:, 1] - grid_origin[1]) / cs[1]).long()
    in_grid = (row >= 0) & (row < num_rows) & (col >= 0) & (col < num_cols)
    tile = torch.where(in_grid, row * num_cols + col, torch.full_like(row, -1))
    return tile, in_grid


@pytest.fixture(scope="module")
def flat_terrain():
    """Flat-plane terrain grid for isolation/uniformity/benchmark tests.

    Using a flat terrain removes difficulty variation so any per-tile imbalance
    is attributable to the sampler + binning pipeline, not terrain geometry.
    """
    cfg = TerrainGeneratorCfg(
        size=CELL_SIZE,
        num_rows=NUM_ROWS,
        num_cols=NUM_COLS,
        border_width=0.0,
        use_cache=False,
        seed=42,
        curriculum=False,
        sub_terrains={"flat": MeshPlaneTerrainCfg()},
    )
    return TerrainGenerator(cfg=cfg, device=DEVICE)


@pytest.fixture(scope="module")
def flat_task_table(flat_terrain, kinematics_cfg, articulation_cfg, simple_commands):
    """Shared task-table build on the flat grid (family runs once per module).

    ``pool_spacing = 1.0`` yields about 20 states per tile, so the same
    result covers the uniformity-CV check (which needs many states per tile)
    as well as the isolation / border-leak checks.
    """
    if not wp.is_device_available("cuda:0"):
        pytest.skip("GPU required")
    origins = torch.tensor(flat_terrain.terrain_origins, device=DEVICE, dtype=torch.float32)
    n_tiles = NUM_ROWS * NUM_COLS
    result = build_task_table(
        terrain_mesh=flat_terrain.terrain_mesh,
        terrain_origins=origins,
        cell_size=CELL_SIZE,
        table_cfg=_table_cfg(kinematics_cfg, 1.0),
        articulation_cfg=articulation_cfg,
        asset_name="robot",
        commands=simple_commands,
        device=DEVICE,
        rng=make_task_table_rng(42, DEVICE),
    )
    return {"result": result, "origins": origins, "n_tiles": n_tiles}


def test_position_view_shares_terrain_and_repeats_only_robot(flat_task_table) -> None:
    """Position retains one global terrain while each displayed state owns one robot."""
    kinematics = flat_task_table["result"].view.kinematic_view

    assert kinematics.model_builder_shared is not None
    assert kinematics.model_builder_shared.shape_count == 1
    assert kinematics.model_builder_shared.joint_coord_count == 0
    assert kinematics.model_builder_state.body_count > 0
    assert kinematics.model_builder_state.shape_count > 0
    assert kinematics.model_builder_state.joint_coord_count == kinematics.joint_q_default.numel()
    assert kinematics.world_spacing == (0.0, 0.0, 0.0)


class TestIsolation:
    """Regression guard: every task's spawn/target must live in the tile reported by tile_index."""

    def test_spawn_target_match_tile_index(self, flat_task_table):
        result = flat_task_table["result"]
        origins = flat_task_table["origins"]
        root_pose = result.states.root_pose[:, 0]
        spawn_idx = result.spawn_index
        target_idx = result.target_index
        tile_idx = result.tile_index

        spawn_tile, spawn_in = _bin_xy(
            root_pose[spawn_idx, :2],
            origins,
            CELL_SIZE,
            NUM_ROWS,
            NUM_COLS,
        )
        target_tile, target_in = _bin_xy(
            root_pose[target_idx, :2],
            origins,
            CELL_SIZE,
            NUM_ROWS,
            NUM_COLS,
        )
        assert bool(spawn_in.all()), "spawn index references an out-of-grid state"
        assert bool(target_in.all()), "target index references an out-of-grid state"
        assert bool((spawn_tile == tile_idx).all()), "spawn state not in its recorded tile"
        assert bool((target_tile == tile_idx).all()), "target state not in its recorded tile"


class TestUniformity:
    """FPS + CSR binning should spread final states ~uniformly across tiles on flat terrain."""

    def test_coverage_and_cv(self, flat_task_table):
        result = flat_task_table["result"]
        origins = flat_task_table["origins"]
        n_tiles = flat_task_table["n_tiles"]
        root_pose = result.states.root_pose[:, 0]
        tile, in_grid = _bin_xy(root_pose[:, :2], origins, CELL_SIZE, NUM_ROWS, NUM_COLS)
        counts = torch.bincount(tile[in_grid], minlength=n_tiles).float()

        coverage = (counts > 0).float().mean().item()
        mean = counts.mean().item()
        cv = (counts.std() / counts.mean().clamp_min(1.0)).item()
        n_border = int((~in_grid).sum().item())
        print(
            f"\n  Uniformity: {int(counts.sum())} in-grid states over {n_tiles} tiles "
            f"(border={n_border}), coverage={coverage:.1%}, "
            f"per-tile mean={mean:.1f} cv={cv:.2f} min={int(counts.min())} max={int(counts.max())}"
        )
        # Flat terrain + FPS should cover every tile and the spread should be tight.
        assert coverage >= 0.9, f"Only {coverage:.1%} of tiles have any states"
        assert cv <= 0.6, f"Per-tile count CV {cv:.2f} too high (non-uniform sampling)"


# Production-like terrain grid: 10x10 cells at 10x10m with 20m border reproduces
# the full SubTerrainPreset.all mix at the smaller end of production grid sizes
# (training ranges 10x10 to 10x20). Halving the columns halves mesh area and
# sampler runtime without sacrificing coverage of the sub-terrain mix.
REAL_NUM_ROWS = 10
REAL_NUM_COLS = 10
REAL_CELL_SIZE = (10.0, 10.0)
REAL_BORDER_WIDTH = 20.0
REAL_POOL_SPACING = 1.0


@pytest.fixture(scope="module")
def realistic_terrain():
    """Production-scale terrain: 10x20 grid, 20m border, full sub-terrain mix.

    Mirrors what actually runs in training via :class:`SceneCfg.terrain`, so
    uniformity / speed / accuracy numbers here are representative rather than
    best-case flat-plane numbers.
    """
    from isaaclab_tasks.core.multi_task.terrain.mdp_presets.terrain_presets import (
        SubTerrainPresetCfg,
    )

    cfg = TerrainGeneratorCfg(
        size=REAL_CELL_SIZE,
        num_rows=REAL_NUM_ROWS,
        num_cols=REAL_NUM_COLS,
        border_width=REAL_BORDER_WIDTH,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        seed=42,
        curriculum=True,
        sub_terrains=SubTerrainPresetCfg().terrain_curriculum,
    )
    return TerrainGenerator(cfg=cfg, device=DEVICE)


class TestRealistic:
    """End-to-end characterization on the production terrain grid.

    Three things under one fixture (terrain gen is expensive, ~seconds):

    * uniformity   - per-tile state distribution on the real mix
    * speed        - end-to-end build_task_table wall time
    * accuracy     - base-to-ground clearance for every IK-solved state
    """

    @pytest.fixture(scope="class")
    def real_result(self, realistic_terrain, kinematics_cfg, articulation_cfg, simple_commands):
        """Run the Position family once at production scale; share across tests.

        One ``build_task_table`` invocation covers all three assertions. The
        pipeline dominates ~97% of total time, so a separate timed
        ``pipeline.run`` is redundant, and a warm-up pass would repeat the
        same dominant cost without improving the measurement -- this runs
        once at env init in production, so JIT/graph-capture cost is part
        of the honest number.
        """
        if not wp.is_device_available("cuda:0"):
            pytest.skip("GPU required")

        origins = torch.tensor(realistic_terrain.terrain_origins, device=DEVICE, dtype=torch.float32)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        table = build_task_table(
            terrain_mesh=realistic_terrain.terrain_mesh,
            terrain_origins=origins,
            cell_size=REAL_CELL_SIZE,
            table_cfg=_table_cfg(kinematics_cfg, REAL_POOL_SPACING),
            articulation_cfg=articulation_cfg,
            asset_name="robot",
            commands=simple_commands,
            device=DEVICE,
            rng=make_task_table_rng(42, DEVICE),
        )
        torch.cuda.synchronize()
        t_total = time.perf_counter() - t0

        return {
            "origins": origins,
            "terrain": realistic_terrain,
            "table": table,
            "t_total": t_total,
        }

    def test_uniformity(self, real_result):
        """Per-tile coverage + count CV on the real terrain mix."""
        table = real_result["table"]
        origins = real_result["origins"]
        root_pose = table.states.root_pose[:, 0]
        n_tiles = REAL_NUM_ROWS * REAL_NUM_COLS

        tile, in_grid = _bin_xy(
            root_pose[:, :2],
            origins,
            REAL_CELL_SIZE,
            REAL_NUM_ROWS,
            REAL_NUM_COLS,
        )
        counts = torch.bincount(tile[in_grid], minlength=n_tiles).float()

        coverage = (counts > 0).float().mean().item()
        mean = counts.mean().item()
        cv = (counts.std() / counts.mean().clamp_min(1.0)).item()
        n_border = int((~in_grid).sum().item())
        print(
            f"\n  Realistic uniformity (pool_spacing={REAL_POOL_SPACING}, grid={REAL_NUM_ROWS}x{REAL_NUM_COLS}):"
            f"\n    in-grid states:  {int(counts.sum())} / {root_pose.shape[0]} "
            f"(border/out-of-grid dropped: {n_border})"
            f"\n    tile coverage:   {coverage:.1%}  ({int((counts > 0).sum())}/{n_tiles} tiles)"
            f"\n    per-tile counts: mean={mean:.1f} cv={cv:.2f} "
            f"min={int(counts.min())} max={int(counts.max())}"
        )
        # Real terrain is heterogeneous (some sub-terrains have less walkable
        # support-polygon area than others) so the bar is looser than flat.
        # These thresholds are regression guards, not targets.
        assert coverage >= 0.7, f"Only {coverage:.1%} of tiles have any states"
        assert cv <= 1.5, f"Per-tile count CV {cv:.2f} suggests major imbalance"

    def test_speed(self, real_result):
        """End-to-end wall time at production scale."""
        t_total = real_result["t_total"]
        n_states = real_result["table"].states.row_count
        n_tasks = real_result["table"].num_tasks
        print(
            f"\n  Realistic speed (pool_spacing={REAL_POOL_SPACING}):"
            f"\n    build_task_table: {t_total * 1000:8.1f} ms "
            f"({n_states} states, {n_tasks} tasks)"
        )
        # Production setup runs this once on env init, so seconds are tolerable.
        # Flag obvious regressions only.
        assert t_total < 120.0, f"build_task_table took {t_total:.1f}s (regression?)"

    def test_accuracy_ground_clearance(self, real_result):
        """Every spawn base must sit at a plausible stance height above its support surface."""
        import trimesh

        table = real_result["table"]
        terrain_mesh = real_result["terrain"].terrain_mesh
        states = table.states
        root_pose = states.root_pose[:, 0]

        bases = root_pose[:, :3].cpu().numpy()
        # Cast downward starting just above the base: the first hit is the
        # support surface the IK solved for. Starting from high altitude would
        # catch obstacle tops on multi-level sub-terrains (floating_island,
        # climbing_box) instead of the support below the feet.
        ray_origins = np.stack([bases[:, 0], bases[:, 1], bases[:, 2] + 0.05], axis=-1)
        ray_dirs = np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float64), (bases.shape[0], 1))

        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(terrain_mesh)
        locations, index_ray, _ = intersector.intersects_location(
            ray_origins,
            ray_dirs,
            multiple_hits=True,
        )
        # For every base, take the highest intersection whose Z <= base_z
        # (the support surface directly beneath the body).
        terrain_z = np.full(bases.shape[0], np.nan, dtype=np.float64)
        for loc, ri in zip(locations, index_ray):
            if loc[2] > bases[ri, 2] + 1e-3:
                continue
            if np.isnan(terrain_z[ri]) or loc[2] > terrain_z[ri]:
                terrain_z[ri] = loc[2]
        hit = ~np.isnan(terrain_z)
        gap = bases[:, 2] - terrain_z
        gap_hit = gap[hit]

        n = bases.shape[0]
        quat_norm = root_pose[:, 3:7].norm(dim=-1)
        finite = (
            torch.isfinite(states.root_pose).all(dim=(1, 2))
            & torch.isfinite(states.root_velocity).all(dim=(1, 2))
            & torch.isfinite(states.joint_position).all(dim=1)
            & torch.isfinite(states.joint_velocity).all(dim=1)
        )
        print(
            f"\n  Realistic accuracy ({n} states):"
            f"\n    finite-state fraction:     {finite.float().mean().item():.1%}"
            f"\n    quat norm mean/min/max:    {quat_norm.mean().item():.4f} / "
            f"{quat_norm.min().item():.4f} / {quat_norm.max().item():.4f}"
            f"\n    support-ray hit fraction:  {hit.mean():.1%}"
            f"\n    base-to-support gap [m]:   "
            f"mean={gap_hit.mean():.3f} median={np.median(gap_hit):.3f} "
            f"p5={np.percentile(gap_hit, 5):.3f} p95={np.percentile(gap_hit, 95):.3f} "
            f"min={gap_hit.min():.3f} max={gap_hit.max():.3f}"
        )

        assert bool(finite.all()), "Position reset-state bank contains non-finite values"
        torch.testing.assert_close(
            quat_norm,
            torch.ones_like(quat_norm),
            atol=0.01,
            rtol=0,
        )
        # Most bases should have a support surface below them (steep overhang-
        # free sub-terrains always do; a handful may miss on corner cases).
        # Threshold is 0.85 rather than 0.9 because the realistic mix at
        # ``pool_spacing=1.0`` lands ~10% of bases on cell edges where the
        # downward ray hits no triangle within numerical tolerance --
        # observed range across runs is ~89-91%, so 0.85 leaves a noise
        # margin without masking actual quality regressions (a real
        # regression would drop well below 80%).
        assert hit.mean() >= 0.85, f"Only {hit.mean():.1%} of bases have a support surface below"
        # ANYmal-C stance height ~0.55 m; IK deviation on uneven terrain
        # ranges ~0.3-0.9 m. Median outside [0.3, 1.0] m is suspicious.
        med = float(np.median(gap_hit))
        assert 0.3 <= med <= 1.0, f"Median base-to-support gap {med:.3f} m outside [0.3, 1.0] m"
