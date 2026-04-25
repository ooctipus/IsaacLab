# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration test: terrain generation -> IK pipeline -> task table builder.

No IsaacSim required -- uses TerrainGenerator + Newton + Warp directly.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.terrain_generator import TerrainGenerator
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg

from isaaclab_tasks.manager_based.multi_task.mdp.util.kinematics import NewtonKinematicsCfg
from isaaclab_tasks.manager_based.multi_task.terrain.mdp.commands.task_table_builder import build_task_table
from isaaclab_tasks.manager_based.multi_task.terrain.mdp.retarget import RetargetPipelineCfg
from isaaclab_tasks.manager_based.multi_task.terrain.mdp.retarget.cfg import SamplerCfg


@pytest.fixture(scope="module", autouse=True)
def _init_warp():
    wp.init()


DEVICE = "cuda:0"
NUM_ROWS = 3
NUM_COLS = 4
CELL_SIZE = (8.0, 8.0)


@pytest.fixture(scope="module")
def terrain():
    """Generate a small terrain grid for testing."""
    from isaaclab.terrains import HfPyramidStairsTerrainCfg

    cfg = TerrainGeneratorCfg(
        size=CELL_SIZE,
        num_rows=NUM_ROWS,
        num_cols=NUM_COLS,
        border_width=0.0,
        use_cache=False,
        seed=42,
        curriculum=True,
        sub_terrains={
            "stairs": HfPyramidStairsTerrainCfg(
                step_height_range=(0.05, 0.15),
                step_width=0.3,
                platform_width=2.0,
                inverted=True,
            )
        },
    )
    return TerrainGenerator(cfg=cfg, device=DEVICE)


@pytest.fixture(scope="module")
def pipeline_cfg():
    """Pipeline cfg for ANYmal-C."""
    from isaaclab.utils.assets import check_file_path, retrieve_file_path

    usd = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Robots/ANYbotics/anymal_c.usd"
    fs = check_file_path(usd)
    if fs == 2:
        usd = retrieve_file_path(usd, force_download=False)
    elif fs == 0:
        pytest.skip("ANYmal USD not available")

    return RetargetPipelineCfg(
        kin=NewtonKinematicsCfg(
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
        ),
        sampler=SamplerCfg(),
        foot_body_names=["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"],
    )


@pytest.fixture(scope="module")
def simple_commands():
    """Minimal command dict for testing."""
    from isaaclab_tasks.manager_based.multi_task.terrain.mdp.commands.commands_cfg import RelativeStateCommandCfg

    return {
        "walk": RelativeStateCommandCfg.TerrainCommands(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            pos_z=(0.0, 0.0),
            duration=(2.0, 2.0),
        )
    }


@pytest.fixture(scope="module")
def stairs_task_table(terrain, pipeline_cfg, simple_commands):
    """Shared task-table build on the stairs grid (pipeline runs once per module)."""
    if not wp.is_device_available("cuda:0"):
        pytest.skip("GPU required")
    origins = torch.tensor(terrain.terrain_origins, device=DEVICE, dtype=torch.float32)
    result = build_task_table(
        terrain_mesh=terrain.terrain_mesh,
        terrain_origins=origins,
        cell_size=CELL_SIZE,
        pipeline_cfg=pipeline_cfg,
        commands=simple_commands,
        num_joints=12,
        pool_size=200,
        device=DEVICE,
    )
    return {"result": result, "origins": origins}


class TestBuildTaskTable:
    """Shape-level smoke test for the task table builder."""

    def test_produces_valid_table(self, stairs_task_table):
        """State validity (finite + unit quats) is covered by
        :meth:`TestRealistic.test_accuracy_ground_clearance`, and index validity
        is covered by :meth:`TestIsolation.test_spawn_target_match_tile_index`
        (which must dereference every index to run). This is just the shape
        contract on ``params``.
        """
        result = stairs_task_table["result"]
        assert result["num_tasks"] > 0
        assert result["spawn_states"].shape[0] > 0
        assert result["spawn_index"].shape == result["target_index"].shape
        assert result["params"].shape[1] == 13
        print(f"  Tasks: {result['num_tasks']}, States: {result['spawn_states'].shape[0]}")


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
def flat_task_table(flat_terrain, pipeline_cfg, simple_commands):
    """Shared task-table build on the flat grid (pipeline runs once per module).

    ``pool_size = 20 * n_tiles`` so the same result covers the uniformity-CV
    check (which needs many states per tile) as well as the isolation /
    border-leak checks.
    """
    if not wp.is_device_available("cuda:0"):
        pytest.skip("GPU required")
    origins = torch.tensor(flat_terrain.terrain_origins, device=DEVICE, dtype=torch.float32)
    n_tiles = NUM_ROWS * NUM_COLS
    result = build_task_table(
        terrain_mesh=flat_terrain.terrain_mesh,
        terrain_origins=origins,
        cell_size=CELL_SIZE,
        pipeline_cfg=pipeline_cfg,
        commands=simple_commands,
        num_joints=12,
        pool_size=20 * n_tiles,
        device=DEVICE,
    )
    return {"result": result, "origins": origins, "n_tiles": n_tiles}


class TestIsolation:
    """Regression guard: every task's spawn/target must live in the tile reported by tile_index."""

    def test_spawn_target_match_tile_index(self, flat_task_table):
        result = flat_task_table["result"]
        origins = flat_task_table["origins"]
        spawn_states = result["spawn_states"]
        spawn_idx = result["spawn_index"]
        target_idx = result["target_index"]
        tile_idx = result["tile_index"]

        spawn_tile, spawn_in = _bin_xy(
            spawn_states[spawn_idx, :2],
            origins,
            CELL_SIZE,
            NUM_ROWS,
            NUM_COLS,
        )
        target_tile, target_in = _bin_xy(
            spawn_states[target_idx, :2],
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
        spawn_states = result["spawn_states"]
        tile, in_grid = _bin_xy(spawn_states[:, :2], origins, CELL_SIZE, NUM_ROWS, NUM_COLS)
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
REAL_POOL_SIZE = 5000  # matches RelativeStateCommandCfg.pool_size


@pytest.fixture(scope="module")
def realistic_terrain():
    """Production-scale terrain: 10x20 grid, 20m border, full sub-terrain mix.

    Mirrors what actually runs in training via :class:`SceneCfg.terrain`, so
    uniformity / speed / accuracy numbers here are representative rather than
    best-case flat-plane numbers.
    """
    from isaaclab_tasks.manager_based.multi_task.terrain.mdp_presets.terrain_presets import (
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
        sub_terrains=SubTerrainPresetCfg().all,
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
    def real_result(self, realistic_terrain, pipeline_cfg, simple_commands):
        """Run the full pipeline once at production scale; share across tests.

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
            pipeline_cfg=pipeline_cfg,
            commands=simple_commands,
            num_joints=12,
            pool_size=REAL_POOL_SIZE,
            device=DEVICE,
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
        spawn_states = table["spawn_states"]
        n_tiles = REAL_NUM_ROWS * REAL_NUM_COLS

        tile, in_grid = _bin_xy(
            spawn_states[:, :2],
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
            f"\n  Realistic uniformity (pool_size={REAL_POOL_SIZE}, grid={REAL_NUM_ROWS}x{REAL_NUM_COLS}):"
            f"\n    in-grid states:  {int(counts.sum())} / {spawn_states.shape[0]} "
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
        n_states = real_result["table"]["spawn_states"].shape[0]
        n_tasks = real_result["table"]["num_tasks"]
        print(
            f"\n  Realistic speed (pool_size={REAL_POOL_SIZE}):"
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
        spawn_states = table["spawn_states"]

        bases = spawn_states[:, :3].cpu().numpy()
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
        quat_norm = spawn_states[:, 3:7].norm(dim=-1)
        finite = torch.isfinite(spawn_states).all(dim=-1)
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

        assert bool(finite.all()), "spawn_states contains non-finite values"
        torch.testing.assert_close(
            quat_norm,
            torch.ones_like(quat_norm),
            atol=0.01,
            rtol=0,
        )
        # Most bases should have a support surface below them (steep overhang-
        # free sub-terrains always do; a handful may miss on corner cases).
        assert hit.mean() >= 0.9, f"Only {hit.mean():.1%} of bases have a support surface below"
        # ANYmal-C stance height ~0.55 m; IK deviation on uneven terrain
        # ranges ~0.3-0.9 m. Median outside [0.3, 1.0] m is suspicious.
        med = float(np.median(gap_hit))
        assert 0.3 <= med <= 1.0, f"Median base-to-support gap {med:.3f} m outside [0.3, 1.0] m"
