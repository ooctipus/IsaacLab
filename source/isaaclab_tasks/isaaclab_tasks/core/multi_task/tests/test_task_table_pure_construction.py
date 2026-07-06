# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure task-table construction boundaries shared by Position, Factory, and Motion."""

import inspect
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from isaaclab.terrains import HfDiscreteObstaclesTerrainCfg, TerrainGeneratorCfg
from isaaclab.terrains.terrain_generator import TerrainGenerator

from isaaclab_tasks.core.multi_task.factory.mdp.reset_state_task_table import (
    build_factory_reset_state_task_table,
)
from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table import build_motion_task_table
from isaaclab_tasks.core.multi_task.terrain.mdp.commands.task_table_builder import (
    build_relative_state_task_table,
)
from isaaclab_tasks.core.multi_task.terrain.terrains.trimesh import (
    FlatBeamCfg,
    MeshBalanceBeamsTerrainCfg,
    MeshContourTerrainCfg,
    MeshFloatingIslandTerrainCfg,
    MeshMazeTerrainCfg,
    MeshRadiatingBeamTerrainCfg,
    MeshStonesEverywhereTerrainCfg,
)


@pytest.mark.parametrize(
    "builder",
    (
        build_relative_state_task_table,
        build_factory_reset_state_task_table,
        build_motion_task_table,
    ),
)
def test_task_table_builder_does_not_accept_a_live_environment(builder) -> None:
    """Builders consume resolved configuration and a device, never a live environment."""
    parameters = inspect.signature(builder).parameters
    assert "env" not in parameters
    assert tuple(parameters) == ("command_cfg", "scene_cfg", "device")


@pytest.mark.parametrize(
    "builder",
    (
        build_relative_state_task_table,
        build_factory_reset_state_task_table,
        build_motion_task_table,
    ),
)
def test_task_table_builder_source_does_not_read_live_environment_state(builder) -> None:
    """The production builder body contains no live-environment access."""
    source = inspect.getsource(builder)
    for forbidden in ("env.scene", "env.cfg", "env.device"):
        assert forbidden not in source


def test_position_builder_uses_the_declared_generator_and_one_table_rng() -> None:
    """Position terrain and stance sampling consume one table-owned random stream."""
    source = inspect.getsource(build_relative_state_task_table)
    assert "terrain_generator_cfg.class_type" in source
    assert "rng=rng.numpy" in source
    assert "TerrainGenerator(" not in source


def test_custom_terrains_have_no_ambient_random_calls() -> None:
    """Custom terrain functions may consume only their explicit ``rng`` argument."""
    source_path = Path(__file__).parents[1] / "terrain" / "terrains" / "trimesh" / "mesh_terrains.py"
    source = source_path.read_text()
    for forbidden in (
        "import random",
        "random.choice",
        "random.randint",
        "random.random",
        "random.shuffle",
        "random.uniform",
        "np.random.uniform",
        "torch.rand",
        ".uniform_(",
    ):
        assert forbidden not in source


def _terrain_cfg(sub_terrain) -> TerrainGeneratorCfg:
    return TerrainGeneratorCfg(
        seed=17,
        size=(4.0, 4.0),
        num_rows=1,
        num_cols=1,
        curriculum=False,
        use_cache=False,
        horizontal_scale=0.1,
        vertical_scale=0.01,
        sub_terrains={"test": sub_terrain},
    )


@pytest.mark.parametrize(
    "cfg",
    (
        _terrain_cfg(MeshMazeTerrainCfg(grid_cols=3, grid_rows=3, wall_height=1.0)),
        _terrain_cfg(
            MeshStonesEverywhereTerrainCfg(
                w_gap=(0.1, 0.1),
                w_stone=(0.6, 0.6),
                s_max=(0.05, 0.05),
                h_max=(0.02, 0.02),
                holes_depth=-1.0,
                platform_width=1.0,
            )
        ),
        _terrain_cfg(
            MeshBalanceBeamsTerrainCfg(
                platform_width=1.0,
                h_offset=(0.02, 0.02),
                w_stone=(0.25, 0.25),
                mid_gap=(0.25, 0.25),
            )
        ),
        _terrain_cfg(
            MeshContourTerrainCfg(
                num_levels=3,
                level_height=0.1,
                stones=MeshContourTerrainCfg.StoneCfg(num_stones=2),
            )
        ),
        _terrain_cfg(
            MeshRadiatingBeamTerrainCfg(
                platform_width=1.0,
                num_bars=(2, 2),
                beam_distribution="random",
                border=MeshRadiatingBeamTerrainCfg.SquareBorderCfg(inner_size=(3.0, 3.0)),
                bar_width_range=(0.4, 0.4),
                bar_height_range=(0.5, 0.5),
            )
        ),
        _terrain_cfg(
            MeshFloatingIslandTerrainCfg(
                num_islands=3,
                island_style=MeshFloatingIslandTerrainCfg.CylinderIslandCfg(radius=0.4),
                island_margin=0.2,
                graph=MeshFloatingIslandTerrainCfg.DelaunayGraphCfg(),
                passway_style=FlatBeamCfg(),
                passway_width=0.3,
            )
        ),
        _terrain_cfg(
            HfDiscreteObstaclesTerrainCfg(
                num_obstacles=8,
                obstacle_height_mode="fixed",
                obstacle_width_range=(0.2, 0.8),
                obstacle_height_range=(1.0, 1.0),
                platform_width=0.5,
            )
        ),
    ),
)
def test_seeded_terrain_generation_is_repeatable_without_ambient_rng(cfg) -> None:
    """Representative Position terrains are seed-exact and preserve ambient RNG state."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        first = TerrainGenerator(cfg.copy())
        assert random.getstate() == python_state
        assert all(np.array_equal(left, right) for left, right in zip(np.random.get_state(), numpy_state))
        assert torch.equal(torch.random.get_rng_state(), torch_state)

        random.seed(7919)
        np.random.seed(7919)
        torch.manual_seed(7919)
        perturbed_python_state = random.getstate()
        perturbed_numpy_state = np.random.get_state()
        perturbed_torch_state = torch.random.get_rng_state()
        second = TerrainGenerator(cfg.copy())
        assert random.getstate() == perturbed_python_state
        assert all(np.array_equal(left, right) for left, right in zip(np.random.get_state(), perturbed_numpy_state))
        assert torch.equal(torch.random.get_rng_state(), perturbed_torch_state)
        assert np.array_equal(first.terrain_mesh.vertices, second.terrain_mesh.vertices)
        assert np.array_equal(first.terrain_mesh.faces, second.terrain_mesh.faces)
        assert np.array_equal(first.terrain_origins, second.terrain_origins)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
