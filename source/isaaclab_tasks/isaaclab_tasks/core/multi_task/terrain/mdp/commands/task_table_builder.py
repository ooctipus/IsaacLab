# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build the task table: IK pipeline -> bin by terrain cell -> Cartesian product.

Pure function module -- no classes, no state. Called once during command
term initialization.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.utils.warp import convert_to_warp_mesh

from isaaclab_tasks.core.multi_task.utils.trace import trace_span

from ....utils.grid_downsample import grid_bucket_downsample
from ...retarget import apply_final_fps

if TYPE_CHECKING:
    import trimesh

    from isaaclab.envs import ManagerBasedEnv

    from ....mdp.commands.state_command.state_command_cfg import StateCommandCfg
    from ...retarget.cfg import RetargetPipelineCfg
    from .commands_cfg import Commands, TerrainCommands


_COMMAND_PARAM_NAMES = (
    "pos_x",
    "pos_y",
    "pos_z",
    "roll",
    "pitch",
    "yaw",
    "lin_vel_x",
    "lin_vel_y",
    "lin_vel_z",
    "ang_vel_x",
    "ang_vel_y",
    "ang_vel_z",
    "duration",
)


@dataclass
class RelativeStateTaskTable:
    """Index-based task table for the locomotion :class:`~...mdp.commands.StateCommand`."""

    num_tasks: int
    spawn_index: torch.Tensor
    """Index into ``spawn_states`` for each task's spawn point."""
    target_index: torch.Tensor
    """Index into ``spawn_states`` for each task's target point."""
    tile_index: torch.Tensor
    """Terrain tile (``row * num_cols + col``) each task belongs to ``[num_tasks]``."""
    params: torch.Tensor
    """Per-task sampled parameters ``[num_tasks, 13]``:
    ``[0:3]`` pos offset, ``[3:6]`` rot, ``[6:9]`` lin_vel,
    ``[9:12]`` ang_vel, ``[12]`` hold_time."""
    task_mask: torch.Tensor
    """Active command mask ``[num_tasks, 12 + num_joints]``."""
    payload_flags: torch.Tensor
    """Opaque payload flags ``[num_tasks, num_payload_flags]``."""
    offsets: torch.Tensor
    """CSR offsets ``[num_cmd_types + 1]`` into the task table."""
    task_partition: torch.Tensor
    """Command type id for each task row ``[num_tasks]``."""
    kind: torch.Tensor
    """Command type tag ``[num_cmd_types]``: 0=pos, 1=pose, 2=vel."""
    spawn_states: torch.Tensor
    """Zero-copy reference to reset states ``[num_states, 13 + 2 * num_joints]``."""

    target_fk_kin: object | None = None
    """Newton kinematics used to FK target feet."""
    newton_joint_names: list[str] | None = None
    """Newton joint names matching :attr:`target_fk_kin`."""
    foot_body_names: list[str] | None = None
    """Foot body names resolved in Newton order."""
    newton_foot_body_ids: list[int] | None = None
    """Foot body ids in Newton order."""
    isaac_to_newton_joint_order: torch.Tensor | None = None
    """Index map from Isaac joint order to Newton joint order."""
    foot_body_ids: list[int] | None = None
    """Foot body ids in Isaac articulation order."""

    def sample_rows(self, count: int) -> torch.Tensor:
        """Sample task rows uniformly on the table device."""
        return torch.randint(0, self.num_tasks, (count,), device=self.spawn_index.device)

    def gather(self, task_rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(spawn_states, target_states)`` rows for the selected tasks.

        Args:
            task_rows: Task-table row indices ``[n]``.

        Returns:
            Spawn and target reset-state rows, each ``[n, 13 + 2 * num_joints]``.
        """
        spawn_states = self.spawn_states[self.spawn_index[task_rows]].clone()
        target_states = self.spawn_states[self.target_index[task_rows]].clone()
        return spawn_states, target_states


def build_relative_state_task_table(cfg: StateCommandCfg, env: ManagerBasedEnv) -> RelativeStateTaskTable:
    """Build the relative-state command task table from a command cfg and environment."""
    table_cfg = cfg.task_table
    if table_cfg.pipeline_cfg.asset_cfg is None:
        raise ValueError("RelativeStateCommand requires cfg.task_table.pipeline_cfg.asset_cfg.")
    robot = env.scene[table_cfg.pipeline_cfg.asset_cfg.name]
    terrain = env.scene.terrain
    if terrain.terrain_mesh is None:
        raise RuntimeError(
            "RelativeStateCommand requires a terrain with a mesh. Set terrain_type='generator' in TerrainImporterCfg."
        )
    terrain_origins = terrain.terrain_origins
    if terrain_origins is None:
        terrain_gen = terrain.cfg.terrain_generator
        terrain_origins = _synthesize_terrain_origins(
            num_rows=int(terrain_gen.num_rows),
            num_cols=int(terrain_gen.num_cols),
            cell_size=terrain_gen.size,
            device=env.device,
        )

    table_data = build_task_table(
        terrain_mesh=terrain.terrain_mesh,
        terrain_origins=terrain_origins,
        cell_size=terrain.cfg.terrain_generator.size,
        pipeline_cfg=table_cfg.pipeline_cfg,
        env=env,
        commands=cfg.commands,
        num_joints=robot.num_joints,
        device=env.device,
        pool_spacing=table_cfg.pool_spacing,
        pool_spacing_area_divisor=table_cfg.pool_spacing_area_divisor,
        pool_sampling_size=table_cfg.pool_sampling_size,
        robot_joint_names=robot.joint_names,
        exclude_self_pairs=table_cfg.exclude_self_pairs,
        max_spawns_per_cell=table_cfg.max_spawns_per_cell,
        num_targets_per_cell=table_cfg.num_targets_per_cell,
    )
    target_fk_kin = table_data.pop("kin")
    newton_joint_names = table_data.pop("newton_joint_names")
    foot_body_names_expected = table_data.pop("foot_body_names")
    newton_foot_body_ids = table_data.pop("foot_body_ids")
    isaac_to_newton_joint_order = torch.tensor(
        _joint_order_from_names(robot.joint_names, newton_joint_names),
        device=env.device,
        dtype=torch.long,
    )
    foot_body_ids, foot_body_names = robot.find_bodies(foot_body_names_expected, preserve_order=True)
    if foot_body_names != foot_body_names_expected:
        raise RuntimeError(
            "PhysX foot body order does not match Newton foot body order: "
            f"physx={foot_body_names}, newton={foot_body_names_expected}."
        )
    table_data["task_partition"] = torch.bucketize(
        torch.arange(int(table_data["num_tasks"]), device=env.device),
        table_data["offsets"][1:-1],
        right=True,
    )
    return RelativeStateTaskTable(
        **table_data,
        target_fk_kin=target_fk_kin,
        newton_joint_names=newton_joint_names,
        foot_body_names=foot_body_names_expected,
        newton_foot_body_ids=newton_foot_body_ids,
        isaac_to_newton_joint_order=isaac_to_newton_joint_order,
        foot_body_ids=foot_body_ids,
    )


def _timing_checkpoint(device: str) -> float:
    """Return a synchronized wall-clock timestamp for startup diagnostics."""
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(torch_device)
    return time.perf_counter()


def _terrain_grid_bounds(
    terrain_origins: torch.Tensor,
    cell_size: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return inner terrain-grid XY bounds [m]."""
    half_x = 0.5 * cell_size[0]
    half_y = 0.5 * cell_size[1]
    x_range = (
        float(terrain_origins[..., 0].amin().item() - half_x),
        float(terrain_origins[..., 0].amax().item() + half_x),
    )
    y_range = (
        float(terrain_origins[..., 1].amin().item() - half_y),
        float(terrain_origins[..., 1].amax().item() + half_y),
    )
    return x_range, y_range


def _synthesize_terrain_origins(
    num_rows: int,
    num_cols: int,
    cell_size: tuple[float, float],
    device: str,
) -> torch.Tensor:
    """Return terrain-generator tile origins when importer env origins are disabled."""
    origins = torch.zeros(num_rows, num_cols, 3, device=device)
    rows = (torch.arange(num_rows, device=device, dtype=torch.float32) + 0.5) * cell_size[0]
    cols = (torch.arange(num_cols, device=device, dtype=torch.float32) + 0.5) * cell_size[1]
    rows -= cell_size[0] * num_rows * 0.5
    cols -= cell_size[1] * num_cols * 0.5
    origins[..., 0], origins[..., 1] = torch.meshgrid(rows, cols, indexing="ij")
    return origins


def _state_count_from_spacing(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    spacing: float,
    area_divisor: float,
) -> int:
    """Derive terrain-state count from spacing and sampling area."""
    if spacing <= 0.0:
        raise ValueError(f"pool_spacing must be positive, got {spacing}.")
    if area_divisor <= 0.0:
        raise ValueError(f"pool_spacing_area_divisor must be positive, got {area_divisor}.")
    area = max((x_range[1] - x_range[0]) * (y_range[1] - y_range[0]), 0.0)
    return max(1, int(area / (area_divisor * spacing**2)))


def _centered_sampling_bounds(
    grid_x_range: tuple[float, float],
    grid_y_range: tuple[float, float],
    sampling_size: tuple[float, float] | None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return full-grid or centered clipped sampling bounds [m]."""
    if sampling_size is None:
        return grid_x_range, grid_y_range

    size_x, size_y = sampling_size
    if size_x <= 0.0 or size_y <= 0.0:
        raise ValueError(f"pool_sampling_size must be positive, got {sampling_size}.")

    center_x = 0.5 * (grid_x_range[0] + grid_x_range[1])
    center_y = 0.5 * (grid_y_range[0] + grid_y_range[1])
    half_x = 0.5 * size_x
    half_y = 0.5 * size_y
    x_range = (max(grid_x_range[0], center_x - half_x), min(grid_x_range[1], center_x + half_x))
    y_range = (max(grid_y_range[0], center_y - half_y), min(grid_y_range[1], center_y + half_y))
    return x_range, y_range


def _pipeline_with_inner_sampling_bounds(
    pipeline_cfg: RetargetPipelineCfg,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    *,
    override: bool = False,
) -> RetargetPipelineCfg:
    """Return a pipeline cfg whose sampler patch is bounded to the requested range."""
    sampler_cfg = pipeline_cfg.sampler
    patch_cfg = getattr(sampler_cfg, "patch", None)
    if patch_cfg is None:
        return pipeline_cfg

    patch_updates = {}
    if override or patch_cfg.x_range is None:
        patch_updates["x_range"] = x_range
    if override or patch_cfg.y_range is None:
        patch_updates["y_range"] = y_range
    if not patch_updates:
        return pipeline_cfg

    return pipeline_cfg.replace(sampler=sampler_cfg.replace(patch=patch_cfg.replace(**patch_updates)))


def _joint_order_from_names(source_joint_names: Sequence[str], target_joint_names: Sequence[str]) -> list[int]:
    """Return indices that reorder source-joint columns to target-joint order."""
    source_to_index: dict[str, int] = {}
    duplicates = set()
    for idx, name in enumerate(source_joint_names):
        if name in source_to_index:
            duplicates.add(name)
        source_to_index[name] = idx
    if duplicates:
        raise ValueError(f"Duplicate source joint names cannot be remapped: {sorted(duplicates)}")

    missing = [name for name in target_joint_names if name not in source_to_index]
    if missing:
        raise ValueError(
            "Retargeted Newton joints do not cover all Isaac articulation joints. "
            f"Missing: {missing}. Available Newton joints: {list(source_joint_names)}"
        )

    return [source_to_index[name] for name in target_joint_names]


def _newton_revolute_joint_names(kin) -> list[str]:
    """Return Newton revolute joint names in ``joint_q[7:]`` coordinate order."""
    n_revolute = int(kin.model.joint_coord_count) - 7
    joint_names = [""] * n_revolute
    joint_q_start = kin.model.joint_q_start.numpy()
    joint_type = kin.model.joint_type.numpy()
    for joint_index in range(1, len(kin.joint_names)):
        if int(joint_type[joint_index]) != 1:
            continue
        coord_index = int(joint_q_start[joint_index]) - 7
        if 0 <= coord_index < n_revolute:
            joint_names[coord_index] = kin.joint_names[joint_index]

    missing = [idx for idx, name in enumerate(joint_names) if not name]
    if missing:
        raise RuntimeError(f"Could not resolve Newton revolute joint names for coordinate indices: {missing}")
    return joint_names


def _reorder_spawn_state_joints(
    spawn_states: torch.Tensor,
    source_joint_names: Sequence[str],
    target_joint_names: Sequence[str] | None,
) -> torch.Tensor:
    """Return spawn states with joint columns reordered to the target articulation."""
    if target_joint_names is None:
        return spawn_states

    n_source = len(source_joint_names)
    n_target = len(target_joint_names)
    if spawn_states.shape[1] != 7 + n_source:
        raise RuntimeError(
            "Retargeted state width does not match Newton joint-coordinate count: "
            f"spawn_states.shape[1]={spawn_states.shape[1]}, expected {7 + n_source}."
        )
    if n_source != n_target:
        raise RuntimeError(
            "Retargeted Newton joint count does not match Isaac articulation joint count: "
            f"Newton={n_source}, Isaac={n_target}."
        )

    joint_order = torch.tensor(
        _joint_order_from_names(source_joint_names, target_joint_names),
        device=spawn_states.device,
        dtype=torch.long,
    )
    return torch.cat([spawn_states[:, :7], spawn_states[:, 7:].index_select(1, joint_order)], dim=1)


def _print_retarget_timing_table(timings: dict[str, float], retarget_dt: float, final_fps_dt: float) -> None:
    """Print a compact timing summary for task-table retargeting."""
    rows = [
        ("sampler total", timings.get("sampler", 0.0)),
        ("  fused projection", timings.get("    sampler.project.fused_kernel", 0.0)),
        ("  morph patch pool", timings.get("  sampler.morph", 0.0)),
        ("  polygon projection", timings.get("  sampler.project", 0.0)),
        ("  polygon FPS", timings.get("  sampler.sampler_fps", 0.0)),
        ("IK build", timings.get("ik_build", 0.0)),
        ("IK solve", timings.get("ik_solve", 0.0)),
        ("FK eval", timings.get("fk_eval", 0.0)),
        ("criteria", timings.get("criteria", 0.0)),
        ("final FPS", final_fps_dt),
    ]
    denom = max(retarget_dt + final_fps_dt, 1.0e-12)
    width = max(len(name) for name, _ in rows)
    print(f"  Retarget timings (pipeline={retarget_dt:.3f}s, final_fps={final_fps_dt:.3f}s)", flush=True)
    for name, dt in rows:
        if dt <= 0.0:
            continue
        print(f"    {name:<{width}} {dt:>8.3f}s  {100.0 * dt / denom:>5.1f}%", flush=True)


def build_task_table(
    terrain_mesh: trimesh.Trimesh,
    terrain_origins: torch.Tensor,
    cell_size: tuple[float, float],
    pipeline_cfg: RetargetPipelineCfg,
    env: ManagerBasedEnv | None,
    commands: dict[str, Commands | TerrainCommands],
    num_joints: int,
    device: str,
    pool_spacing: float,
    pool_spacing_area_divisor: float = 3.0,
    pool_sampling_size: tuple[float, float] | None = None,
    robot_joint_names: Sequence[str] | None = None,
    exclude_self_pairs: bool = True,
    max_spawns_per_cell: int = 0,
    num_targets_per_cell: int = 0,
) -> dict:
    """Run the IK pipeline, bin states by terrain cell, build task table.

    Args:
        terrain_mesh: Combined terrain trimesh.
        terrain_origins: Per-cell origins ``[num_rows, num_cols, 3]``.
        cell_size: ``(width, height)`` of each terrain cell [m].
        pipeline_cfg: Retarget pipeline configuration.
        env: Environment passed through to env-bound retarget pipelines.
        commands: Command type dict from the command cfg.
        num_joints: Number of robot joints.
        device: Torch/Warp device string.
        pool_spacing: Target spacing between final IK-solved terrain states
            [m].
        pool_spacing_area_divisor: Area divisor used for spacing-mode pool
            sizing.
        pool_sampling_size: Optional centered XY sampling window size [m].
            ``None`` samples over the full terrain grid.
        robot_joint_names: Isaac articulation joint names in simulation order.
            When provided, retargeted Newton joint columns are reordered to
            this order before the states are stored in the table.
        exclude_self_pairs: When ``True`` (default), drop pairs where the
            spawn and target reference the same state, removing the diagonal
            of each per-cell Cartesian product. Cells with fewer than two
            valid states contribute no pairs.
        max_spawns_per_cell: Optional cap on per-cell spawn states for
            spawn × target pairing. ``0`` keeps every valid state in the cell.
            A positive integer ``N`` picks ``min(N, n_c)`` spawn states via
            :func:`~isaaclab_tasks.core.multi_task.utils.grid_downsample.grid_bucket_downsample`.
            Target states are then selected from the remaining non-spawn states
            when any remain, falling back to the full cell state pool otherwise.
        num_targets_per_cell: Optional cap on per-cell target states for
            the spawn × target pairing. ``0`` keeps the full per-cell
            Cartesian product (``n_c × n_c``). A positive integer ``N``
            picks ``min(N, n_c)`` targets per cell via
            :func:`~isaaclab_tasks.core.multi_task.utils.grid_downsample.grid_bucket_downsample`
            on the cell's spawn xy and pairs every spawn with each picked
            target.

    Returns:
        Dict with keys matching :class:`TaskTable` fields plus FK metadata:
        ``spawn_states``, ``spawn_index``, ``target_index``,
        ``params``, ``task_mask``, ``offsets``, ``kind``,
        ``payload_flags``, ``num_tasks``, ``kin``,
        ``newton_joint_names``, ``foot_body_names``, ``foot_body_ids``.
    """
    from .commands_cfg import PoseCommands, PositionCommands, TerrainCommands, VelocityCommands

    timing_start = _timing_checkpoint(device)

    num_rows, num_cols = terrain_origins.shape[0], terrain_origins.shape[1]
    num_subterrains = num_rows * num_cols
    cell_size_t = torch.tensor(cell_size, device=device)
    grid_x_range, grid_y_range = _terrain_grid_bounds(terrain_origins, cell_size)
    sampling_x_range, sampling_y_range = _centered_sampling_bounds(
        grid_x_range,
        grid_y_range,
        pool_sampling_size,
    )

    # --- Step 1: Run IK pipeline on terrain mesh ---
    terrain_area = (grid_x_range[1] - grid_x_range[0]) * (grid_y_range[1] - grid_y_range[0])
    sampling_area = (sampling_x_range[1] - sampling_x_range[0]) * (sampling_y_range[1] - sampling_y_range[0])
    total_states = _state_count_from_spacing(
        sampling_x_range,
        sampling_y_range,
        pool_spacing,
        pool_spacing_area_divisor,
    )
    if pool_sampling_size is None:
        print(
            f"  Spacing mode: area={sampling_area:.1f} m^2, spacing={pool_spacing:.3f} m "
            f"-> total_states={total_states}",
            flush=True,
        )
    else:
        print(
            f"  Spacing mode: sample_area={sampling_area:.1f} m^2 "
            f"(terrain_area={terrain_area:.1f} m^2), spacing={pool_spacing:.3f} m "
            f"-> total_states={total_states}",
            flush=True,
        )

    with trace_span(
        "task_table.convert_mesh",
        vertices=len(terrain_mesh.vertices),
        faces=len(terrain_mesh.faces),
    ):
        wp_mesh = convert_to_warp_mesh(terrain_mesh.vertices, terrain_mesh.faces, device=device)

    # Single global pipeline.run over the entire terrain. Per-cell binning
    # happens after FK so we get the same CSR layout, but without the
    # 200-cell-loop fixed overhead (solver build + criteria setup amortized
    # across one big batch instead of 200 small ones).
    grid_pipeline_cfg = _pipeline_with_inner_sampling_bounds(
        pipeline_cfg,
        sampling_x_range,
        sampling_y_range,
        override=True,
    )
    retarget_t0 = _timing_checkpoint(device)
    with trace_span(
        "task_table.retarget_pipeline",
        requested_states=total_states,
        sampling_area=sampling_area,
        pool_spacing=pool_spacing,
    ):
        pipeline = grid_pipeline_cfg.class_type(grid_pipeline_cfg, env=env, device=device)
        grid_origin = np.zeros(3, dtype=np.float32)
        buffer = pipeline.run(wp_mesh, grid_origin, total_states)
        last_rejection_summary = pipeline.rejection_summary
    retarget_dt = _timing_checkpoint(device) - retarget_t0
    # ``buffer.num_selected`` == post-criteria count (pipeline emits the
    # un-thinned set; thinning is the caller's job).
    n_candidates = int(buffer.num_selected)
    if n_candidates == 0:
        raise RuntimeError(
            f"Retarget pipeline produced no valid terrain states for RelativeStateCommand.\n{last_rejection_summary}"
        )

    # FPS spatial-thinning via the one-shot helper. ``apply_final_fps``
    # rewrites ``buffer._selected`` / ``buffer.num_selected`` in place
    # to the thinned subset; we then read survivors directly. Avoids
    # the StateBuffer indirection (allocates ``[N, state_dim]`` zeros
    # plus an explicit copy + clone) that's overkill for a one-shot.
    sizing = grid_pipeline_cfg.sampler.sizing
    final_fps_t0 = _timing_checkpoint(device)
    with trace_span("task_table.final_fps", n_candidates=n_candidates):
        apply_final_fps(
            buffer,
            n_desired=total_states,
            extractor=getattr(sizing, "fps_features", None),
            spacing=getattr(sizing, "fps_spacing", None),
        )
        target_count = int(buffer.num_selected)
        survivors_idx = buffer._selected[:target_count].long()
        spawn_states_raw = buffer.joint_q_result_t[survivors_idx].clone()
    final_fps_dt = _timing_checkpoint(device) - final_fps_t0

    print(
        f"  Global retargeting: requested {total_states}, selected {target_count} "
        f"({100.0 * target_count / max(1, total_states):.1f}%)",
        flush=True,
    )
    chunk_meta = pipeline._chunk_profile_meta
    if chunk_meta:
        print(
            f"  IK chunks: {chunk_meta['n_chunks']} @ chunk_size={chunk_meta['chunk_size']:,}"
            f" (N={chunk_meta['N']:,}, max_iters={chunk_meta['max_iters']}, "
            f"solve={pipeline._timings.get('ik_solve', 0.0):.3f}s)",
            flush=True,
        )
    _print_retarget_timing_table(pipeline._timings, retarget_dt, final_fps_dt)

    with trace_span("task_table.pack_spawn_states", selected_states=target_count):
        newton_joint_names = _newton_revolute_joint_names(pipeline.kin)
        spawn_states = _reorder_spawn_state_joints(
            spawn_states_raw,
            newton_joint_names,
            robot_joint_names,
        )
        root_state = torch.zeros(spawn_states.shape[0], 13, device=spawn_states.device, dtype=spawn_states.dtype)
        root_state[:, :7] = spawn_states[:, :7]
        spawn_states = torch.cat([root_state, spawn_states[:, 7:], torch.zeros_like(spawn_states[:, 7:])], dim=-1)

    print(f"  Task table: {spawn_states.shape[0]} IK-solved states", flush=True)

    # --- Step 2: Bin states by terrain cell (CSR, no padding) ---
    # Drop states that fall outside the sub-terrain grid (e.g. the flat border
    # around the terrain). The IK sampler feeds on the full mesh and has no
    # notion of sub-terrain boundaries, so a raw ``.clamp`` would silently
    # push border states into edge cells and break geometric isolation.
    with trace_span("task_table.bin_states", states=int(spawn_states.shape[0]), cells=num_subterrains):
        grid_origin = terrain_origins[0, 0, :2].to(device) - cell_size_t * 0.5

        base_xy = spawn_states[:, :2]
        cell_xy = (base_xy - grid_origin.unsqueeze(0)) / cell_size_t.unsqueeze(0)
        row_idx = torch.floor(cell_xy[:, 0]).long()
        col_idx = torch.floor(cell_xy[:, 1]).long()
        in_grid = (row_idx >= 0) & (row_idx < num_rows) & (col_idx >= 0) & (col_idx < num_cols)
        dropped = int((~in_grid).sum().item())

        kept_state_idx = in_grid.nonzero(as_tuple=False).squeeze(-1)
        flat_cell_kept = row_idx[in_grid] * num_cols + col_idx[in_grid]

        # CSR layout: cell_values[cell_offsets[c]:cell_offsets[c+1]] -> global spawn_states indices in cell c.
        sort_order = flat_cell_kept.argsort()
        cell_values = kept_state_idx[sort_order]

        counts_per_cell = torch.bincount(flat_cell_kept, minlength=num_subterrains)
        cell_offsets = torch.zeros(num_subterrains + 1, device=device, dtype=torch.long)
        cell_offsets[1:] = counts_per_cell.cumsum(0)

        non_empty = int((counts_per_cell > 0).sum())
        cmin = int(counts_per_cell.min().item())
        cmax = int(counts_per_cell.max().item())
        cmean = float(counts_per_cell.float().mean().item())
    if dropped > 0:
        print(f"  Dropped {dropped} border/out-of-grid states", flush=True)
    print(
        f"  Binned into {num_rows}x{num_cols} grid: {non_empty}/{num_subterrains} non-empty cells, "
        f"counts min={cmin} mean={cmean:.1f} max={cmax}",
        flush=True,
    )

    # --- Step 3: Per-cell spawn × target pairing (shared across command types) ---
    # For each cell c with n_c states, the spawn set is either the full cell
    # or a downsampled subset of size ``min(max_spawns_per_cell, n_c)``. The
    # target set is independently either the full target-candidate pool or a
    # downsampled subset of size ``min(num_targets_per_cell, n_candidates)``.
    # Cell layout is built once and reused for every command type.
    pair_spawn_parts: list[torch.Tensor] = []
    pair_target_parts: list[torch.Tensor] = []
    pair_tile_parts: list[torch.Tensor] = []
    selected_spawn_counts: list[int] = []
    selected_target_counts: list[int] = []
    offsets_cpu = cell_offsets.cpu().tolist()
    spawn_xy = spawn_states[:, :2]
    with trace_span("task_table.build_pairs", non_empty_cells=non_empty):
        for cell in range(num_subterrains):
            start = offsets_cpu[cell]
            end = offsets_cpu[cell + 1]
            n_c = end - start
            if n_c == 0:
                continue
            if exclude_self_pairs and n_c < 2:
                # Need at least two distinct states to form a non-self pair.
                continue
            ids = cell_values[start:end]
            if max_spawns_per_cell == 0:
                spawn_ids_in_cell = ids
                target_candidate_ids = ids
            else:
                if max_spawns_per_cell < 1:
                    raise ValueError(f"max_spawns_per_cell must be >= 1 or 0 for unlimited, got {max_spawns_per_cell}.")
                n_spawns = min(int(max_spawns_per_cell), n_c)
                local_idx = grid_bucket_downsample(spawn_xy[ids], n_spawns)
                spawn_ids_in_cell = ids[local_idx]
                spawn_mask = torch.zeros(n_c, device=device, dtype=torch.bool)
                spawn_mask[local_idx] = True
                target_candidate_ids = ids[~spawn_mask]
                if target_candidate_ids.numel() == 0:
                    target_candidate_ids = ids
            if num_targets_per_cell <= 0:
                target_ids_in_cell = target_candidate_ids
            else:
                n_targets = min(int(num_targets_per_cell), int(target_candidate_ids.shape[0]))
                local_idx = grid_bucket_downsample(spawn_xy[target_candidate_ids], n_targets)
                target_ids_in_cell = target_candidate_ids[local_idx]
            n_t = int(target_ids_in_cell.shape[0])
            spawn_ids = spawn_ids_in_cell.repeat_interleave(n_t)
            target_ids = target_ids_in_cell.repeat(int(spawn_ids_in_cell.shape[0]))
            if exclude_self_pairs:
                keep = spawn_ids != target_ids
                spawn_ids = spawn_ids[keep]
                target_ids = target_ids[keep]
            pair_count = int(spawn_ids.shape[0])
            if pair_count == 0:
                continue
            selected_spawn_counts.append(int(spawn_ids_in_cell.shape[0]))
            selected_target_counts.append(int(target_ids_in_cell.shape[0]))
            pair_spawn_parts.append(spawn_ids)
            pair_target_parts.append(target_ids)
            pair_tile_parts.append(torch.full((pair_count,), cell, device=device, dtype=torch.long))

        if pair_spawn_parts:
            pair_spawn = torch.cat(pair_spawn_parts)
            pair_target = torch.cat(pair_target_parts)
            pair_tile = torch.cat(pair_tile_parts)
        else:
            pair_spawn = torch.zeros(0, device=device, dtype=torch.long)
            pair_target = torch.zeros(0, device=device, dtype=torch.long)
            pair_tile = torch.zeros(0, device=device, dtype=torch.long)
        num_pairs_per_type = int(pair_spawn.shape[0])
    if num_pairs_per_type == 0:
        raise RuntimeError("No terrain cells contained valid retargeted states; cannot build a task table.")
    n_pair_cells = len(selected_spawn_counts)
    spawn_min = min(selected_spawn_counts)
    spawn_max = max(selected_spawn_counts)
    spawn_mean = sum(selected_spawn_counts) / n_pair_cells
    target_min = min(selected_target_counts)
    target_max = max(selected_target_counts)
    target_mean = sum(selected_target_counts) / n_pair_cells
    spawn_cap_msg = "unlimited" if max_spawns_per_cell == 0 else str(max_spawns_per_cell)
    target_cap_msg = "unlimited" if num_targets_per_cell <= 0 else str(num_targets_per_cell)
    print(
        "  Pair layout: "
        f"spawn_cap={spawn_cap_msg}, target_cap={target_cap_msg}, cells={n_pair_cells}/{num_subterrains}, "
        f"spawns/cell min={spawn_min} mean={spawn_mean:.1f} max={spawn_max}, "
        f"targets/cell min={target_min} mean={target_mean:.1f} max={target_max}, "
        f"pairs/command={num_pairs_per_type}",
        flush=True,
    )

    with trace_span("task_table.compact_spawn_states", states=int(spawn_states.shape[0]), pairs=num_pairs_per_type):
        used_state_ids, inverse = torch.unique(torch.cat([pair_spawn, pair_target]), sorted=True, return_inverse=True)
        pair_spawn = inverse[:num_pairs_per_type]
        pair_target = inverse[num_pairs_per_type:]
        if int(used_state_ids.shape[0]) < int(spawn_states.shape[0]):
            print(
                f"  Compacted task states: {int(spawn_states.shape[0])} -> {int(used_state_ids.shape[0])}",
                flush=True,
            )
        spawn_states = spawn_states.index_select(0, used_state_ids)

    # --- Step 4: Replicate pair layout per command type; sample per-type params ---
    with trace_span(
        "task_table.sample_command_params",
        command_types=len(commands),
        pairs_per_type=num_pairs_per_type,
    ):
        ranges = torch.zeros((len(commands), 13, 2), device=device)
        mask = torch.zeros((len(commands), 12), device=device, dtype=torch.bool)
        kind = torch.zeros(len(commands), dtype=torch.int32, device=device)

        spawn_indices_list = []
        target_indices_list = []
        tile_indices_list = []
        params_list = []
        mask_list = []
        payload_flags_list = []
        row_counts = []

        for cmd_id, val in enumerate(commands.values()):
            is_terrain_command = isinstance(val, TerrainCommands)
            if is_terrain_command:
                if val.match_base_pos:
                    mask[cmd_id, :3] = True
                if val.match_base_rot:
                    mask[cmd_id, 3:6] = True
                ranges[cmd_id, 12, 0] = val.duration[0]
                ranges[cmd_id, 12, 1] = val.duration[1]
                kind[cmd_id] = 1 if val.match_base_rot else 0
            else:
                for data_id, name in enumerate(_COMMAND_PARAM_NAMES):
                    data = getattr(val, name)
                    if data is not None and isinstance(data, tuple):
                        if data_id < 12:
                            mask[cmd_id, data_id] = True
                        ranges[cmd_id, data_id, 0] = data[0]
                        ranges[cmd_id, data_id, 1] = data[1]
                if isinstance(val, PositionCommands):
                    kind[cmd_id] = 0
                elif isinstance(val, PoseCommands):
                    kind[cmd_id] = 1
                elif isinstance(val, VelocityCommands):
                    kind[cmd_id] = 2

            range_min = ranges[cmd_id, :, 0].view(1, 13)
            range_span = ranges[cmd_id, :, 1] - ranges[cmd_id, :, 0]
            task_params = torch.rand(num_pairs_per_type, 13, device=device) * range_span.view(1, 13) + range_min

            full_mask = torch.zeros(num_pairs_per_type, 12 + num_joints, device=device, dtype=torch.bool)
            full_mask[:, :12] = mask[cmd_id].view(1, 12)
            if is_terrain_command:
                full_mask[:, 12:] = True

            spawn_indices_list.append(pair_spawn)
            target_indices_list.append(pair_target)
            tile_indices_list.append(pair_tile)
            params_list.append(task_params)
            mask_list.append(full_mask)
            payload_flags = torch.zeros(num_pairs_per_type, 1, device=device, dtype=torch.bool)
            payload_flags[:, 0] = is_terrain_command
            payload_flags_list.append(payload_flags)
            row_counts.append(num_pairs_per_type)

        all_spawn = torch.cat(spawn_indices_list, dim=0)
        all_target = torch.cat(target_indices_list, dim=0)
        all_tile = torch.cat(tile_indices_list, dim=0)
        all_params = torch.cat(params_list, dim=0)
        all_masks = torch.cat(mask_list, dim=0)
        all_payload_flags = torch.cat(payload_flags_list, dim=0)

        counts_t = torch.tensor(row_counts, device=device, dtype=torch.long)
        offsets = torch.zeros(len(commands) + 1, device=device, dtype=torch.long)
        offsets[1:] = torch.cumsum(counts_t, dim=0)

    print(f"  {int(all_spawn.shape[0])} tasks ({len(commands)} command types)", flush=True)
    print(f"  Task table built in {_timing_checkpoint(device) - timing_start:.3f} s", flush=True)

    return {
        "num_tasks": int(all_spawn.shape[0]),
        "spawn_index": all_spawn,
        "target_index": all_target,
        "tile_index": all_tile,
        "params": all_params,
        "task_mask": all_masks,
        "payload_flags": all_payload_flags,
        "offsets": offsets,
        "kind": kind,
        "spawn_states": spawn_states,
        "kin": pipeline.kin,
        "newton_joint_names": newton_joint_names,
        "foot_body_names": pipeline.foot_body_names,
        "foot_body_ids": pipeline.foot_body_ids,
    }
