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
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.utils.warp import convert_to_warp_mesh

from isaaclab_tasks.manager_based.multi_task.curriculum import pack_articulation_reset_state
from isaaclab_tasks.manager_based.multi_task.trace import trace_span

from ...grid_downsample import grid_bucket_downsample

if TYPE_CHECKING:
    import trimesh

    from ...retarget.cfg import RetargetPipelineCfg
    from .commands_cfg import RelativeStateCommandCfg


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


def _timing_checkpoint(device: str) -> float:
    """Return a synchronized wall-clock timestamp for startup diagnostics."""
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(torch_device)
    return time.perf_counter()


def synthesize_terrain_origins(
    num_rows: int,
    num_cols: int,
    size: tuple[float, float],
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Reproduce :attr:`isaaclab.terrains.TerrainGenerator.terrain_origins`.

    IsaacLab's :class:`~isaaclab.terrains.TerrainGenerator` lays out tiles at
    ``((row + 0.5 - num_rows/2) * size_x, (col + 0.5 - num_cols/2) * size_y)``
    (inline in :meth:`_add_sub_terrain` + the post-loop centering transform).
    There's no public utility that exposes just the centers, so we compute
    them from the cfg here. The resulting tensor matches what
    :attr:`isaaclab.terrains.TerrainImporter.terrain_origins` would hold when
    :paramref:`~isaaclab.terrains.TerrainImporterCfg.use_terrain_origins` is
    ``True`` (and is needed when it is ``False``, since the importer leaves
    ``terrain_origins = None`` in that case).

    Args:
        num_rows: Sub-terrain row count.
        num_cols: Sub-terrain column count.
        size: ``(size_x, size_y)`` per-tile in metres.
        device: Tensor device.

    Returns:
        Per-tile centres of shape ``(num_rows, num_cols, 3)``.
    """
    origins = torch.zeros(num_rows, num_cols, 3, device=device)
    row_centers = (torch.arange(num_rows, device=device) - (num_rows - 1) / 2.0) * size[0]
    col_centers = (torch.arange(num_cols, device=device) - (num_cols - 1) / 2.0) * size[1]
    origins[..., 0] = row_centers.view(-1, 1)
    origins[..., 1] = col_centers.view(1, -1)
    return origins


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


def build_task_table(
    terrain_mesh: trimesh.Trimesh,
    terrain_origins: torch.Tensor,
    cell_size: tuple[float, float],
    pipeline_cfg: RetargetPipelineCfg,
    commands: dict[str, RelativeStateCommandCfg.Commands | RelativeStateCommandCfg.TerrainCommands],
    num_joints: int,
    device: str,
    pool_spacing: float,
    pool_spacing_area_divisor: float = 3.0,
    pool_sampling_size: tuple[float, float] | None = None,
    robot_joint_names: Sequence[str] | None = None,
    exclude_self_pairs: bool = True,
    num_targets_per_cell: int = 0,
) -> dict:
    """Run the IK pipeline, bin states by terrain cell, build task table.

    Args:
        terrain_mesh: Combined terrain trimesh.
        terrain_origins: Per-cell origins ``[num_rows, num_cols, 3]``.
        cell_size: ``(width, height)`` of each terrain cell [m].
        pipeline_cfg: Retarget pipeline configuration.
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
        num_targets_per_cell: Optional cap on per-cell target states for
            the spawn × target pairing. ``0`` keeps the full per-cell
            Cartesian product (``n_c × n_c``). A positive integer ``N``
            picks ``min(N, n_c)`` targets per cell via
            :func:`~isaaclab_tasks.manager_based.multi_task.terrain.grid_downsample.grid_bucket_downsample`
            on the cell's spawn xy and pairs every spawn with each picked
            target.

    Returns:
        Dict with keys matching :class:`TaskTable` fields plus FK metadata:
        ``spawn_states``, ``spawn_index``, ``target_index``,
        ``params``, ``task_mask``, ``offsets``, ``kind``,
        ``task_is_terrain``, ``task_uses_feet``, ``num_tasks``, ``kin``,
        ``newton_joint_names``, ``foot_body_names``, ``foot_body_ids``.
    """
    from .commands_cfg import RelativeStateCommandCfg

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
    with trace_span(
        "task_table.retarget_pipeline",
        requested_states=total_states,
        sampling_area=sampling_area,
        pool_spacing=pool_spacing,
    ):
        pipeline = grid_pipeline_cfg.class_type(grid_pipeline_cfg)
        grid_origin = np.zeros(3, dtype=np.float32)
        buffer = pipeline.run(wp_mesh, grid_origin, total_states)
        last_rejection_summary = pipeline.rejection_summary
    if buffer.num_selected == 0:
        raise RuntimeError(
            f"Retarget pipeline produced no valid terrain states for RelativeStateCommand.\n{last_rejection_summary}"
        )
    print(
        f"  Global retargeting: requested {total_states}, selected {buffer.num_selected} "
        f"({100.0 * buffer.num_selected / max(1, total_states):.1f}%)",
        flush=True,
    )

    with trace_span("task_table.pack_spawn_states", selected_states=int(buffer.num_selected)):
        selected = buffer._selected[: buffer.num_selected].long()
        spawn_states = buffer.joint_q_result_t[selected].clone()
        newton_joint_names = _newton_revolute_joint_names(pipeline.kin)
        spawn_states = _reorder_spawn_state_joints(
            spawn_states,
            newton_joint_names,
            robot_joint_names,
        )
        spawn_states = pack_articulation_reset_state(spawn_states[:, :7], spawn_states[:, 7:])

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
    # For each cell c with n_c spawn states, the target set is either the
    # full cell (``num_targets_per_cell == 0`` -> n_c × n_c pairs) or an
    # FPS-thinned subset of size ``min(N, n_c)`` (-> n_c × min(N, n_c) pairs).
    # Cell layout is built once and reused for every command type.
    pair_spawn_parts: list[torch.Tensor] = []
    pair_target_parts: list[torch.Tensor] = []
    pair_tile_parts: list[torch.Tensor] = []
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
            if num_targets_per_cell <= 0:
                target_ids_in_cell = ids
            else:
                n_targets = min(int(num_targets_per_cell), n_c)
                local_idx = grid_bucket_downsample(spawn_xy[ids], n_targets)
                target_ids_in_cell = ids[local_idx]
            n_t = int(target_ids_in_cell.shape[0])
            spawn_ids = ids.repeat_interleave(n_t)
            target_ids = target_ids_in_cell.repeat(n_c)
            if exclude_self_pairs:
                keep = spawn_ids != target_ids
                spawn_ids = spawn_ids[keep]
                target_ids = target_ids[keep]
            pair_count = int(spawn_ids.shape[0])
            if pair_count == 0:
                continue
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
        task_is_terrain_list = []
        task_uses_feet_list = []
        row_counts = []

        for cmd_id, val in enumerate(commands.values()):
            is_terrain_command = isinstance(val, RelativeStateCommandCfg.TerrainCommands)
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
                if isinstance(val, RelativeStateCommandCfg.PositionCommands):
                    kind[cmd_id] = 0
                elif isinstance(val, RelativeStateCommandCfg.PoseCommands):
                    kind[cmd_id] = 1
                elif isinstance(val, RelativeStateCommandCfg.VelocityCommands):
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
            task_is_terrain_list.append(
                torch.full((num_pairs_per_type,), is_terrain_command, device=device, dtype=torch.bool)
            )
            task_uses_feet_list.append(
                torch.full(
                    (num_pairs_per_type,),
                    is_terrain_command and val.match_feet,
                    device=device,
                    dtype=torch.bool,
                )
            )
            row_counts.append(num_pairs_per_type)

        all_spawn = torch.cat(spawn_indices_list, dim=0)
        all_target = torch.cat(target_indices_list, dim=0)
        all_tile = torch.cat(tile_indices_list, dim=0)
        all_params = torch.cat(params_list, dim=0)
        all_masks = torch.cat(mask_list, dim=0)
        all_task_is_terrain = torch.cat(task_is_terrain_list, dim=0)
        all_task_uses_feet = torch.cat(task_uses_feet_list, dim=0)

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
        "task_is_terrain": all_task_is_terrain,
        "task_uses_feet": all_task_uses_feet,
        "offsets": offsets,
        "kind": kind,
        "spawn_states": spawn_states,
        "kin": pipeline.kin,
        "newton_joint_names": newton_joint_names,
        "foot_body_names": pipeline.foot_body_names,
        "foot_body_ids": pipeline.foot_body_ids,
    }
