# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build the task table: IK pipeline -> bin by terrain cell -> Cartesian product.

Pure function module -- no classes, no state. Called once during command
term initialization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.utils.warp import convert_to_warp_mesh

if TYPE_CHECKING:
    import trimesh

    from ...mdp.retarget.cfg import RetargetPipelineCfg
    from .commands_cfg import RelativeStateCommandCfg


def build_task_table(
    terrain_mesh: trimesh.Trimesh,
    terrain_origins: torch.Tensor,
    cell_size: tuple[float, float],
    pipeline_cfg: RetargetPipelineCfg,
    commands: dict[str, RelativeStateCommandCfg.Commands],
    num_joints: int,
    pool_size: int,
    device: str,
) -> dict:
    """Run the IK pipeline, bin states by terrain cell, build task table.

    Args:
        terrain_mesh: Combined terrain trimesh.
        terrain_origins: Per-cell origins ``[num_rows, num_cols, 3]``.
        cell_size: ``(width, height)`` of each terrain cell [m].
        pipeline_cfg: Retarget pipeline configuration.
        commands: Command type dict from the command cfg.
        num_joints: Number of robot joints.
        pool_size: Total IK-solved states to generate.
        device: Torch/Warp device string.

    Returns:
        Dict with keys matching :class:`TaskTable` fields:
        ``spawn_states``, ``spawn_index``, ``target_index``,
        ``params``, ``task_mask``, ``offsets``, ``kind``, ``num_tasks``.
    """
    from .commands_cfg import RelativeStateCommandCfg

    num_rows, num_cols = terrain_origins.shape[0], terrain_origins.shape[1]
    num_subterrains = num_rows * num_cols
    cell_size_t = torch.tensor(cell_size, device=device)

    # --- Step 1: Run IK pipeline on full terrain mesh ---
    wp_mesh = convert_to_warp_mesh(terrain_mesh.vertices, terrain_mesh.faces, device=device)
    pipeline = pipeline_cfg.class_type(pipeline_cfg)
    buffer = pipeline.run(wp_mesh, np.zeros(3), pool_size)
    print(pipeline.rejection_summary)

    if buffer.num_selected > 0:
        selected = buffer._selected[: buffer.num_selected].long()
        spawn_states = buffer.joint_q_result_t[selected].clone()
    else:
        origins_flat = terrain_origins.reshape(-1, 3).to(device)
        identity_quat = torch.zeros(origins_flat.shape[0], 4, device=device)
        identity_quat[:, 3] = 1.0
        zeros_joints = torch.zeros(origins_flat.shape[0], num_joints, device=device)
        spawn_states = torch.cat([origins_flat, identity_quat, zeros_joints], dim=-1)

    print(f"  Task table: {spawn_states.shape[0]} IK-solved states")

    # --- Step 2: Bin states by terrain cell (CSR, no padding) ---
    # Drop states that fall outside the sub-terrain grid (e.g. the flat border
    # around the terrain). The IK sampler feeds on the full mesh and has no
    # notion of sub-terrain boundaries, so a raw ``.clamp`` would silently
    # push border states into edge cells and break geometric isolation.
    grid_origin = terrain_origins[0, 0, :2].to(device) - cell_size_t * 0.5

    base_xy = spawn_states[:, :2]
    cell_xy = (base_xy - grid_origin.unsqueeze(0)) / cell_size_t.unsqueeze(0)
    row_idx = cell_xy[:, 0].long()
    col_idx = cell_xy[:, 1].long()
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
    if dropped > 0:
        print(f"  Dropped {dropped} border/out-of-grid states")
    cmin = int(counts_per_cell.min().item())
    cmax = int(counts_per_cell.max().item())
    cmean = float(counts_per_cell.float().mean().item())
    print(
        f"  Binned into {num_rows}x{num_cols} grid: {non_empty}/{num_subterrains} non-empty cells, "
        f"counts min={cmin} mean={cmean:.1f} max={cmax}"
    )

    # --- Step 3: Per-cell Cartesian product (shared across command types) ---
    # For each cell c with n_c states, produce n_c x n_c (spawn, target) pairs.
    # Cell layout is built once and reused for every command type.
    pair_spawn_parts: list[torch.Tensor] = []
    pair_target_parts: list[torch.Tensor] = []
    pair_tile_parts: list[torch.Tensor] = []
    offsets_cpu = cell_offsets.cpu().tolist()
    for cell in range(num_subterrains):
        start = offsets_cpu[cell]
        end = offsets_cpu[cell + 1]
        n_c = end - start
        if n_c == 0:
            continue
        ids = cell_values[start:end]
        pair_spawn_parts.append(ids.repeat_interleave(n_c))
        pair_target_parts.append(ids.repeat(n_c))
        pair_tile_parts.append(torch.full((n_c * n_c,), cell, device=device, dtype=torch.long))

    if pair_spawn_parts:
        pair_spawn = torch.cat(pair_spawn_parts)
        pair_target = torch.cat(pair_target_parts)
        pair_tile = torch.cat(pair_tile_parts)
    else:
        pair_spawn = torch.zeros(0, device=device, dtype=torch.long)
        pair_target = torch.zeros(0, device=device, dtype=torch.long)
        pair_tile = torch.zeros(0, device=device, dtype=torch.long)
    num_pairs_per_type = int(pair_spawn.shape[0])

    # --- Step 4: Replicate pair layout per command type; sample per-type params ---
    ranges = torch.zeros((len(commands), 13, 2), device=device)
    mask = torch.zeros((len(commands), 12), device=device, dtype=torch.bool)
    kind = torch.zeros(len(commands), dtype=torch.int32, device=device)

    spawn_indices_list = []
    target_indices_list = []
    tile_indices_list = []
    params_list = []
    mask_list = []
    row_counts = []

    for cmd_id, val in enumerate(commands.values()):
        for data_id, data in enumerate(val.__dict__.values()):
            if data is not None and isinstance(data, tuple):
                if data_id < 12:
                    mask[cmd_id, data_id] = True
                ranges[cmd_id, data_id, 0] = data[0]
                ranges[cmd_id, data_id, 1] = data[1]

        if isinstance(val, RelativeStateCommandCfg.TerrainCommands):
            val.pos_x = val.pos_y = val.pos_z = None
            kind[cmd_id] = 1 if (val.roll or val.pitch or val.yaw) else 0
        elif isinstance(val, RelativeStateCommandCfg.PositionCommands):
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
        if isinstance(val, RelativeStateCommandCfg.TerrainCommands):
            full_mask[:, 12:] = True

        spawn_indices_list.append(pair_spawn)
        target_indices_list.append(pair_target)
        tile_indices_list.append(pair_tile)
        params_list.append(task_params)
        mask_list.append(full_mask)
        row_counts.append(num_pairs_per_type)

    all_spawn = torch.cat(spawn_indices_list, dim=0)
    all_target = torch.cat(target_indices_list, dim=0)
    all_tile = torch.cat(tile_indices_list, dim=0)
    all_params = torch.cat(params_list, dim=0)
    all_masks = torch.cat(mask_list, dim=0)

    counts_t = torch.tensor(row_counts, device=device, dtype=torch.long)
    offsets = torch.zeros(len(commands) + 1, device=device, dtype=torch.long)
    offsets[1:] = torch.cumsum(counts_t, dim=0)

    print(f"  {int(all_spawn.shape[0])} tasks ({len(commands)} command types)")

    return {
        "num_tasks": int(all_spawn.shape[0]),
        "spawn_index": all_spawn,
        "target_index": all_target,
        "tile_index": all_tile,
        "params": all_params,
        "task_mask": all_masks,
        "offsets": offsets,
        "kind": kind,
        "spawn_states": spawn_states,
    }
