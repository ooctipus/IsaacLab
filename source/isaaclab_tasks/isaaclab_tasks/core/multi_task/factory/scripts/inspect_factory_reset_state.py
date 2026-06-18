# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect the configured Factory reset-state table without launching simulation.

The script resolves the same ``commands.reset_state`` config used by training,
builds the geometric Newton-IK table, applies the configured geometric filters
and target pairing, prints the table summary, and opens a viser preview by
default. It deliberately does not use ``env.scene.num_envs``.

Run:
  SCRIPT=source/isaaclab_tasks/isaaclab_tasks/core/multi_task/factory/scripts/inspect_factory_reset_state.py
  ./isaaclab.sh -p $SCRIPT presets=franka,nut_thread_m16
  ./isaaclab.sh -p $SCRIPT presets=franka,nut_thread_m16 --no_viewer
  # writes /tmp/factory_success_grid_*.png by default
  ./isaaclab.sh -p $SCRIPT presets=franka,nut_thread_m16 --no_viewer
"""

from __future__ import annotations

import argparse
import importlib
import re
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import warp as wp

from isaaclab.utils.assets import check_file_path, retrieve_file_path

from isaaclab_tasks.core.multi_task.curriculum import StateLayout
from isaaclab_tasks.core.multi_task.factory.mdp.reset_state_task_table import (
    FactoryResetStateTaskTable,
    _nut_bounds_mask,
    _pair_within_boards,
)
from isaaclab_tasks.core.multi_task.factory.retarget import (
    FactoryIKPipeline,
    collision_min_sd,
    posed_collision_min_sd,
    self_collision_min_sd,
)
from isaaclab_tasks.utils.hydra import resolve_task_config

COMMAND_NAME = "reset_state"
_TAG_PALETTE = (
    (0.20, 0.55, 1.00),
    (0.30, 0.85, 0.35),
    (1.00, 0.45, 0.20),
    (0.75, 0.40, 0.95),
    (0.20, 0.85, 0.85),
    (0.95, 0.80, 0.20),
    (0.95, 0.35, 0.65),
    (0.55, 0.75, 0.30),
)


class _PreviewCommandManager:
    """Tiny command-manager facade for the production sampler visual logger."""

    def __init__(self, table: FactoryResetStateTaskTable):
        self._reset_state = SimpleNamespace(table=table)

    def get_term(self, name: str):
        if name != COMMAND_NAME:
            raise KeyError(name)
        return self._reset_state


class _PreviewEnv:
    """Minimal env facade consumed by the configured logger and sampler."""

    def __init__(self, table: FactoryResetStateTaskTable):
        self.command_manager = _PreviewCommandManager(table)
        self.common_step_counter = 0
        self.extras: dict[str, dict[str, object]] = {}


def _resolve_factory_cfg(task: str, agent: str):
    """Resolve the same Factory env cfg path used by training, without Kit."""
    importlib.import_module("isaaclab_tasks.core.multi_task.factory.config")
    env_cfg, agent_cfg = resolve_task_config(task, agent)
    command_cfg = getattr(env_cfg.commands, COMMAND_NAME)
    table_cfg = command_cfg.task_table
    pcfg = table_cfg.pipeline_cfg
    pcfg.scene = env_cfg.scene
    usd_path = env_cfg.scene.robot.spawn.usd_path
    status = check_file_path(usd_path)
    if status == 0:
        raise FileNotFoundError(f"robot USD not found: {usd_path}")
    if status == 2:
        usd_path = retrieve_file_path(usd_path, force_download=False)
    pcfg.robot.usd_path = usd_path
    return env_cfg, agent_cfg, command_cfg, table_cfg


def _print_config(command_cfg, table_cfg) -> None:
    """Print the reset-state table config that was resolved."""
    pcfg = table_cfg.pipeline_cfg
    print("\n=== configured Factory reset-state command ===")
    print(f"  command              : commands.{COMMAND_NAME}")
    print(f"  command keys         : {list(command_cfg.commands.keys())}")
    print(f"  reset assets         : {list(command_cfg.payload.reset_assets)}")
    print(f"  board configurations : {int(pcfg.board.num_boards)}")
    print(f"  yield ratio          : {float(pcfg.yield_ratio):.3f}")
    print(f"  diversity knob       : {float(pcfg.diversity_knob):.2f}")
    print(f"  rows/board           : {int(table_cfg.rows_per_board)}")
    print(f"  targets/board        : {int(table_cfg.targets_per_board)}")
    print(f"  nut bounds           : {table_cfg.nut_bounds}")
    print(f"  stash viz geometry   : {bool(table_cfg.stash_viz_geometry)}")


def _ordered_joint_names(model) -> list[str]:
    """Return the robot joint order used by the geometric reset-state rows."""
    names: list[str] = []
    for name in [*model.arm_joint_names, *model.finger_joint_names]:
        if name not in names:
            names.append(name)
    return names


def _robot_joint_pos(model, result, valid: torch.Tensor, squeeze: torch.Tensor) -> torch.Tensor:
    """Map Newton joint coordinates into reset-state joint-position rows."""
    joint_names = _ordered_joint_names(model)
    joint_pos = model.default_joint_q().expand(int(valid.sum()), -1).clone()
    arm_pairs = [(coord, joint_names.index(name)) for coord, name in zip(model.arm_coords, model.arm_joint_names)]
    finger_pairs = [
        (coord, joint_names.index(name)) for coord, name in zip(model.finger_coords, model.finger_joint_names)
    ]
    joint_q = result.joint_q[valid]
    joint_pos[:, [idx for _, idx in arm_pairs]] = joint_q[:, [coord for coord, _ in arm_pairs]]
    joint_pos[:, [idx for _, idx in finger_pairs]] = joint_q[:, [coord for coord, _ in finger_pairs]]
    joint_pos[:, [idx for _, idx in finger_pairs]] += squeeze[valid].unsqueeze(-1)
    return joint_pos


def _root_state_from_pose(pose: torch.Tensor) -> torch.Tensor:
    """Create root-state rows from env-local pose rows and zero root velocity."""
    root_state = torch.zeros(pose.shape[0], 13, device=pose.device, dtype=pose.dtype)
    root_state[:, :7] = pose
    return root_state


def _build_geometric_table(command_cfg, table_cfg, pipeline: FactoryIKPipeline, result) -> FactoryResetStateTaskTable:
    """Build the configured geometric table without touching a live env."""
    pcfg = table_cfg.pipeline_cfg
    model = pipeline.model
    valid = _nut_bounds_mask(table_cfg, result.nut_pose[:, :3])
    n_rows = int(valid.sum())
    n_base = len(pipeline.placement_sampler.tag_names)
    grasped = result.tag < n_base
    squeeze = torch.where(result.family % 2 == 1, table_cfg.finger_squeeze, -table_cfg.finger_squeeze)
    squeeze = torch.where(grasped, squeeze, torch.zeros_like(squeeze))

    robot_root = torch.zeros(n_rows, 13, device=pcfg.device)
    robot_root[:, 6] = 1.0
    robot_joint_pos = _robot_joint_pos(model, result, valid, squeeze)
    robot_joint_vel = torch.zeros_like(robot_joint_pos)
    pose_by_asset = {
        pcfg.board.board_asset_cfg.name: result.board_pose[valid],
        pcfg.board.fixed_asset_cfg.name: result.bolt_pose[valid],
        pcfg.placement.held_asset_cfg.name: result.nut_pose[valid],
    }
    rigid_states = [
        _root_state_from_pose(pose_by_asset[name])
        for name in command_cfg.payload.reset_assets
        if name != pcfg.robot.asset_cfg.name
    ]
    state_data = torch.cat([robot_root, robot_joint_pos, robot_joint_vel, *rigid_states], dim=-1).contiguous()
    state_tag_indices = result.tag[valid].contiguous()
    state_board_indices = result.board_index[valid].contiguous()
    coords, spawn_index, target_index, slot_indices, task_tag_indices = _pair_within_boards(
        table_cfg, state_data, state_tag_indices, state_board_indices
    )
    viz_geom = {}
    if bool(table_cfg.stash_viz_geometry) and n_rows > 0:
        from isaaclab_tasks.core.multi_task.factory.viz.geometry import build_success_grid_geometry

        viz_geom = build_success_grid_geometry(
            pipeline.model,
            result.joint_q[valid],
            result.nut_pose[valid],
            result.bolt_pose[valid],
            result.board_pose[valid],
            result.board_index[valid],
        )
    return FactoryResetStateTaskTable(
        state_data=state_data,
        state_tag_indices=state_tag_indices,
        state_board_indices=state_board_indices,
        state_tag_names=list(result.tag_names),
        state_coords=coords,
        spawn_index=spawn_index,
        target_index=target_index,
        slot_indices=slot_indices,
        task_tag_indices=task_tag_indices,
        num_states=n_rows,
        built_size=int(result.joint_q.shape[0]),
        target_size=int(table_cfg.rows_per_board) * int(pcfg.board.num_boards),
        **viz_geom,
    )


def _print_table_summary(table: FactoryResetStateTaskTable, pipeline: FactoryIKPipeline) -> None:
    """Print row, target, slot, and tag counts for the built table."""
    print(f"\n{pipeline.rejection_summary}")
    num_goals = int(torch.unique(table.target_index).numel()) if table.target_index.numel() else 0
    boards = torch.unique(table.state_board_indices).sort().values if table.num_states > 0 else torch.empty(0)
    survival = table.num_states / table.built_size if table.built_size > 0 else 0.0
    print("\n=== built reset-state table ===")
    print(f"  states       : {table.num_states} kept / {table.built_size} built ({survival:.1%})")
    print(f"  target size  : {int(table.target_size)}")
    print(f"  boards       : {int(boards.numel())} with at least one state")
    print(f"  goal states  : {num_goals}")
    print(f"  task slots   : {table.num_tasks}")
    target_mask = torch.zeros(table.num_states, dtype=torch.bool, device=table.state_data.device)
    if table.target_index.numel() > 0:
        target_mask[torch.unique(table.target_index)] = True
    for tag, name in enumerate(table.state_tag_names):
        mask = table.state_tag_indices == tag
        if not bool(mask.any()):
            continue
        print(f"  {name:24s}: {int(mask.sum()):5d} states, {int((mask & target_mask).sum()):5d} goals")


def _safe_image_name(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", tag).strip("_")


def _write_success_grid(env_cfg, table: FactoryResetStateTaskTable, out_prefix: str) -> None:
    """Run the configured training sampler visual logger against the simless preview table."""
    # Keep this path wired through the resolved env config. The inspector should
    # reproduce training behavior, not override visual logger render settings.
    sampler_term = env_cfg.curriculum.reset_sampler
    sampler_cfg = sampler_term.params["sampling"]
    logger = sampler_term.params["sampler_visual_logger"]
    if logger is None:
        raise RuntimeError("resolved Factory env cfg has no reset_sampler sampler_visual_logger configured.")
    preview_env = _PreviewEnv(table)
    success_rates = torch.zeros(table.num_tasks, device=table.state_data.device)
    layout = StateLayout(coords=table.state_coords, spawn_index=table.spawn_index, target_index=table.target_index)
    sampler = sampler_cfg.class_type(sampler_cfg, layout, env=preview_env, success_rates=success_rates)
    probs = sampler.probabilities()
    logger(preview_env, sampler, success_rates, probs)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images = preview_env.extras.get("log_images", {})
    if not images:
        raise RuntimeError("configured sampler_visual_logger did not emit any images.")
    for tag, image in images.items():
        path = Path(f"{out_prefix}_{_safe_image_name(tag)}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(path, image)
        print(f"[factory_reset_state] wrote {tag} -> {path} ({image.shape[1]}x{image.shape[0]})")


def _collision_mesh_table(model) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Return robot collision mesh shapes in body-local frames."""
    import newton

    flags = model.shape_flags.numpy()
    shape_body = model.shape_body.numpy()
    shape_tf = wp.to_torch(model.shape_transform).cpu().numpy()
    collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
    meshes = []
    for shape_id in range(model.shape_count):
        if (int(flags[shape_id]) & collide) == 0:
            continue
        source = model.shape_source[shape_id] if shape_id < len(model.shape_source) else None
        verts = getattr(source, "vertices", None) if source is not None else None
        faces = getattr(source, "indices", None) if source is not None else None
        if verts is None or faces is None:
            continue
        meshes.append(
            (
                int(shape_body[shape_id]),
                np.asarray(verts, dtype=np.float32).reshape(-1, 3),
                np.asarray(faces, dtype=np.int32).reshape(-1, 3),
                shape_tf[shape_id, :3].astype(np.float32, copy=True),
                shape_tf[shape_id, 3:7].astype(np.float32, copy=True),
            )
        )
    return meshes


def _run_viewer(pipeline: FactoryIKPipeline, result, valid: torch.Tensor, port: int, collision_geom: bool) -> None:
    """Render board cells, robot solutions, held asset, contact pads, and sampler funnel in viser."""
    import viser

    from isaaclab_tasks.core.multi_task.factory.retarget import points_vs_body_meshes_min_sd
    from isaaclab_tasks.core.multi_task.factory.retarget.criteria import posed_points
    from isaaclab_tasks.core.multi_task.factory.retarget.model import _quat_xyzw_rot as _rot
    from isaaclab_tasks.core.multi_task.factory.retarget.model import load_collider_mesh

    cfg = pipeline.cfg
    fmodel = pipeline.model
    sel = torch.nonzero(valid, as_tuple=False).flatten()
    row_tags = result.tag[sel].cpu().numpy()
    board_ids = result.board_index[sel].cpu().numpy()
    row_count = int(sel.numel())
    lib_board, lib_bolt = pipeline.board_library
    n_boards = int(lib_board.shape[0])
    print(f"[viz] rendering all {row_count} configured table rows across {n_boards} board-configuration cells")
    geom_mode = "collision/simple" if collision_geom else "visual"
    print(f"[viz] geometry mode: {geom_mode}")
    print("[viz] robot and held rows rendered as collision meshes tinted by placement tag")

    body_q_t = fmodel.eval_fk(result.joint_q[sel])
    grip_probes = wp.array(fmodel.gripper_probes, dtype=wp.vec3, device=cfg.device)
    grip_bodies = wp.array(fmodel.gripper_probe_bodies, dtype=wp.int32, device=cfg.device)
    static_sd_mm = {
        name: (
            collision_min_sd(
                body_q_t,
                pipeline._robot_probe_bodies_wp,
                pipeline._robot_probes_wp,
                mesh.id,
                0.05,
                cfg.device,
            )
            * 1e3
        )
        .cpu()
        .numpy()
        for name, mesh in fmodel.static_obstacles.items()
    }
    held_probes_t = torch.tensor(fmodel.held_probes, device=cfg.device)
    nut_pose_t = result.nut_pose[sel]
    nut_pts = posed_points(held_probes_t, nut_pose_t)
    fwd = posed_collision_min_sd(body_q_t, grip_bodies, grip_probes, fmodel.held_mesh.id, nut_pose_t, 0.05, cfg.device)
    rev = points_vs_body_meshes_min_sd(
        nut_pts,
        body_q_t,
        pipeline._grip_target_body_wp,
        pipeline._grip_target_mesh_wp,
        pipeline._grip_target_tf_wp,
        0.05,
        cfg.device,
    )
    nut_sd_mm = (torch.minimum(fwd, rev) * 1e3).cpu().numpy()
    self_sd_mm = (
        (
            self_collision_min_sd(
                body_q_t,
                pipeline._self_probe_body_wp,
                pipeline._self_probes_wp,
                pipeline._self_target_body_wp,
                pipeline._self_target_mesh_wp,
                pipeline._self_target_tf_wp,
                pipeline._self_adj_wp,
                pipeline._n_bodies,
                0.05,
                cfg.device,
            )
            * 1e3
        )
        .cpu()
        .numpy()
    )

    body_q = body_q_t.cpu().numpy()
    nut_pose = result.nut_pose[sel].cpu().numpy()
    if collision_geom:
        obstacle_render = dict(fmodel.obstacle_geom)
        board_v, board_f = fmodel.board_verts, fmodel.board_faces
        bolt_v, bolt_f = fmodel.fixed_verts, fmodel.fixed_faces
    else:
        obstacle_render = {}
        for name, (usd_path, spawn_scale, pos, quat) in fmodel.obstacle_spec.items():
            verts, faces = load_collider_mesh(usd_path, cfg.device, scale=spawn_scale, visual=True)
            obstacle_render[name] = (_rot(quat, verts) + pos, faces)
        board_v, board_f = load_collider_mesh(fmodel.board_spec[0], cfg.device, scale=fmodel.board_spec[1], visual=True)
        bolt_v, bolt_f = load_collider_mesh(fmodel.fixed_spec[0], cfg.device, scale=fmodel.fixed_spec[1], visual=True)
    lib_board_np = lib_board.cpu().numpy()
    lib_bolt_np = lib_bolt.cpu().numpy()

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.scene.set_up_direction("+z")
    row_handles_by_tag: dict[int, list[Any]] = {tag_idx: [] for tag_idx in range(len(result.tag_names))}
    cols = int(np.ceil(np.sqrt(n_boards)))
    spacing = 1.6

    def add(name, verts, faces, color, world_pos, world_quat_xyzw, offset, opacity=1.0):
        wxyz = np.array([world_quat_xyzw[3], world_quat_xyzw[0], world_quat_xyzw[1], world_quat_xyzw[2]])
        return server.scene.add_mesh_simple(
            name,
            verts,
            faces,
            color=color,
            wxyz=wxyz,
            position=world_pos + offset,
            opacity=opacity,
        )

    ident = np.array([0.0, 0.0, 0.0, 1.0])
    n_tags = len(result.tag_names)
    robot_meshes = _collision_mesh_table(fmodel.model)
    if not robot_meshes:
        raise RuntimeError("robot USD did not expose collision mesh geometry for the viser preview")
    robot_verts_by_tag: list[list[np.ndarray]] = [[] for _ in range(n_tags)]
    robot_faces_by_tag: list[list[np.ndarray]] = [[] for _ in range(n_tags)]
    robot_vert_base_by_tag = [0 for _ in range(n_tags)]
    held_verts_by_tag: list[list[np.ndarray]] = [[] for _ in range(n_tags)]
    held_faces_by_tag: list[list[np.ndarray]] = [[] for _ in range(n_tags)]
    held_vert_base_by_tag = [0 for _ in range(n_tags)]

    for board_id in range(n_boards):
        off = np.array([(board_id % cols) * spacing, (board_id // cols) * spacing, 0.0], dtype=np.float32)
        grp = f"/board_{board_id}"
        add(
            f"{grp}/board",
            board_v,
            board_f,
            (0.35, 0.35, 0.4),
            lib_board_np[board_id, :3],
            lib_board_np[board_id, 3:7],
            off,
        )
        add(
            f"{grp}/bolt",
            bolt_v,
            bolt_f,
            (0.5, 0.5, 0.55),
            lib_bolt_np[board_id, :3],
            lib_bolt_np[board_id, 3:7],
            off,
        )
        for obstacle_name, (obstacle_v, obstacle_f) in obstacle_render.items():
            add(f"{grp}/{obstacle_name}", obstacle_v, obstacle_f, (0.6, 0.6, 0.6), np.zeros(3), ident, off, 0.35)

        cell_rows = np.nonzero(board_ids == board_id)[0]
        for cell_row in cell_rows.tolist():
            tag = int(row_tags[cell_row])
            for body_id, shape_verts, shape_faces, shape_pos, shape_quat in robot_meshes:
                body_pos = body_q[cell_row, body_id, :3]
                body_quat = body_q[cell_row, body_id, 3:7]
                body_verts = _rot(body_quat, _rot(shape_quat, shape_verts) + shape_pos) + body_pos + off
                robot_verts_by_tag[tag].append(body_verts.astype(np.float32, copy=False))
                robot_faces_by_tag[tag].append(shape_faces + robot_vert_base_by_tag[tag])
                robot_vert_base_by_tag[tag] += shape_verts.shape[0]

            held_pos = nut_pose[cell_row, :3]
            held_quat = nut_pose[cell_row, 3:7]
            held_verts = _rot(held_quat, fmodel.held_verts) + held_pos + off
            held_verts_by_tag[tag].append(held_verts.astype(np.float32, copy=False))
            held_faces_by_tag[tag].append(fmodel.held_faces + held_vert_base_by_tag[tag])
            held_vert_base_by_tag[tag] += fmodel.held_verts.shape[0]

        if cell_rows.size:
            tag_txt = " ".join(
                f"{result.tag_names[tag]} x{count}"
                for tag, count in zip(*np.unique(row_tags[cell_rows], return_counts=True))
            )
            worst_static = min(static_sd_mm[name][cell_rows].min() for name in static_sd_mm)
            label = (
                f"b{board_id} | {cell_rows.size} rows | {tag_txt} | "
                f"worst sd[mm]: static {worst_static:.0f} nut {nut_sd_mm[cell_rows].min():.1f} "
                f"self {self_sd_mm[cell_rows].min():.0f}"
            )
        else:
            label = f"b{board_id} | 0 rows (configuration unused by the kept table)"
        server.scene.add_label(
            f"{grp}/label",
            text=label,
            position=lib_board_np[board_id, :3] + off + np.array([0.0, 0.0, 0.35]),
        )

    for tag, tag_name in enumerate(result.tag_names):
        if not robot_verts_by_tag[tag]:
            continue
        arm_color = _TAG_PALETTE[tag % len(_TAG_PALETTE)]
        row_handles_by_tag[tag].append(
            server.scene.add_mesh_simple(
                f"/rows/{tag_name}/franka_collision",
                np.concatenate(robot_verts_by_tag[tag], axis=0),
                np.concatenate(robot_faces_by_tag[tag], axis=0),
                color=arm_color,
                opacity=0.38,
            )
        )
        row_handles_by_tag[tag].append(
            server.scene.add_mesh_simple(
                f"/rows/{tag_name}/held_asset_collision",
                np.concatenate(held_verts_by_tag[tag], axis=0),
                np.concatenate(held_faces_by_tag[tag], axis=0),
                color=(1.0, 0.84, 0.0),
                opacity=0.62,
            )
        )
    tag_counts = (
        np.bincount(row_tags, minlength=len(result.tag_names))
        if row_tags.size
        else np.zeros(len(result.tag_names), dtype=np.int64)
    )
    server.gui.add_markdown("### Placement tag filters")

    def set_tag_visible(tag_idx: int, visible: bool) -> None:
        for handle in row_handles_by_tag[tag_idx]:
            handle.visible = visible

    for tag_idx, tag_name in enumerate(result.tag_names):
        if tag_counts[tag_idx] == 0:
            continue
        checkbox = server.gui.add_checkbox(f"{tag_name} ({int(tag_counts[tag_idx])})", initial_value=True)

        @checkbox.on_update
        def _on_tag_update(event, tag_idx=tag_idx) -> None:
            set_tag_visible(tag_idx, bool(event.target.value))

    gs = pipeline.grasp_sampler
    scale = 10.0
    ex_off = np.array([-2.5, 0.0, 0.5], dtype=np.float32)
    surf = gs.surface_points.cpu().numpy() * scale + ex_off
    cand = np.concatenate([gs.candidate_pair_a.cpu().numpy(), gs.candidate_pair_b.cpu().numpy()])
    max_candidate_points = 4096
    if cand.shape[0] > max_candidate_points:
        cand = cand[np.linspace(0, cand.shape[0] - 1, max_candidate_points, dtype=np.int64)]
    cand = cand * scale + ex_off
    seg = np.stack([gs.pair_a.cpu().numpy(), gs.pair_b.cpu().numpy()], axis=1) * scale + ex_off
    fam = gs.pair_family.cpu().numpy()
    fam_palette = np.array(
        [
            [0.9, 0.2, 0.2],
            [0.2, 0.6, 1.0],
            [0.2, 0.9, 0.3],
            [1.0, 0.6, 0.1],
            [0.8, 0.3, 0.9],
            [0.2, 0.9, 0.9],
        ],
        dtype=np.float32,
    )
    seg_colors = np.repeat(fam_palette[fam % len(fam_palette)][:, None, :], 2, axis=1)
    server.scene.add_mesh_simple(
        "/sampler/held_asset",
        fmodel.held_verts * scale,
        fmodel.held_faces,
        color=(1.0, 0.84, 0.0),
        position=ex_off,
        opacity=0.3,
    )
    server.scene.add_point_cloud(
        "/sampler/surface_samples", points=surf, colors=np.full_like(surf, 0.45), point_size=0.004
    )
    server.scene.add_point_cloud(
        "/sampler/antipodal_candidates",
        points=cand,
        colors=np.tile(np.array([[0.3, 0.55, 1.0]], dtype=np.float32), (cand.shape[0], 1)),
        point_size=0.002,
    )
    server.scene.add_line_segments("/sampler/retained_pairs", points=seg, colors=seg_colors, line_width=2.0)
    server.scene.add_point_cloud(
        "/sampler/retained_contacts",
        points=seg.reshape(-1, 3),
        colors=np.tile(np.array([[1.0, 0.1, 0.1]], dtype=np.float32), (seg.shape[0] * 2, 1)),
        point_size=0.006,
    )
    server.scene.add_label(
        "/sampler/label",
        text=(
            f"sampler funnel ({scale:.0f}x): {surf.shape[0]} surface samples -> "
            f"{cand.shape[0]} shown candidate contacts -> {seg.shape[0]} retained pairs"
        ),
        position=ex_off + np.array([0.0, 0.0, 0.35]),
    )

    actual_port = server.get_port()
    hostname = socket.gethostname()
    print(
        f"\n[factory_reset_state] viser ready: http://localhost:{actual_port} "
        f"({n_boards} board cells, {row_count} table rows overlaid)"
    )
    print(f"[factory_reset_state] remote URL:  http://{hostname}:{actual_port}")
    print("[factory_reset_state] robot and held-asset collision meshes are tinted by placement tag.")
    print("[factory_reset_state] use the viser placement-tag checkboxes to show/hide groups.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        return


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect the configured Factory reset-state table without simulation.")
    ap.add_argument("--task", type=str, default="Isaac-Factory-v0", help="Gym task id to resolve.")
    ap.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent cfg entry point to resolve.")
    ap.add_argument("--no_viewer", action="store_true", help="Print diagnostics and exit without starting viser.")
    ap.add_argument("--viewer_port", type=int, default=8080, help="Viser port.")
    ap.add_argument("--success_grid_out", type=str, default="/tmp/factory_success_grid", help="Output PNG path prefix.")
    ap.add_argument("--no_success_grid", action="store_true", help="Skip writing the success-grid PNG.")
    ap.add_argument(
        "--visual_meshes",
        action="store_false",
        dest="collision_geom",
        help="Render expensive visual meshes for the board/bolt/obstacles instead of collision/simple geometry.",
    )
    ap.add_argument("--collision_geom", action="store_true", help=argparse.SUPPRESS)
    ap.set_defaults(collision_geom=True)
    ap.add_argument("--headless", action="store_true", help=argparse.SUPPRESS)

    from isaaclab_tasks.utils import setup_preset_cli

    args, remaining = setup_preset_cli(ap)
    if not any(tok.startswith("presets=") for tok in remaining):
        remaining = ["presets=franka,nut_thread_m16"] + remaining
        print("[factory_reset_state] no presets= given, defaulting to presets=franka,nut_thread_m16")
    sys.argv = [sys.argv[0]] + remaining

    env_cfg, _, command_cfg, table_cfg = _resolve_factory_cfg(args.task, args.agent)
    _print_config(command_cfg, table_cfg)

    pipeline = FactoryIKPipeline(table_cfg.pipeline_cfg)
    target_size = int(table_cfg.rows_per_board) * int(table_cfg.pipeline_cfg.board.num_boards)
    result = pipeline.build_balanced_table(target_size)
    table = _build_geometric_table(command_cfg, table_cfg, pipeline, result)
    _print_table_summary(table, pipeline)
    valid = _nut_bounds_mask(table_cfg, result.nut_pose[:, :3])

    if not args.no_success_grid:
        _write_success_grid(env_cfg, table, args.success_grid_out)
    if not args.no_viewer:
        _run_viewer(pipeline, result, valid, args.viewer_port, args.collision_geom)


if __name__ == "__main__":
    main()
