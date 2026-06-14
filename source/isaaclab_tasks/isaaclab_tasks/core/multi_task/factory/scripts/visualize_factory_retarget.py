# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize the ``factory/retarget`` board-configuration library and its reset states.

The locomotion-terrain analogy made literal: builds the balanced table via
:class:`FactoryIKPipeline.build_balanced_table`, lays the board+bolt
configuration LIBRARY out as a grid (one cell per configuration, like terrain
cells), and overlays every accepted nut+robot solution pair inside its own
configuration's cell: Franka collider meshes (tinted by placement tag), the nut
(gold), and the fingertip contact targets (red = +jaw-y pad, green = -jaw-y
pad). Empty cells are configurations the table holds no rows for. Each cell's
label reports the row count, per-tag breakdown, and the worst measured
clearances among its rows. Driven by the PRODUCTION env cfg (the same preset
resolution the train script uses); ``--num_boards``/``--rows_per_board``
override the cfg values for a readable grid. Isaacsim-free (no Kit launch).

Run:
  S=source/isaaclab_tasks/isaaclab_tasks/core/multi_task/factory/scripts/visualize_factory_retarget.py
  ./isaaclab.sh -p $S presets=franka,nut_thread_m16 --num_boards 16   # viser on :8080
  ./isaaclab.sh -p $S presets=franka,nut_thread_m16 --headless        # build + funnel, no server
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import warp as wp

from isaaclab_tasks.core.multi_task.factory.retarget import (
    FactoryIKPipeline,
    collision_min_sd,
    posed_collision_min_sd,
    self_collision_min_sd,
)
from isaaclab_tasks.core.multi_task.factory.retarget.cfg import resolve_from_task

PORT = 8080
# distinct tint per tag (near_seated, mid_insertion, above_tip, on_table, in_air)
TAG_COLORS = [
    (0.2, 0.55, 1.0),
    (0.3, 0.85, 0.35),
    (1.0, 0.5, 0.05),
    (0.75, 0.4, 0.95),
    (0.2, 0.85, 0.85),
]


def _quat_xyzw_rot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    t = 2.0 * np.cross([x, y, z], v)
    return v + w * t + np.cross([x, y, z], t)


def _quat_xyzw_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def _mesh_table(model):
    """(body, verts, faces, local_pos, local_quat) for every mesh shape."""
    shape_body = model.shape_body.numpy()
    shape_tf = wp.to_torch(model.shape_transform).cpu().numpy()
    out = []
    for si in range(model.shape_count):
        src = model.shape_source[si] if si < len(model.shape_source) else None
        v = getattr(src, "vertices", None) if src is not None else None
        f = getattr(src, "indices", None) if src is not None else None
        if v is None or f is None:
            continue
        out.append(
            (
                int(shape_body[si]),
                np.asarray(v, dtype=np.float32).reshape(-1, 3),
                np.asarray(f, dtype=np.int32).reshape(-1, 3),
                shape_tf[si, :3],
                shape_tf[si, 3:7],
            )
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize the factory/retarget board library + its reset states.")
    ap.add_argument(
        "--num_boards", type=int, default=None, help="Board-library size override (default: the env cfg's)."
    )
    ap.add_argument(
        "--rows_per_board", type=int, default=None, help="Pairs-per-configuration override (default: the env cfg's)."
    )
    ap.add_argument(
        "--collision_geom",
        action="store_true",
        help="Render the obstacles' collision approximations instead of their visual meshes.",
    )
    ap.add_argument("--headless", action="store_true", help="Build and print the funnel without the viser server.")
    import sys

    from isaaclab_tasks.utils import setup_preset_cli

    args, remaining = setup_preset_cli(ap)
    if not any(tok.startswith("presets=") for tok in remaining):
        remaining = ["presets=franka,nut_thread_m16"] + remaining
        print("[viz] no presets= given, defaulting to presets=franka,nut_thread_m16")
    sys.argv = [sys.argv[0]] + remaining

    table_cfg = resolve_from_task()
    cfg = table_cfg.pipeline_cfg
    if args.num_boards is not None:
        cfg.board.num_boards = args.num_boards
    rows_per_board = args.rows_per_board if args.rows_per_board is not None else int(table_cfg.rows_per_board)
    pipeline = FactoryIKPipeline(cfg)
    fmodel = pipeline.model
    result = pipeline.build_balanced_table(rows_per_board * cfg.board.num_boards)
    print(f"\n{pipeline.rejection_summary}")

    # Every kept row renders, overlaid in its board-configuration's cell.
    sel = torch.arange(result.joint_q.shape[0], device=cfg.device)
    row_tags = result.tag[sel].cpu().numpy()
    m = sel.shape[0]
    lib_board, lib_bolt = pipeline.board_library
    n_boards = lib_board.shape[0]
    print(f"[viz] rendering {m} solution pairs across {n_boards} board-configuration cells")

    # FK the selected solutions + per-cell clearances (so "is anything colliding?"
    # is answerable on sight).
    body_q_t = fmodel.eval_fk(result.joint_q[sel])  # [m, body_count, 7]
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
    # symmetric gripper<->nut clearance: worst of (gripper probes vs nut mesh) and
    # (nut probes vs gripper colliders)
    from isaaclab_tasks.core.multi_task.factory.retarget import points_vs_body_meshes_min_sd
    from isaaclab_tasks.core.multi_task.factory.retarget.criteria import posed_points

    held_probes_t = torch.tensor(fmodel.held_probes, device=cfg.device)
    nq = result.nut_pose[sel]
    nut_pts = posed_points(held_probes_t, nq)

    fwd = posed_collision_min_sd(body_q_t, grip_bodies, grip_probes, fmodel.held_mesh.id, nq, 0.05, cfg.device)
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
    pad_targets = result.pad_targets[sel].cpu().numpy()
    board_ids = result.board_index[sel].cpu().numpy()

    if args.headless:
        print("\n--headless: done.")
        return

    import viser

    from isaaclab_tasks.core.multi_task.factory.retarget.model import _quat_xyzw_rot as _rot
    from isaaclab_tasks.core.multi_task.factory.retarget.model import load_collider_mesh

    franka_meshes = _mesh_table(fmodel.model)
    # Obstacles render their VISUAL meshes by default (the collision geometry on the
    # table/board assets is a box approximation); --collision_geom shows what the
    # signed-distance queries actually see. The board + bolt are per-row posed.
    if args.collision_geom:
        obstacle_render = dict(fmodel.obstacle_geom)
        board_v, board_f = fmodel.board_verts, fmodel.board_faces
        bolt_v, bolt_f = fmodel.fixed_verts, fmodel.fixed_faces
    else:
        obstacle_render = {}
        for name, (usd_path, spawn_scale, pos, quat) in fmodel.obstacle_spec.items():
            v, f = load_collider_mesh(usd_path, cfg.device, scale=spawn_scale, visual=True)
            obstacle_render[name] = (_rot(quat, v) + pos, f)
        board_v, board_f = load_collider_mesh(fmodel.board_spec[0], cfg.device, scale=fmodel.board_spec[1], visual=True)
        bolt_v, bolt_f = load_collider_mesh(fmodel.fixed_spec[0], cfg.device, scale=fmodel.fixed_spec[1], visual=True)
    lib_board_np = lib_board.cpu().numpy()
    lib_bolt_np = lib_bolt.cpu().numpy()

    server = viser.ViserServer(host="0.0.0.0", port=PORT)
    server.scene.set_up_direction("+z")
    cols = int(np.ceil(np.sqrt(n_boards)))
    spacing = 1.6  # [m] cell pitch (> Franka reach so cells stay separated)

    def add(name, verts, faces, color, world_pos, world_quat_xyzw, offset, opacity=1.0):
        wxyz = np.array([world_quat_xyzw[3], world_quat_xyzw[0], world_quat_xyzw[1], world_quat_xyzw[2]])
        server.scene.add_mesh_simple(
            name, verts, faces, color=color, wxyz=wxyz, position=world_pos + offset, opacity=opacity
        )

    ident = np.array([0.0, 0.0, 0.0, 1.0])
    # one cell per board CONFIGURATION (terrain-cell analogy): the configuration
    # geometry renders once, all its solution pairs overlay inside it.
    for b in range(n_boards):
        off = np.array([(b % cols) * spacing, (b // cols) * spacing, 0.0], dtype=np.float32)
        grp = f"/board_{b}"
        add(f"{grp}/board", board_v, board_f, (0.35, 0.35, 0.4), lib_board_np[b, :3], lib_board_np[b, 3:7], off)
        add(f"{grp}/bolt", bolt_v, bolt_f, (0.5, 0.5, 0.55), lib_bolt_np[b, :3], lib_bolt_np[b, 3:7], off)
        for oname, (ov, of) in obstacle_render.items():
            add(f"{grp}/{oname}", ov, of, (0.6, 0.6, 0.6), np.zeros(3), ident, off, opacity=0.35)
        cell_rows = np.nonzero(board_ids == b)[0]
        for c in cell_rows.tolist():
            tag = int(row_tags[c])
            arm_color = TAG_COLORS[tag % len(TAG_COLORS)]
            row = f"{grp}/row_{c}_{result.tag_names[tag]}"
            for body, v, f, lp, lq in franka_meshes:
                bp, bq = body_q[c, body, :3], body_q[c, body, 3:7]
                add(f"{row}/franka/b{body}", v, f, arm_color, bp + _quat_xyzw_rot(bq, lp), _quat_xyzw_mul(bq, lq), off)
            add(
                f"{row}/nut",
                fmodel.held_verts,
                fmodel.held_faces,
                (1.0, 0.84, 0.0),
                nut_pose[c, :3],
                nut_pose[c, 3:7],
                off,
            )
            # fingertip contact targets: red = +jaw-y pad, green = -jaw-y pad
            for k, col in ((0, (1.0, 0.1, 0.1)), (1, (0.1, 0.9, 0.1))):
                server.scene.add_icosphere(f"{row}/pad{k}", radius=0.004, color=col, position=pad_targets[c, k] + off)
        if cell_rows.size:
            tag_txt = " ".join(
                f"{result.tag_names[t]} x{n}" for t, n in zip(*np.unique(row_tags[cell_rows], return_counts=True))
            )
            worst_static = min(static_sd_mm[name][cell_rows].min() for name in static_sd_mm)
            label = (
                f"b{b} | {cell_rows.size} rows | {tag_txt} | worst sd[mm]: static {worst_static:.0f}"
                f" nut {nut_sd_mm[cell_rows].min():.1f} self {self_sd_mm[cell_rows].min():.0f}"
            )
        else:
            label = f"b{b} | 0 rows (configuration unused by the kept table)"
        server.scene.add_label(
            f"{grp}/label", text=label, position=lib_board_np[b, :3] + off + np.array([0.0, 0.0, 0.35])
        )

    # --- sampler exhibit: the full sampling funnel on the nut, magnified ---
    # surface samples (gray) -> antipodal candidate contacts (blue) -> FPS-retained
    # pairs (segments colored by grasp family, contact points red).
    gs = pipeline.grasp_sampler
    scale = 10.0
    ex_off = np.array([-2.5, 0.0, 0.5], dtype=np.float32)
    surf = gs.surface_points.cpu().numpy() * scale + ex_off
    cand = np.concatenate([gs.candidate_pair_a.cpu().numpy(), gs.candidate_pair_b.cpu().numpy()]) * scale + ex_off
    seg = np.stack([gs.pair_a.cpu().numpy(), gs.pair_b.cpu().numpy()], axis=1) * scale + ex_off
    fam = gs.pair_family.cpu().numpy()
    fam_palette = np.array(
        [[0.9, 0.2, 0.2], [0.2, 0.6, 1.0], [0.2, 0.9, 0.3], [1.0, 0.6, 0.1], [0.8, 0.3, 0.9], [0.2, 0.9, 0.9]],
        dtype=np.float32,
    )
    seg_colors = np.repeat(fam_palette[fam % len(fam_palette)][:, None, :], 2, axis=1)
    server.scene.add_mesh_simple(
        "/sampler/nut",
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
    fam_legend = {gs.family_names[f]: tuple(np.round(fam_palette[f % len(fam_palette)], 2)) for f in np.unique(fam)}
    server.scene.add_label(
        "/sampler/label",
        text=(
            f"sampler funnel ({scale:.0f}x): {surf.shape[0]} surface samples -> "
            f"{gs.candidate_pair_a.shape[0]} antipodal pairs -> {seg.shape[0]} retained"
        ),
        position=ex_off + np.array([0.0, 0.0, 0.35]),
    )

    print(
        f"\n[viz] READY -> http://localhost:{PORT}  ({n_boards} configuration cells, {m} solution pairs overlaid,"
        " Franka tinted by placement tag)",
        flush=True,
    )
    print(
        f"[viz] tags + colors: {list(zip(result.tag_names, ['blue', 'green', 'orange', 'purple', 'cyan']))}", flush=True
    )
    print("[viz] pad targets: red sphere = +jaw-y pad, green sphere = -jaw-y pad", flush=True)
    print(f"[viz] sampler exhibit at x=-2.5 ({scale:.0f}x): gray=surface, blue=candidates, colored segments=retained")
    print(f"[viz] family colors: {fam_legend}", flush=True)
    while True:
        time.sleep(30)


if __name__ == "__main__":
    main()
