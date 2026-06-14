# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone validation of the ``factory/retarget`` package.

Driven by the PRODUCTION env cfg (mirroring terrain's validate_spawn_points.py):
resolves ``Isaac-Factory-v0`` with the same preset system the train script uses
and pulls ``commands.reset_state.task_table.pipeline_cfg`` from it, so this
tool validates exactly what training builds. Builds the offline reset-state
table and reports the rejection funnel, per-tag yield, per-grasp-family yield,
and approach-direction diversity. Isaacsim-free (no Kit launch).

Run:
  S=source/isaaclab_tasks/isaaclab_tasks/core/multi_task/factory/scripts/validate_factory_retarget.py
  ./isaaclab.sh -p $S presets=franka,nut_thread_m16
  ./isaaclab.sh -p $S presets=franka,nut_thread_m16 --placements_per_board 8 --rows_per_board 4
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

import isaaclab.utils.math as math_utils

from isaaclab_tasks.core.multi_task.factory.retarget import FactoryIKPipeline
from isaaclab_tasks.core.multi_task.factory.retarget.cfg import resolve_from_task


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate the factory/retarget offline IK pipeline.")
    ap.add_argument(
        "--placements_per_board", type=int, default=None, help="Nut placements per board (default: the env cfg's)."
    )
    ap.add_argument("--grasps_per_placement", type=int, default=8, help="Antipodal pairs per nut placement.")
    ap.add_argument(
        "--collision_objective",
        type=str,
        choices=["on", "off"],
        default=None,
        help="Override the collision-avoidance objectives (default: keep the cfg default).",
    )
    ap.add_argument(
        "--rows_per_board",
        type=int,
        default=None,
        help=(
            "Exercise build_balanced_table at this rows-per-configuration density (total ="
            " rows_per_board x board.num_boards) instead of one build_table pass."
        ),
    )
    import sys

    from isaaclab_tasks.utils import setup_preset_cli

    args, remaining = setup_preset_cli(ap)
    if not any(tok.startswith("presets=") for tok in remaining):
        remaining = ["presets=franka,nut_thread_m16"] + remaining
        print("[retarget] no presets= given, defaulting to presets=franka,nut_thread_m16")
    sys.argv = [sys.argv[0]] + remaining

    cfg = resolve_from_task().pipeline_cfg
    if args.placements_per_board is not None:
        cfg.placement.placements_per_board = args.placements_per_board
    if args.collision_objective is not None:
        from isaaclab_tasks.core.multi_task.factory.retarget import CollisionAvoidanceCfg

        objs = [o for o in cfg.robot.solve.objectives if not isinstance(o, CollisionAvoidanceCfg)]
        cfg.robot.solve.objectives = objs + ([CollisionAvoidanceCfg()] if args.collision_objective == "on" else [])
    pipeline = FactoryIKPipeline(cfg)
    m = pipeline.model
    print(
        f"[retarget] model: {m.body_count} bodies (robot only, ee={m.ee_body}), coords={m.nq}; "
        f"obstacles={list(m.static_obstacles)}; {pipeline.grasp_sampler.pair_a.shape[0]} antipodal pairs; "
        f"tags={pipeline.tag_names}"
    )

    table_size = args.rows_per_board * cfg.board.num_boards if args.rows_per_board is not None else None
    if table_size is not None:
        result = pipeline.build_balanced_table(table_size)
    else:
        result = pipeline.build_table(grasps_per_placement=args.grasps_per_placement)
    print(f"\n{pipeline.rejection_summary}")

    n_attempted = pipeline._n_worlds * pipeline._n_grasps * pipeline._n_seeds
    n_ok = result.joint_q.shape[0]
    print("\n=== per-tag accepted rows + arm-config diversity ===")
    for t, name in enumerate(result.tag_names):
        mask = result.tag == t
        n_tag = int(mask.sum())
        if n_tag == 0:
            print(f"  {name:16s}: (no rows)")
            continue
        arm = result.joint_q[mask][:, 0:7]
        spread = arm.std(dim=0).mean().item() if n_tag >= 2 else 0.0
        print(f"  {name:16s}: {n_tag:5d} accepted  | arm-config std {np.degrees(spread):5.1f} deg")

    print("\n=== per-grasp-family accepted rows ===")
    for f in torch.unique(result.family).tolist():
        n_f = int((result.family == f).sum())
        print(f"  {result.family_names[f]:24s}: {n_f}")

    # approach-direction diversity (EE-z = approach axis) among accepted rows
    body_q = m.eval_fk(result.joint_q)
    ee_quat = body_q[:, m.ee_body, 3:7]
    ee_z = math_utils.quat_apply(ee_quat, torch.tensor([0.0, 0.0, 1.0], device=cfg.device).expand(n_ok, 3))
    down = -ee_z[:, 2]
    print(
        f"\n=== approach diversity (accepted rows) ===\n"
        f"  top-down {int((down > 0.866).sum())}, oblique {int(((down <= 0.866) & (down >= 0.5)).sum())}, "
        f"sideways {int((down.abs() < 0.5).sum())}, from-below {int((down <= -0.5).sum())}"
    )
    ap_mm = result.aperture * 1e3
    print(f"  aperture: min {ap_mm.min():.1f} / median {ap_mm.median():.1f} / max {ap_mm.max():.1f} mm")

    if table_size is not None:
        # balanced mode: the gates are density (boards keep rows_per_board on
        # average; empty boards are genuinely infeasible worlds) and PER-BOARD
        # diversity (each multi-row board must mix state kinds, not cluster in one)
        boards = torch.unique(result.board_index)
        tags_per_board = torch.tensor(
            [torch.unique(result.tag[result.board_index == b]).numel() for b in boards.tolist()]
        ).float()
        multi = tags_per_board[torch.tensor([int((result.board_index == b).sum()) >= 2 for b in boards.tolist()])]
        mean_tags = float(multi.mean()) if multi.numel() else 0.0
        # a board can carry at most rows_per_board distinct tags; demand 75% of that, capped at 2
        tag_gate = min(2.0, 0.75 * args.rows_per_board)
        ok = n_ok >= 0.5 * table_size and torch.unique(result.family).numel() >= 2 and mean_tags >= tag_gate
        print(
            f"\n[retarget] balanced table: requested {table_size} (= {args.rows_per_board}/board), built {n_ok} rows"
            f" on {boards.numel()} boards; avg {mean_tags:.1f} distinct tags per multi-row board"
        )
    else:
        overall = n_ok / n_attempted if n_attempted else 0.0
        # the full board pose/tilt randomization makes many sub-worlds genuinely
        # infeasible (unreachable or board-through-arm); ~10%+ TRUE yield with all
        # tags populated is the healthy regime -- raw yield is bought with num_worlds.
        ok = (
            n_ok > 0
            and overall > 0.08
            and torch.unique(result.tag).numel() >= 2
            and torch.unique(result.family).numel() >= 2
        )
        print(f"\n[retarget] {n_attempted} candidates -> {n_ok} accepted reset states ({overall:.1%})")
    print(
        "\n[retarget] "
        + (
            "PASS: pipeline builds a diverse multi-tag, multi-family reset-state table via batched Newton IK."
            if ok
            else "WARN: low yield, too few tags, or a single grasp family -- inspect the funnel above."
        )
    )
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
