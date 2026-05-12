# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Aggregate parity reports from ``reports/<task>.json`` into a punch list.

Run with the IsaacLab venv (no IsaacLab imports required — pure JSON I/O)::

    ./isaaclab.sh -p source/.../utils/parity/aggregate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
REPORTS = _HERE / "reports"


def main() -> None:
    rows = []
    for p in sorted(REPORTS.glob("*.json")):
        d = json.loads(p.read_text())
        if "error" in d:
            rows.append({"task": d["task"], "status": "ERR", "error": d["error"]})
            continue

        L1 = d["L1_placement"]["deltas_mm"]
        L3 = d["L3_sampling"]
        rows.append(
            {
                "task": d["task"],
                "status": "OK",
                "tcp_mm": L1.get("tcp_mm"),
                "goal_mm": L1.get("goal_marker_mm"),
                "keypoint_mm": L1.get("keypoint_to_mw_obj_mm") or L1.get("cube_to_mw_obj_mm"),
                "joint_max": d["L2_joint_state"].get("max_abs_delta", 0.0),
                "isaac_obj_std": float(np.sum(L3["isaac_obj_init"]["std"])),
                "mw_obj_std": float(np.sum(L3["mw_obj_init"]["std"])),
                "obj_mean_mm": L3["mean_delta_obj_mm"],
                "tgt_mean_mm": L3["mean_delta_target_mm"],
                "rew_delta": d["L56_rollout"].get("reward_mean_delta") or float("nan"),
            }
        )

    print(f"# Parity Summary — {len(rows)} tasks\n")
    err = [r for r in rows if r["status"] == "ERR"]
    ok = [r for r in rows if r["status"] == "OK"]
    print(f"errors: {len(err)} / {len(rows)}")
    for r in err:
        print(f"  [ERR] {r['task']}: {r['error']}")

    print()
    print("=" * 100)
    print("TOP OFFENDERS — sorted by metric")
    print("=" * 100)

    def _top(key: str, n: int = 10, descending: bool = True, fmt="{:.1f}", postfix=""):
        sorted_rows = sorted(ok, key=lambda r: (r[key] if r[key] is not None else 0.0), reverse=descending)
        for r in sorted_rows[:n]:
            v = r[key]
            v_str = fmt.format(v) if v is not None else "n/a"
            print(f"  {r['task']:>32s}  {key}={v_str}{postfix}")

    print("\n## L1 Placement: TCP world delta vs MW (top 10) — dominated by Sawyer reset noise")
    _top("tcp_mm", 10, postfix=" mm")

    print("\n## L1 Placement: Goal marker delta (top 10) — TASK_SPECS.goal vs MW _target_pos")
    _top("goal_mm", 10, postfix=" mm")

    print("\n## L1 Placement: Keypoint vs MW obj_init delta (top 10) — manipulandum reset divergence")
    _top("keypoint_mm", 10, postfix=" mm")

    print("\n## L2 Joint state max abs delta (top 5)")
    _top("joint_max", 5, fmt="{:.4f}")

    print("\n## L3 Sampling: IsaacLab obj std (BOTTOM 10) — should be ~0.06 like MW; near-0 = deterministic")
    _top("isaac_obj_std", 10, descending=False, fmt="{:.4f}")

    print("\n## L3 Sampling: obj_init mean delta (top 10)")
    _top("obj_mean_mm", 10, postfix=" mm")

    print("\n## L3 Sampling: target mean delta (top 10)")
    _top("tgt_mean_mm", 10, postfix=" mm")

    print("\n## L6 Reward delta (top 10) — mean |Δr| over the rollout")
    _top("rew_delta", 10, fmt="{:.3f}")

    print()
    print("=" * 100)
    print("CATEGORY HISTOGRAMS (env-count exceeding tolerance)")
    print("=" * 100)
    THRESH = {
        "tcp_mm": 50.0,
        "goal_mm": 30.0,
        "keypoint_mm": 50.0,
        "joint_max": 0.05,
        "obj_mean_mm": 50.0,
        "tgt_mean_mm": 30.0,
    }
    for k, t in THRESH.items():
        bad = sum(1 for r in ok if (r[k] or 0.0) > t)
        print(f"  {k:>16s} > {t}: {bad} / {len(ok)}")

    n_deterministic = sum(1 for r in ok if (r["isaac_obj_std"] or 0.0) < 0.005)
    print(f"  {'isaac_obj_std':>16s} < 0.005 (deterministic): {n_deterministic} / {len(ok)}")

    print()
    print("=" * 100)
    print("RAW TABLE")
    print("=" * 100)
    cols = [
        "task",
        "tcp_mm",
        "goal_mm",
        "keypoint_mm",
        "joint_max",
        "isaac_obj_std",
        "obj_mean_mm",
        "tgt_mean_mm",
        "rew_delta",
    ]
    print("  ".join(f"{c:>16s}" if c != "task" else f"{c:>32s}" for c in cols))
    for r in ok:
        cells = []
        for c in cols:
            v = r[c]
            if c == "task":
                cells.append(f"{v:>32s}")
            elif v is None:
                cells.append(f"{'n/a':>16s}")
            elif c in ("joint_max", "isaac_obj_std", "rew_delta"):
                cells.append(f"{v:>16.4f}")
            else:
                cells.append(f"{v:>16.1f}")
        print("  ".join(cells))


if __name__ == "__main__":
    main()
