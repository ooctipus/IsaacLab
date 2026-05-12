# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Extract per-task (obj_init, target) sampling ranges from MW dump.

Produces ``mw_ranges.json`` with min/max for each task — useful as a
ground-truth source when populating ``TASK_SPECS`` ranges. Also prints the
results so we can hand-paste the per-task numbers into a TASK_SPECS update.

Run with the IsaacLab venv (no IsaacLab imports needed)::

    ./isaaclab.sh -p source/.../utils/parity/extract_mw_ranges.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
DATA = _HERE / "data"
OUT = _HERE / "mw_ranges.json"


def main() -> None:
    out = {}
    for p in sorted(DATA.glob("*.json")):
        if p.name.startswith("_"):
            continue
        d = json.loads(p.read_text())
        samples = d.get("samples", [])
        if not samples:
            continue
        obj = np.array([s["obj_init_pos"] for s in samples])
        tgt = np.array([s["target_pos"] for s in samples])
        out[d["meta"]["task"]] = {
            "obj_init_min": obj.min(axis=0).tolist(),
            "obj_init_max": obj.max(axis=0).tolist(),
            "obj_init_mean": obj.mean(axis=0).tolist(),
            "obj_init_std": obj.std(axis=0).tolist(),
            "target_min": tgt.min(axis=0).tolist(),
            "target_max": tgt.max(axis=0).tolist(),
            "target_mean": tgt.mean(axis=0).tolist(),
            "target_std": tgt.std(axis=0).tolist(),
        }

    OUT.write_text(json.dumps(out, indent=2))
    print(f"[ranges] wrote {OUT} ({len(out)} tasks)")
    for task, r in sorted(out.items()):
        print(f"  {task}")
        print(
            f"    obj_init: mean={[round(v, 3) for v in r['obj_init_mean']]}"
            f"  std={[round(v, 3) for v in r['obj_init_std']]}"
            f"  range=[{[round(v, 3) for v in r['obj_init_min']]}..{[round(v, 3) for v in r['obj_init_max']]}]"
        )
        print(
            f"    target  : mean={[round(v, 3) for v in r['target_mean']]}"
            f"  std={[round(v, 3) for v in r['target_std']]}"
            f"  range=[{[round(v, 3) for v in r['target_min']]}..{[round(v, 3) for v in r['target_max']]}]"
        )


if __name__ == "__main__":
    main()
