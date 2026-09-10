# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare solver-only issued instructions per world per substep between two analyze_ncu.py JSON exports.

usage: inst_compare.py fpgs_ncu.json mjwarp_ncu.json [mjwarp_substeps_covered]
"""

import json
import re
import sys
from pathlib import Path

col = re.compile(
    r"aabb|narrow_phase|broadphase|contact_forces|contact_linear|pack_contact|sensing|sensor|timestamp|verify_narrow|_nxn|convert_newton_contacts|ccd_kernel|primitive_narrowphase|geom_local|mesh_triangle|reduce_buffered|export_reduced|_clear_active",
    re.I,
)


def tot(path, subs, worlds):
    d = json.loads(Path(path).read_text())
    inst = sum(r["inst"] * r["n"] for r in d if r["inst"] == r["inst"] and not col.search(r["kernel"]))
    fl = sum(r["ffma"] * r["n"] for r in d if r["ffma"] == r["ffma"] and not col.search(r["kernel"]))
    us = sum(r["total_us"] for r in d if not col.search(r["kernel"]))
    n = sum(r["n"] for r in d if not col.search(r["kernel"]))
    return inst / subs / worlds, fl / subs / worlds, us / subs, n / subs


SUBS = {"FPGS": 8, "MJWarp": int(sys.argv[3]) if len(sys.argv) > 3 else 8}
for name, path in (("FPGS", sys.argv[1]), ("MJWarp", sys.argv[2])):
    i, f, us, n = tot(path, SUBS[name], 16384)
    print(
        f"{name:7s} solver-only per world per substep: {i:8.0f} warp-inst  {f:8.0f} fp32 thread-ops | "
        f"{us:7.0f} us/substep (base clk) over {n:5.1f} launches"
    )
