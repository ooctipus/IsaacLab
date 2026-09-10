# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Essential-work model for one FeatherPGS substep versus measured kernel time.

The model counts the state that must cross DRAM/L2 at least once per substep and
the arithmetic a reduced-coordinate PGS step inherently needs. It brackets the
solver's floor between a bandwidth bound (everything streams from DRAM once) and a
latency bound (a fused per-world pipeline of a few dependent launches).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# RTX 5090 (GB202), measured device properties on this machine.
GPU = {
    "sms": 170,
    "dram_gbps": 1792.0,
    "l2_mb": 120.0,
    "fp32_tflops": 104.8,
    "l2_gbps": 6000.0,
    "launch_latency_us": 2.5,
    "min_dep_kernel_us": 4.0,
}
F = 4  # bytes per float32


def per_world_bytes(m: dict, rows: float, mf_rows: float) -> dict[str, float]:
    W = m["worlds"]
    bodies = m["bodies"] / W
    dofs = m["joint_dofs"] / W
    coords = m["joint_coords"] / W
    joints = m["joints"] / W
    m["shapes"] / W
    contacts = max(m.get("contacts_active", 0) / W, rows / 3.0)
    out = {}
    # Generalized + maximal state read once, written once.
    out["joint_state_rw"] = 2 * F * (coords + dofs + dofs)  # q, qd, qdd/tau
    out["control_read"] = F * (dofs * 3)  # targets, gains
    out["body_pose_rw"] = 2 * F * bodies * (7 + 6)  # body_q, body_qd
    out["topology_read"] = F * joints * 8  # parent/child/type/axis indices (mostly L2 resident)
    out["joint_frames_read"] = F * joints * (7 + 7 + 3)  # X_p, X_c, axis
    out["inertia_read"] = F * bodies * (1 + 9 + 3)  # mass, inertia, com
    # Mass matrix factor: only if not kept in registers/shared inside a fused kernel.
    out["mass_factor_rw"] = 2 * F * dofs * (dofs + 1) / 2
    # Constraint rows: J (rows x dofs), rhs/diag/type/mu/impulse per row, contact geometry.
    out["contact_read"] = F * contacts * 16
    out["rows_rw"] = 2 * F * rows * (dofs + 6) + 2 * F * mf_rows * (12 + 6)
    out["response_rw"] = 2 * F * rows * dofs  # H^-1 J^T if materialized
    out["total_streaming"] = sum(v for k, v in out.items())
    out["total_fused_min"] = (
        out["joint_state_rw"]
        + out["control_read"]
        + out["body_pose_rw"]
        + out["contact_read"]
        + 0.25 * (out["topology_read"] + out["joint_frames_read"] + out["inertia_read"])
    )
    return out


def per_world_flops(m: dict, rows: float, mf_rows: float, iters: int) -> dict[str, float]:
    W = m["worlds"]
    bodies = m["bodies"] / W
    dofs = m["joint_dofs"] / W
    joints = m["joints"] / W
    out = {
        "fk": joints * 120,
        "inverse_dynamics": bodies * 250,
        "crba": bodies * 200 + dofs * dofs * 12,
        "cholesky": dofs**3 / 3 + dofs * dofs,
        "trisolve": 2 * dofs * dofs,
        "rows_jacobian": rows * dofs * 12,
        "response_hinv_jt": rows * 2 * dofs * dofs,
        "pgs": iters * (rows * (2 * dofs + 12) + mf_rows * 40),
        "integrate": dofs * 20 + bodies * 60,
    }
    out["total"] = sum(out.values())
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_json", type=Path, help="run_profiled.py output")
    p.add_argument("analysis_json", type=Path, help="analyze_nsys.py --json output")
    p.add_argument("--kernels-per-substep", type=float, default=None)
    args = p.parse_args()
    run = json.loads(args.run_json.read_text())
    ana = json.loads(args.analysis_json.read_text())
    m = run["model"]
    W = m["worlds"]
    rows = m.get("constraint_count", {}).get("mean", 0.0)
    mf_rows = m.get("mf_constraint_count", {}).get("mean", 0.0)
    iters = m.get("pgs_iterations", 8)
    subs = m["num_substeps"] * max(run.get("decimation", 1), 1)
    run.get("decimation", 1)

    stages = ana["stages"]
    graph_us = ana["summary"]["graph_busy_us_per_step"]
    collide_us = stages.get("collide|graph", {}).get("us", 0.0)
    sensors_us = stages.get("sensors|graph", {}).get("us", 0.0)
    solver_us = graph_us - collide_us - sensors_us
    solver_us_per_substep = solver_us / subs
    nodes = ana["summary"]["graph_nodes_per_step"]
    collide_nodes = stages.get("collide|graph", {}).get("n", 0.0)
    solver_nodes_per_substep = (nodes - collide_nodes) / subs

    b = per_world_bytes(m, rows, mf_rows)
    f = per_world_flops(m, rows, mf_rows, iters)
    tot_stream_mb = b["total_streaming"] * W / 1e6
    tot_fused_mb = b["total_fused_min"] * W / 1e6
    gflop = f["total"] * W / 1e9
    t_dram_stream = tot_stream_mb / GPU["dram_gbps"] * 1e3  # us
    t_dram_fused = tot_fused_mb / GPU["dram_gbps"] * 1e3
    t_l2_stream = tot_stream_mb / GPU["l2_gbps"] * 1e3
    t_flops = gflop / GPU["fp32_tflops"] * 1e3
    t_latency_now = solver_nodes_per_substep * GPU["min_dep_kernel_us"]
    fused_kernels = 6
    t_latency_fused = fused_kernels * GPU["min_dep_kernel_us"]

    print(f"Task {run['task']}  physics {run['physics']}  worlds {W}  substeps/env-step {subs}")
    print(
        f"  per-world: bodies {m['bodies'] / W:.1f} dofs {m['joint_dofs'] / W:.1f} shapes {m['shapes'] / W:.1f} "
        f"dense rows {rows:.1f} (max {m.get('constraint_count', {}).get('max', 0)}) "
        f"mf rows {mf_rows:.2f} pgs iters {iters}"
    )
    print(
        f"  measured graph busy {graph_us:.0f} us/step: collide {collide_us:.0f}, sensors {sensors_us:.0f}, "
        f"solver {solver_us:.0f} -> {solver_us_per_substep:.0f} us/substep over {solver_nodes_per_substep:.0f} launches"
    )
    print("  essential per-world bytes per substep:")
    for k, v in b.items():
        print(f"    {k:22s} {v:9.0f} B   x{W} = {v * W / 1e6:8.1f} MB")
    print(f"  essential flops per world per substep: {f['total']:.0f}  -> {gflop:.3f} GFLOP total")
    for k, v in f.items():
        if k != "total":
            print(f"    {k:22s} {v:9.0f}")
    print("  floors per substep (us):")
    print(f"    DRAM, everything streamed once   {t_dram_stream:8.1f}")
    print(f"    DRAM, fused (state only)          {t_dram_fused:8.1f}")
    print(f"    L2-resident streaming             {t_l2_stream:8.1f}")
    print(f"    FP32 compute                      {t_flops:8.1f}")
    print(
        f"    launch/dependency, today's chain  {t_latency_now:8.1f}  "
        f"({solver_nodes_per_substep:.0f} x {GPU['min_dep_kernel_us']} us)"
    )
    print(f"    launch/dependency, {fused_kernels}-kernel fused  {t_latency_fused:8.1f}")
    floor_fused = max(t_dram_fused, t_flops, t_latency_fused)
    floor_stream = max(t_dram_stream, t_flops) + t_latency_fused
    print(
        f"  => northstar floor ~{floor_fused:.0f} us/substep (fused, L2-resident), "
        f"practical target ~{floor_stream:.0f} us; today {solver_us_per_substep:.0f} us  "
        f"=> headroom {solver_us_per_substep / floor_stream:.1f}x .. {solver_us_per_substep / floor_fused:.1f}x"
    )
    out_path = Path(str(args.analysis_json).replace("_analysis.json", "_roofline.json"))
    out_path.write_text(
        json.dumps(
            {
                "bytes": b,
                "flops": f,
                "floors_us": {
                    "dram_stream": t_dram_stream,
                    "dram_fused": t_dram_fused,
                    "l2_stream": t_l2_stream,
                    "flops": t_flops,
                    "latency_now": t_latency_now,
                    "latency_fused": t_latency_fused,
                },
                "measured_solver_us_per_substep": solver_us_per_substep,
                "solver_launches_per_substep": solver_nodes_per_substep,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
