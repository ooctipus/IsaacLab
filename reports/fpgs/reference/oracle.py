# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline oracle for the FeatherPGS matrix-free GS sweep.

Loads a capture written by FEATHER_PGS_CAPTURE_KERNEL=1, runs the legacy kernel as the
reference, optionally the new response-block path, and compares outputs bitwise / by tolerance.

usage: uv run python oracle.py <capture_base> [--new] [--reps N] [--stats]
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import newton._src.solvers.feather_pgs.solver_feather_pgs as S
import numpy as np
import warp as wp


def load(base: str):
    with np.load(base + ".npz") as d, open(base + ".json") as metadata_file:
        meta = json.load(metadata_file)
        return {k: d[k] for k in d.files}, meta


def to_wp(a: np.ndarray, device):
    if a.dtype == np.int64:
        a = a.astype(np.int32)
    return wp.array(a, device=device, copy=True)


def legacy_inputs(arrs, meta, device, arrays: dict):
    """Build the mf_gs input list in launch order from (possibly modified) wp arrays."""
    return [
        arrays["local_general_world_count"],
        arrays["local_general_worlds"],
        int(meta["general_block_count"]),
        int(meta["use_general_world_queue"]),
        arrays["rb_class_zero"],
        arrays["constraint_count"],
        arrays["dense_phase_bounds"],
        arrays["local_solve_owner"],
        arrays["world_dof_indices"],
        arrays["world_deferred_dof_mask"],
        arrays["dense_rhs"],
        arrays["diag"],
        arrays["row_w"],
        arrays["impulses"],
        arrays["J_world"],
        arrays["Y_world"],
        arrays["row_type"],
        arrays["row_parent"],
        arrays["row_mu"],
        arrays["drive_target_vel_bias"],
        arrays["drive_vel_multiplier"],
        arrays["drive_impulse_multiplier"],
        arrays["drive_max_impulse"],
        arrays["drive_vel_limit"],
        arrays["mf_constraint_count"],
        arrays["mf_contact_rows_end"],
        arrays["mf_meta"],
        arrays["mf_impulses"],
        arrays["mf_J_a"],
        arrays["mf_J_b"],
        arrays["mf_MiJt_a"],
        arrays["mf_MiJt_b"],
        arrays["mf_row_mu"],
        arrays["mf_row_w"],
        int(meta["iterations"]),
        float(meta["omega"]),
        int(meta["regularize"]),
        int(meta["row_phase"]),
        int(meta["friction_start_iteration"]),
        int(meta["iteration_offset"]),
        int(meta["freeze_drive_rows"]),
        int(meta["defer_dense_response"]),
    ]


def make_arrays(arrs, meta, device):
    out = {}
    for k, v in arrs.items():
        out[k] = to_wp(v, device)
    if "drive_vel_limit" not in out:
        out["drive_vel_limit"] = wp.zeros((1, 1), dtype=wp.float32, device=device)
    out["rb_class_zero"] = wp.zeros(int(meta["world_count"]), dtype=wp.int32, device=device)
    return out


def stats(arrs, meta):
    W = meta["world_count"]
    md = arrs["constraint_count"][:W].astype(int)
    mm = arrs["mf_constraint_count"][:W].astype(int)
    mt = md + mm
    print(
        f"worlds {W}  D={meta['max_world_dofs']}  M_D={meta['dense_max_constraints']}"
        f"  M_MF={meta['mf_max_constraints']}"
    )
    print(
        f"  phase={meta['row_phase']} iters={meta['iterations']} omega={meta['omega']} reg={meta['regularize']} "
        f"fric_start={meta['friction_start_iteration']} defer={meta['defer_dense_response']} "
        f"general_queue={meta['use_general_world_queue']} drive={meta['has_drive_rows']} "
        f"dvlim={meta['has_dense_velocity_limit_rows']} fvl={meta['fuse_vel_limits']}"
    )
    for name, a in (("m_dense", md), ("m_mf", mm), ("m_total", mt)):
        q = np.percentile(a, [0, 10, 50, 90, 99, 100])
        print(f"  {name:8s} mean {a.mean():6.1f}  p0/10/50/90/99/100 = {q.astype(int).tolist()}")
    for b in (16, 24, 32, 48, 64):
        print(f"  worlds with m_total <= {b:2d}: {100.0 * np.mean(mt <= b):5.1f}%")
    rt = arrs["row_type"]
    types = {}
    for w in range(W):
        for t in rt[w, : md[w]]:
            types[int(t)] = types.get(int(t), 0) + 1
    print("  dense row types:", dict(sorted(types.items())))
    mfm = arrs["mf_meta"].reshape(W, -1, 4)
    mtypes = {}
    for w in range(W):
        for i in range(mm[w]):
            t = int(mfm[w, i, 3]) & 0xFFFF
            mtypes[t] = mtypes.get(t, 0) + 1
    print("  mf row types:", dict(sorted(mtypes.items())))
    owner = arrs["local_solve_owner"][:W]
    print("  local_solve_owner histogram:", {int(k): int(v) for k, v in zip(*np.unique(owner, return_counts=True))})
    diag = arrs["diag"]
    nz = sum(int(np.sum(diag[w, : md[w]] <= 0)) for w in range(W))
    print(f"  dense rows with diag<=0: {nz}")
    # symmetry of the response block on a sample of worlds
    D = meta["max_world_dofs"]
    J = arrs["J_world"].reshape(W, -1, D)
    Y = arrs["Y_world"].reshape(W, -1, D)
    worst = 0.0
    for w in np.random.default_rng(0).choice(W, size=min(64, W), replace=False):
        m = md[w]
        if m < 2:
            continue
        A = J[w, :m] @ Y[w, :m].T
        asym = np.abs(A - A.T).max()
        scale = np.abs(A).max() + 1e-30
        worst = max(worst, asym / scale)
    print(f"  dense response block relative asymmetry (64 worlds): {worst:.3e}")


def run_legacy(arrays, arrs, meta, device, reps=1):
    dev_arch = meta["device_arch"]
    shared_metadata = S._use_resident_mfgs_metadata(
        meta["dense_max_constraints"],
        meta["mf_max_constraints"],
        meta["max_world_dofs"],
        int(getattr(wp.get_device(device), "max_shared_memory_per_block", 0)),
        has_drive_rows=meta["has_drive_rows"],
        fuse_vel_limits=meta["fuse_vel_limits"],
    )
    kern = S._get_pgs_solve_mf_gs_kernel(
        meta["dense_max_constraints"],
        meta["mf_max_constraints"],
        meta["max_world_dofs"],
        dev_arch,
        has_drive_rows=meta["has_drive_rows"],
        has_dense_velocity_limit_rows=meta["has_dense_velocity_limit_rows"],
        fuse_vel_limits=meta["fuse_vel_limits"],
        friction_mode=meta["friction_mode"],
        shared_metadata=shared_metadata,
        skip_local_internal_worlds=meta["skip_local_internal_worlds"],
    )
    # RMW arrays: fresh copies per run
    v_out = wp.clone(arrays["v_out"])
    arrays = dict(arrays)
    arrays["impulses"] = wp.clone(arrays["impulses"])
    arrays["mf_impulses"] = wp.clone(arrays["mf_impulses"])
    inputs = legacy_inputs(arrs, meta, device, arrays)
    dim = [int(meta["general_block_count"])]
    wp.launch_tiled(kern, dim=dim, inputs=inputs, outputs=[v_out], block_dim=32, device=device)
    wp.synchronize_device(device)
    res = (v_out.numpy().copy(), arrays["impulses"].numpy().copy(), arrays["mf_impulses"].numpy().copy())
    if reps > 1:
        import newpath

        pristine = (
            wp.array(arrs["v_out"], device=device),
            wp.array(arrs["impulses"], device=device),
            wp.array(arrs["mf_impulses"], device=device),
        )
        work = tuple(wp.clone(x) for x in pristine)
        arrays["impulses"], arrays["mf_impulses"] = work[1], work[2]
        inputs = legacy_inputs(arrs, meta, device, arrays)

        def copies():
            for dst, src in zip(work, pristine):
                wp.copy(dst, src)

        def full():
            copies()
            wp.launch_tiled(kern, dim=dim, inputs=inputs, outputs=[work[0]], block_dim=32, device=device)

        t_c = newpath.graph_time(copies, device, reps)
        t_f = newpath.graph_time(full, device, reps)
        print(f"  legacy GPU us (graph replay, medians of {reps}): kernel {t_f - t_c:.1f} (copies {t_c:.1f})")
    return res


def compare(name, ref, new, arrs, meta, cls=None):
    W = meta["world_count"]
    v_r, l_r, m_r = ref
    v_n, l_n, m_n = new
    dv = np.abs(v_r - v_n)
    dl = np.abs(l_r - l_n)
    dm = np.abs(m_r - m_n)
    print(
        f"  [{name}] max|dv| {dv.max():.3e} (bitwise equal: {np.array_equal(v_r, v_n)})  "
        f"max|dlam| {dl.max():.3e}  max|dmf| {dm.max():.3e}  |v| max {np.abs(v_r).max():.3e}"
        f"  |lam| max {np.abs(l_r).max():.3e}"
    )
    if dv.max() > 0:
        # per-world worst
        wdi = arrs["world_dof_indices"]
        D = meta["max_world_dofs"]
        worst = []
        for w in range(W):
            idx = wdi[w, :D]
            idx = idx[idx >= 0]
            if idx.size:
                worst.append((float(dv[idx].max()), w))
        worst.sort(reverse=True)
        print(
            "   worst worlds (dv, world, class, m_dense):",
            [
                (round(d, 6), w, int(cls[w]) if cls is not None else -1, int(arrs["constraint_count"][w]))
                for d, w in worst[:6]
            ],
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--new", action="store_true", help="also run the response-block path")
    ap.add_argument("--new-rows", type=int, default=64)
    ap.add_argument("--lane-rows", type=int, default=16)
    args = ap.parse_args()
    wp.init()
    device = args.device
    arrs, meta = load(args.base)
    if args.stats:
        stats(arrs, meta)
    arrays = make_arrays(arrs, meta, device)
    t0 = time.time()
    ref = run_legacy(arrays, arrs, meta, device, reps=args.reps)
    print(f"  legacy done in {time.time() - t0:.1f}s (incl. compile)")
    if args.new:
        import newpath  # Local module for the separate response-block experiment.

        new = newpath.run_new(arrays, arrs, meta, device, reps=args.reps, rows=args.new_rows, lane_rows=args.lane_rows)
        v_n, imp_n, mfimp_n, rb_class_np, dev_arrays = new
        # Worlds the build kernel left to the legacy kernel: run it with the class skip on the same buffers.
        v_d, imp_d, mfimp_d, rb_class_d = dev_arrays
        legacy_skip = S._get_pgs_solve_mf_gs_kernel(
            meta["dense_max_constraints"],
            meta["mf_max_constraints"],
            meta["max_world_dofs"],
            meta["device_arch"],
            has_drive_rows=meta["has_drive_rows"],
            has_dense_velocity_limit_rows=meta["has_dense_velocity_limit_rows"],
            fuse_vel_limits=meta["fuse_vel_limits"],
            friction_mode=meta["friction_mode"],
            skip_local_internal_worlds=meta["skip_local_internal_worlds"],
            skip_response_block=True,
        )
        arr2 = dict(arrays)
        arr2["rb_class_zero"] = rb_class_d
        arr2["impulses"] = imp_d
        arr2["mf_impulses"] = mfimp_d
        wp.launch_tiled(
            legacy_skip,
            dim=[int(meta["general_block_count"])],
            inputs=legacy_inputs(arrs, meta, device, arr2),
            outputs=[v_d],
            block_dim=32,
            device=device,
        )
        wp.synchronize_device(device)
        new3 = (v_d.numpy().copy(), imp_d.numpy().copy(), mfimp_d.numpy().copy())
        print(f"  legacy-owned worlds (class 0): {int((rb_class_np == 0).sum())}")
        compare("new+legacy vs legacy", ref, new3, arrs, meta, rb_class_np)
    np.savez(args.base + "_ref.npz", v_out=ref[0], impulses=ref[1], mf_impulses=ref[2])


if __name__ == "__main__":
    sys.exit(main())
