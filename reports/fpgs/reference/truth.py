# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ground truth on a captured substep: what do the real kernels converge to?

Runs the legacy row Gauss-Seidel kernel and the parallel projection kernel on the captured state with N iterations
(24 and 400 for the reference study) and reports per-world relative velocity differences between them, so the
study's numpy models can be checked against the kernels and the two methods' fixed points against each other.

usage: truth.py <capture_base> [rows]
"""

import sys

import newton._src.solvers.feather_pgs.solver_feather_pgs as S
import numpy as np
import oracle as O
import warp as wp

base = sys.argv[1]
rows = int(sys.argv[2]) if len(sys.argv) > 2 else 48
wp.init()
dev = "cuda:0"
arrs, meta = O.load(base)
arrays = O.make_arrays(arrs, meta, dev)
W = meta["world_count"]
D = meta["max_world_dofs"]
M_D = meta["dense_max_constraints"]
common = dict(
    has_drive_rows=meta["has_drive_rows"], has_dense_velocity_limit_rows=meta["has_dense_velocity_limit_rows"]
)


def extra_inputs():
    z3 = wp.zeros((1, 1, 1), dtype=wp.float32, device=dev)
    ink = [wp.zeros(4, dtype=wp.int32, device=dev), z3, z3, z3, z3, wp.zeros((W, M_D), dtype=wp.float32, device=dev)]
    i = wp.zeros(1, dtype=wp.int32, device=dev)
    i2 = wp.zeros((1, 1), dtype=wp.int32, device=dev)
    f = wp.zeros(1, dtype=wp.float32, device=dev)
    v3 = wp.zeros(1, dtype=wp.vec3, device=dev)
    tf = wp.zeros(1, dtype=wp.transform, device=dev)
    sv = wp.zeros(1, dtype=wp.spatial_vector, device=dev)
    u = wp.zeros(1, dtype=wp.uint32, device=dev)
    wr = [
        i2,
        i,
        i,
        i,
        i,
        i,
        i,
        i,
        i,
        v3,
        v3,
        v3,
        f,
        f,
        i,
        tf,
        sv,
        sv,
        i,
        v3,
        i,
        u,
        f,
        f,
        i,
        i,
        0,
        0,
        0.0,
        0,
        0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
    ]
    return ink + wr


def run(kern, iterations, parallel):
    a = dict(arrays)
    a["impulses"] = wp.clone(arrays["impulses"])
    a["mf_impulses"] = wp.clone(arrays["mf_impulses"])
    v = wp.clone(arrays["v_out"])
    inputs = O.legacy_inputs(arrs, meta, dev, a)
    inputs[-8] = int(iterations)  # iterations
    if parallel:
        inputs = inputs + extra_inputs()
    wp.launch_tiled(
        kern,
        dim=[int(meta["general_block_count"])],
        inputs=inputs,
        outputs=[v],
        block_dim=getattr(kern, "_fpgs_block_dim", 32),
        device=dev,
    )
    wp.synchronize_device(dev)
    return v.numpy().copy(), a["impulses"].numpy().copy()


legacy = S._get_pgs_solve_mf_gs_kernel(
    M_D,
    meta["mf_max_constraints"],
    D,
    meta["device_arch"],
    fuse_vel_limits=meta["fuse_vel_limits"],
    friction_mode=meta["friction_mode"],
    skip_local_internal_worlds=meta["skip_local_internal_worlds"],
    **common,
)
res = {}
for it in (24, 400):
    res[("leg", it)] = run(legacy, it, False)
for sw in (24, 400):
    par = S._get_pgs_solve_parallel_kernel(
        M_D,
        meta["mf_max_constraints"],
        D,
        meta["device_arch"],
        rows=rows,
        sweeps=sw,
        nesterov=True,
        min_rows=0,
        tol=1e-7,
        matrix_free=True,
        skip_local_internal_worlds=meta["skip_local_internal_worlds"],
        exact_row_sums=True,
    )
    res[("par", sw)] = run(par, sw, True)

md = arrs["constraint_count"][:W].astype(int)
mm = arrs["mf_constraint_count"][:W].astype(int)
owned = np.flatnonzero((md > 0) & (md <= rows) & (mm == 0))
wdi = arrs["world_dof_indices"]
rt = arrs["row_type"] & 0xFF
has_contact = np.array([np.any(rt[w, : md[w]] == int(S.PGS_CONSTRAINT_TYPE_CONTACT)) for w in owned])
owned = owned[has_contact]
print(f"{base}: {owned.size} owned worlds with contacts (rows <= {rows})")


def vel(v, w):
    dofs = wdi[w, :D]
    ok = dofs >= 0
    return v[dofs[ok]].astype(np.float64)


def compare(a, b, label):
    errs = []
    for w in owned:
        va, vb = vel(res[a][0], w), vel(res[b][0], w)
        errs.append(np.linalg.norm(va - vb) / (np.linalg.norm(vb) + 1e-9))
    e = np.array(errs)
    print(f"  {label:34s} rel |dv|: median {np.median(e):.3e}  p90 {np.percentile(e, 90):.3e}  max {e.max():.3e}")


compare(("leg", 24), ("leg", 400), "legacy 24 vs legacy 400")
compare(("par", 24), ("par", 400), "parallel 24 vs parallel 400")
compare(("par", 400), ("leg", 400), "parallel 400 vs legacy 400")
compare(("par", 24), ("leg", 24), "parallel 24 vs legacy 24")
# stash for the numpy study
np.savez(
    base + "_truth.npz",
    owned=owned,
    v_leg400=res[("leg", 400)][0],
    lam_leg400=res[("leg", 400)][1],
    v_par400=res[("par", 400)][0],
    lam_par400=res[("par", 400)][1],
    v_par24=res[("par", 24)][0],
    lam_par24=res[("par", 24)][1],
    v_leg24=res[("leg", 24)][0],
)
print("saved", base + "_truth.npz")
