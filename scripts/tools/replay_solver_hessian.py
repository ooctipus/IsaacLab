# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reconstruct the constraint-solver Hessian from a recorded pre-Hessian state and replay its NaN.

Given a solver snapshot captured by the ``FACTORY_PRESOLVE_DUMP`` / ``FACTORY_SOLVE_SNAPSHOT`` hooks
in ``mujoco_warp``'s ``solver.py``, this tool rebuilds the constraint Hessian

    H = qM + JTDAJ + JTCJ

*entirely from the recorded pre-Hessian inputs* (mass matrix, constraint rows, the failing-iteration
cone state and iterate) -- it never loads the captured Hessian to build ``H`` -- then factorizes it
with an unpivoted Cholesky that mirrors the solver's tile kernel. An indefinite ``H`` makes the
factorization take ``sqrt`` of a negative pivot and produce NaN, reproducing the solver failure from
the recording rather than merely re-factorizing the captured bad matrix.

The terms:

* ``qM``    -- joint-space mass matrix, densified from the recorded sparse ``d_qM`` using mujoco_warp's
  own sparse structure (``qM_mulm_*``); this carries the runtime values (e.g. large kinematic-body
  armature) that a freshly compiled model would not.
* ``JTDAJ`` -- normal/limit term ``Jᵀ·diag(state_check(D, state))·J`` over rows in the QUADRATIC zone;
  this is what the pyramidal cone builds, and it is positive semi-definite.
* ``JTCJ``  -- elliptic friction-cone curvature, a faithful port of ``update_gradient_h`` (elliptic),
  evaluated at the recorded iterate (``Jaref``) and cone state. This term can be indefinite and is the
  sole source of indefiniteness on a near-singular held body.

Usage::

    # structure recorded in the dump (newer captures):
    ./isaaclab.sh -p scripts/tools/replay_solver_hessian.py <dump.npz>

    # structure from a sidecar (older captures); generate it once from the saved model:
    ./isaaclab.sh -p scripts/tools/replay_solver_hessian.py <dump.npz> --qm_struct qM_struct.npz
    ./isaaclab.sh -p scripts/tools/replay_solver_hessian.py --save_qm_struct qM_struct.npz --mjb model.mjb
"""

from __future__ import annotations

import argparse

import numpy as np

try:
    from mujoco_warp._src import types as _mjt

    CONE = int(_mjt.ConstraintState.CONE.value)
    QUADRATIC = int(_mjt.ConstraintState.QUADRATIC.value)
    MJ_MINVAL = float(_mjt.MJ_MINVAL)
except Exception:  # noqa: BLE001 -- tool may run without mujoco_warp importable
    CONE, QUADRATIC, MJ_MINVAL = 4, 1, 1e-15


# --------------------------------------------------------------------------------------------------
# dump access
# --------------------------------------------------------------------------------------------------
def pick(dump, *keys, required=True):
    """Return the first present key's array (handles snapshot vs raw-efc naming)."""
    for k in keys:
        if k in dump.files:
            return np.asarray(dump[k])
    if required:
        raise KeyError(f"none of {keys} in dump (have: {sorted(dump.files)[:40]} ...)")
    return None


def scalar(dump, *keys, default=None):
    """Return the first present key as a python int."""
    a = pick(dump, *keys, required=False)
    return int(np.asarray(a).reshape(-1)[0]) if a is not None else default


# --------------------------------------------------------------------------------------------------
# linear-algebra helpers
# --------------------------------------------------------------------------------------------------
def mirror_lower(H):
    """Mirror a lower-triangular matrix to full symmetric (solver stores ctx.h lower-triangular)."""
    return np.tril(H) + np.tril(H, -1).T


def unpivoted_cholesky_nan(A):
    """Count NaNs from an unpivoted float32 Cholesky reading only the lower triangle + diagonal.

    This mirrors mujoco_warp's tile Cholesky: an indefinite matrix yields ``sqrt`` of a negative pivot
    and propagates NaN, exactly as the kernel does.
    """
    A = A.astype(np.float32).copy()
    n = A.shape[0]
    L = np.zeros_like(A)
    for i in range(n):
        s = A[i, i] - (L[i, :i] * L[i, :i]).sum()
        L[i, i] = np.sqrt(s)
        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - (L[j, :i] * L[i, :i]).sum()) / L[i, i]
    return int(np.isnan(L).sum())


def min_eig(A):
    """Smallest eigenvalue of the symmetric part of ``A``."""
    return float(np.linalg.eigvalsh(0.5 * (A + A.T)).min())


# --------------------------------------------------------------------------------------------------
# mass matrix
# --------------------------------------------------------------------------------------------------
def densify_mass_matrix(qM_sparse, rowadr, col, madr, nv):
    """Densify mujoco_warp's sparse mass matrix using its own ``qM_mulm_*`` structure.

    ``mul_m_sparse`` computes ``res[dof] = Σ_k qM[madr[k]] · vec[col[k]]`` over
    ``k ∈ [rowadr[dof], rowadr[dof+1])``, hence ``M[dof, col[k]] = qM[madr[k]]``.
    """
    M = np.zeros((nv, nv))
    for dof in range(nv):
        for k in range(int(rowadr[dof]), int(rowadr[dof + 1])):
            M[dof, int(col[k])] = qM_sparse[int(madr[k])]
    return M


def load_qm_structure(dump, qm_struct_path):
    """Return ``(rowadr, col, madr)`` from the dump, else from a sidecar npz."""
    if "m_qM_mulm_rowadr" in dump.files:
        g = lambda nm: pick(dump, "m_" + nm).reshape(-1).astype(int)  # noqa: E731
        return g("qM_mulm_rowadr"), g("qM_mulm_col"), g("qM_mulm_madr")
    if qm_struct_path is None:
        raise SystemExit(
            "Mass-matrix structure not in dump and --qm_struct not given. Generate it once with\n"
            "  --save_qm_struct qM_struct.npz --mjb <model.mjb>"
        )
    s = np.load(qm_struct_path)
    return (
        s["qM_mulm_rowadr"].reshape(-1).astype(int),
        s["qM_mulm_col"].reshape(-1).astype(int),
        s["qM_mulm_madr"].reshape(-1).astype(int),
    )


def save_qm_structure(mjb_path, out_path):
    """Build the model from a saved binary and write its ``qM_mulm_*`` sparse structure to npz."""
    import mujoco
    import mujoco_warp
    import warp as wp

    mjm = mujoco.MjModel.from_binary_path(mjb_path)
    with wp.ScopedDevice("cuda:0"):
        m = mujoco_warp.put_model(mjm)
    g = lambda nm: wp.to_torch(getattr(m, nm)).detach().cpu().numpy()  # noqa: E731
    np.savez(out_path, qM_mulm_rowadr=g("qM_mulm_rowadr"), qM_mulm_col=g("qM_mulm_col"), qM_mulm_madr=g("qM_mulm_madr"))
    print(f"wrote {out_path} (nv={m.nv}, is_sparse={m.is_sparse})")


# --------------------------------------------------------------------------------------------------
# constraint Hessian terms
# --------------------------------------------------------------------------------------------------
def dense_jacobian(dump, nefc, nv):
    """Reconstruct the dense ``(nefc, nv)`` constraint Jacobian from the sparse efc rows."""
    J = pick(dump, "efc_J").reshape(-1)
    rownnz = pick(dump, "efc_J_rownnz").reshape(-1)
    rowadr = pick(dump, "efc_J_rowadr").reshape(-1)
    colind = pick(dump, "efc_J_colind").reshape(-1)
    Jd = np.zeros((nefc, nv))
    for i in range(nefc):
        a, n = int(rowadr[i]), int(rownnz[i])
        for k in range(n):
            c = int(colind[a + k])
            if 0 <= c < nv:
                Jd[i, c] = float(J[a + k])
    return Jd


def normal_hessian(Jd, D, state, nefc, nv):
    """``JTDAJ`` over QUADRATIC-zone rows (state_check masks D to zero elsewhere)."""
    Deff = np.where(state[:nefc] == QUADRATIC, D[:nefc], 0.0)
    H = np.zeros((nv, nv))
    for i in range(nefc):
        H += Deff[i] * np.outer(Jd[i], Jd[i])
    return H


def cone_hessian(dump, Jd, D, state, jaref, nv, impratio):
    """Elliptic friction-cone curvature ``JTCJ`` -- faithful port of ``update_gradient_h`` (elliptic).

    Returns ``(H, n_contacts)``. Handles arbitrary ``condim`` and any number of active cone contacts.
    """
    cdim = pick(dump, "contact_dim").reshape(-1).astype(int)
    cfric = pick(dump, "contact_friction").astype(np.float64)
    cdist = pick(dump, "contact_dist").reshape(-1).astype(np.float64)
    cmarg = pick(dump, "contact_includemargin").reshape(-1).astype(np.float64)
    cefc = pick(dump, "contact_efc_address").astype(int)
    invsqrt = 1.0 / np.sqrt(impratio)
    H = np.zeros((nv, nv))
    n_contacts = 0
    for cid in range(cdim.shape[0]):
        condim = int(cdim[cid])
        if condim == 1 or (cdist[cid] - cmarg[cid]) >= 0.0:  # frictionless or inactive
            continue
        efcid0 = int(cefc[cid, 0])
        if efcid0 < 0 or state[efcid0] != CONE:
            continue
        fri = cfric[cid]
        mu = fri[0] * invsqrt
        mu2 = mu * mu
        dm = D[efcid0] / (mu2 * (1.0 + mu2)) if mu2 * (1.0 + mu2) != 0.0 else 0.0
        if dm == 0.0:
            continue
        n_contacts += 1
        nrm = jaref[efcid0] * mu
        u = np.array([nrm] + [jaref[int(cefc[cid, j])] * fri[j - 1] for j in range(1, condim)])
        tt = float(np.sum(u[1:] ** 2))
        t = max(np.sqrt(tt) if tt > 0 else 0.0, MJ_MINVAL)
        ttt = max(t**3, MJ_MINVAL)
        fv = np.array([mu] + [fri[k - 1] for k in range(1, condim)])
        efcids = [efcid0] + [int(cefc[cid, k]) for k in range(1, condim)]
        Hc = np.zeros((condim, condim))
        for a in range(condim):
            for b in range(a + 1):
                if a == 0 and b == 0:
                    hc = 1.0
                elif b == 0:
                    hc = -(mu / t) * u[a]
                else:
                    hc = mu * (nrm / ttt) * u[a] * u[b]
                    if a == b:
                        hc += mu2 - mu * (nrm / t)
                hc *= dm * fv[a] * fv[b]
                Hc[a, b] = hc
                Hc[b, a] = hc
        Jsub = Jd[efcids, :]
        H += Jsub.T @ Hc @ Jsub
    return H, n_contacts


# --------------------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------------------
def run(dump, qm_struct_path, reg_eps):
    nefc = scalar(dump, "d_nefc", "mjw_nefc")
    cone = scalar(dump, "opt_cone", default=-1)
    rowadr, col, madr = load_qm_structure(dump, qm_struct_path)
    nv = int(rowadr.shape[0] - 1)

    # ---- world consistency (the snapshot world must match the per-world arrays) ----
    snap_world = scalar(dump, "_snap_world", default=-1)
    qacc_world = scalar(dump, "_qacc_nan_world", default=-1)
    nan_world = scalar(dump, "_nan_world", default=-1)
    meta = pick(dump, "snap_meta", required=False)
    print(
        f"== capture ==  cone={'elliptic' if cone == 1 else 'pyramidal' if cone == 0 else cone}  nv={nv}  nefc={nefc}"
    )
    if meta is not None:
        m = np.asarray(meta).reshape(-1)
        print(f"   snapshot: phase={m[1]} niter={m[2]} world={m[3]} (col={m[4]})")
    print(
        f"   world: dumped={nan_world} snapshot={snap_world} post_solve_qacc_nan={qacc_world}"
        f"  -> {'ALIGNED' if snap_world in (-1, nan_world) else 'MIXED (suspect!)'}"
    )

    # ---- terms (all from recorded pre-Hessian inputs) ----
    qM = densify_mass_matrix(pick(dump, "d_qM").reshape(-1).astype(np.float64), rowadr, col, madr, nv)
    Jd = dense_jacobian(dump, nefc, nv)
    state = pick(dump, "snap_efc_state", "efc_state").reshape(-1).astype(int)
    D = pick(dump, "snap_efc_D", "efc_D").reshape(-1).astype(np.float64)
    impratio = float(pick(dump, "opt_impratio", required=False).reshape(-1)[0]) if "opt_impratio" in dump.files else 1.0

    JTDAJ = normal_hessian(Jd, D, state, nefc, nv)
    jaref = pick(dump, "snap_jaref", required=False)
    if jaref is not None and cone == 1:
        jaref = jaref.reshape(-1).astype(np.float64)
        JTCJ, ncone = cone_hessian(dump, Jd, D, state, jaref, nv, impratio)
        have_cone = True
    else:
        JTCJ, ncone, have_cone = np.zeros((nv, nv)), 0, False
        if cone == 1:
            print(
                "   NOTE: elliptic cone, but snap_jaref is not in this capture -> JTCJ skipped; "
                "H_derived = qM+JTDAJ only. Re-capture with the iterate export to reconstruct the\n"
                "         cone term and reproduce the NaN from inputs."
            )

    H_derived = qM + JTDAJ + JTCJ
    H_cap = (
        mirror_lower(pick(dump, "snap_h", "ctx_h").astype(np.float64)[:nv, :nv])
        if ("snap_h" in dump.files or "ctx_h" in dump.files)
        else None
    )

    # ---- spectrum ----
    print("== spectrum ==")
    print(f"   qM                 min_eig={min_eig(qM):+.4e}")
    print(f"   qM+JTDAJ (pyramid) min_eig={min_eig(qM + JTDAJ):+.4e}")
    if have_cone:
        print(f"   JTCJ (elliptic)    min_eig={min_eig(JTCJ):+.4e}  (#cone contacts={ncone})")
        print(f"   H = qM+JTDAJ+JTCJ  min_eig={min_eig(H_derived):+.4e}")
        ev, V = np.linalg.eigh(0.5 * (H_derived + H_derived.T))
        print(f"   H negative mode dominant DOFs: {np.argsort(-np.abs(V[:, 0]))[:6].tolist()}")

    # ---- cross-check against the captured Hessian (not used to build H_derived) ----
    if H_cap is not None:
        diff = np.abs(H_derived - H_cap)
        # high-armature (kinematic) DOFs are near-rigid; the kernel omits their cone self-block.
        kin = np.where(np.diag(qM) > 1e6)[0]
        free = np.setdiff1d(np.arange(nv), kin)
        print("== reconstruction vs captured snap_h ==")
        print(f"   max|H_derived - snap_h| overall        = {diff.max():.3e}")
        if free.size:
            print(f"   max| ... | on non-kinematic block       = {diff[np.ix_(free, free)].max():.3e}")
        print(f"   min_eig(snap_h) = {min_eig(H_cap):+.4e}")

    # ---- reproduction ----
    print("== reproduction (float32 unpivoted Cholesky) ==")
    print(f"   H_derived (from inputs only)  -> #nan={unpivoted_cholesky_nan(H_derived)}")
    if H_cap is not None:
        print(f"   snap_h    (recorded Hessian)  -> #nan={unpivoted_cholesky_nan(H_cap)}")
    print(f"   qM+JTDAJ  (pyramidal-equiv)    -> #nan={unpivoted_cholesky_nan(qM + JTDAJ)}")
    reg = reg_eps * float(np.max(np.diag(H_derived)))
    print(f"   H_derived + {reg_eps:g}*max(diag)     -> #nan={unpivoted_cholesky_nan(H_derived + reg * np.eye(nv))}")

    # ---- iterate provenance ----
    qws = pick(dump, "d_qacc_warmstart", required=False)
    qsnap = pick(dump, "snap_qacc", required=False)
    if jaref is not None and qws is not None and qsnap is not None:
        aref = pick(dump, "efc_aref").reshape(-1)[:nefc].astype(np.float64)
        jaref_ws = Jd @ qws.reshape(-1)[:nv].astype(np.float64) - aref
        jaref_snap = Jd @ qsnap.reshape(-1)[:nv].astype(np.float64) - aref
        print("== iterate provenance (the Hessian is built at the post-linesearch iterate) ==")
        print(f"   |Jaref(qacc_warmstart) - snap_jaref| = {np.max(np.abs(jaref_ws - jaref[:nefc])):.3e}  (large)")
        print(f"   |Jaref(snap_qacc)      - snap_jaref| = {np.max(np.abs(jaref_snap - jaref[:nefc])):.3e}  (~0)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", nargs="?", help="FACTORY_PRESOLVE_DUMP npz (pre-Hessian state + solve snapshot).")
    ap.add_argument("--qm_struct", default=None, help="Sidecar npz with qM_mulm_* (for dumps lacking it).")
    ap.add_argument("--reg_eps", type=float, default=1e-4, help="Diagonal regularization eps for the fix test.")
    ap.add_argument("--save_qm_struct", default=None, help="Write the model's qM_mulm_* structure here and exit.")
    ap.add_argument("--mjb", default=None, help="Saved MuJoCo model binary (for --save_qm_struct).")
    args = ap.parse_args()

    if args.save_qm_struct:
        if not args.mjb:
            ap.error("--save_qm_struct requires --mjb")
        save_qm_structure(args.mjb, args.save_qm_struct)
        return 0
    if not args.dump:
        ap.error("dump is required (or use --save_qm_struct --mjb)")
    run(np.load(args.dump, allow_pickle=True), args.qm_struct, args.reg_eps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
