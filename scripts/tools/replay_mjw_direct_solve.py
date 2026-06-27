# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Level-0 NaN replay: run ONLY the MuJoCo-Warp constraint solve from captured solver inputs.

This is the most faithful / lowest-level reproduction. Instead of re-deriving contacts and the
constraint Jacobian (which the higher-level Newton replay does via the Newton->mjw conversion and
``make_constraint``, and which does not reproduce the knife-edge NaN), this loads the EXACT
captured pre-solve solver state for the NaN frame:

  * mass matrix qM (factorized in-tool) and qfrc_smooth (net unconstrained force),
  * the constraint rows efc.{J(sparse),D,aref,pos,vel,Jqvel,type,margin} + nefc/ne/nf/nl,
  * the contact friction/solref/solimp/dim + efc_address (for the elliptic cone),
  * the qacc_warmstart (solver initial guess),

then calls ``mujoco_warp.solver.solve(m, d)`` on a single-world model and checks for NaN. Because
the inputs ARE the original solve's inputs, this reproduces the original NaN deterministically.

    ./isaaclab.sh -p scripts/tools/replay_mjw_direct_solve.py <npz> --device cuda:0
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import warp as wp

import scripts.tools.replay_step_transition_newton as R  # noqa: E402


def _set_row(dst_wp, src_np, n: int | None = None):
    """Copy src_np into the first elements of world-0 of a warp array (shape [nworld, ...] or [N])."""
    t = wp.to_torch(dst_wp)
    s = torch.as_tensor(np.asarray(src_np), dtype=t.dtype if t.is_floating_point else t.dtype, device=t.device)
    if t.dim() >= 1 and t.shape[0] == 1 and (s.dim() + 1 == t.dim() or (s.dim() == t.dim() and s.shape[0] != 1)):
        # world-batched [1, ...]: write into row 0
        row = t[0]
        m = s.shape[0] if n is None else n
        row.reshape(-1)[: s.reshape(-1).shape[0]].copy_(s.reshape(-1)[: row.reshape(-1).shape[0]])
    else:
        flat_t = t.reshape(-1)
        flat_s = s.reshape(-1)
        k = min(flat_t.shape[0], flat_s.shape[0]) if n is None else n
        flat_t[:k].copy_(flat_s[:k])


def _g(data, key):
    return np.asarray(data[key]) if key in data.files else None


def run_from_presolve_dump(args) -> int:
    """Direct solve from a COMPLETE consistent pre-solve dump (FACTORY_PRESOLVE_DUMP npz).

    Unlike the mixed cold-path/replay_pre path, every field here is from the SAME snapshot taken
    right before the failing _solve, so the solve reproduces the NaN exactly.
    """
    import mujoco_warp
    from mujoco_warp._src import solver as mjw_solver
    from mujoco_warp._src import smooth

    dump = np.load(args.presolve_dump, allow_pickle=True)
    f = set(dump.files)
    print(f"presolve dump keys: {sorted(f)}")

    exact_model = bool(args.mj_model)
    if exact_model:
        # Build the EXACT mjw model from the saved compiled MuJoCo model (FACTORY_SAVE_MJMODEL).
        # geom/body/efc indices then match the capture, so efc_id and the cone linking transplant
        # faithfully -> the isolated solve can reproduce the original NaN.
        import mujoco as _mj

        mjm = _mj.MjModel.from_binary_path(args.mj_model)
        mjd = _mj.MjData(mjm)
        _mj.mj_forward(mjm, mjd)
        with wp.ScopedDevice(args.device):
            m = mujoco_warp.put_model(mjm)
            d = mujoco_warp.put_data(mjm, mjd, nworld=1, nconmax=args.nconmax, njmax=args.njmax)
        print(f"built EXACT mjw model from {args.mj_model}: nv={m.nv}")
    else:
        usd = R._find_usd_path(args.npz_path) if args.npz_path else R._find_usd_path(args.presolve_dump)
        model = R._build_model(usd, args.device)
        R._set_shape_material(model, args.contact_ke, args.contact_kd, 0.6923, args.contact_gap, args.contact_margin, verbose=False)
        solver = R._make_solver(model, args)
        m, d = solver.mjw_model, solver.mjw_data

    def setrow(wp_arr, key):
        if key not in f or wp_arr is None:
            return False
        t = wp.to_torch(wp_arr)
        s = torch.as_tensor(np.asarray(dump[key]), dtype=t.dtype, device=t.device)
        if t.dim() >= 1 and t.shape[0] == 1:
            t[0].reshape(-1)[: s.reshape(-1).shape[0]].copy_(s.reshape(-1)[: t[0].reshape(-1).shape[0]])
        else:
            t.reshape(-1)[: s.reshape(-1).shape[0]].copy_(s.reshape(-1)[: t.reshape(-1).shape[0]])
        return True

    # state -> kinematics (we still need m structure; qM is injected next)
    setrow(d.qpos, "d_qpos")
    setrow(d.qvel, "d_qvel")
    mujoco_warp.fwd_position(m, d)
    mujoco_warp.fwd_velocity(m, d)
    # dynamics
    for fld, key in (("qfrc_smooth", "d_qfrc_smooth"), ("qacc_smooth", "d_qacc_smooth"),
                     ("qacc_warmstart", "d_qacc_warmstart"), ("qacc", "d_qacc_warmstart"), ("qM", "d_qM")):
        if hasattr(d, fld):
            setrow(getattr(d, fld), key)
    smooth.factor_m(m, d)
    # constraint rows
    # NOTE: efc_id is intentionally NOT injected: it references contact/equality OBJECT indices
    # from the original 8192-world layout that do not exist in this single-env model (injecting it
    # causes an illegal memory access). The efc<->contact<->object linking is context-specific and
    # cannot be transplanted, which is itself why the isolated solve cannot fully reproduce the cone.
    for fld, key in (("D", "efc_D"), ("aref", "efc_aref"), ("pos", "efc_pos"), ("vel", "efc_vel"),
                     ("Jqvel", "efc_Jqvel"), ("type", "efc_type"), ("margin", "efc_margin"),
                     ("frictionloss", "efc_frictionloss"),
                     ("J", "efc_J"), ("J_rownnz", "efc_J_rownnz"), ("J_rowadr", "efc_J_rowadr"),
                     ("J_colind", "efc_J_colind")):
        if hasattr(d.efc, fld):
            setrow(getattr(d.efc, fld), key)
    if exact_model and hasattr(d.efc, "id") and "efc_id" in f:
        # With the exact model the constraint-object indices are valid; the cone linking transplants.
        setrow(d.efc.id, "efc_id")
    for fld, key in (("nefc", "d_nefc"), ("ne", "d_ne"), ("nf", "d_nf"), ("nl", "d_nl")):
        if hasattr(d, fld) and key in f:
            wp.to_torch(getattr(d, fld))[:] = int(np.asarray(dump[key]).reshape(-1)[0])
    # contacts (filtered to the NaN world in the dump) -> place at 0..ncon, set nacon
    csel = dump["_contact_sel"] if "_contact_sel" in f else None
    ncon = int(csel.shape[0]) if csel is not None else 0
    contact_flds = [("friction", "contact_friction"), ("dim", "contact_dim"), ("solref", "contact_solref"),
                    ("solimp", "contact_solimp"), ("efc_address", "contact_efc_address"),
                    ("dist", "contact_dist"), ("solreffriction", "contact_solreffriction"),
                    ("pos", "contact_pos"), ("frame", "contact_frame"), ("includemargin", "contact_includemargin")]
    if exact_model:
        contact_flds.append(("geom", "contact_geom"))  # geom indices valid only with the exact model
    for fld, key in contact_flds:
        if hasattr(d.contact, fld) and key in f and getattr(d.contact, fld) is not None:
            t = wp.to_torch(getattr(d.contact, fld))
            s = torch.as_tensor(np.asarray(dump[key]), dtype=t.dtype, device=t.device)
            t[: s.shape[0]].copy_(s)
    if hasattr(d, "nacon") and ncon:
        wp.to_torch(d.nacon)[:] = ncon
    # opts
    for fld, key in (("iterations", "opt_iterations"), ("ls_iterations", "opt_ls_iterations"),
                     ("tolerance", "opt_tolerance"), ("ls_tolerance", "opt_ls_tolerance"), ("impratio", "opt_impratio")):
        if key in f and hasattr(m.opt, fld):
            try:
                ov = getattr(m.opt, fld)
                val = float(np.asarray(dump[key]).reshape(-1)[0])
                wp.to_torch(ov)[:] = val
            except Exception:
                pass

    if args.force_all_iters:
        # In the original 8192-world batch the solver's shared while-loop ran all 15 iterations
        # (other worlds had not converged), forcing this world to over-iterate on its near-singular
        # system and diverge to NaN. In isolation it converges at iter 1, so we pin tolerance=0.
        for opt_fld in ("tolerance", "ls_tolerance"):
            if hasattr(m.opt, opt_fld):
                try:
                    wp.to_torch(getattr(m.opt, opt_fld))[:] = 0.0
                except Exception:
                    setattr(m.opt, opt_fld, 0.0)
        print("  forced m.opt.tolerance=0 (run all iterations, matching the batched solve)")

    nefc = int(np.asarray(dump["d_nefc"]).reshape(-1)[0]) if "d_nefc" in f else -1
    print(f"loaded: nefc={nefc} nacon(set)={ncon}  qLDiagInv max={float(np.max(np.abs(wp.to_torch(d.qLDiagInv).detach().cpu().numpy()))):.4e}")
    mjw_solver.solve(m, d)
    qacc = wp.to_torch(d.qacc).detach().cpu().numpy()
    niter = int(wp.to_torch(d.solver_niter).reshape(-1)[0].item())
    print("\n=== DIRECT SOLVE FROM CONSISTENT PRE-SOLVE DUMP ===")
    print(f"  solver_niter={niter}  qacc #nan={int(np.isnan(qacc).sum())}/{qacc.size}")
    if int(np.isnan(qacc).sum()) > 0:
        print("  >>> NaN REPRODUCED from the exact consistent pre-solve state. <<<")
    else:
        print(f"  >>> No NaN: qacc range [{np.nanmin(qacc):.3e}, {np.nanmax(qacc):.3e}] <<<")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_path", nargs="?", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--njmax", type=int, default=1200)
    ap.add_argument("--nconmax", type=int, default=2000)
    ap.add_argument("--rigid_contact_max", type=int, default=400000)
    ap.add_argument("--max_triangle_pairs", type=int, default=30_000_000)
    ap.add_argument("--broad_phase", default="sap")
    ap.add_argument("--solver", default="newton")
    ap.add_argument("--integrator", default="implicitfast")
    ap.add_argument("--cone", default="elliptic")
    ap.add_argument("--impratio", type=float, default=1.0)
    ap.add_argument("--iterations", type=int, default=15)
    ap.add_argument("--ls_iterations", type=int, default=100)
    ap.add_argument("--ls_parallel", action="store_true")
    ap.add_argument("--update_data_interval", type=int, default=1)
    ap.add_argument("--contact_ke", type=float, default=1e7)
    ap.add_argument("--contact_kd", type=float, default=1e4)
    ap.add_argument("--contact_gap", type=float, default=0.001)
    ap.add_argument("--contact_margin", type=float, default=0.0)
    ap.add_argument("--frame", type=int, default=-1, help="replay frame whose pre-state to load (default last).")
    ap.add_argument("--force_all_iters", action="store_true", help="Pin solver tolerance to 0 so all iterations run (reproduces the original non-convergent solve).")
    ap.add_argument("--presolve_dump", default=None, help="Path to a FACTORY_PRESOLVE_DUMP npz (complete consistent pre-solve state). If set, solve directly from it.")
    ap.add_argument("--mj_model", default=None, help="Path to a FACTORY_SAVE_MJMODEL binary. If set, build the EXACT mjw model from it (faithful indices) instead of rebuilding from USD.")
    args = ap.parse_args()

    if args.presolve_dump:
        return run_from_presolve_dump(args)

    import mujoco_warp
    from mujoco_warp._src import solver as mjw_solver

    data = np.load(args.npz_path, allow_pickle=True)
    usd = R._find_usd_path(args.npz_path)
    replay_size = int(data["replay_buffer_size"])
    frame = args.frame if args.frame >= 0 else replay_size - 1
    captured_env = int(data["exported_env_ids"][0]) if "exported_env_ids" in data.files else 0

    friction = 0.6923
    mf = _g(data, "mjw_contact_friction")
    if mf is not None and mf.size:
        friction = float(mf.reshape(-1)[0])

    print(f"Loading USD: {usd}")
    model = R._build_model(usd, args.device)
    R._set_shape_material(model, args.contact_ke, args.contact_kd, friction, args.contact_gap, args.contact_margin)
    state = model.state()
    control = model.control()
    solver = R._make_solver(model, args)
    R._restore_mjw_model(solver, data)

    m = solver.mjw_model
    d = solver.mjw_data

    # 1) inject pre-solve Newton state -> mjw qpos/qvel
    translation = R._compute_replay_translation(data, model, state, frame)
    free_idx = R._free_joint_pos_indices(data, int(np.asarray(data["replay_pre_joint_q"]).shape[1]))
    R._assign_pre_state(state, data, frame, translation, free_idx)
    solver._update_mjc_data(d, model, state)

    # 2) run position/velocity to factorize M and lay out kinematics (factor_m -> qLD used by solve)
    mujoco_warp.fwd_position(m, d)
    mujoco_warp.fwd_velocity(m, d)

    # compare my (recomputed) mass matrix to the captured one; inject the captured qM and
    # re-factorize so the solve sees the EXACT (near-singular) conditioning the original had.
    from mujoco_warp._src import smooth

    cap_qM = _g(data, "mjw_qM")
    my_qM = wp.to_torch(d.qM).detach().cpu().numpy().reshape(-1)
    if cap_qM is not None:
        c = cap_qM.reshape(-1)
        k = min(c.size, my_qM.size)
        print(f"qM: my{my_qM.shape} captured{c.shape} maxdiff(first {k})={np.max(np.abs(my_qM[:k]-c[:k])):.4e}")
        _set_row(d.qM, c)
        smooth.factor_m(m, d)
        cap_ld = _g(data, "mjw_qLDiagInv")
        my_ld = wp.to_torch(d.qLDiagInv).detach().cpu().numpy().reshape(-1)
        if cap_ld is not None:
            print(f"qLDiagInv: captured max={np.max(np.abs(cap_ld)):.4e}  mine-after-refactor max={np.max(np.abs(my_ld)):.4e}")
        print("injected captured qM + refactored M")

    # 3) OVERWRITE the constraint + dynamics with the captured pre-solve solver state for this frame.
    nefc = int(_g(data, "mjw_nefc").reshape(-1)[0])
    ne = int(_g(data, "mjw_ne").reshape(-1)[0])
    nf = int(_g(data, "mjw_nf").reshape(-1)[0])
    nl = int(_g(data, "mjw_nl").reshape(-1)[0])
    print(f"captured nefc={nefc} ne={ne} nf={nf} nl={nl} solver_niter={int(_g(data,'mjw_solver_niter').reshape(-1)[0])}")

    for arr_name, key in (("nefc", "mjw_nefc"), ("ne", "mjw_ne"), ("nf", "mjw_nf"), ("nl", "mjw_nl")):
        if hasattr(d, arr_name) and _g(data, key) is not None:
            wp.to_torch(getattr(d, arr_name))[:] = int(_g(data, key).reshape(-1)[0])

    # dynamics
    for fld, key in (("qfrc_smooth", "mjw_qfrc_smooth"), ("qacc_smooth", "mjw_qacc_smooth"), ("qacc_warmstart", None)):
        if fld == "qacc_warmstart":
            ws = np.asarray(data["replay_pre_mjw_qacc_warmstart"][frame])
            ws = ws[captured_env] if ws.ndim == 2 else ws
            _set_row(d.qacc_warmstart, ws)
            if hasattr(d, "qacc"):
                _set_row(d.qacc, ws)  # also seed qacc with the warmstart (solver initial guess)
        else:
            v = _g(data, key)
            if v is not None and hasattr(d, fld):
                _set_row(getattr(d, fld), v.reshape(-1))

    # constraint rows
    efc = d.efc
    for fld, key in (
        ("D", "mjw_efc_D"), ("aref", "mjw_efc_aref"), ("pos", "mjw_efc_pos"),
        ("vel", "mjw_efc_vel"), ("Jqvel", "mjw_efc_Jqvel"), ("frictionloss", None),
        ("type", "mjw_efc_type"), ("margin", "mjw_efc_margin"),
    ):
        v = _g(data, key)
        if v is not None and hasattr(efc, fld):
            _set_row(getattr(efc, fld), v.reshape(-1))
    # sparse Jacobian
    jrow = _g(data, "mjw_efc_J_rownnz")
    jadr = _g(data, "mjw_efc_J_rowadr")
    jval = _g(data, "mjw_efc_J")
    jcol = _g(data, "mjw_efc_J_colind")
    if all(x is not None for x in (jrow, jadr, jval, jcol)):
        if hasattr(efc, "J_rownnz"):
            _set_row(efc.J_rownnz, jrow.reshape(-1))
            _set_row(efc.J_rowadr, jadr.reshape(-1))
            _set_row(efc.J, jval.reshape(-1))
            _set_row(efc.J_colind, jcol.reshape(-1))
        print(f"injected sparse efc.J nnz={jval.reshape(-1).shape[0]}")

    # contacts (for the elliptic cone friction referenced by efc_address)
    contact = d.contact
    ncon = 0
    for fld, key in (
        ("friction", "mjw_contact_friction"), ("solref", "mjw_contact_solref"),
        ("solimp", "mjw_contact_solimp"), ("dim", "mjw_contact_dim"),
        ("dist", "mjw_contact_dist"), ("efc_address", "mjw_contact_efc_address"),
    ):
        v = _g(data, key)
        if v is not None and hasattr(contact, fld):
            t = wp.to_torch(getattr(contact, fld))
            s = torch.as_tensor(v, dtype=t.dtype, device=t.device)
            ncon = s.shape[0]
            t[:ncon].copy_(s)
    # CRITICAL: set the detected-contact count d.nacon. The elliptic-cone kernel loops
    # conid in [0, nacon) and applies the cone coupling via contact.efc_address; without
    # nacon the cone is never applied and the (otherwise singular) solve converges.
    cce = _g(data, "mjw_contact_count_env")
    nacon = int(cce.reshape(-1)[0]) if cce is not None else ncon
    if hasattr(d, "nacon"):
        wp.to_torch(d.nacon)[:] = nacon
        print(f"  set d.nacon={nacon}")

    # Verify the efc injection actually took effect (read back vs captured).
    for fld, key in (("D", "mjw_efc_D"), ("aref", "mjw_efc_aref"), ("pos", "mjw_efc_pos"), ("vel", "mjw_efc_vel"), ("Jqvel", "mjw_efc_Jqvel"), ("type", "mjw_efc_type")):
        v = _g(data, key)
        if v is not None and hasattr(efc, fld):
            got = wp.to_torch(getattr(efc, fld)).detach().cpu().numpy().reshape(-1)[:nefc]
            print(f"  efc.{fld}[:{nefc}] injected={np.array2string(got, precision=4)}  captured={np.array2string(v.reshape(-1)[:nefc], precision=4)}")
    # sparse J: shape + per-row nnz + first row values
    if hasattr(efc, "J"):
        jt = wp.to_torch(efc.J)
        print(f"  efc.J shape={tuple(jt.shape)}  captured J shape={jval.shape if jval is not None else None}")
        if hasattr(efc, "J_rownnz") and jrow is not None:
            rn = wp.to_torch(efc.J_rownnz).detach().cpu().numpy().reshape(-1)[:nefc]
            ra = wp.to_torch(efc.J_rowadr).detach().cpu().numpy().reshape(-1)[:nefc]
            print(f"  efc.J_rownnz[:{nefc}] injected={rn} captured={jrow.reshape(-1)[:nefc]}")
            print(f"  efc.J_rowadr[:{nefc}] injected={ra} captured={jadr.reshape(-1)[:nefc]}")
            j_flat = jt.detach().cpu().numpy().reshape(-1)
            r0n, r0a = int(rn[0]), int(ra[0])
            print(f"  efc.J row0 injected={np.array2string(j_flat[r0a:r0a+r0n], precision=4)}")
            print(f"  efc.J row0 captured={np.array2string(jval.reshape(-1)[int(jadr.reshape(-1)[0]):int(jadr.reshape(-1)[0])+int(jrow.reshape(-1)[0])], precision=4)}")
    # contact friction (elliptic cone)
    if hasattr(contact, "friction"):
        cf = wp.to_torch(contact.friction).detach().cpu().numpy()
        print(f"  contact.friction[0] injected={np.array2string(cf[0], precision=4)}")
    print(f"  nefc(d)={int(wp.to_torch(d.nefc).reshape(-1)[0])}")

    if args.force_all_iters:
        # Force the solver to run every iteration (no early convergence). The original ran the
        # full 15 (never converged -> diverged to NaN); m.stat.meaninertia (build-time, from the
        # rebuilt USD masses) makes our convergence threshold differ, so we pin tolerance to 0.
        for opt_fld in ("tolerance", "ls_tolerance"):
            if hasattr(m.opt, opt_fld):
                ov = getattr(m.opt, opt_fld)
                try:
                    wp.to_torch(ov)[:] = 0.0
                except Exception:
                    setattr(m.opt, opt_fld, 0.0)
        print("  forced m.opt.tolerance=0 (run all iterations)")

    # 4) run the constraint solve only
    print(f"[pre-solve] qacc finite={bool(torch.isfinite(wp.to_torch(d.qacc)).all())}  "
          f"efc.D finite={bool(torch.isfinite(wp.to_torch(efc.D)).all())}")
    mjw_solver.solve(m, d)

    # 5) report
    qacc = wp.to_torch(d.qacc).detach().cpu().numpy()
    efc_force = wp.to_torch(efc.force).detach().cpu().numpy()
    niter = int(wp.to_torch(d.solver_niter).reshape(-1)[0].item())
    exp_qacc = _g(data, "mjw_qacc")
    print("\n=== DIRECT SOLVE RESULT ===")
    print(f"  solver_niter={niter}  (captured 15)")
    print(f"  qacc: #nan={int(np.isnan(qacc).sum())}/{qacc.size}   (captured #nan={int(np.isnan(exp_qacc).sum()) if exp_qacc is not None else '?'})")
    print(f"  efc.force[:nefc]: #nan={int(np.isnan(efc_force[:max(nefc,1)]).sum())}/{nefc}")
    if int(np.isnan(qacc).sum()) > 0:
        print("  >>> NaN REPRODUCED via direct solve from captured constraint rows. <<<")
    else:
        print("  >>> No NaN: solve converged. qacc range "
              f"[{np.nanmin(qacc):.3e}, {np.nanmax(qacc):.3e}] <<<")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
