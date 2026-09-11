# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run an Isaac Lab task with phase-annotated stepping for nsys/ncu profiling.

Emits NVTX ranges around every env.step phase, brackets the measured window with
cudaProfilerStart/Stop, and writes host-side timing plus model/solver metadata.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import warp as wp

wp.config.enable_backward = False

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab.benchmark.stepping import sample_random_actions  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

HOST = defaultdict(float)
COUNTS = defaultdict(int)
_NVTX = True


def _wrap(obj, name, label):
    if obj is None or not hasattr(obj, name):
        return
    fn = getattr(obj, name)

    @functools.wraps(fn)
    def wrapped(*a, **k):
        if _NVTX:
            torch.cuda.nvtx.range_push(label)
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            HOST[label] += time.perf_counter() - t0
            COUNTS[label] += 1
            if _NVTX:
                torch.cuda.nvtx.range_pop()

    setattr(obj, name, wrapped)


def _instrument(env):
    u = env.unwrapped

    def g(name):
        return getattr(u, name, None)

    # Manager-based envs.
    _wrap(g("action_manager"), "process_action", "action.process")
    _wrap(g("action_manager"), "apply_action", "action.apply")
    _wrap(g("termination_manager"), "compute", "termination")
    _wrap(g("reward_manager"), "compute", "reward")
    _wrap(g("observation_manager"), "compute", "observation")
    _wrap(g("command_manager"), "compute", "command")
    _wrap(g("event_manager"), "apply", "event.apply")
    # Direct envs.
    _wrap(u, "_pre_physics_step", "action.process")
    _wrap(u, "_apply_action", "action.apply")
    _wrap(u, "_get_dones", "termination")
    _wrap(u, "_get_rewards", "reward")
    _wrap(u, "_get_observations", "observation")
    # Shared.
    _wrap(u.scene, "write_data_to_sim", "scene.write")
    _wrap(u.sim, "step", "sim.step")
    _wrap(u.scene, "update", "scene.update")
    _wrap(u, "_reset_idx", "reset_idx")
    # Physics manager internals.
    try:
        from isaaclab_newton.physics import NewtonManager

        _wrap(NewtonManager, "forward", "newton.forward")
        _wrap(NewtonManager, "_reset_solver_internals_delegate", "newton.reset_internals")
        real_launch = wp.capture_launch

        def timed_launch(graph, *a, **k):
            # Sensors can launch separate graphs from observation updates.
            label = "physics_graph" if graph is NewtonManager._graph else "auxiliary_graph"
            if _NVTX:
                torch.cuda.nvtx.range_push(label)
            t0 = time.perf_counter()
            try:
                return real_launch(graph, *a, **k)
            finally:
                HOST[label] += time.perf_counter() - t0
                COUNTS[label] += 1
                if _NVTX:
                    torch.cuda.nvtx.range_pop()

        wp.capture_launch = timed_launch
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Newton instrumentation skipped: {exc}")


def _trace_step(env, i: int, physics: str) -> None:
    """Print one line of physics statistics after env step ``i``."""
    torch.cuda.synchronize()
    m = _model_meta(physics)
    resets = int(env.unwrapped.reset_buf.sum().item()) if hasattr(env.unwrapped, "reset_buf") else -1
    extra = ""
    if physics == "feather_pgs":
        # Consistency of the published poses with joint coordinates, and the FK/ID cache state.
        from isaaclab_newton.physics.newton_manager import NewtonManager
        from newton import eval_fk

        st = NewtonManager.get_state_0()
        model = NewtonManager._model
        scratch = getattr(_trace_step, "_scratch", None)
        if scratch is None:
            scratch = model.state()
            _trace_step._scratch = scratch
        eval_fk(model, st.joint_q, st.joint_qd, scratch)
        dq = np.abs(scratch.body_q.numpy() - st.body_q.numpy()).max()
        dqd = np.abs(scratch.body_qd.numpy() - st.body_qd.numpy()).max()
        s = NewtonManager._solver
        valid = getattr(s, "_fk_id_cache_valid", None)
        nvalid = int(valid.numpy().sum()) if valid is not None else -1
        extra = f" body_q_vs_fk={dq:.3e} body_qd_vs_fk={dqd:.3e} cache_valid={nvalid}"
        diff = getattr(s, "_debug_cache_diff", None)
        sc = getattr(s, "_debug_cache_scratch", None)
        if diff is not None and sc is not None and "cached_f" in sc:
            fc = sc["cached_f"].numpy()
            ff = sc["fresh_f"].numpy()
            d = np.abs(fc - ff).max(axis=1)
            top = np.argsort(-d)[:4]
            print("  in-graph stage-1 detail (body, art, |df|, mass, cached f, fresh f, |da|, |dqcom|):")
            mass = model.body_mass.numpy()
            b2a = s.body_to_articulation.numpy()
            da = np.abs(sc["cached_a"].numpy() - sc["fresh_a"].numpy()).max(axis=1)
            dq = np.abs(sc["cached_qcom"].numpy() - sc["fresh_qcom"].numpy()).max(axis=1)
            for b in top:
                print(
                    f"    {b} art={b2a[b]} {d[b]:.3e} m={mass[b]:.3f} {np.round(fc[b], 2)} {np.round(ff[b], 2)} "
                    f"da={da[b]:.2e} dq={dq[b]:.2e}"
                )
            print(
                f"  bodies with |df|>1: {int((d > 1).sum())} of {d.size}; per-body-slot histogram of |df|>1: "
                f"{np.bincount((np.flatnonzero(d > 1) % 17), minlength=17).tolist()}"
            )
        if diff is not None:
            d = diff.numpy()
            extra += " cache_vs_fresh[q,qcom,origin,S,v,a,f,f_sub0,f_sub1]=" + ",".join(f"{x:.2e}" for x in d[:9])
            diff.zero_()
    print(
        f"TRACE step={i} contacts={m.get('contacts_active')} pairs={m.get('broad_phase_pairs')} "
        f"gjk={m.get('gjk_items')} manifold={m.get('manifold_items')} resets={resets} "
        f"z_min={m.get('body_z_min'):.4f} z_max={m.get('body_z_max'):.4f} "
        f"qd_max={m.get('joint_qd_abs_max'):.2f}{extra}",
        flush=True,
    )


def _model_meta(physics: str) -> dict:
    meta: dict = {}
    try:
        from isaaclab_newton.physics import NewtonManager

        m = NewtonManager.get_model()
        meta.update(
            worlds=int(m.world_count),
            bodies=int(m.body_count),
            shapes=int(m.shape_count),
            joints=int(m.joint_count),
            joint_dofs=int(m.joint_dof_count),
            joint_coords=int(m.joint_coord_count),
            articulations=int(m.articulation_count),
            rigid_contact_max=int(getattr(m, "rigid_contact_max", 0) or 0),
        )
        s = NewtonManager._solver
        meta["solver_class"] = type(s).__name__
        meta["decimation"] = int(NewtonManager._decimation)
        meta["num_substeps"] = int(NewtonManager._num_substeps)
        meta["solver_dt"] = float(NewtonManager._solver_dt)
        c = NewtonManager.get_contacts()
        if c is not None and getattr(c, "rigid_contact_count", None) is not None:
            meta["contacts_active"] = int(c.rigid_contact_count.numpy().sum())
        st_types = getattr(m, "shape_type", None)
        if st_types is not None and not getattr(_trace_step, "_shapes_printed", False):
            import collections

            hist = collections.Counter(int(t) for t in st_types.numpy().tolist())
            print(f"SHAPES per model: {dict(sorted(hist.items()))} (GeoType ints), shape_count={int(m.shape_count)}")
            for attr in ("shape_gap", "shape_collision_radius"):
                arr = getattr(m, attr, None)
                if arr is not None:
                    v = arr.numpy()
                    print(f"SHAPES {attr}: min={v.min():.4g} max={v.max():.4g} mean={v.mean():.4g}")
            _trace_step._shapes_printed = True
        cp = getattr(NewtonManager, "_collision_pipeline", None)
        if cp is not None:
            bp = getattr(cp, "broad_phase_pair_count", None)
            if bp is not None:
                meta["broad_phase_pairs"] = int(bp.numpy()[0])
            npz = getattr(cp, "narrow_phase", None)
            for attr, key in (("split_gjk_work_count", "gjk_items"), ("split_manifold_work_count", "manifold_items")):
                arr = getattr(npz, attr, None) if npz is not None else None
                if arr is not None:
                    meta[key] = int(arr.numpy()[0])
        st = NewtonManager.get_state_0()
        bq = st.body_q.numpy()
        jq = st.joint_q.numpy()
        jqd = st.joint_qd.numpy()
        meta["state_finite"] = bool(np.isfinite(bq).all() and np.isfinite(jq).all() and np.isfinite(jqd).all())
        meta["joint_qd_abs_max"] = float(np.abs(jqd).max()) if jqd.size else 0.0
        meta["body_z_min"] = float(bq[:, 2].min()) if bq.size else 0.0
        meta["body_z_max"] = float(bq[:, 2].max()) if bq.size else 0.0
        if physics == "feather_pgs":
            for attr in (
                "size_groups",
                "n_arts_by_size",
                "dense_max_constraints",
                "mf_max_constraints",
                "max_world_dofs",
                "pgs_mode",
                "pgs_iterations",
                "use_parallel_streams",
                "_paired_response_primary_size",
                "_paired_response_secondary_size",
                "_sparse_diagonal_response_size",
                "_compact_diagonal_mass_size",
                "_has_free_rigid_bodies",
                "_jy_world_aliased",
                "_hinv_jt_writes_world",
                "articulated_contact_response",
                "world_count",
            ):
                v = getattr(s, attr, None)
                if isinstance(v, dict):
                    v = {str(k): int(x) for k, x in v.items()}
                elif isinstance(v, (list, tuple)):
                    v = [int(x) for x in v]
                elif isinstance(v, (np.integer, int)) and not isinstance(v, bool):
                    v = int(v)
                elif isinstance(v, np.ndarray):
                    v = v.tolist()
                meta[attr] = v
            for name in ("constraint_count", "mf_constraint_count", "propagation_constraint_count"):
                arr = getattr(s, name, None)
                if arr is not None:
                    a = arr.numpy()
                    meta[name] = {
                        "sum": int(a.sum()),
                        "max": int(a.max()),
                        "mean": float(a.mean()),
                        "nonzero_worlds": int((a > 0).sum()),
                    }
    except Exception as exc:  # noqa: BLE001
        meta["error"] = repr(exc)
    return meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="Isaac-Lift-Franka")
    p.add_argument("--physics", default="feather_pgs")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-envs", type=int, default=16384)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--profile-steps", type=int, default=10, help="Env steps inside cudaProfilerStart/Stop.")
    p.add_argument("--no-nvtx", action="store_true")
    p.add_argument("--solver-nvtx", action="store_true", help="Enable solver-internal NVTX (eager only).")
    p.add_argument("--no-graph", action="store_true", help="Disable CUDA graphs (eager launches).")
    p.add_argument("--trace-stats", action="store_true", help="Print per-step contact/reset statistics.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--override", action="append", default=[], help="Extra hydra override.")
    p.add_argument(
        "--solver-attr", action="append", default=[], help="key=value set on sim.physics.solver_cfg (python literal)."
    )
    p.add_argument("--physics-attr", action="append", default=[], help="key=value set on sim.physics (python literal).")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    global _NVTX
    _NVTX = not args.no_nvtx

    torch.manual_seed(args.seed)
    env_cfg = parse_env_cfg(
        args.task, device=args.device, num_envs=args.num_envs, overrides=[f"physics={args.physics}", *args.override]
    )
    env_cfg.seed = args.seed
    if args.no_graph:
        env_cfg.sim.physics.use_cuda_graph = False
    import ast as _ast

    for kv in args.solver_attr:
        k, v = kv.split("=", 1)
        with contextlib.suppress(Exception):
            v = _ast.literal_eval(v)
        setattr(env_cfg.sim.physics.solver_cfg, k, v)
        print(f"[CFG] solver_cfg.{k} = {v!r}")
    for kv in args.physics_attr:
        k, v = kv.split("=", 1)
        with contextlib.suppress(Exception):
            v = _ast.literal_eval(v)
        setattr(env_cfg.sim.physics, k, v)
        print(f"[CFG] physics.{k} = {v!r}")
    result_cfg = {"solver_attr": args.solver_attr, "physics_attr": args.physics_attr}
    if args.solver_nvtx and hasattr(env_cfg.sim.physics.solver_cfg, "nvtx"):
        env_cfg.sim.physics.solver_cfg.nvtx = True
    env_cfg.validate()
    launcher_args = {"device": args.device, "headless": True, "visualizer": None, "visualizer_explicit": True}

    result: dict = {
        "task": args.task,
        "physics": args.physics,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "repeats": args.repeats,
        "profile_steps": args.profile_steps,
        "cuda_graph": not args.no_graph,
        **result_cfg,
    }
    with launch_simulation(env_cfg, launcher_args):
        with contextlib.closing(gym.make(args.task, cfg=env_cfg)) as env:
            n = env.unwrapped.num_envs
            result["decimation"] = int(env.unwrapped.cfg.decimation)
            result["sim_dt"] = float(env.unwrapped.cfg.sim.dt)
            with torch.inference_mode():
                env.reset()
                for i in range(args.warmup_steps):
                    env.step(sample_random_actions(env))
                    if args.trace_stats:
                        _trace_step(env, i, args.physics)
                torch.cuda.synchronize()
                wp.synchronize_device(args.device)
                result["model"] = _model_meta(args.physics)
                _instrument(env)
                HOST.clear()
                COUNTS.clear()
                # Timed repeats (no profiler range).
                fps = []
                step_wall = []
                for r in range(args.repeats):
                    torch.cuda.synchronize()
                    wp.synchronize_device(args.device)
                    t0 = time.perf_counter()
                    for i in range(args.steps):
                        a = sample_random_actions(env)
                        ts = time.perf_counter()
                        env.step(a)
                        step_wall.append(time.perf_counter() - ts)
                        if args.trace_stats:
                            _trace_step(env, args.warmup_steps + r * args.steps + i, args.physics)
                    torch.cuda.synchronize()
                    wp.synchronize_device(args.device)
                    el = time.perf_counter() - t0
                    fps.append(n * args.steps / el)
                    print(
                        f"RESULT repeat={r + 1} fps={fps[-1]:.1f} ms_per_step={1e3 * el / args.steps:.3f}", flush=True
                    )
                total_steps = args.steps * args.repeats
                result["fps"] = {
                    "mean": statistics.fmean(fps),
                    "std": statistics.stdev(fps) if len(fps) > 1 else 0.0,
                    "per_repeat": fps,
                }
                result["ms_per_step_sync"] = 1e3 * n / statistics.fmean(fps)
                result["host_ms_per_step"] = {
                    k: 1e3 * v / total_steps for k, v in sorted(HOST.items(), key=lambda kv: -kv[1])
                }
                result["host_calls_per_step"] = {k: v / total_steps for k, v in COUNTS.items()}
                result["host_step_return_ms"] = {
                    "mean": 1e3 * statistics.fmean(step_wall),
                    "median": 1e3 * statistics.median(step_wall),
                }
                # Profiler window.
                torch.cuda.synchronize()
                wp.synchronize_device(args.device)
                torch.cuda.cudart().cudaProfilerStart()
                for i in range(args.profile_steps):
                    if _NVTX:
                        torch.cuda.nvtx.range_push(f"env_step:{i}")
                    a = sample_random_actions(env)
                    env.step(a)
                    if _NVTX:
                        torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize()
                wp.synchronize_device(args.device)
                torch.cuda.cudart().cudaProfilerStop()
                result["model_after"] = _model_meta(args.physics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(
        "FINAL " + json.dumps({k: result[k] for k in ("fps", "host_ms_per_step", "host_step_return_ms")}, default=str),
        flush=True,
    )


if __name__ == "__main__":
    main()
