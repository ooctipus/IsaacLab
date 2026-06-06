# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load and replay NaN state snapshots exported by the NewtonManager debug buffer.

Usage:
    # Print summary (state + episode_length + contacts + mjwarp solver internals + nut-in-bolt pose)
    ./isaaclab.sh -p scripts/tools/replay_nan_state.py /tmp/factory_bench/nan_debug/nan_replay_20260604_203254_627726.npz

    # Print summary + matplotlib plots (save PNGs to a dir so it works headless)
    ./isaaclab.sh -p scripts/tools/replay_nan_state.py /tmp/factory_bench/nan_debug/nan_replay_20260604_203254_627726.npz --plot --output_dir /tmp/factory_bench/nan_debug

    # Replay in Newton viewer (auto-discovers .usd next to .npz; needs a display)
    ./isaaclab.sh -p scripts/tools/replay_nan_state.py /tmp/factory_bench/nan_debug/nan_replay_20260604_203254_627726.npz --visualizer newton

    # Mid-episode incident: replay once, slow (0.1x), to watch the last clean frames before the NaN
    ./isaaclab.sh -p scripts/tools/replay_nan_state.py /tmp/factory_bench/nan_debug/nan_replay_20260604_203254_627726.npz --visualizer newton --speed 0.1 --no-loop
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def _print_summary(npz_path: str) -> None:
    """Print a text summary of the npz contents."""
    import numpy as np

    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"Failed to load {npz_path}: {e}", file=sys.stderr)
        sys.exit(1)

    n = int(data.get("buffer_size", data["joint_q"].shape[0] if "joint_q" in data else 0))
    if n == 0 and "joint_q" in data:
        n = data["joint_q"].shape[0]
    sim_time = float(data.get("sim_time", 0.0))

    print("=== NaN replay summary ===")
    print(f"File: {npz_path}")
    print(f"Buffer size (steps): {n}")
    print(f"Sim time at export: {sim_time}")
    if "exported_env_ids" in data:
        env_ids = data["exported_env_ids"]
        print(f"Exported env(s) only: {env_ids.tolist()}")

    usd_path = _find_usd_path(npz_path)
    if usd_path:
        print(f"Scene USD: {usd_path}")

    for key in ("body_q", "body_qd", "joint_q", "joint_qd"):
        if key not in data:
            continue
        arr = data[key]
        print(f"\n{key}: shape {arr.shape}, dtype {arr.dtype}")
        nan_per_step = np.isnan(arr).reshape(arr.shape[0], -1).any(axis=1)
        first_nan = np.where(nan_per_step)[0]
        if len(first_nan) > 0:
            print(f"  First step with NaN: {int(first_nan[0])} (last step {n-1} is the incident)")
        print(f"  Min: {np.nanmin(arr):.6g}, Max: {np.nanmax(arr):.6g}")

    _print_failure_analysis(data, n)


def _quat_rel_pose(bolt, nut):
    """Return (pos_in_bolt_frame, euler_xyz_deg) of ``nut`` expressed in ``bolt``'s frame.

    Both are 7-vectors ``[px,py,pz, qx,qy,qz,qw]`` (Newton/Warp xyzw convention).
    Returns ``None`` if scipy is unavailable.
    """
    try:
        from scipy.spatial.transform import Rotation as R
    except Exception:
        return None
    import numpy as np

    Rb = R.from_quat(bolt[3:7])
    Rn = R.from_quat(nut[3:7])
    pos = Rb.inv().apply(nut[:3] - bolt[:3])
    eul = (Rb.inv() * Rn).as_euler("xyz", degrees=True)
    return pos, eul


def _print_failure_analysis(data, n: int) -> None:
    """Print the deep failure context: reset/episode state, the held-asset pose relative
    to the fixed asset, the contacts/forces that fed the failing solve, and where in the
    mjwarp solver the NaN actually lives (contact-gen vs constraint assembly vs solve)."""
    import numpy as np

    def has(*keys):
        return all(k in data for k in keys)

    # --- episode / reset context -------------------------------------------------
    print("\n--- failure context ---")
    if "episode_length" in data:
        print(f"episode_length (NaN env): {data['episode_length'].tolist()}  "
              "(0/1 => just reset; large => mid-episode)")
    if "per_substep_capture" in data:
        print(f"PER-SUBSTEP capture: NaN born on substep {int(data.get('nan_substep_idx', -1))} "
              f"(last finite substep {int(data.get('last_finite_substep_idx', -1))}, "
              f"monotonic #{int(data.get('substep_counter', -1))}); pre_* = that last finite substep, "
              "so pre->post here is a SINGLE solver substep")
    if has("pre_body_q", "body_q") and data["body_q"].shape[0] >= 2:
        d = float(np.nanmax(np.abs(data["pre_body_q"] - data["body_q"][-2])))
        print(f"|pre_body_q - post[-2]| = {d:.3e}  "
              "(0 => pre-step == previous-post: continuous OR reset captured pre-flush)")

    # --- bodies: identify nut/bolt by label, derive held-relative-to-fixed pose ---
    labels = [str(x) for x in data["body_label"]] if "body_label" in data else None
    nut_i = bolt_i = None
    if labels:
        for i, lb in enumerate(labels):
            u = lb.upper()
            if nut_i is None and ("NUT" in u or "HELD" in u):
                nut_i = i
            if bolt_i is None and ("BOLT" in u or "THREAD" in u or "FIXED" in u):
                bolt_i = i
        print(f"\nbodies ({len(labels)}): " + ", ".join(f"[{i}]{lb.split('/')[-2] if '/' in lb else lb}"
                                                         for i, lb in enumerate(labels)))
        if nut_i is not None and bolt_i is not None and "body_q" in data:
            print(f"detected held/nut=[{nut_i}], fixed/bolt=[{bolt_i}]")
            tags = ([("pre", None)] if "pre_body_q" in data else []) + [("post[-2] (last clean)", -2)]
            for tag, t in tags:
                bq = data["pre_body_q"] if t is None else data["body_q"][t]
                rp = _quat_rel_pose(bq[bolt_i], bq[nut_i])
                if rp is not None:
                    p, e = rp
                    print(f"  nut-in-bolt {tag:22s} pos(m)=[{p[0]:+.4f} {p[1]:+.4f} {p[2]:+.4f}] "
                          f"euler(deg)=[{e[0]:+.1f} {e[1]:+.1f} {e[2]:+.1f}]")

    # --- where the NaN lives in the solver (localizes the failure) ----------------
    if has("mjw_nan_summary_keys", "mjw_nan_summary_counts"):
        keys = [str(k) for k in data["mjw_nan_summary_keys"]]
        cnts = data["mjw_nan_summary_counts"]
        order = np.argsort(-cnts)
        hot = [(keys[i], int(cnts[i])) for i in order if cnts[i] > 0]
        print("\nmjwarp NaN localization (array -> #NaN in NaN'd world):")
        if hot:
            for k, c in hot:
                print(f"  {k:28s} {c}")
            print("  ^ earliest in the chain (contact.* -> efc.* -> qacc/qfrc) is the origin")
        else:
            print("  (no NaN in dumped solver arrays — NaN may be in an undumped field)")
    if "mjw_solver_niter" in data:
        ni = np.asarray(data["mjw_solver_niter"]).reshape(-1)
        print(f"solver_niter (NaN world): {ni.tolist()}  (high/maxed => failed to converge)")

    # --- contacts / forces that fed the failing solve ----------------------------
    for pfx, label in (("contact", "live post-step"), ("pre_contact", "pre-state/input")):
        if f"{pfx}_count_env" in data:
            print(f"\n{label} Newton contacts on NaN env: {int(data[f'{pfx}_count_env'])}"
                  f" (of {int(data.get(f'{pfx}_count_total', -1))} total)")
            if f"{pfx}_margin0" in data and f"{pfx}_margin1" in data and len(data[f"{pfx}_margin0"]):
                sep = data[f"{pfx}_margin0"] + data[f"{pfx}_margin1"]
                print(f"  separation(margin0+1): min={np.nanmin(sep):+.5f} max={np.nanmax(sep):+.5f} "
                      "(neg => interpenetration)")
    if "mjw_contact_dist" in data and len(np.asarray(data["mjw_contact_dist"]).reshape(-1)):
        dist = np.asarray(data["mjw_contact_dist"]).reshape(-1)
        print(f"\nmjw contact.dist (neg=penetration), NaN world: n={dist.size} "
              f"min={np.nanmin(dist):+.5f} (deepest penetration) finite={np.isfinite(dist).sum()}")
    if "mjw_efc_force" in data:
        f = np.asarray(data["mjw_efc_force"]).reshape(-1)
        fin = f[np.isfinite(f)]
        print(f"mjw efc.force (constraint forces), NaN world: n={f.size} "
              f"max|finite|={np.abs(fin).max() if fin.size else float('nan'):.3e} "
              f"#NaN={int(np.isnan(f).sum())}")

    # The active (force-bearing) solver contact(s), identified directly by Newton shape
    # label — answers "which pair is the failing contact" with no frame-matching inference.
    if "mjw_contact_efc_address" in data and "mjw_contact_newton_shape0" in data:
        sl = [str(x) for x in data["shape_label"]] if "shape_label" in data else []

        def _lbl(i):
            i = int(i)
            if 0 <= i < len(sl):
                p = sl[i].split("/")
                return "/".join(p[-3:-1]) if len(p) > 2 else sl[i]
            return f"shape{i}"

        addr = np.asarray(data["mjw_contact_efc_address"])
        active = (addr.reshape(addr.shape[0], -1) >= 0).any(1) if addr.size else np.array([], bool)
        ns0 = np.asarray(data["mjw_contact_newton_shape0"]).reshape(-1)
        ns1 = np.asarray(data["mjw_contact_newton_shape1"]).reshape(-1)
        dist = np.asarray(data.get("mjw_contact_dist", [])).reshape(-1)
        idxs = np.where(active)[0]
        print("\nACTIVE (force-bearing) solver contact(s) — the failing pair, by Newton shape:")
        if len(idxs):
            for ci in idxs:
                dd = f"  dist={dist[ci]:+.2e}m" if ci < len(dist) else ""
                print(f"  {_lbl(ns0[ci]):28s} <-> {_lbl(ns1[ci]):28s}{dd}")
        else:
            print("  (no contact flagged active via efc_address)")


def _plot(npz_path: str, output_dir: str | None) -> None:
    """Generate matplotlib plots of the state trajectories."""
    import numpy as np

    data = np.load(npz_path, allow_pickle=True)
    n = int(data.get("buffer_size", data["joint_q"].shape[0] if "joint_q" in data else 0))

    try:
        import matplotlib

        if output_dir:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping --plot", file=sys.stderr)
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    t = np.arange(n)

    if "body_q" in data:
        bq = data["body_q"]
        if bq.ndim == 3:
            pos = bq[:, 0, :3]
        elif bq.ndim == 2 and bq.shape[1] >= 3:
            pos = bq[:, :3]
        else:
            pos = None
        if pos is not None:
            for i, label in enumerate("xyz"):
                axes[0].plot(t, pos[:, i], label=label)
            axes[0].set_ylabel("Body 0 position [m]")
            axes[0].legend(loc="upper right")
            axes[0].set_title("First body position (world)")
            axes[0].grid(True, alpha=0.3)

    if "joint_qd" in data:
        jqd = data["joint_qd"]
        ndof = min(3, jqd.shape[1])
        for i in range(ndof):
            axes[1].plot(t, jqd[:, i], label=f"qd[{i}]")
        axes[1].set_ylabel("Joint velocity")
        axes[1].set_xlabel("Step (last = NaN incident)")
        axes[1].legend(loc="upper right")
        axes[1].set_title("First 3 joint velocities")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "nan_replay_plot.png")
        plt.savefig(out_path, dpi=150)
        print(f"\nPlot saved to {out_path}")
    else:
        plt.show()


def _find_usd_path(npz_path: str) -> str | None:
    """Find the companion .usd file for a .npz export."""
    base = os.path.splitext(npz_path)[0]
    for ext in (".usd", ".usda", ".usdc"):
        candidate = base + ext
        if os.path.isfile(candidate):
            return candidate
    return None


def _prepare_stage(usd_path: str):
    """Open the exported USD and fix ``over`` ancestors so the stage composes.

    The NaN exporter uses ``Sdf.CopySpec`` which preserves the original
    specifier.  Ancestor prims (``/World``, ``/World/envs``) are stored as
    ``over`` because they were overrides on the live stage.  Converting them
    to ``def`` lets USD compose and traverse the env subtree.

    Returns:
        (stage, env_root_path) — the opened stage and the path to the
        environment prim (e.g. ``/World/envs/env_1303``).
    """
    import re

    from pxr import Sdf, Usd

    stage = Usd.Stage.Open(usd_path)
    layer = stage.GetRootLayer()

    for ancestor in ("/World", "/World/envs"):
        spec = layer.GetPrimAtPath(ancestor)
        if spec and spec.specifier == Sdf.SpecifierOver:
            spec.specifier = Sdf.SpecifierDef

    env_re = re.compile(r"/World/envs/(env_\d+)$")
    env_root = None

    def _visitor(path):
        nonlocal env_root
        if env_root is None and env_re.match(str(path)):
            env_root = str(path)

    layer.Traverse(Sdf.Path.absoluteRootPath, _visitor)
    return stage, env_root


def _replay_newton(npz_path: str, speed: float, loop: bool) -> None:
    """Replay state snapshots in the Newton ViewerGL.

    Builds a Newton Model from the exported USD scene, creates a State, then
    steps through each recorded snapshot writing body_q/body_qd/joint_q/joint_qd
    into the state and rendering each frame.
    """
    import numpy as np
    import warp as wp
    from newton import ModelBuilder
    from newton.viewer import ViewerGL
    from pxr import UsdGeom

    data = np.load(npz_path, allow_pickle=True)
    n_steps = int(data.get("buffer_size", 0))
    if n_steps == 0 and "joint_q" in data:
        n_steps = data["joint_q"].shape[0]

    usd_path = _find_usd_path(npz_path)
    if usd_path is None:
        print(
            "No companion .usd file found next to the .npz.\n"
            "The Newton viewer requires the exported scene USD to build a Model.\n"
            "Re-run training with the NaN replay buffer enabled to export both .npz and .usd.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Building Newton model from: {usd_path}")
    stage, env_root = _prepare_stage(usd_path)
    up_axis = UsdGeom.GetStageUpAxis(stage)

    builder = ModelBuilder(up_axis=up_axis)
    if env_root:
        builder.add_usd(stage, root_path=env_root, load_visual_shapes=False)
    else:
        builder.add_usd(stage, load_visual_shapes=False)
    # Visualization needs geometry only, not physics. The captured Factory scene
    # carries NewtonSDFCollisionAPI on the nut/bolt: building those texture SDFs
    # requires CUDA (irrelevant to rendering), and a blanket convex-hull would raise
    # on SDF shapes. Strip SDF / hydroelastic intent so the model finalizes on CPU
    # with the raw collision meshes.
    from newton._src.geometry.flags import ShapeFlags

    for _i in range(len(builder.shape_sdf_max_resolution)):
        builder.shape_sdf_max_resolution[_i] = None
        builder.shape_sdf_target_voxel_size[_i] = None
        builder.shape_sdf_padding[_i] = None
        builder.shape_flags[_i] = int(builder.shape_flags[_i]) & ~int(ShapeFlags.HYDROELASTIC)
    model = builder.finalize(device="cpu")
    state = model.state()

    if model.body_count == 0:
        print("Warning: model has 0 bodies. The USD may not contain resolvable physics prims.", file=sys.stderr)

    body_q_all = data.get("body_q")
    body_qd_all = data.get("body_qd")
    joint_q_all = data.get("joint_q")
    joint_qd_all = data.get("joint_qd")
    sim_time_at_export = float(data.get("sim_time", 0.0))

    dt = 1.0 / 60.0
    env_ids = data["exported_env_ids"].tolist() if "exported_env_ids" in data else []
    print(f"Model: {model.body_count} bodies, {model.joint_count} joints")

    viewer = ViewerGL(width=1920, height=1080, headless=False)
    viewer.set_model(model)
    viewer.set_world_offsets((0.0, 0.0, 0.0))
    viewer.up_axis = 2  # Z-up

    # Work around Newton ViewerGL imgui color_edit3 incompatibility (expects
    # ImVec4 but receives a plain tuple).  Patch before the first frame so we
    # never leave imgui in a half-finished state.
    viewer._render_left_panel = lambda: None

    # Point camera at the robot using the first frame's body positions.
    if body_q_all is not None and body_q_all.shape[0] > 0:
        first_frame = body_q_all[0]
        valid_mask = np.isfinite(first_frame).all(axis=-1)
        if valid_mask.any():
            positions = first_frame[valid_mask, :3]
            center = positions.mean(axis=0)
            extent = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
            cam_dist = max(extent * 2.0, 1.5)
            # Use the camera's own vec type (pyglet Vec3), NOT wp.vec3: Newton's
            # camera does PyVec3 arithmetic internally (e.g. sync_pivot_to_view on
            # mouse-drag), so a wp.vec3 here crashes the first time you orbit.
            from pyglet.math import Vec3 as _PyVec3

            viewer.camera.pos = _PyVec3(
                float(center[0]) + cam_dist * 0.5,
                float(center[1]) - cam_dist * 0.5,
                float(center[2]) + cam_dist * 0.4,
            )
            # look_at() sets yaw/pitch and the orbit pivot using the camera's math.
            viewer.camera.look_at((float(center[0]), float(center[1]), float(center[2])))
            print(f"Camera target: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")

    print(f"Replaying {n_steps} snapshots at {speed}x speed (loop={loop})")
    if env_ids:
        print(f"Exported env_id(s): {env_ids}")
    print("Press ESC in the viewer window to exit.")

    frame_delay = dt / speed
    sim_time = sim_time_at_export - n_steps * dt

    def _assign_if_compatible(target, frame_data, dtype):
        """Assign frame data to a warp state array if shapes are compatible."""
        if target is None or frame_data is None:
            return
        if frame_data.shape[0] == target.shape[0]:
            target.assign(wp.array(frame_data, dtype=dtype, device="cpu"))

    running = True
    while running and viewer.is_running():
        for step_idx in range(n_steps):
            if not viewer.is_running():
                running = False
                break

            if body_q_all is not None and step_idx < body_q_all.shape[0]:
                _assign_if_compatible(state.body_q, body_q_all[step_idx], wp.transform)
            if body_qd_all is not None and step_idx < body_qd_all.shape[0]:
                _assign_if_compatible(state.body_qd, body_qd_all[step_idx], wp.spatial_vector)
            if joint_q_all is not None and step_idx < joint_q_all.shape[0]:
                _assign_if_compatible(state.joint_q, joint_q_all[step_idx], float)
            if joint_qd_all is not None and step_idx < joint_qd_all.shape[0]:
                _assign_if_compatible(state.joint_qd, joint_qd_all[step_idx], float)

            sim_time += dt
            viewer.begin_frame(sim_time)
            viewer.log_state(state)
            viewer.end_frame()

            time.sleep(frame_delay)

        if not loop:
            print("Replay complete. Close the viewer window to exit.")
            while viewer.is_running():
                viewer.begin_frame(sim_time)
                viewer.log_state(state)
                viewer.end_frame()
                time.sleep(frame_delay)
            break


def main():
    parser = argparse.ArgumentParser(
        description="Load NaN replay npz and print summary, plot, or replay in Newton viewer."
    )
    parser.add_argument(
        "npz_path",
        type=str,
        help="Path to the nan_replay_*.npz file exported when NaN was detected.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot body_q positions (first body, xyz) and joint_qd (first 3 dof) over time.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save plot images. If not set, plots are shown interactively.",
    )
    parser.add_argument(
        "--visualizer",
        type=str,
        default=None,
        choices=["newton"],
        help="Visualizer backend for replay. Currently supports: newton.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (e.g. 0.25 for quarter speed). Default: 1.0.",
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="Play the replay only once instead of looping.",
    )
    args = parser.parse_args()

    _print_summary(args.npz_path)

    if args.plot:
        _plot(args.npz_path, args.output_dir)

    if args.visualizer == "newton":
        _replay_newton(args.npz_path, speed=args.speed, loop=not args.no_loop)


if __name__ == "__main__":
    main()
