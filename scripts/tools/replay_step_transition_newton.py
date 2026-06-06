# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay captured Newton/MJWarp transitions without IsaacLab.

This script uses only Newton/MJWarp plus the exported ``nan_replay_*.usd`` and
``nan_replay_*.npz`` files.  It builds a single-env Newton model from the USD,
injects replay frame pre-state/control/contact data, runs solver substeps, and
compares the output with the recorded replay post-state.

Repro (faithful stable-trajectory replay of frames 0-6; uses current npz + companion usd).
CUDA_VISIBLE_DEVICES picks the GPU (use a free one; --device cuda:0 maps to it):
    CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p scripts/tools/replay_step_transition_newton.py \\
        /tmp/factory_bench/nan_debug/nan_replay_20260605_050616_958767.npz \\
        --start_frame 0 --end_frame 7 --contact_must_involve NUT_M16 --device cuda:0 \\
        --njmax 1200000 --nconmax 2000000 --rigid_contact_max 4000000

Expected: frames 0-6 -> actual_nan=0 with body_q max_abs_diff ~1e-5..1e-4 (faithful);
frame 7 -> actual_nan=0 vs expected_nan=87 (the isolated per-frame solve stays finite; the
literal NaN only occurs in the full 8192-world batched run).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import warp as wp


def _find_usd_path(npz_path: str) -> str | None:
    base = os.path.splitext(npz_path)[0]
    for ext in (".usd", ".usda", ".usdc"):
        candidate = base + ext
        if os.path.isfile(candidate):
            return candidate
    return None


def _prepare_stage(usd_path: str):
    import re

    from pxr import Sdf, Usd

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {usd_path}")
    layer = stage.GetRootLayer()
    for ancestor in ("/World", "/World/envs"):
        spec = layer.GetPrimAtPath(ancestor)
        if spec and spec.specifier == Sdf.SpecifierOver:
            spec.specifier = Sdf.SpecifierDef

    env_re = re.compile(r"/World/envs/(env_\\d+)$")
    env_root = None

    def _visitor(path):
        nonlocal env_root
        if env_root is None and env_re.match(str(path)):
            env_root = str(path)

    layer.Traverse(Sdf.Path.absoluteRootPath, _visitor)
    return stage, env_root


def _torch_assign(dst, value: np.ndarray) -> None:
    dst_t = wp.to_torch(dst)
    src_t = torch.as_tensor(value, dtype=dst_t.dtype, device=dst_t.device)
    if tuple(dst_t.shape) != tuple(src_t.shape):
        raise ValueError(f"Shape mismatch: destination {tuple(dst_t.shape)} vs source {tuple(src_t.shape)}")
    dst_t.copy_(src_t)


def _torch_assign_row(dst, value: np.ndarray, row: int = 0) -> None:
    dst_t = wp.to_torch(dst)
    src_t = torch.as_tensor(value, dtype=dst_t.dtype, device=dst_t.device)
    if tuple(dst_t[row].shape) != tuple(src_t.shape):
        raise ValueError(f"Shape mismatch: destination row {tuple(dst_t[row].shape)} vs source {tuple(src_t.shape)}")
    dst_t[row].copy_(src_t)


def _build_model(usd_path: str, device: str):
    from newton import ModelBuilder
    from pxr import UsdGeom

    stage, env_root = _prepare_stage(usd_path)
    up_axis = UsdGeom.GetStageUpAxis(stage) or "Z"
    builder = ModelBuilder(up_axis=up_axis)
    if env_root:
        builder.add_usd(stage, root_path=env_root, load_visual_shapes=False)
    else:
        builder.add_usd(stage, load_visual_shapes=False)
    model = builder.finalize(device=device)
    return model


def _make_solver(model, args):
    from newton.solvers import SolverMuJoCo

    return SolverMuJoCo(
        model,
        solver=args.solver,
        integrator=args.integrator,
        njmax=args.njmax,
        nconmax=args.nconmax,
        impratio=args.impratio,
        cone=args.cone,
        update_data_interval=args.update_data_interval,
        iterations=args.iterations,
        ls_iterations=args.ls_iterations,
        ls_parallel=args.ls_parallel,
        use_mujoco_contacts=False,
    )


def _set_shape_material(model, ke: float, kd: float, mu: float, gap: float, margin: float, verbose: bool = True) -> None:
    """Set the rebuilt model's shape material + contact gap/margin to the Factory runtime values.

    The single-env USD does NOT carry the Factory ke/kd/friction/gap/margin (those are applied
    at runtime via ``NewtonShapeCfg``), so a rebuilt model has default material AND a default
    contact gap of 0. Two consequences:
      * The Newton->mjw conversion derives ``geom_solref``/friction from the shape material
        (``convert_solref(ke, kd, ...)``); default material gives ``solref=[0.02,1.0]`` / mu=1.0
        instead of the captured ``[2e-4,1.581]`` / 0.6923.
      * Contact activation uses ``includemargin`` (from gap/margin); with gap=0 a contact at
        ``dist~1e-4`` (>0) is INACTIVE, so the bolt-nut contacts generate no efc rows (nefc=4)
        and the constraint solve that NaNs in the original never even forms. Factory gap=0.001
        makes those contacts active.
    Must be set BEFORE building the solver (geom_* are computed at solver-build time).
    """
    set_attrs = []
    for attr, val in (
        ("shape_material_ke", ke),
        ("shape_material_kd", kd),
        ("shape_material_mu", mu),
        ("shape_gap", gap),
        ("shape_margin", margin),
    ):
        arr = getattr(model, attr, None)
        if arr is not None:
            wp.to_torch(arr)[:] = float(val)
            set_attrs.append(attr)
    if verbose:
        print(f"  [material] set ke={ke:.3g} kd={kd:.3g} mu={mu:.4g} gap={gap:.4g} margin={margin:.4g} on {set_attrs}")


def _restore_mjw_model(solver, data: np.lib.npyio.NpzFile, verbose: bool = True) -> None:
    """Overwrite the rebuilt model's solver-level constants with the captured ones.

    The single-env USD rebuild does not reliably preserve the kinematic pinning
    (``dof_armature`` of ``1e10`` on the Table/NistBoard/Bolt free joints) nor the
    true body masses/inertias (note the "invalid inertia -> sphere approximation"
    warnings on Table/NistBoard). We restore those exact arrays from the capture so
    the mass matrix and pinning match the original run bit-for-bit.
    """
    mjw = solver.mjw_model
    restored = []
    field_to_key = {
        "dof_armature": "mjw_model_dof_armature",
        "body_mass": "mjw_model_body_mass",
        "body_inertia": "mjw_model_body_inertia",
        "body_invweight0": "mjw_model_body_invweight0",
        "dof_invweight0": "mjw_model_dof_invweight0",
    }
    for field, key in field_to_key.items():
        if key not in data or not hasattr(mjw, field):
            continue
        captured = np.asarray(data[key])
        target = getattr(mjw, field)
        target_t = wp.to_torch(target)
        # mjw_model arrays may be [N] (single world) or [nworld, N]. Broadcast the
        # captured single-world array across the leading world dim if present.
        src = torch.as_tensor(captured, dtype=target_t.dtype, device=target_t.device)
        if target_t.shape == src.shape:
            target_t.copy_(src)
        elif target_t.dim() == src.dim() + 1 and tuple(target_t.shape[1:]) == tuple(src.shape):
            target_t.copy_(src.unsqueeze(0).expand_as(target_t))
        elif target_t.numel() == src.numel():
            target_t.copy_(src.reshape(target_t.shape))
        else:
            if verbose:
                print(f"  [restore] SKIP {field}: shape {tuple(target_t.shape)} vs captured {tuple(src.shape)}")
            continue
        restored.append(field)
    if verbose:
        print(f"  [restore] mjw_model fields restored from capture: {restored}")


_NEWTON_JOINT_QPOS_PER_TYPE = {0: 1, 1: 1, 2: 4, 3: 0, 4: 7}  # PRISMATIC, REVOLUTE, BALL, FIXED, FREE


def _free_joint_pos_indices(data: np.lib.npyio.NpzFile, qpos_len: int) -> list[int]:
    """Indices in joint_q that hold free-joint linear positions (the first 3 of each FREE joint).

    These are the only joint_q entries expressed in env world frame; everything else
    (revolute/prismatic) is frame-invariant. Derived from the Newton joint-type list.
    """
    if "nmodel_joint_type" not in data:
        return []
    jtypes = np.asarray(data["nmodel_joint_type"]).reshape(-1)
    out: list[int] = []
    off = 0
    for jt in jtypes:
        n = _NEWTON_JOINT_QPOS_PER_TYPE.get(int(jt), 0)
        if int(jt) == 4:  # FREE joint -> first 3 qpos are linear position
            out.extend([off, off + 1, off + 2])
        off += n
        if off >= qpos_len:
            break
    return out


def _make_contacts(model, args):
    import newton

    pipeline = newton.CollisionPipeline(
        model,
        broad_phase=args.broad_phase,
        max_triangle_pairs=args.max_triangle_pairs,
        rigid_contact_max=args.rigid_contact_max,
    )
    contacts = pipeline.contacts()
    return pipeline, contacts


def _short_shape_label(label: str) -> str:
    for token in (
        "BOLT_M16",
        "NUT_M16",
        "NistBoard",
        "Table",
        "panda_link0",
        "panda_link1",
        "panda_link2",
        "panda_link3",
        "panda_link4",
        "panda_link5",
        "panda_link6",
        "panda_link7",
        "panda_hand",
        "panda_leftfinger",
        "panda_rightfinger",
        "force_sensor",
    ):
        if token in label:
            return "/".join(part for part in label.split("/") if token in part or part.startswith("collisions") or part.startswith("collision"))
    if "GroundPlane" in label:
        return "Ground"
    return label.split("/")[-1]


def _shape_remap(data: np.lib.npyio.NpzFile, model) -> dict[int, int]:
    captured_labels = [str(x) for x in data["shape_label"]] if "shape_label" in data else []
    model_labels = [str(x) for x in model.shape_label]
    model_by_short: dict[str, int] = {}
    for idx, label in enumerate(model_labels):
        model_by_short.setdefault(_short_shape_label(label), idx)
    out: dict[int, int] = {}
    for idx, label in enumerate(captured_labels):
        key = _short_shape_label(label)
        if key in model_by_short:
            out[idx] = model_by_short[key]
    return out


def _body_remap(data: np.lib.npyio.NpzFile, model) -> dict[int, int]:
    captured_labels = [str(x) for x in data["body_label"]]
    model_labels = [str(x) for x in model.body_label]
    model_by_short: dict[str, int] = {}
    for idx, label in enumerate(model_labels):
        model_by_short.setdefault(_short_shape_label(label), idx)
    out: dict[int, int] = {}
    for idx, label in enumerate(captured_labels):
        key = _short_shape_label(label)
        if key in model_by_short:
            out[idx] = model_by_short[key]
    return out


def _compute_replay_translation(data: np.lib.npyio.NpzFile, model, default_state, frame: int) -> np.ndarray:
    """Return translation (captured_world - model_world) of the env-grid origin.

    Anchor on the robot BASE (``panda_link0``), which is a fixed joint pinned at the env
    origin and therefore reflects the pure env-grid offset. The Table/NistBoard/Bolt are
    reset-randomized per env, so their captured-vs-default offset is contaminated by that
    randomization (observed ~0.165 m error) and must NOT be used as the frame anchor.
    """
    body_remap = _body_remap(data, model)
    captured_labels = [str(x) for x in data["body_label"]]
    captured_q = np.asarray(data["replay_pre_body_q"][frame])
    model_q = wp.to_torch(default_state.body_q).detach().cpu().numpy()
    # Prefer the fixed robot base; fall back to any robot link, then any matched body.
    for tokens in (("panda_link0",), ("panda_link",), ()):
        offsets = []
        for captured_idx, model_idx in body_remap.items():
            label = captured_labels[captured_idx]
            if not tokens or any(t in label for t in tokens):
                offsets.append(captured_q[captured_idx, :3] - model_q[model_idx, :3])
        if offsets:
            return np.median(np.asarray(offsets), axis=0)
    raise RuntimeError("Could not compute captured-env to replay-USD translation")


def _remap_shape_array(values: np.ndarray, remap: dict[int, int]) -> np.ndarray:
    out = np.asarray(values, dtype=np.int32).copy()
    for i, value in enumerate(out):
        out[i] = remap.get(int(value), int(value))
    return out


def _translate_body_q(body_q: np.ndarray, translation: np.ndarray) -> np.ndarray:
    out = np.asarray(body_q).copy()
    out[..., :3] -= translation
    return out


def _translate_contact_points(name: str, values: np.ndarray, translation: np.ndarray) -> np.ndarray:
    out = np.asarray(values).copy()
    if name in ("rigid_contact_point0", "rigid_contact_point1"):
        out -= translation
    return out


_CONTACT_FIELD_NAMES = (
    "rigid_contact_shape0",
    "rigid_contact_shape1",
    "rigid_contact_point0",
    "rigid_contact_point1",
    "rigid_contact_normal",
    "rigid_contact_offset0",
    "rigid_contact_offset1",
    "rigid_contact_margin0",
    "rigid_contact_margin1",
    "rigid_contact_thickness0",
    "rigid_contact_thickness1",
    "rigid_contact_stiffness",
    "rigid_contact_damping",
    "rigid_contact_friction",
)


def _build_contact_cache(
    data: np.lib.npyio.NpzFile,
    model,
    captured_env: int,
    start_frame: int,
    end_frame: int,
    replay_translation: np.ndarray,
    drop_unmappable: bool = True,
    must_involve: str | None = None,
) -> dict | None:
    if "replay_pre_contact_rigid_contact_count" not in data:
        return None
    remap = _shape_remap(data, model)
    recorded_counts = np.asarray(data["replay_pre_contact_rigid_contact_count"])
    shape_world = np.asarray(data["shape_world"]).reshape(-1)
    shape0_all = np.asarray(data["replay_pre_contact_rigid_contact_shape0"])
    shape1_all = np.asarray(data["replay_pre_contact_rigid_contact_shape1"])
    labels = [str(x) for x in data["shape_label"]] if "shape_label" in data else []

    selected_by_frame = {}
    dropped_pairs = {}
    frames = range(start_frame, end_frame + 1)
    for frame in frames:
        total = int(recorded_counts[frame].reshape(-1)[0])
        shape0 = shape0_all[frame][:total]
        shape1 = shape1_all[frame][:total]
        world0 = np.where((shape0 >= 0) & (shape0 < len(shape_world)), shape_world[np.clip(shape0, 0, len(shape_world) - 1)], -2)
        world1 = np.where((shape1 >= 0) & (shape1 < len(shape_world)), shape_world[np.clip(shape1, 0, len(shape_world) - 1)], -2)
        # Require BOTH shapes to belong to the captured env. A contact with one global shape
        # (e.g. the shared ground plane, shape_world < 0) cannot be faithfully replayed in the
        # single-env model: we translate the env's geometry by the env-grid offset, but the
        # global ground does not move, so a translated ground contact is inconsistent and
        # destabilizes the solve. Such contacts involve pinned bodies and are dynamically inert.
        selected = np.where((world0 == captured_env) & (world1 == captured_env))[0]
        if drop_unmappable and remap:
            # Drop contacts whose shapes are not present in the single-env model (e.g. the
            # global ground plane). Keeping them would leave invalid shape ids in the buffer
            # and corrupt the constraint solve. Pinned-body contacts (Table<->ground) are
            # dynamically inert anyway.
            mappable = np.array(
                [i for i in selected if int(shape0[i]) in remap and int(shape1[i]) in remap],
                dtype=np.int64,
            )
            for i in selected:
                if int(shape0[i]) not in remap or int(shape1[i]) not in remap:
                    a = labels[int(shape0[i])].split("/")[-1] if 0 <= int(shape0[i]) < len(labels) else str(int(shape0[i]))
                    b = labels[int(shape1[i])].split("/")[-1] if 0 <= int(shape1[i]) < len(labels) else str(int(shape1[i]))
                    dropped_pairs[tuple(sorted((a, b)))] = dropped_pairs.get(tuple(sorted((a, b))), 0) + 1
            selected = mappable
        if must_involve:
            # Keep only contacts where at least one shape belongs to the given body token
            # (e.g. "NUT_M16"). The original mjw solve culls pinned<->pinned contacts
            # (e.g. Bolt<->NistBoard), so injecting them adds constraints the real solve never
            # had and over-stabilizes it. Matching the dynamic body's contacts reproduces the
            # original constraint set.
            selected = np.array(
                [i for i in selected if must_involve in labels[int(shape0[i])] or must_involve in labels[int(shape1[i])]],
                dtype=np.int64,
            )
        selected_by_frame[frame] = (total, selected)
    if dropped_pairs:
        print(f"  [contacts] dropped unmappable contact pairs (not in single-env model): {dropped_pairs}")

    fields = {}
    for name in _CONTACT_FIELD_NAMES:
        key = f"replay_pre_contact_{name}"
        if key not in data:
            continue
        all_values = np.asarray(data[key])
        frame_values = {}
        for frame, (total, selected) in selected_by_frame.items():
            values = all_values[frame][:total][selected]
            if name in ("rigid_contact_shape0", "rigid_contact_shape1"):
                values = _remap_shape_array(values, remap)
            # NOTE: contact point0/point1 are NOT translated. Newton stores rigid_contact
            # points in a body/contact-local frame near the origin (verified: recomputed and
            # recorded centroids match to ~1e-4), not env world frame, so the env-grid
            # translation must not be applied to them.
            frame_values[frame] = values.copy()
        fields[name] = frame_values
    return {
        "count_dtype": recorded_counts.dtype,
        "selected_by_frame": selected_by_frame,
        "fields": fields,
        "contact_counts": {frame: int(len(selected)) for frame, (_, selected) in selected_by_frame.items()},
    }


def _assign_contact_buffer(
    contacts, contact_cache: dict | None, frame: int, material: dict | None = None
) -> tuple[bool, int, int]:
    if contact_cache is None:
        return False, 0, 0
    total, selected = contact_cache["selected_by_frame"][frame]
    n = int(len(selected))
    if n > contacts.rigid_contact_max:
        raise ValueError(f"Recorded contact count {n} exceeds replay contact capacity {contacts.rigid_contact_max}")

    def assign(name: str, values: np.ndarray) -> None:
        target = getattr(contacts, name)
        target_t = wp.to_torch(target)
        src = torch.as_tensor(values, dtype=target_t.dtype, device=target_t.device)
        target_t[: src.shape[0]].copy_(src)

    count = np.array([n], dtype=contact_cache["count_dtype"])
    assign("rigid_contact_count", count)
    present = set(contact_cache["fields"].keys())
    for name, frame_values in contact_cache["fields"].items():
        assign(name, frame_values[frame])
    # The capture does not record contact material (stiffness/damping/friction). Fill them
    # with the known Factory values so the Newton->mjw conversion produces the original
    # solref/friction; otherwise the injected contacts have zero stiffness (degenerate).
    if material and n > 0:
        fill = {
            "rigid_contact_stiffness": material.get("ke"),
            "rigid_contact_damping": material.get("kd"),
            "rigid_contact_friction": material.get("friction"),
        }
        for name, val in fill.items():
            if val is None or name in present or not hasattr(contacts, name):
                continue
            arr = getattr(contacts, name)
            if arr is None:
                continue  # array not allocated on this buffer; nothing to fill
            target_t = wp.to_torch(arr)
            target_t[:n] = float(val)
    return True, n, total


def _translate_joint_q(joint_q: np.ndarray, translation: np.ndarray, free_pos_idx: list[int]) -> np.ndarray:
    """Translate the free-joint linear-position entries of joint_q into the model env frame.

    With ``update_data_interval>=1`` joint_q (not body_q) is what reaches ``mjw_data.qpos``,
    so the env-frame offset must be applied here. Only FREE-joint position triples move;
    revolute/prismatic angles and all quaternions are left untouched.
    """
    out = np.asarray(joint_q).copy()
    if free_pos_idx:
        for k in range(0, len(free_pos_idx), 3):
            triple = free_pos_idx[k : k + 3]
            out[triple] -= translation
    return out


def _assign_pre_state(state, data, frame: int, replay_translation: np.ndarray, free_pos_idx: list[int]) -> None:
    _torch_assign(state.body_q, _translate_body_q(data["replay_pre_body_q"][frame], replay_translation))
    _torch_assign(state.body_qd, data["replay_pre_body_qd"][frame])
    _torch_assign(state.joint_q, _translate_joint_q(data["replay_pre_joint_q"][frame], replay_translation, free_pos_idx))
    _torch_assign(state.joint_qd, data["replay_pre_joint_qd"][frame])


def _assign_pre_control_and_solver(control, solver, data, frame: int, captured_env: int, include_solver_vectors: bool = True) -> None:
    if "replay_pre_control_joint_f" in data:
        joint_f = data["replay_pre_control_joint_f"][frame]
        if joint_f.ndim == 1 and joint_f.shape[0] != control.joint_f.shape[0]:
            n_dof = control.joint_f.shape[0]
            joint_f = joint_f[captured_env * n_dof : (captured_env + 1) * n_dof]
        _torch_assign(control.joint_f, joint_f)

    if not include_solver_vectors:
        # Open-loop chained frames: apply only the recorded control and let the solver's
        # qpos/qvel/qacc_warmstart carry forward from the previous step's output.
        return

    mjw = solver.mjw_data
    for name in ("qacc", "qvel", "qacc_warmstart", "qacc_smooth", "qfrc_smooth", "qfrc_constraint"):
        key = f"replay_pre_mjw_{name}"
        if key not in data:
            continue
        arr = data[key][frame]
        if arr.ndim == 2:
            arr = arr[captured_env]
        _torch_assign_row(getattr(mjw, name), arr, 0)


def _assign_pre_frame(
    state, control, solver, data, frame: int, captured_env: int, replay_translation: np.ndarray, free_pos_idx: list[int]
) -> None:
    _assign_pre_state(state, data, frame, replay_translation, free_pos_idx)
    _assign_pre_control_and_solver(control, solver, data, frame, captured_env)


def _compare_array(name: str, actual: np.ndarray, expected: np.ndarray, prefix: str = "") -> dict:
    finite = np.isfinite(actual) & np.isfinite(expected)
    actual_nan = int(np.isnan(actual).sum()) if np.issubdtype(actual.dtype, np.floating) else 0
    expected_nan = int(np.isnan(expected).sum()) if np.issubdtype(expected.dtype, np.floating) else 0
    max_diff = float(np.max(np.abs(actual[finite] - expected[finite]))) if finite.any() else float("nan")
    print(f"{prefix}{name}: actual_nan={actual_nan} expected_nan={expected_nan} max_abs_diff={max_diff:.6e}")
    return {
        "name": name,
        "actual_nan": actual_nan,
        "expected_nan": expected_nan,
        "max_abs_diff": max_diff,
    }


def _compare_post(
    state,
    solver,
    data,
    frame: int,
    captured_env: int,
    replay_translation: np.ndarray,
    free_pos_idx: list[int],
    prefix: str = "",
) -> list[dict]:
    results = [
        _compare_array(
            "body_q",
            wp.to_torch(state.body_q).detach().cpu().numpy(),
            _translate_body_q(data["replay_post_body_q"][frame], replay_translation),
            prefix,
        ),
        _compare_array("body_qd", wp.to_torch(state.body_qd).detach().cpu().numpy(), data["replay_post_body_qd"][frame], prefix),
        _compare_array(
            "joint_q",
            wp.to_torch(state.joint_q).detach().cpu().numpy(),
            _translate_joint_q(data["replay_post_joint_q"][frame], replay_translation, free_pos_idx),
            prefix,
        ),
        _compare_array("joint_qd", wp.to_torch(state.joint_qd).detach().cpu().numpy(), data["replay_post_joint_qd"][frame], prefix),
    ]
    mjw = solver.mjw_data
    for name in ("qacc", "qvel", "qacc_warmstart", "qacc_smooth", "qfrc_smooth", "qfrc_constraint"):
        key = f"replay_post_mjw_{name}"
        if key not in data:
            continue
        expected = data[key][frame]
        if expected.ndim == 2:
            expected = expected[captured_env]
        actual = wp.to_torch(getattr(mjw, name))[0].detach().cpu().numpy()
        results.append(_compare_array(f"mjw_{name}", actual, expected, prefix))
    return results


def _replay_one_frame(
    state,
    control,
    solver,
    pipeline,
    contacts,
    data,
    frame: int,
    captured_env: int,
    model,
    contact_cache: dict | None,
    replay_translation: np.ndarray,
    free_pos_idx: list[int],
    material: dict | None,
    dt: float,
    inject_state: bool,
    skip_compare: bool,
) -> list[dict]:
    t0 = time.perf_counter()
    if inject_state:
        _assign_pre_frame(state, control, solver, data, frame, captured_env, replay_translation, free_pos_idx)
    else:
        # Open-loop chained frame: only apply control; state + warmstart carry forward.
        _assign_pre_control_and_solver(control, solver, data, frame, captured_env, include_solver_vectors=False)

    # Always run collision first: it computes contacts from the (injected) geometry AND lazily
    # allocates the contact buffer's material arrays (stiffness/damping/friction). For the
    # recorded path we then overwrite the geometry + material with the captured values; the
    # collide call ensures those material arrays exist so they can be filled.
    pipeline.collide(state, contacts)
    used_recorded_contacts, contact_count, total_contact_count = _assign_contact_buffer(contacts, contact_cache, frame, material)
    if used_recorded_contacts:
        print(f"Frame {frame}: overwrote recomputed buffer with {contact_count} recorded contacts (of {total_contact_count}) for env {captured_env}.")
    t1 = time.perf_counter()
    solver.step(state, state, control, contacts, dt)
    state.clear_forces()
    t2 = time.perf_counter()

    if os.environ.get("FACTORY_DUMP_MJW_CONTACTS") == "1":
        d = solver.mjw_data
        ncon = int(wp.to_torch(d.ncon)[0].item()) if hasattr(d, "ncon") else -1
        print(f"  [mjw] frame {frame} ncon={ncon} nefc={int(wp.to_torch(d.nefc)[0].item()) if hasattr(d,'nefc') else -1}")
        for fld in ("solref", "friction", "dist"):
            if hasattr(d.contact, fld):
                arr = wp.to_torch(getattr(d.contact, fld)).detach().cpu().numpy()
                n = max(ncon, 0)
                print(f"    contact.{fld}[:3] = {np.array2string(arr[:min(3, n if n>0 else 3)], precision=5, max_line_width=160)}")

    if skip_compare:
        print(f"Frame {frame} timing: prepare={t1 - t0:.3f}s solve={t2 - t1:.3f}s compare=0.000s")
        return []
    results = _compare_post(state, solver, data, frame, captured_env, replay_translation, free_pos_idx, prefix=f"frame {frame} ")
    t3 = time.perf_counter()
    print(f"Frame {frame} timing: prepare={t1 - t0:.3f}s solve={t2 - t1:.3f}s compare={t3 - t2:.3f}s")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz_path")
    parser.add_argument("--frame", type=int, default=-1)
    parser.add_argument("--start_frame", type=int, default=None)
    parser.add_argument("--end_frame", type=int, default=None)
    parser.add_argument(
        "--open_loop",
        action="store_true",
        help="Inject only the first frame's pre-state, then chain subsequent solver outputs.",
    )
    parser.add_argument(
        "--reuse_solver",
        action="store_true",
        help="Reuse solver/data across injected frames. Faster, but NaNs can contaminate later frames.",
    )
    parser.add_argument(
        "--no_align",
        action="store_true",
        help="Disable translating captured world-space poses/contacts into the rebuilt-USD env frame. "
        "Alignment is ON by default and is required for faithful replay (the franka base is fixed at the "
        "USD env_0 position while the captured free joints/contacts live in the original env's world frame).",
    )
    parser.add_argument(
        "--no_restore_model",
        action="store_true",
        help="Disable overwriting the rebuilt model's dof_armature/body_mass/body_inertia with the captured "
        "values. Restore is ON by default and is required to preserve kinematic pinning (1e10 armature) and "
        "true inertias that the single-env USD rebuild does not reproduce.",
    )
    parser.add_argument(
        "--recompute_contacts",
        action="store_true",
        help="Ignore the recorded contact buffer and recompute contacts with the CollisionPipeline from the "
        "(injected, correctly-placed) geometry. Avoids the fragile shape-id remap and is what the real "
        "pipeline does; required for a clean Level-2 open-loop replay.",
    )
    parser.add_argument(
        "--contact_must_involve",
        type=str,
        default=None,
        help="Keep only recorded contacts where a shape label contains this token (e.g. NUT_M16). "
        "Matches the original mjw solve which culls pinned<->pinned contacts.",
    )
    parser.add_argument("--contact_ke", type=float, default=1e7, help="Shape stiffness ke (Factory default 1e7).")
    parser.add_argument("--contact_kd", type=float, default=1e4, help="Shape damping kd (Factory default 1e4).")
    parser.add_argument("--contact_gap", type=float, default=0.001, help="Shape contact gap (Factory default 0.001); activates near-touching contacts.")
    parser.add_argument("--contact_margin", type=float, default=0.0, help="Shape contact margin (Factory default 0.0).")
    parser.add_argument(
        "--contact_friction",
        type=float,
        default=-1.0,
        help="Injected-contact friction coefficient. <0 (default) reads it from the captured "
        "mjw_contact_friction, else falls back to 0.6923.",
    )
    parser.add_argument(
        "--update_data_interval",
        type=int,
        default=1,
        help="SolverMuJoCo update_data_interval. MUST be >=1 for injected joint_q/joint_qd to reach mjw_data "
        "qpos/qvel; 0 (the old default) leaves qpos at the USD rest pose and NaNs on every frame.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--solver", default="newton")
    parser.add_argument("--integrator", default="implicitfast")
    parser.add_argument("--cone", default="elliptic")
    parser.add_argument("--njmax", type=int, default=1200)
    parser.add_argument("--nconmax", type=int, default=300)
    parser.add_argument("--impratio", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--ls_iterations", type=int, default=100)
    parser.add_argument("--ls_parallel", action="store_true")
    parser.add_argument("--broad_phase", default="sap")
    parser.add_argument("--max_triangle_pairs", type=int, default=30_000_000)
    parser.add_argument("--rigid_contact_max", type=int, default=4_000_000)
    parser.add_argument("--skip_compare", action="store_true")
    args = parser.parse_args()

    data = np.load(args.npz_path, allow_pickle=True)
    usd_path = _find_usd_path(args.npz_path)
    if usd_path is None:
        print(f"No companion USD found for {args.npz_path}", file=sys.stderr)
        return 1
    replay_size = int(data["replay_buffer_size"])
    if args.start_frame is not None or args.end_frame is not None:
        start_frame = 0 if args.start_frame is None else args.start_frame
        end_frame = replay_size - 1 if args.end_frame is None else args.end_frame
    else:
        frame = args.frame if args.frame >= 0 else replay_size - 1
        start_frame = frame
        end_frame = frame
    if start_frame < 0 or end_frame >= replay_size or start_frame > end_frame:
        raise ValueError(f"Invalid frame range [{start_frame}, {end_frame}] for replay buffer size {replay_size}")
    captured_env = int(data["exported_env_ids"][0]) if "exported_env_ids" in data else 0

    # Resolve the contact friction (env-randomized, not in the rebuilt model) from the capture.
    friction = args.contact_friction
    if friction < 0:
        friction = 0.6923
        if "mjw_contact_friction" in data.files:
            mf = np.asarray(data["mjw_contact_friction"]).reshape(-1)
            if mf.size:
                friction = float(mf[0])
    material = {"ke": args.contact_ke, "kd": args.contact_kd, "friction": friction}

    print(f"Loading USD: {usd_path}")
    model = _build_model(usd_path, args.device)
    if not args.no_restore_model:
        # Set shape material + gap/margin BEFORE building the solver so geom_solref/friction and
        # contact activation match the Factory runtime (the conversion reads model shape props).
        _set_shape_material(model, material["ke"], material["kd"], material["friction"], args.contact_gap, args.contact_margin)
    state = model.state()
    control = model.control()
    pipeline, contacts = _make_contacts(model, args)
    solver = _make_solver(model, args)
    if not args.no_restore_model:
        _restore_mjw_model(solver, data)
    replay_translation = (
        np.zeros(3, dtype=np.float32)
        if args.no_align
        else _compute_replay_translation(data, model, state, start_frame)
    )
    print(f"Replay translation captured_env->USD frame: {replay_translation.tolist()}")
    free_pos_idx = _free_joint_pos_indices(data, int(np.asarray(data["replay_pre_joint_q"]).shape[1]))
    print(f"Free-joint position indices in joint_q (translated to env frame): {free_pos_idx}")
    print(f"update_data_interval={args.update_data_interval} (must be >=1 for injected state to reach qpos)")
    print(f"Injected-contact material: ke={material['ke']:.3g} kd={material['kd']:.3g} friction={material['friction']:.4g}")
    if args.recompute_contacts:
        print("Recomputing contacts via CollisionPipeline (recorded contact buffer ignored).")
        contact_cache = None
    else:
        print(f"Preparing recorded contact cache for frames {start_frame}:{end_frame} and env {captured_env}.")
        contact_cache = _build_contact_cache(
            data, model, captured_env, start_frame, end_frame, replay_translation, must_involve=args.contact_must_involve
        )

    print(f"Model: bodies={model.body_count} joints={model.joint_count} dofs={model.joint_dof_count}")
    print(f"Replay frames={start_frame}:{end_frame} captured_env={captured_env} open_loop={args.open_loop}")
    if contact_cache is not None:
        print(f"Cached contact counts: {contact_cache['contact_counts']}")
    substeps = data["replay_meta_substep_idx"] if "replay_meta_substep_idx" in data else None
    fresh_solver_per_frame = not args.open_loop and not args.reuse_solver
    all_results: list[dict] = []
    for frame in range(start_frame, end_frame + 1):
        substep = int(substeps[frame]) if substeps is not None else -1
        print(f"Replay frame={frame} substep={substep}")
        if fresh_solver_per_frame:
            state = model.state()
            control = model.control()
            pipeline, contacts = _make_contacts(model, args)
            solver = _make_solver(model, args)
            if not args.no_restore_model:
                _restore_mjw_model(solver, data, verbose=False)
        inject_state = (not args.open_loop) or frame == start_frame
        all_results.extend(
            _replay_one_frame(
                state,
                control,
                solver,
                pipeline,
                contacts,
                data,
                frame,
                captured_env,
                model,
                contact_cache,
                replay_translation,
                free_pos_idx,
                material,
                args.dt,
                inject_state,
                args.skip_compare,
            )
        )
    if all_results:
        nan_mismatches = [
            result for result in all_results if result["actual_nan"] != result["expected_nan"]
        ]
        finite_mismatches = [
            result for result in all_results if result["max_abs_diff"] != 0.0 and np.isfinite(result["max_abs_diff"])
        ]
        print(
            "Summary:"
            f" compared={len(all_results)}"
            f" nan_count_mismatches={len(nan_mismatches)}"
            f" finite_value_mismatches={len(finite_mismatches)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
