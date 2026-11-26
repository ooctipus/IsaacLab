# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Identical-state probes and physical diagnostics, with explicit unsupported rows."""

import time
from pathlib import Path

import newton
import numpy as np
import torch
import warp as wp
from experiments.neio import SolverNeIO
from experiments.neio.deformable import element_edges
from experiments.neio.rigid import quat_rotate
from optimizer import StudySolver


def make_solver(scene, name, sweeps, network=None, history="clear"):
    model = scene.model
    if name == "shared":
        return StudySolver(model, network, iterations=sweeps, history=history, **scene.options)
    if name == "supplied":
        root = Path(__file__).parent / "_source/experiments/neio-reference/checkpoints"
        options = {"history_at_step_start": history, "group_by": "world", "inference": "torch"}
        rigid = dict(options, checkpoint=str(root / "rigid_e/weights_270000.pt")) if model.body_count else None
        particle = (
            dict(options, checkpoint=str(root / "particle_local_iso_500k/weights_500000.pt"))
            if model.particle_count
            else None
        )
        return SolverNeIO(
            model,
            iterations=sweeps,
            **scene.options,
            rigid_optimizer_options=rigid,
            particle_optimizer_options=particle,
        )
    if name == "vbd":
        return newton.solvers.SolverVBD(
            model,
            iterations=sweeps,
            rigid_compliant_alm=True,
            rigid_body_contact_buffer_size=1024,
            rigid_body_particle_contact_buffer_size=2048,
            **scene.options,
        )
    if name == "pgs":
        raise NotImplementedError("The pinned Newton MJWarp adapter accepts Newton/CG only; PGS is unsupported")
    if name == "mjwarp":
        if model.particle_count or scene.case["family"] == "cable":
            raise NotImplementedError("MJWarp does not implement this particle/rod objective")
        return newton.solvers.SolverMuJoCo(
            model, solver="newton", iterations=sweeps, use_mujoco_contacts=False, njmax=4096, nconmax=2048
        )
    raise ValueError(name)


def snapshot(state):
    return {
        name: wp.clone(value)
        for name in ("particle_q", "particle_qd", "body_q", "body_qd", "joint_q", "joint_qd")
        if (value := getattr(state, name)) is not None
    }


def restore(state, saved):
    for name, value in saved.items():
        getattr(state, name).assign(value)


def state_distance(first, second):
    result = {}
    for name in ("particle_q", "body_q"):
        if name not in first or not first[name].size:
            continue
        a, b = first[name].numpy(), second[name].numpy()
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            return {"finite": False}
        distance = np.linalg.norm(a[:, :3] - b[:, :3], axis=-1)
        result[f"{name}_max_m"] = float(distance.max())
        result[f"{name}_rms_m"] = float(np.sqrt(np.mean(distance**2)))
        if name == "body_q":
            qa, qb = a[:, 3:].astype(np.float64), b[:, 3:].astype(np.float64)
            dot = np.abs(np.sum(qa * qb, axis=-1)) / np.maximum(
                np.linalg.norm(qa, axis=-1) * np.linalg.norm(qb, axis=-1), 1e-30
            )
            result["body_angle_max_rad"] = float((2 * np.arccos(dot.clip(0, 1))).max())
    return {"finite": True, **result}


def metrics(scene, state, pipeline, contacts):
    """Geometric diagnostics recompute collision candidates at the measured state."""
    model = scene.model
    arrays = [
        getattr(state, n).numpy()
        for n in ("body_q", "body_qd", "particle_q", "particle_qd")
        if getattr(state, n) is not None and getattr(state, n).size
    ]
    finite = all(np.isfinite(a).all() for a in arrays)
    bounded = finite and all(np.max(np.abs(a), initial=0) < 1e4 for a in arrays)
    result = {"finite": bool(finite), "bounded": bool(bounded)}
    if bounded and scene.case.get("rest", False):
        drift = [
            np.linalg.norm(getattr(state, key).numpy()[..., :3] - getattr(model, key).numpy()[..., :3], axis=-1).max()
            for key in ("particle_q", "body_q")
            if getattr(model, key) is not None and getattr(model, key).size
        ]
        result["rest_drift_m"] = float(max(drift, default=0))
    if not bounded:
        return result
    pipeline.collide(state, contacts)
    counts = {
        "rigid": int(contacts.rigid_contact_count.numpy()[0]),
        "soft": int(contacts.soft_contact_count.numpy()[0]),
    }
    result["contacts"] = counts
    result["contact_overflow"] = (
        counts["rigid"] > contacts.rigid_contact_max or counts["soft"] > contacts.soft_contact_max
    )
    if result["contact_overflow"]:
        return result
    device = wp.device_to_torch(model.device)
    identity = torch.tensor([[0, 0, 0, 0, 0, 0, 1]], device=device, dtype=torch.float32)
    poses = torch.cat((wp.to_torch(state.body_q), identity)) if model.body_count else identity
    shape_body = wp.to_torch(model.shape_body).long()
    if counts["rigid"]:
        n = counts["rigid"]
        a = poses[shape_body[wp.to_torch(contacts.rigid_contact_shape0)[:n].long()]]
        b = poses[shape_body[wp.to_torch(contacts.rigid_contact_shape1)[:n].long()]]
        p0 = a[:, :3] + quat_rotate(a[:, 3:], wp.to_torch(contacts.rigid_contact_point0)[:n])
        p1 = b[:, :3] + quat_rotate(b[:, 3:], wp.to_torch(contacts.rigid_contact_point1)[:n])
        gap = ((p1 - p0) * wp.to_torch(contacts.rigid_contact_normal)[:n]).sum(-1)
        gap -= wp.to_torch(contacts.rigid_contact_margin0)[:n] + wp.to_torch(contacts.rigid_contact_margin1)[:n]
        result["rigid_penetration_m"] = float((-gap).clamp_min(0).max())
    if counts["soft"]:
        n = counts["soft"]
        particle = wp.to_torch(contacts.soft_contact_particle)[:n].long()
        valid = particle >= 0
        ids = wp.to_torch(contacts.soft_contact_shape)[:n].long()[valid]
        p = poses[shape_body[ids]]
        surface = p[:, :3] + quat_rotate(p[:, 3:], wp.to_torch(contacts.soft_contact_body_pos)[:n][valid])
        gap = (
            (wp.to_torch(state.particle_q)[particle[valid]] - surface)
            * wp.to_torch(contacts.soft_contact_normal)[:n][valid]
        ).sum(-1)
        gap -= wp.to_torch(model.particle_radius)[particle[valid]] + wp.to_torch(model.shape_margin)[ids]
        result["soft_penetration_m"] = float((-gap).clamp_min(0).max()) if len(gap) else None
    if model.particle_count:
        edges = element_edges(model)
        if len(edges):
            points, rest = state.particle_q.numpy(), model.particle_q.numpy()
            length = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=-1)
            initial = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=-1)
            result["max_edge_strain"] = float(np.max(np.abs(length / np.maximum(initial, 1e-12) - 1)))
    if scene.case["family"] == "m12":
        # The imported nut is the only dynamic body. Report escapes even when a
        # separated nut has zero penetration and all arithmetic remains finite.
        displacement = state.body_q.numpy()[:, :3] - model.body_q.numpy()[:, :3]
        result["m12_axial_displacement_m"] = float(np.abs(displacement[:, 2]).max())
        result["m12_lateral_displacement_m"] = float(np.linalg.norm(displacement[:, :2], axis=-1).max())
    if model.joint_count:
        parent, child = wp.to_torch(model.joint_parent).long(), wp.to_torch(model.joint_child).long()
        xp, xc = wp.to_torch(model.joint_X_p), wp.to_torch(model.joint_X_c)
        pp, pc = poses[parent], poses[child]
        separation = pp[:, :3] + quat_rotate(pp[:, 3:], xp[:, :3]) - pc[:, :3] - quat_rotate(pc[:, 3:], xc[:, :3])
        # Free joints have unconstrained anchors and do not contribute.
        valid = wp.to_torch(model.joint_type) != int(newton.JointType.FREE)
        result["joint_anchor_gap_m"] = float(separation[valid].norm(dim=-1).max()) if bool(valid.any()) else None
    limits = scene.case.get("acceptance", {})
    result["physical_threshold_failures"] = [
        key for key, limit in limits.items() if result.get(key) is not None and result[key] > limit
    ]
    return result


def probe(scene, solver, saved, time_seconds, repeats=3, capture=False):
    """Cold-start, identical input, eager execution. Diagnostic force passes are untimed."""
    model = scene.model
    a, b, control = model.state(), model.state(), model.control()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    timings = []
    for i in range(repeats + 1):
        restore(a, saved)
        solver.reset(a, flags=0)
        scene.prepare(a, time_seconds, control)
        wp.synchronize_device(model.device)
        start = time.perf_counter()
        pipeline.collide(a, contacts)
        solver.step(a, b, control, contacts, scene.dt)
        wp.synchronize_device(model.device)
        if i:
            timings.append(1000 * (time.perf_counter() - start))
    result = {
        "eager_substep_ms": float(np.median(timings)),
        "timing_repeats": repeats,
        "collision_frequency": "each_substep",
        **metrics(scene, b, pipeline, contacts),
    }
    if capture:
        eager_state = snapshot(b)
        restore(a, saved)
        solver.reset(a, flags=0)
        scene.prepare(a, time_seconds, control)
        wp.synchronize_device(model.device)
        capture_start = time.perf_counter()
        with wp.ScopedCapture(device=model.device) as graph:
            pipeline.collide(a, contacts)
            solver.step(a, b, control, contacts, scene.dt)
        result["capture_seconds"] = time.perf_counter() - capture_start
        captured_ms = []
        for _ in range(repeats):
            restore(a, saved)
            solver.reset(a, flags=0)
            scene.prepare(a, time_seconds, control)
            wp.synchronize_device(model.device)
            start = time.perf_counter()
            wp.capture_launch(graph.graph)
            wp.synchronize_device(model.device)
            captured_ms.append(1000 * (time.perf_counter() - start))
        result["captured_substep_ms"] = float(np.median(captured_ms))
        result["capture_parity"] = state_distance(eager_state, snapshot(b))
        if not result["capture_parity"]["finite"] or any(
            value > (1e-3 if key.endswith("rad") else 1e-5)
            for key, value in result["capture_parity"].items()
            if key != "finite"
        ):
            raise RuntimeError(f"Captured step differs from eager step: {result['capture_parity']}")
    if isinstance(solver, SolverNeIO):
        restore(a, saved)
        solver.reset(a, flags=0)
        scene.prepare(a, time_seconds, control)
        pipeline.collide(a, contacts)
        solver.record_convergence, solver.convergence_slot = True, 0
        solver.step(a, b, control, contacts, scene.dt)
        values = [v if np.isfinite(v) else None for v in solver.convergence_buffer[0].tolist()]
        solver.record_convergence = False
        result["neio_objective_gradient"] = dict(
            zip(
                ("particle_initial_n", "particle_final_n", "body_initial_mixed_units", "body_final_mixed_units"), values
            )
        )
        result["particle_residual_ratio"] = (
            values[1] / values[0] if values[0] is not None and values[1] is not None and values[0] > 1e-12 else None
        )
        result["body_residual_ratio"] = (
            values[3] / values[2] if values[2] is not None and values[3] is not None and values[2] > 1e-12 else None
        )
    elif isinstance(solver, newton.solvers.SolverMuJoCo):
        result["mjwarp_actual_iterations"] = solver.mjw_data.solver_niter.numpy().tolist()
    return result, snapshot(b)


def reference_states(scene, steps, stride):
    """VBD-100 is only the probe-state generator, never labeled ground truth."""
    model = scene.model
    a, b, control = model.state(), model.state(), model.control()
    solver = make_solver(scene, "vbd", 100)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    saved = [(0.0, snapshot(a))]
    for step in range(steps):
        scene.prepare(a, step * scene.dt, control)
        pipeline.collide(a, contacts)
        solver.step(a, b, control, contacts, scene.dt)
        a, b = b, a
        if (step + 1) % stride == 0 or step + 1 == steps:
            check = metrics(scene, a, pipeline, contacts)
            if not check["bounded"] or check.get("contact_overflow", False):
                raise RuntimeError(f"Probe state generator failed at step {step + 1}: {check}")
            if not model.particle_count and scene.case["family"] != "cable":
                # VBD evolves maximal coordinates, while MJWarp reads joint
                # coordinates. Project each saved probe once, then give exactly
                # that same pose/velocity to every compared solver.
                projected = model.state()
                restore(projected, snapshot(a))
                newton.eval_ik(model, projected, projected.joint_q, projected.joint_qd)
                newton.eval_fk(model, projected.joint_q, projected.joint_qd, projected)
                saved.append(((step + 1) * scene.dt, snapshot(projected)))
                scene.case["probe_state_projection"] = "Newton IK/FK; common state for all solvers"
            else:
                saved.append(((step + 1) * scene.dt, snapshot(a)))
    return saved


def rollout(scene, solver, steps):
    model = scene.model
    a, b, control = model.state(), model.state(), model.control()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    failures, peak, threshold_failures = None, {}, set()
    for step in range(steps):
        scene.prepare(a, step * scene.dt, control)
        pipeline.collide(a, contacts)
        solver.step(a, b, control, contacts, scene.dt)
        a, b = b, a
        if step % 16 == 0 or step + 1 == steps:
            values = metrics(scene, a, pipeline, contacts)
            if not values["bounded"] or values.get("contact_overflow"):
                failures = step
                break
            threshold_failures.update(values.get("physical_threshold_failures", []))
            for key in (
                "rigid_penetration_m",
                "soft_penetration_m",
                "max_edge_strain",
                "joint_anchor_gap_m",
                "rest_drift_m",
                "m12_axial_displacement_m",
                "m12_lateral_displacement_m",
            ):
                if values.get(key) is not None:
                    peak[key] = max(peak.get(key, 0), values[key])
    return {
        "failed_at_step": failures,
        "completed_steps": step + 1,
        "requested_steps": steps,
        "sample_interval_steps": 16,
        "sampled_peaks": peak,
        "physical_threshold_failures": sorted(threshold_failures),
    }
