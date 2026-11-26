# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Identical-state probes and physical diagnostics, with explicit unsupported rows."""

import json
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


def make_solver(scene, name, sweeps, network=None, history="clear", learn=False):
    model = scene.model
    if name == "shared":
        solver = StudySolver(
            model,
            network,
            iterations=sweeps,
            history=history,
            learn=learn,
            length_unit_m=scene.case.get("length_unit_m", 1.0),
            **scene.options,
        )
    elif name == "supplied":
        root = Path(__file__).parent / "_source/experiments/neio-reference/checkpoints"
        options = {"history_at_step_start": history, "group_by": "world", "inference": "torch"}
        rigid = dict(options, checkpoint=str(root / "rigid_e/weights_270000.pt")) if model.body_count else None
        particle = (
            dict(options, checkpoint=str(root / "particle_local_iso_500k/weights_500000.pt"))
            if model.particle_count
            else None
        )
        solver = SolverNeIO(
            model,
            iterations=sweeps,
            **scene.options,
            rigid_optimizer_options=rigid,
            particle_optimizer_options=particle,
        )
    elif name == "vbd":
        native = scene.native.solver if scene.native is not None else None
        return newton.solvers.SolverVBD(
            model,
            iterations=sweeps,
            rigid_compliant_alm=True,
            rigid_body_contact_buffer_size=1024,
            rigid_body_particle_contact_buffer_size=2048,
            rigid_contact_history=native.rigid_contact_history if native else False,
            particle_topological_contact_filter_threshold=native.particle_topological_contact_filter_threshold
            if native and model.particle_count
            else 3,
            **scene.options,
        )
    elif name == "pgs":
        raise NotImplementedError("The pinned Newton MJWarp adapter accepts Newton/CG only; PGS is unsupported")
    elif name == "mjwarp":
        if model.particle_count or scene.case["family"] == "cable" or "cross_slide" in scene.case.get("example", ""):
            raise NotImplementedError("MJWarp does not implement this particle/rod objective")
        return newton.solvers.SolverMuJoCo(
            model, solver="newton", iterations=sweeps, use_mujoco_contacts=False, njmax=4096, nconmax=2048
        )
    else:
        raise ValueError(name)
    # The port omits the inherited VBD allocation options. Configure the owned
    # arrays before any capture so unused adjacency construction cannot overflow
    # or flood timing runs with device printf. NeIO forces use global contacts.
    if model.body_count:
        solver.body_body_contact_buffer_pre_alloc = 1024
        solver.body_body_contact_indices = wp.zeros(model.body_count * 1024, dtype=wp.int32, device=model.device)
        solver.body_particle_contact_buffer_pre_alloc = 4096 if model.particle_count else 0
        solver.body_particle_contact_indices = wp.zeros(
            model.body_count * solver.body_particle_contact_buffer_pre_alloc, dtype=wp.int32, device=model.device
        )
    return solver


def snapshot(state, control=None):
    saved = {
        name: wp.clone(value)
        for name in ("particle_q", "particle_qd", "particle_f", "body_q", "body_qd", "body_f", "joint_q", "joint_qd")
        if (value := getattr(state, name)) is not None
    }
    if control is not None:
        saved.update(
            {f"control.{name}": wp.clone(value) for name, value in vars(control).items() if isinstance(value, wp.array)}
        )
    return saved


def restore(state, saved, control=None):
    for name, value in saved.items():
        if name.startswith("control."):
            if control is not None:
                getattr(control, name.removeprefix("control.")).assign(value)
        else:
            getattr(state, name).assign(value)


def state_distance(first, second, length_unit_m=1):
    result = {}
    for name in ("particle_q", "body_q"):
        if name not in first or not first[name].size:
            continue
        a, b = first[name].numpy(), second[name].numpy()
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            return {"finite": False}
        distance = np.linalg.norm(a[:, :3] - b[:, :3], axis=-1) * length_unit_m
        result[f"{name}_max_m"] = float(distance.max())
        result[f"{name}_rms_m"] = float(np.sqrt(np.mean(distance**2)))
        if name == "body_q":
            qa, qb = a[:, 3:].astype(np.float64), b[:, 3:].astype(np.float64)
            dot = np.abs(np.sum(qa * qb, axis=-1)) / np.maximum(
                np.linalg.norm(qa, axis=-1) * np.linalg.norm(qb, axis=-1), 1e-30
            )
            result["body_angle_max_rad"] = float((2 * np.arccos(dot.clip(0, 1))).max())
    return {"finite": True, **result}


def metrics(scene, state, pipeline, contacts, contact_geometry=False):
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
        body0 = shape_body[wp.to_torch(contacts.rigid_contact_shape0)[:n].long()]
        body1 = shape_body[wp.to_torch(contacts.rigid_contact_shape1)[:n].long()]
        active = gap < 0
        inverse_mass = torch.cat((wp.to_torch(model.body_inv_mass), torch.zeros(1, device=device)))
        dynamic = active & (body0 != body1) & (inverse_mass[body0] > 0) & (inverse_mass[body1] > 0)
        pairs = torch.stack((body0[dynamic], body1[dynamic]), -1).sort(-1).values.unique(dim=0)
        result["active_rigid_contacts"] = int(active.sum())
        result["dynamic_contact_pairs"] = pairs.cpu().tolist()
        result["dynamic_contact_pair_count"] = len(pairs)
        if contact_geometry:
            result["contact_points"] = ((p0[active] + p1[active]) * 0.5).cpu().tolist()
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
        prismatic = wp.to_torch(model.joint_type) == int(newton.JointType.PRISMATIC)
        if bool(prismatic.any()):
            starts = wp.to_torch(model.joint_qd_start).long()[: model.joint_count][prismatic]
            local_axes = quat_rotate(xp[prismatic, 3:], wp.to_torch(model.joint_axis)[starts])
            axes = quat_rotate(pp[prismatic, 3:], local_axes)
            delta = separation[prismatic]
            separation[prismatic] = delta - (delta * axes).sum(-1, keepdim=True) * axes
        d6 = wp.to_torch(model.joint_type) == int(newton.JointType.D6)
        if bool(d6.any()):
            dimensions = model.joint_dof_dim.numpy()
            starts = model.joint_qd_start.numpy()
            for joint in torch.nonzero(d6).flatten().tolist():
                for axis in range(dimensions[joint, 0]):
                    direction = wp.to_torch(model.joint_axis)[starts[joint] + axis].unsqueeze(0)
                    local = quat_rotate(xp[joint : joint + 1, 3:], direction)
                    world = quat_rotate(pp[joint : joint + 1, 3:], local)[0]
                    separation[joint] -= (separation[joint] * world).sum() * world
        # Free joints have unconstrained anchors and do not contribute.
        valid = wp.to_torch(model.joint_type) != int(newton.JointType.FREE)
        result["joint_anchor_gap_m"] = float(separation[valid].norm(dim=-1).max()) if bool(valid.any()) else None
    limits = scene.case.get("acceptance", {})
    for key in tuple(result):
        if key.endswith("_m") and result[key] is not None:
            result[key] *= scene.case.get("length_unit_m", 1.0)
    result["physical_threshold_failures"] = [
        key for key, limit in limits.items() if result.get(key) is not None and result[key] > limit
    ]
    return result


def probe(scene, solver, saved, time_seconds, repeats=3, capture=False):
    """Cold-start, identical input, eager execution. Diagnostic force passes are untimed."""
    model = scene.model
    a, b, control = model.state(), model.state(), model.control()
    pipeline = newton.CollisionPipeline(model, contact_matching="latest")
    contacts = pipeline.contacts()
    timings = []
    for i in range(repeats + 1):
        restore(a, saved, control)
        restore(b, saved, control)
        solver.reset(a, flags=0)
        pipeline.reset_contact_matching()
        scene.prepare(a, time_seconds, control, sampled=True)
        wp.synchronize_device(model.device)
        start = time.perf_counter()
        pipeline.collide(a, contacts)
        solver.step(a, b, control, contacts, scene.dt)
        wp.synchronize_device(model.device)
        if i:
            timings.append(1000 * (time.perf_counter() - start))
    eager_state = snapshot(b)
    result = {
        "eager_substep_ms": float(np.median(timings)),
        "timing_repeats": repeats,
        "collision_frequency": "each_substep",
        **metrics(scene, b, pipeline, contacts),
    }
    if capture:
        restore(a, saved, control)
        restore(b, saved, control)
        solver.reset(a, flags=0)
        pipeline.reset_contact_matching()
        scene.prepare(a, time_seconds, control, sampled=True)
        wp.synchronize_device(model.device)
        capture_start = time.perf_counter()
        with wp.ScopedCapture(device=model.device) as graph:
            pipeline.collide(a, contacts)
            solver.step(a, b, control, contacts, scene.dt)
        result["capture_seconds"] = time.perf_counter() - capture_start
        captured_ms = []
        for _ in range(repeats):
            restore(a, saved, control)
            restore(b, saved, control)
            solver.reset(a, flags=0)
            pipeline.reset_contact_matching()
            scene.prepare(a, time_seconds, control, sampled=True)
            wp.synchronize_device(model.device)
            start = time.perf_counter()
            wp.capture_launch(graph.graph)
            wp.synchronize_device(model.device)
            captured_ms.append(1000 * (time.perf_counter() - start))
        result["capture_parity"] = state_distance(eager_state, snapshot(b))
        result["capture_parity_pass"] = result["capture_parity"]["finite"] and not any(
            value > (1e-3 if key.endswith("rad") else 1e-5)
            for key, value in result["capture_parity"].items()
            if key != "finite"
        )
        # Preserve the independently measured eager solve. A capture mismatch
        # invalidates accelerated timing and selects eager rollout execution.
        result["captured_substep_ms"] = float(np.median(captured_ms)) if result["capture_parity_pass"] else None
    if isinstance(solver, SolverNeIO):
        restore(a, saved, control)
        restore(b, saved, control)
        solver.reset(a, flags=0)
        pipeline.reset_contact_matching()
        scene.prepare(a, time_seconds, control, sampled=True)
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
    return result, eager_state


def native_states(scene, output=None):
    """Sample complete native trajectories, preserving controls and external forces.

    These are cold-start timestep problems, not VBD target solutions. Native
    history produces the trajectory but is deliberately absent from each probe.
    """
    native, case = scene.native, scene.case
    frames = int(np.ceil((case["duration"] - native.sim_time) / native.frame_dt))
    stride = max(1, frames // case.get("bank_samples", 96))
    saved, diagnostics = [], []
    start = time.perf_counter()
    for frame in range(frames):
        native.step()
        if frame % stride and frame + 1 != frames:
            continue
        state = native.state_0
        row = {
            "time": float(native.sim_time),
            "rigid_contacts": int(native.contacts.rigid_contact_count.numpy()[0]) if native.contacts else 0,
            "soft_contacts": int(native.contacts.soft_contact_count.numpy()[0]) if native.contacts else 0,
        }
        for name in ("particle_q", "particle_qd", "body_q", "body_qd"):
            value = getattr(state, name)
            if value is not None and value.size:
                data = value.numpy()
                if not np.isfinite(data).all() or np.max(np.abs(data)) > 1e5:
                    raise RuntimeError(f"Native {case['name']} diverged at {native.sim_time:.3f}s ({name})")
        if native.contacts and (
            row["rigid_contacts"] > native.contacts.rigid_contact_max
            or row["soft_contacts"] > native.contacts.soft_contact_max
        ):
            raise RuntimeError(f"Native {case['name']} exceeded contact capacity")
        if native.solver.particle_enable_self_contact:
            info = native.solver.trimesh_collision_detector.collision_info
            for kind in ("vertex_colliding_triangles", "edge_colliding_edges"):
                counts = getattr(info, f"{kind}_count").numpy()
                capacities = getattr(info, f"{kind}_buffer_sizes").numpy()
                if np.any(counts > capacities):
                    raise RuntimeError(f"Native {case['name']} dropped {kind} pairs at {native.sim_time:.3f}s")
                distance = getattr(info, f"{kind}_min_dist").numpy()
                row[f"{kind}_near"] = int(np.count_nonzero(distance < native.solver.particle_self_contact_margin))
                row[f"{kind}_peak"] = int(counts.max(initial=0))
        if case["family"] == "rigid" and not native.model.particle_count:
            projected = native.model.state()
            restore(projected, snapshot(state))
            newton.eval_ik(native.model, projected, projected.joint_q, projected.joint_qd)
            newton.eval_fk(native.model, projected.joint_q, projected.joint_qd, projected)
            state = projected
            case["probe_state_projection"] = "Newton IK/FK; common state for all solvers"
        saved.append((float(native.sim_time), snapshot(state, native.control)))
        diagnostics.append(row)
    evidence = {
        "case": case,
        "duration_seconds": float(native.sim_time),
        "states": len(saved),
        "generation_seconds": time.perf_counter() - start,
        "particles": native.model.particle_count,
        "bodies": native.model.body_count,
        "native_iterations": native.solver.iterations,
        "substeps_per_frame": native.sim_substeps,
        "dt": native.sim_dt,
        "diagnostics": diagnostics,
    }
    try:
        if hasattr(native, "test_final"):
            native.test_final()
        evidence["native_test_pass"] = True
    except (AssertionError, ValueError) as error:
        evidence.update(native_test_pass=False, native_test_error=str(error))
    if "plectoneme" in case["example"]:
        evidence["out_of_plane_span_m"] = max(float(np.ptp(s["body_q"].numpy()[:, 1])) for _, s in saved)
        if (
            evidence["out_of_plane_span_m"] < 4 * native.cable_radius
            or max(r["rigid_contacts"] for r in diagnostics) == 0
        ):
            raise RuntimeError("Plectoneme bank did not reach buckling and self-contact")
    if "cross_slide" in case["example"]:
        evidence["tracking_max_error_m"] = float(native.table_tracking_max_error)
        evidence["tracking_rms_error_m"] = float(
            np.sqrt(native.table_tracking_error_sq_sum / native.table_tracking_sample_count)
        )
        if not evidence["native_test_pass"]:
            raise RuntimeError(f"Cable-driven table failed its tracking check: {evidence['native_test_error']}")
    if "cloth_twist" in case["example"] or "cloth_rollers" in case["example"]:
        evidence["self_contact_observed"] = any(
            r.get("vertex_colliding_triangles_near", 0) + r.get("edge_colliding_edges_near", 0) > 0 for r in diagnostics
        )
        if not evidence["self_contact_observed"]:
            raise RuntimeError("Cloth bank did not reach self-contact")
    scene.evidence = evidence
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        (output / "coverage.json").write_text(json.dumps(evidence, indent=2) + "\n")
        arrays = {name: np.stack([state[name].numpy() for _, state in saved]) for name in saved[0][1]}
        arrays["time"] = np.asarray([t for t, _ in saved])
        for name in ("tri_indices", "shape_body", "shape_transform", "shape_scale", "shape_type"):
            if (value := getattr(native.model, name)) is not None:
                arrays[name] = value.numpy()
        for i, mesh in enumerate(scene.model.shape_source):
            if isinstance(mesh, newton.Mesh):
                arrays[f"mesh_vertices_{i}"] = np.asarray(mesh.vertices)
                arrays[f"mesh_indices_{i}"] = np.asarray(mesh.indices)
        np.savez_compressed(output / "states.npz", **arrays)
    return saved


def reference_states(scene, steps, stride, output=None):
    """Generate timestep inputs with the declared physics solver, never target solutions."""
    if scene.native is not None:
        return native_states(scene, output)
    if scene.case.get("workload") or scene.case["family"] == "m12":
        stride = min(stride, max(1, int(1 / (30 * scene.dt))))
    model = scene.model
    a, b, control = model.state(), model.state(), model.control()
    if model.joint_count:
        newton.eval_fk(model, model.joint_q, model.joint_qd, a)
        b.assign(a)
    generator = scene.case.get("generator", "vbd")
    solver = make_solver(scene, generator, 100)
    pipeline = newton.CollisionPipeline(model, contact_matching="latest")
    contacts = pipeline.contacts()
    diagnostic_pipeline = newton.CollisionPipeline(model)
    diagnostic_contacts = diagnostic_pipeline.contacts()
    saved = [(0.0, snapshot(a, control))]
    diagnostics, recorded_poses = [], [a.body_q.numpy()] if model.body_count else []
    graph = None
    if scene.case["family"] == "m12" or scene.case.get("workload"):
        # Capture the expensive solve, leaving time-dependent loading outside.
        scene.prepare(a, 0, control)
        pipeline.collide(a, contacts)
        solver.step(a, b, control, contacts, scene.dt)
        restore(a, saved[0][1])
        restore(b, saved[0][1])
        solver.reset(a, flags=0)
        with wp.ScopedCapture(device=model.device) as capture:
            pipeline.collide(a, contacts)
            solver.step(a, b, control, contacts, scene.dt)
            a.assign(b)
        graph = capture.graph
    for step in range(steps):
        scene.prepare(a, step * scene.dt, control)
        record = (step + 1) % stride == 0 or step + 1 == steps
        applied_force = wp.clone(a.body_f) if record and model.body_count else None
        if graph is not None:
            wp.capture_launch(graph)
        else:
            pipeline.collide(a, contacts)
            solver.step(a, b, control, contacts, scene.dt)
            a, b = b, a
        if record:
            if applied_force is not None:
                a.body_f.assign(applied_force)
            check = metrics(scene, a, diagnostic_pipeline, diagnostic_contacts, contact_geometry=output is not None)
            if not check["bounded"] or check.get("contact_overflow", False):
                raise RuntimeError(f"Probe state generator failed at step {step + 1}: {check}")
            diagnostics.append({"time": (step + 1) * scene.dt, **check})
            if model.body_count:
                recorded_poses.append(a.body_q.numpy())
            if not model.particle_count and scene.case["family"] != "cable":
                # VBD evolves maximal coordinates, while MJWarp reads joint
                # coordinates. Project each saved probe once, then give exactly
                # that same pose/velocity to every compared solver.
                projected = model.state()
                restore(projected, snapshot(a))
                newton.eval_ik(model, projected, projected.joint_q, projected.joint_qd)
                newton.eval_fk(model, projected.joint_q, projected.joint_qd, projected)
                saved.append(((step + 1) * scene.dt, snapshot(projected, control)))
                scene.case["probe_state_projection"] = "Newton IK/FK; common state for all solvers"
            else:
                saved.append(((step + 1) * scene.dt, snapshot(a, control)))
    scene.evidence = {
        "case": scene.case,
        "generator": generator,
        "duration_seconds": steps * scene.dt,
        "states": len(saved),
        "diagnostics": diagnostics,
    }
    arrays = {name: np.stack([state[name].numpy() for _, state in saved]) for name in saved[0][1]}
    arrays["time"] = np.asarray([t for t, _ in saved])
    for name in ("tri_indices", "shape_body", "shape_transform", "shape_scale", "shape_type", "shape_flags"):
        if (value := getattr(model, name)) is not None:
            arrays[name] = value.numpy()
    if recorded_poses:
        arrays["body_q_recorded"] = np.stack(recorded_poses)
    pairs = {tuple(pair) for row in diagnostics for pair in row.get("dynamic_contact_pairs", [])}
    contact_fraction = np.mean([row.get("active_rigid_contacts", 0) > 0 for row in diagnostics])
    scene.evidence.update(distinct_dynamic_pairs=len(pairs), active_contact_fraction=float(contact_fraction))
    if scene.case.get("workload") == "rigid_pile":
        fraction = np.mean([row.get("dynamic_contact_pair_count", 0) >= 3 for row in diagnostics])
        scene.evidence["multi_body_contact_fraction"] = float(fraction)
        if len(pairs) < 15 or fraction < 0.5:
            raise RuntimeError(f"Rigid pile lacks sustained object-object collisions: {len(pairs)} pairs, {fraction=}")
    if scene.joint_amplitude is not None:
        moving = model.joint_type.numpy() == int(newton.JointType.REVOLUTE)
        coordinates = model.joint_q_start.numpy()[:-1][moving]
        excursions = np.ptp(np.unwrap(arrays["joint_q"][:, coordinates], axis=0), axis=0)
        scene.evidence.update(
            revolute_joints=int(moving.sum()),
            moving_joints=int((excursions > 0.1).sum()),
            joint_excursion_rad=excursions.tolist(),
            load_n=scene.case["load_n"],
            peak_applied_force_n=float(np.linalg.norm(arrays["body_f"][:, scene.load_body, :3], axis=-1).max()),
        )
        if scene.evidence["moving_joints"] < 6:
            raise RuntimeError("Robot workload did not exercise at least six dynamic revolute joints")
        if scene.evidence["peak_applied_force_n"] < 0.95 * scene.case["load_n"]:
            raise RuntimeError("Robot workload did not apply its declared load to the selected link")
        anchor_gap = max(row.get("joint_anchor_gap_m", 0) for row in diagnostics)
        scene.evidence["max_joint_anchor_gap_m"] = anchor_gap
        if anchor_gap > 0.005:
            raise RuntimeError(f"Robot state generator separated joint anchors by {anchor_gap} m")
        if scene.case["workload"] == "anymal_disturbance" and contact_fraction < 0.5:
            raise RuntimeError("Quadruped workload did not maintain ground contact")
    if scene.case["family"] == "m12":
        angles = np.unwrap(2 * np.arctan2(arrays["body_q"][:, 0, 5], arrays["body_q"][:, 0, 6]))
        turns = (angles[-1] - angles[0]) / (2 * np.pi)
        travel = arrays["body_q"][-1, 0, 2] - arrays["body_q"][0, 0, 2]
        scene.evidence.update(
            nut_turns=float(turns),
            axial_travel_m=float(travel),
            travel_per_turn_m=float(travel / turns) if abs(turns) > 0.01 else None,
            total_turns=float(np.abs(np.diff(angles)).sum() / (2 * np.pi)),
            turn_span=float(np.ptp(angles) / (2 * np.pi)),
            thread_lead_error_m=float(
                np.abs(
                    arrays["body_q"][:, 0, 2] - arrays["body_q"][0, 0, 2] - 0.00175 * (angles - angles[0]) / (2 * np.pi)
                ).max()
            ),
            peak_applied_force_n=float(np.linalg.norm(arrays["body_f"][:, 0, :3], axis=-1).max()),
            peak_applied_torque_nm=float(np.abs(arrays["body_f"][:, 0, 5]).max()),
        )
        if scene.case.get("minimum_turns", 0) > scene.evidence["total_turns"]:
            raise RuntimeError("Threading workload did not complete the required accumulated turns")
        if scene.case.get("minimum_turns", 0) and scene.evidence["turn_span"] < 1.5:
            raise RuntimeError("Threading workload only oscillated locally instead of completing full turns")
        if scene.case.get("minimum_turns", 0) and (
            scene.evidence["thread_lead_error_m"] > 0.002 or contact_fraction < 0.9
        ):
            raise RuntimeError("Threading workload rotated without sustained thread engagement")
        if scene.evidence["peak_applied_force_n"] < 0.95 * scene.case.get("load_n", 0):
            raise RuntimeError("Threaded workload did not apply its declared axial/lateral load")
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        (output / "coverage.json").write_text(json.dumps(scene.evidence, indent=2) + "\n")
        for i, mesh in enumerate(model.shape_source):
            if isinstance(mesh, newton.Mesh):
                arrays[f"mesh_vertices_{i}"] = np.asarray(mesh.vertices)
                arrays[f"mesh_indices_{i}"] = np.asarray(mesh.indices)
        np.savez_compressed(output / "states.npz", **arrays)
    return saved


def rollout(scene, solver, steps, capture=True):
    if scene.native is not None:
        native = scene.native
        native.solver = solver
        if hasattr(native, "capture"):
            scene.retained_graphs.append(native.graph)
            if capture:
                native.capture()
            else:
                native.graph = None
        frames = int(np.ceil((scene.case["duration"] - native.sim_time) / native.frame_dt))
        pipeline = newton.CollisionPipeline(scene.model)
        contacts = pipeline.contacts()
        failures, peak, threshold_failures = None, {}, set()
        for frame in range(frames):
            native.step()
            if frame % 12 and frame + 1 != frames:
                continue
            values = metrics(scene, native.state_0, pipeline, contacts)
            if not values["bounded"] or values.get("contact_overflow"):
                failures = frame * native.sim_substeps
                break
            threshold_failures.update(values.get("physical_threshold_failures", []))
            for key in ("rigid_penetration_m", "soft_penetration_m", "max_edge_strain", "joint_anchor_gap_m"):
                if values.get(key) is not None:
                    peak[key] = max(peak.get(key, 0), values[key])
        native_check = None
        if failures is None and hasattr(native, "test_final"):
            try:
                native.test_final()
                native_check = True
            except (AssertionError, ValueError) as error:
                native_check = False
                threshold_failures.add(str(error))
        return {
            "failed_at_step": failures,
            "completed_steps": (frame + 1) * native.sim_substeps,
            "requested_steps": frames * native.sim_substeps,
            "duration_seconds": native.sim_time,
            "native_test_pass": native_check,
            "sampled_peaks": peak,
            "physical_threshold_failures": sorted(threshold_failures),
        }
    model = scene.model
    a, b, control = model.state(), model.state(), model.control()
    if model.joint_count:
        newton.eval_fk(model, model.joint_q, model.joint_qd, a)
        b.assign(a)
    pipeline = newton.CollisionPipeline(model, contact_matching="latest")
    contacts = pipeline.contacts()
    diagnostic_pipeline = newton.CollisionPipeline(model)
    diagnostic_contacts = diagnostic_pipeline.contacts()
    failures, peak, threshold_failures = None, {}, set()
    poses = [a.body_q.numpy()[0].copy()] if scene.case["family"] == "m12" else []
    graph = None
    if capture and (scene.case["family"] == "m12" or scene.case.get("workload")):
        initial = snapshot(a)
        scene.prepare(a, 0, control)
        pipeline.collide(a, contacts)
        solver.step(a, b, control, contacts, scene.dt)
        restore(a, initial)
        restore(b, initial)
        solver.reset(a, flags=0)
        with wp.ScopedCapture(device=model.device) as capture:
            pipeline.collide(a, contacts)
            solver.step(a, b, control, contacts, scene.dt)
            a.assign(b)
        graph = capture.graph
    for step in range(steps):
        scene.prepare(a, step * scene.dt, control)
        if graph is not None:
            wp.capture_launch(graph)
        else:
            pipeline.collide(a, contacts)
            solver.step(a, b, control, contacts, scene.dt)
            a, b = b, a
        if step % 16 == 0 or step + 1 == steps:
            values = metrics(scene, a, diagnostic_pipeline, diagnostic_contacts)
            if not values["bounded"] or values.get("contact_overflow"):
                failures = step
                break
            if poses:
                poses.append(a.body_q.numpy()[0].copy())
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
    threading = {}
    if len(poses) > 1:
        poses = np.asarray(poses)
        angle = np.unwrap(2 * np.arctan2(poses[:, 5], poses[:, 6]))
        turns = (angle[-1] - angle[0]) / (2 * np.pi)
        travel = poses[-1, 2] - poses[0, 2]
        threading = {
            "nut_turns": float(turns),
            "total_turns": float(np.abs(np.diff(angle)).sum() / (2 * np.pi)),
            "axial_travel_m": float(travel),
            "travel_per_turn_m": float(travel / turns) if abs(turns) > 0.01 else None,
            "thread_lead_error_m": float(
                np.max(np.abs(poses[:, 2] - poses[0, 2] - 0.00175 * (angle - angle[0]) / (2 * np.pi)))
            ),
        }
        if scene.case.get("thread_speed_rad_s") and threading["thread_lead_error_m"] > 0.002:
            threshold_failures.add("thread_lead_error_m")
    return {
        "failed_at_step": failures,
        "completed_steps": step + 1,
        "requested_steps": steps,
        "sample_interval_steps": 16,
        "sampled_peaks": peak,
        "physical_threshold_failures": sorted(threshold_failures),
        "threading": threading,
    }
