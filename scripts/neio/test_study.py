# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Architecture, physical derivative and cross-family integration gates."""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "_source"), str(ROOT)]

import newton
from benchmark import make_solver, metrics
from optimizer import StudySolver, body_inertia_pullback, new_network
from scenes import make_scene

DEVICE = os.environ.get("NEIO_TEST_DEVICE", "cuda:0")


def test_architecture_and_splits():
    expected = {"run.py", "scenes.py", "optimizer.py", "benchmark.py", "test_study.py"}
    assert {p.name for p in ROOT.glob("*.py")} == expected
    assert "optimizer" not in (ROOT / "scenes.py").read_text()
    for file in ("scenes.py", "benchmark.py"):
        tree = ast.parse((ROOT / file).read_text())
        assert not any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("new_network", "NeIONetwork")
            for n in ast.walk(tree)
        )
    study = json.loads((ROOT / "study.json").read_text())
    seeds = study["seeds"]
    assert not (
        set(seeds["train"]) & set(seeds["test"])
        | set(seeds["train"]) & set(seeds["validation"])
        | set(seeds["validation"]) & set(seeds["test"])
    )
    assert {c["family"] for c in study["validation"]} == set(study["arms"]["mixed"])
    assert {c["name"].removesuffix("_validation") for c in study["validation"]} == {
        c["name"].removesuffix("_train") for c in study["train"]
    }
    assert study["arms"]["cloth"] == ["cloth"]
    assert set(study["arms"]["cloth"]) < set(study["arms"]["deformables"]) < set(study["arms"]["mixed"])
    loads = {c["load_n"] for c in study["test"] if c["family"] == "m12" and "threading" not in c["name"]}
    assert loads == {200, 400}
    assert study["rollout_sweeps"].keys() == study["sweeps"].keys()
    for name, budgets in study["rollout_sweeps"].items():
        assert 1 <= len(budgets) <= 2, "Long rollouts must stay a bounded subset of the full probe matrix"
        assert set(budgets) <= set(study["sweeps"][name])


def test_rejected_suite_cannot_launch_training(tmp_path):
    study = json.loads((ROOT / "study.json").read_text())
    study["status"] = "superseded-pilot"
    path = tmp_path / "rejected.json"
    path.write_text(json.dumps(study))
    result = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "--mode=train", f"--study={path}"],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "needs verified replacement workloads" in result.stderr


def test_training_records_metrics_and_checkpoint_tracking_identity(tmp_path):
    from run import start_tracking, train

    study = json.loads((ROOT / "study.json").read_text())
    # Small fixture for the logging contract; it cannot pass the production CLI gate.
    study["train"] = [
        {"name": "numerical_fixture", "family": "cloth"},
        {"name": "rigid_fixture", "family": "rigid", "count": 1},
    ]
    study["arms"]["mixed"] = ["cloth", "rigid"]
    study["training"]["samples_per_step"] = 2
    study["training"]["checkpoint_every"] = 8
    study["training"]["trajectory_curriculum"] = [[0, 1], [8, 2], [16, 8]]
    study["validation"] = [
        {"name": "numerical_fixture", "family": "cloth"},
        {"name": "rigid_fixture", "family": "rigid", "count": 1},
    ]
    args = SimpleNamespace(
        arm="mixed",
        seed=101,
        worlds=1,
        sweeps=1,
        steps=32,
        mode="train",
        run_name="logging-verification",
        wandb_project="neio-verification",
        wandb_entity=None,
        wandb_group=None,
        wandb_mode="offline",
    )
    wp.init()
    with wp.ScopedDevice(DEVICE):
        torch.manual_seed(101)
        network = new_network(DEVICE)
        tracking, identity = start_tracking(args, study, {}, tmp_path)
        try:
            train(args, study, network, DEVICE, 0, 1, tmp_path, {"wandb": identity}, tracking)
        finally:
            tracking.finish()
    checkpoint = torch.load(tmp_path / "latest.pt", map_location="cpu", weights_only=False)
    assert checkpoint["step"] == 32
    assert checkpoint["wandb"] == json.loads((tmp_path / "wandb.json").read_text())
    rows = [json.loads(line) for line in (tmp_path / "training.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [8, 16, 24, 32]
    assert len(rows[-1]["rank_metrics"]) == 2
    assert all({sample["family"] for sample in row["rank_metrics"]} == {"cloth", "rigid"} for row in rows)
    assert rows[-1]["sampled_timesteps"] == 64
    assert [row["trajectory_horizon"] for row in rows] == [1, 2, 8, 8]
    assert rows[0]["rank_metrics"][0]["trajectory_step"] == 0
    assert rows[-1]["rank_metrics"][0]["trajectory_step"] == 7
    assert rows[-1]["rank_metrics"][0]["processed_nodes"] > 0
    assert rows[-1]["rank_metrics"][0]["particle_residual_ratio"] >= 0
    validation = json.loads((tmp_path / "validation.json").read_text())
    assert [json.loads(line)["step"] for line in (tmp_path / "validation.jsonl").read_text().splitlines()] == [0, 32]
    assert len(validation["cases"]) == 10
    assert any(row.get("particle_residual_ratio") is not None for row in validation["cases"])
    assert any(row.get("body_residual_ratio") is not None for row in validation["cases"])
    assert validation["score"] == pytest.approx(np.mean([row["score"] for row in validation["cases"]]))
    assert rows[-1]["rank_metrics"][-1]["oracle_calls"] == 136
    assert rows[-1]["rank_metrics"][-1]["oracle_backward_calls"] == 64
    assert all(np.isfinite(row["loss"]) for row in rows)
    baseline = json.loads((tmp_path / "validation.jsonl").read_text().splitlines()[0])
    assert checkpoint["best"] <= baseline["score"]


@pytest.mark.parametrize(
    "name",
    [
        "cloth_twist_train",
        "rigid_dense_pile_train",
        "franka_payload_train",
        "anymal_disturbance_train",
        "m12_threading_train",
        "loaded_cloth_bag_train",
    ],
)
def test_bank_requires_physical_interactions_and_supports_learning(tmp_path, name):
    from benchmark import probe, reference_states, restore

    study = json.loads((ROOT / "study.json").read_text())
    case = dict(next(case for case in study["train"] if case["name"] == name))
    if case.get("workload"):
        case["duration"] = 2.0
    if case.get("workload") == "rigid_pile":
        case.update(pusher_start_s=0.25, pusher_duration_s=0.75)
    if case["family"] == "m12":
        case.update(duration=2.0, minimum_turns=1)
    family, duration = case["family"], case["duration"]
    wp.init()
    with wp.ScopedDevice(DEVICE):
        scene = make_scene(case, 101, DEVICE)
        if name == "franka_payload_train":
            # With drives at their rest targets and gravity disabled, the arm
            # must not try to correct a fictitious rest-pose joint error.
            gravity = wp.clone(scene.model.gravity)
            scene.model.gravity.fill_(wp.vec3(0))
            rest_in, rest_out = scene.model.state(), scene.model.state()
            rest_solver = make_solver(scene, "shared", 1, new_network(DEVICE))
            rest_solver.step(rest_in, rest_out, scene.model.control(), None, scene.dt)
            assert float(rest_solver.rigid_optimizer._gradient.norm()) < 1
            wp.copy(scene.model.gravity, gravity)
        states = reference_states(scene, round(duration / scene.dt), max(1, round(1 / (30 * scene.dt))), tmp_path)
        if family == "cloth":
            assert scene.evidence["self_contact_observed"]
            assert scene.evidence["native_test_pass"]
        elif family == "rigid":
            assert scene.evidence["distinct_dynamic_pairs"] >= 15
            assert scene.evidence["multi_body_contact_fraction"] >= 0.5
            assert scene.evidence["peak_external_force_n"] == 0
            assert scene.evidence["pusher_excursion_m"] == pytest.approx(case["pusher_stroke_m"], abs=0.001)
            assert scene.evidence["final_rms_speed_m_s"] < case["settling_rms_speed_m_s"]
            assert np.linalg.norm(states[-1][1]["body_qd"].numpy()[scene.pusher_body]) < 1e-6
        elif family == "m12":
            assert scene.evidence["total_turns"] >= 1
            assert scene.evidence["thread_lead_error_m"] < 0.002
            assert scene.evidence["peak_applied_force_n"] == case["load_n"]
        elif family == "articulation":
            assert scene.evidence["revolute_joints"] >= 7
            assert scene.evidence["moving_joints"] >= 6
            # Prismatic gripper joints must not break geometric diagnostics.
            values, _ = probe(scene, make_solver(scene, "vbd", 4), states[-2][1], states[-2][0], repeats=1)
            assert values["finite"] and values["joint_anchor_gap_m"] >= 0
        assert states[-1][0] >= duration - 1e-6
        # A late, folded state must still furnish finite nonzero learning gradients.
        # In the bag's early free fall, rigid residuals are effectively zero.
        # Its inactive rigid rows must not poison the cloth learning derivative.
        for index in [1, -2] if family == "coupled" else [-2]:
            a, b, control = scene.model.state(), scene.model.state(), scene.model.control()
            restore(a, states[index][1], control)
            restore(b, states[index][1], control)
            network = new_network(DEVICE)
            solver = StudySolver(scene.model, network, learn=True, iterations=2, **scene.options)
            pipeline = newton.CollisionPipeline(scene.model)
            contacts = pipeline.contacts()
            pipeline.collide(a, contacts)
            solver.step(a, b, control, contacts, scene.dt)
            assert solver.failed_updates == 0
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in network.parameters())
            assert sum(float(p.grad.abs().sum()) for p in network.parameters()) > 0
        if family == "cloth":
            # Cold probes must retain the sampled moving-pin poses in both buffers.
            solver = make_solver(scene, "vbd", 4)
            _, result = probe(scene, solver, states[-2][1], states[-2][0], repeats=1)
            fixed = (wp.to_torch(scene.model.particle_flags).long() & 1) == 0
            torch.testing.assert_close(
                wp.to_torch(result["particle_q"])[fixed], wp.to_torch(states[-2][1]["particle_q"])[fixed]
            )


@pytest.mark.parametrize("family", ["particle", "rigid", "friction"])
def test_residual_loss_gradient_matches_weight_finite_difference(family):
    wp.init()
    with wp.ScopedDevice(DEVICE):
        builder = newton.ModelBuilder(gravity=wp.vec3(0))
        builder.begin_world()
        if family == "particle":
            builder.add_particle(pos=wp.vec3(0), vel=wp.vec3(0), mass=1)
        elif family == "friction":
            body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.09), wp.quat_identity()))
            builder.add_shape_sphere(body, radius=0.1)
            builder.add_ground_plane()
        else:
            body = builder.add_body()
            builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        builder.end_world()
        builder.color()
        model = builder.finalize()
        torch.manual_seed(12)
        net = new_network(DEVICE)
        parameter = net.direction_head[-1].bias
        axis = 2 if family == "friction" else 0
        original = float(parameter[axis].detach())
        dt = 0.01
        a, b = model.state(), model.state()
        solver = StudySolver(model, net, iterations=4)
        solver.step(a, b, model.control(), None, dt)
        key = "particle_q" if family == "particle" else "body_q"
        torch.testing.assert_close(wp.to_torch(getattr(b, key)), wp.to_torch(getattr(model, key)), atol=1e-8, rtol=0)
        model.gravity.fill_(wp.vec3(1, 2, -9.81))

        def solve(value, learn):
            with torch.no_grad():
                parameter[axis] = value
            net.zero_grad()
            a, b, control = model.state(), model.state(), model.control()
            solver = StudySolver(model, net, iterations=1, learn=learn, initialization="inertia-velocity-only")
            solver.record_convergence = True
            contacts = None
            if family == "friction":
                wp.to_torch(a.body_qd)[:, 0] = 2.0
                pipeline = newton.CollisionPipeline(model)
                contacts = pipeline.contacts()
                pipeline.collide(a, contacts)
                assert contacts.rigid_contact_count.numpy()[0] > 0
            solver.step(a, b, control, contacts, dt)
            if family == "particle":
                squared_residual = wp.to_torch(solver.particle_forces).square().sum()
            else:
                length_squared = wp.to_torch(model.body_inertia).diagonal(dim1=-2, dim2=-1).sum(-1) / wp.to_torch(
                    model.body_mass
                )
                squared_residual = (
                    wp.to_torch(solver.body_forces).square().sum()
                    + (wp.to_torch(solver.body_torques).square().sum(-1) / length_squared).sum()
                )
            adapter = solver.particle_optimizer if family == "particle" else solver.rigid_optimizer
            initial = adapter._gradient.detach().square().sum().clamp_min(1e-12)
            return float(torch.log1p(squared_residual / initial))

        solve(original, True)
        analytical = float(parameter.grad[axis])
        eps = 0.002
        high = solve(original + eps, False)
        low = solve(original - eps, False)
        numerical = (high - low) / (2 * eps)
        assert analytical == pytest.approx(numerical, rel=0.03, abs=1e-3)
        assert abs(numerical) > 1e-9
        if family == "friction":
            # A sliding body can have a nonzero contact torque while its
            # inertial rotation is exactly zero. The derivative of axis * angle
            # must retain the rotational inertia in that state.
            from experiments.neio.kernels import evaluate_body_inertia_force

            pose = wp.to_torch(solver.body_inertia_q).clone()
            force_seed = torch.zeros_like(wp.to_torch(solver.body_forces))
            torque_seed = torch.zeros_like(force_seed)
            torque_seed[:, 1] = 1
            analytical = float(body_inertia_pullback(solver, pose, force_seed, torque_seed)[0, 4])
            values = []
            for sign in (1, -1):
                perturbed = pose.clone()
                perturbed[0, 4] += sign * 1e-4
                wp.launch(
                    evaluate_body_inertia_force,
                    model.body_count,
                    inputs=[
                        dt,
                        wp.from_torch(perturbed, dtype=wp.transform),
                        solver.body_inertia_q,
                        model.body_mass,
                        solver.body_inv_mass_effective,
                        model.body_inertia,
                        model.body_com,
                    ],
                    outputs=[solver.body_forces, solver.body_torques],
                )
                values.append(float(wp.to_torch(solver.body_torques)[0, 1]))
            numerical = (values[0] - values[1]) / 2e-4
            assert abs(numerical) > 1e-3
            assert analytical == pytest.approx(numerical, rel=1e-4)
        if family == "particle":
            # The same physical motion expressed in meters or centimeters must
            # produce the same updates AND training gradients through history.
            positions, gradients = [], []
            for unit in (1.0, 0.01):
                model.gravity.fill_(wp.vec3(1 / unit, 2 / unit, -9.81 / unit))
                net.zero_grad(set_to_none=True)
                a, b = model.state(), model.state()
                solver = StudySolver(
                    model, net, iterations=4, learn=True, length_unit_m=unit, initialization="inertia-velocity-only"
                )
                solver.step(a, b, model.control(), None, dt)
                positions.append(wp.to_torch(b.particle_q).clone() * unit)
                gradients.append(torch.cat([p.grad.flatten() for p in net.parameters()]))
            torch.testing.assert_close(positions[0], positions[1], rtol=1e-3, atol=1e-7)
            torch.testing.assert_close(gradients[0], gradients[1], rtol=0.03, atol=1e-4)
            model.gravity.fill_(wp.vec3(1, 2, -9.81))
        # A saturated scale controller at a larger evaluation sweep budget must
        # not turn a bounded displacement into an exponentially growing one.
        with torch.no_grad():
            net.scale_head[-1].bias.fill_(100)
        a, b = model.state(), model.state()
        solver = StudySolver(model, net, iterations=64, initialization="inertia-velocity-only")
        solver.step(a, b, model.control(), None, dt)
        positions = wp.to_torch(getattr(b, key))[..., :3]
        assert torch.isfinite(positions).all() and positions.abs().max() < 1


@pytest.mark.parametrize("family", ["cloth", "rigid", "articulation", "cable", "soft", "coupled"])
def test_shared_parameters_train_and_geometry_is_measurable(family):
    wp.init()
    with wp.ScopedDevice(DEVICE):
        torch.manual_seed(101)
        scene = make_scene({"family": family, "amplitude": 0.4}, 101, DEVICE, worlds=2)
        model = scene.model
        net = new_network(DEVICE)
        if family == "articulation":
            flags = model.body_flags.numpy()
            flags[0] = int(newton.BodyFlags.KINEMATIC)
            model.body_flags.assign(flags)
            assert model.body_inv_mass.numpy()[0] > 0
            with torch.no_grad():
                net.direction_head[-1].bias.fill_(0.2)
        solver = StudySolver(model, net, learn=True, iterations=2, **scene.options)
        if family == "articulation":
            solver.initialization = "inertia-velocity-only"
        parameters = {id(p) for p in net.parameters()}
        for optimizer in (solver.particle_optimizer, solver.rigid_optimizer):
            if optimizer is not None:
                assert optimizer.network is net
                assert {id(p) for p in optimizer.parameters()} == parameters
        a, b, control = model.state(), model.state(), model.control()
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        for step in range(2):
            scene.prepare(a, step * scene.dt, control)
            fixed_pose = wp.to_torch(a.body_q)[0].clone() if family == "articulation" else None
            pipeline.collide(a, contacts)
            net.zero_grad()
            solver.step(a, b, control, contacts, scene.dt)
            if fixed_pose is not None:
                torch.testing.assert_close(wp.to_torch(b.body_q)[0], fixed_pose, atol=0, rtol=0)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
            assert any(p.grad.abs().sum() > 0 for p in net.parameters())
            check = metrics(scene, b, pipeline, contacts)
            assert check["bounded"] and not check["contact_overflow"]
            a, b = b, a
        with pytest.raises(NotImplementedError, match="PGS"):
            make_solver(scene, "pgs", 8)


def test_capture_matches_eager_and_rigid_probe_coordinates_are_consistent():
    from benchmark import probe, reference_states, snapshot, state_distance

    wp.init()
    with wp.ScopedDevice(DEVICE):
        scene = make_scene({"family": "rigid", "count": 1}, 101, DEVICE)
        time, saved = reference_states(scene, 3, 3)[-1]
        reconstructed = scene.model.state()
        newton.eval_fk(scene.model, saved["joint_q"], saved["joint_qd"], reconstructed)
        distance = state_distance(saved, snapshot(reconstructed))
        assert distance["body_q_max_m"] < 1e-7
        assert distance["body_angle_max_rad"] < 1e-6
        net = new_network(DEVICE)
        for name in ("shared", "vbd", "mjwarp"):
            values, _ = probe(scene, make_solver(scene, name, 2, net), saved, time, repeats=1, capture=True)
            assert values["capture_parity"]["finite"]
            assert values["captured_substep_ms"] > 0


def test_m12_uses_thread_contacts_without_vbd_list_truncation():
    wp.init()
    with wp.ScopedDevice(DEVICE):
        scene = make_scene(
            {
                "family": "m12",
                "load_n": 400,
                "thread_speed_rad_s": -6.28,
                "torque_integral_gain": 0.02,
                "torque_limit_nm": 2.0,
                "guided": True,
                "thread_start_height_m": 0.0105,
            },
            20001,
            DEVICE,
        )
        model = scene.model
        solver = make_solver(scene, "vbd", 4)
        a, b, control = model.state(), model.state(), model.control()
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        scene.prepare(a, 0.5, control)
        command = a.body_f.numpy().copy()
        assert command[0, 2] == -400 and -2 <= command[0, 5] < 0
        scene.prepare(a, 2.0, control, sampled=True)
        assert (a.body_f.numpy() == command).all(), "Cold probes must freeze the recorded motor/load command"
        pipeline.collide(a, contacts)
        solver.step(a, b, control, contacts, scene.dt)
        count = solver.body_body_contact_counts.numpy().max()
        assert count > 64  # catches accidental replacement with a primitive collider
        assert count <= solver.body_body_contact_buffer_pre_alloc
        assert int(contacts.rigid_contact_count.numpy()[0]) <= contacts.rigid_contact_max
