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

import pytest
import torch
import warp as wp

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "_source"), str(ROOT)]

import newton
from benchmark import make_solver, metrics
from experiments.neio.rigid import quat_rotate
from optimizer import StudySolver, new_network, quaternion_energy_gradient, quaternion_product, rotation_quaternion
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
    study["train"] = [{"name": "numerical_fixture", "family": "cloth"}]
    study["validation"] = [{"name": "numerical_fixture", "family": "cloth"}]
    args = SimpleNamespace(
        arm="cloth",
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
    assert len(rows[-1]["rank_metrics"]) == 1
    assert rows[-1]["rank_metrics"][0]["processed_nodes"] > 0
    assert rows[-1]["rank_metrics"][0]["oracle_calls"] == 64


@pytest.mark.parametrize(
    "family,example,duration",
    [
        ("cloth", "cloth.example_cloth_twist", 12),
        ("rigid", "vbd.example_vbd_rigid_rigid_contact", 6),
    ],
)
def test_native_bank_preserves_difficult_states_and_supports_learning(tmp_path, family, example, duration):
    from benchmark import probe, reference_states, restore

    wp.init()
    with wp.ScopedDevice(DEVICE):
        scene = make_scene(
            {
                "name": "native_bank",
                "family": family,
                "example": example,
                "duration": duration,
                "bank_samples": 24,
                "material_range": [1.1, 1.1],
            },
            101,
            DEVICE,
        )
        states = reference_states(scene, 0, 1, tmp_path)
        if family == "cloth":
            assert scene.evidence["self_contact_observed"]
        else:
            positions = states[-1][1]["body_q"].numpy()[:, :3]
            assert abs(positions - scene.model.body_q.numpy()[:, :3]).max() > 0.1
        assert scene.evidence["native_test_pass"]
        assert states[-1][0] >= duration - 1e-6
        # A late, folded state must still furnish finite nonzero learning gradients.
        a, b, control = scene.model.state(), scene.model.state(), scene.model.control()
        restore(a, states[-2][1], control)
        restore(b, states[-2][1], control)
        network = new_network(DEVICE)
        solver = StudySolver(scene.model, network, learn=True, iterations=2, **scene.options)
        pipeline = newton.CollisionPipeline(scene.model)
        contacts = pipeline.contacts()
        pipeline.collide(a, contacts)
        solver.step(a, b, control, contacts, scene.dt)
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


def test_world_rotation_gradient_matches_independent_energy_derivative():
    torch.manual_seed(7)
    q0 = torch.nn.functional.normalize(torch.randn(5, 4, dtype=torch.float64), dim=-1)
    rotation = (0.6 * torch.randn(5, 3, dtype=torch.float64)).requires_grad_()
    point, target = torch.randn(5, 3, dtype=torch.float64), torch.randn(5, 3, dtype=torch.float64)
    q = quaternion_product(rotation_quaternion(rotation), q0)
    rotated = quat_rotate(q, point)
    energy = 0.5 * (rotated - target).square().sum()
    expected = torch.autograd.grad(energy, rotation, retain_graph=True)[0]
    torque_gradient = torch.linalg.cross(rotated.detach(), rotated.detach() - target)
    actual = torch.autograd.grad(q, rotation, quaternion_energy_gradient(q.detach(), torque_gradient))[0]
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-10)


def test_particle_force_to_weight_gradient_matches_energy_finite_difference():
    wp.init()
    with wp.ScopedDevice(DEVICE):
        builder = newton.ModelBuilder(gravity=wp.vec3(0))
        builder.begin_world()
        builder.add_particle(pos=wp.vec3(0), vel=wp.vec3(0), mass=1)
        builder.end_world()
        builder.color()
        model = builder.finalize()
        torch.manual_seed(12)
        net = new_network(DEVICE)
        parameter = net.direction_head[-1].bias
        original = float(parameter[0].detach())
        dt = 0.01

        def solve(value, learn):
            with torch.no_grad():
                parameter[0] = value
            net.zero_grad()
            a, b, control = model.state(), model.state(), model.control()
            solver = StudySolver(model, net, iterations=1, learn=learn)
            solver.step(a, b, control, None, dt)
            q = wp.to_torch(b.particle_q)
            energy = float(0.5 * q.square().sum() / dt**2)
            scale = float(solver.particle_optimizer.energy_scale[0, 0])
            return energy, scale

        _, scale = solve(original, True)
        analytical = float(parameter.grad[0]) * scale
        eps = 0.002
        high, _ = solve(original + eps, False)
        low, _ = solve(original - eps, False)
        numerical = (high - low) / (2 * eps)
        assert analytical == pytest.approx(numerical, rel=0.02, abs=1e-10)
        assert abs(numerical) > 1e-9


@pytest.mark.parametrize("family", ["cloth", "rigid", "articulation", "cable", "soft", "coupled"])
def test_shared_parameters_train_and_geometry_is_measurable(family):
    wp.init()
    with wp.ScopedDevice(DEVICE):
        torch.manual_seed(101)
        scene = make_scene({"family": family, "amplitude": 0.4}, 101, DEVICE, worlds=2)
        model = scene.model
        net = new_network(DEVICE)
        solver = StudySolver(model, net, learn=True, iterations=2, **scene.options)
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
            pipeline.collide(a, contacts)
            net.zero_grad()
            solver.step(a, b, control, contacts, scene.dt)
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
        scene = make_scene({"family": "m12", "load_n": 400}, 20001, DEVICE)
        model = scene.model
        solver = make_solver(scene, "vbd", 4)
        a, b, control = model.state(), model.state(), model.control()
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        scene.prepare(a, 0.1, control)
        pipeline.collide(a, contacts)
        solver.step(a, b, control, contacts, scene.dt)
        count = solver.body_body_contact_counts.numpy().max()
        assert count > 64  # catches accidental replacement with a primitive collider
        assert count <= solver.body_body_contact_buffer_pre_alloc
        assert int(contacts.rigid_contact_count.numpy()[0]) <= contacts.rigid_contact_max
