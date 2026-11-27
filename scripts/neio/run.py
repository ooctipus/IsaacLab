# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Composition root for the NeIO transfer study. Run from the image's Python environment."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "_source"))

import newton
import numpy as np
import torch
import torch.distributed as dist
import wandb
import warp as wp
from benchmark import make_solver, probe, reference_states, restore, rollout, state_distance
from optimizer import new_network
from scenes import make_scene


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def start_tracking(args, study, metadata, output, previous=None):
    """Keep experiment identity with the checkpoint, including across cluster restarts."""
    if previous and (
        previous["project"] != args.wandb_project or (args.wandb_entity and previous["entity"] != args.wandb_entity)
    ):
        raise ValueError("Resume must use the checkpoint's W&B project and entity")
    tracking = wandb.init(
        project=args.wandb_project,
        entity=previous["entity"] if previous else args.wandb_entity,
        id=previous["id"] if previous else None,
        resume="must" if previous else "never",
        name=args.run_name,
        group=args.wandb_group or f"neio-{args.arm}",
        job_type=args.mode,
        tags=["superseded-pilot", "backfill"] if args.mode == "upload" else ["learned-solver", args.arm],
        config=None
        if previous
        else {"arguments": metadata.get("arguments", vars(args)), "study": study, "source": metadata},
        mode=args.wandb_mode,
        dir=str(output),
        settings=wandb.Settings(init_timeout=60),
    )
    tracking.define_metric("train/step")
    tracking.define_metric("train/*", step_metric="train/step")
    tracking.define_metric("validation/step")
    tracking.define_metric("validation/*", step_metric="validation/step")
    tracking.define_metric("eval/sweeps")
    tracking.define_metric("eval/*", step_metric="eval/sweeps")
    tracking.summary["objective"] = study["training"].get("objective", "legacy_energy_gradient_surrogate")
    identity = {"id": tracking.id, "project": tracking.project, "entity": tracking.entity, "url": tracking.url}
    write_json(output / "wandb.json", identity)
    return tracking, identity


def log_checkpoint(tracking, output, step):
    """Snapshot checkpoint and its evidence together; W&B versions the immutable bundle."""
    artifact = wandb.Artifact(
        f"neio-{tracking.id}",
        type="model",
        metadata={"step": step, "primary_weights": "best.pt" if (output / "best.pt").exists() else "latest.pt"},
    )
    for name in (
        "latest.pt",
        "best.pt",
        "training.jsonl",
        "validation.json",
        "validation.jsonl",
        "provenance.json",
        "wandb.json",
    ):
        path = output / name
        if path.exists():
            artifact.add_file(str(path), name=name)
    if "latest.pt" not in artifact.manifest.entries:
        raise ValueError("Cannot log a model artifact without latest.pt")
    return tracking.log_artifact(artifact, aliases=["latest", f"step-{step}"])


def log_training(tracking, row):
    """Publish scalar learning curves; retain detailed sampling records in JSONL."""
    values = {f"train/{key}": value for key, value in row.items() if key != "rank_metrics"}
    samples = {}
    for index, measured in enumerate(row.get("rank_metrics", [])):
        rank = measured.get("rank", index)  # Historical uploads contain one sample per rank.
        for key, value in measured.items():
            if (
                key
                in (
                    "particle_residual_ratio",
                    "body_residual_ratio",
                    "residual_loss",
                    "rejected_updates",
                    "gradient_norm_before_clipping",
                )
                or "last_sweep_" in key
            ):
                samples.setdefault(f"train/{measured['workload']}/rank_{rank}/{key}", []).append(value)
    for key, sampled in samples.items():
        values[key] = float(np.mean(sampled)) if all(value is not None for value in sampled) else None
    tracking.log(values)


def upload_existing(args, output):
    """Retain the rejected pilots as explicitly labeled historical records, without rerunning physics."""
    metadata = json.loads((output / "provenance.json").read_text())
    checkpoint = torch.load(output / "latest.pt", map_location="cpu", weights_only=False)
    args.arm, args.seed = metadata["arguments"]["arm"], metadata["arguments"]["seed"]
    identity_path = output / "wandb.json"
    previous = json.loads(identity_path.read_text()) if identity_path.exists() else None
    tracking, _ = start_tracking(args, metadata["study"], metadata, output, previous)
    try:
        tracking.summary["scientific_status"] = "Superseded pilot; insufficient physical scene coverage"
        tracking.summary["original_workflow_status"] = "FAILED_CANCELED"
        tracking.summary["checkpoint_step"] = checkpoint["step"]
        imported = tracking.summary.get("imported_through_step", 0)
        for line in (output / "training.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["step"] > imported:
                log_training(tracking, row)
                tracking.summary["imported_through_step"] = row["step"]
        validation_path = output / "validation.json"
        if validation_path.exists():
            validation = json.loads(validation_path.read_text())
            tracking.log(
                {"validation/step": validation["step"], "validation/cloth_selection_score": validation["score"]}
            )
        artifact = log_checkpoint(tracking, output, checkpoint["step"])
        artifact.wait()
        tracking.summary["checkpoint_artifact"] = artifact.qualified_name
    except BaseException:
        tracking.finish(exit_code=1)
        raise
    tracking.finish()


def source_provenance():
    dependencies = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    for requirement in dependencies:
        name, version = requirement.split("==")
        if importlib.metadata.version(name).split("+")[0] != version:
            raise RuntimeError(
                f"Expected {requirement}; use python_project=scripts/neio or uv run --project scripts/neio"
            )
    lock = json.loads((ROOT / "source.lock.json").read_text())
    manifest = json.loads((ROOT / "source.manifest.json").read_text())
    for relative, expected in manifest.items():
        path = ROOT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Dependency does not match its manifest: {relative}")
    if Path(newton.__file__).resolve() != ROOT / "_source/newton/__init__.py":
        raise RuntimeError("The study must use its pinned Newton dependency")
    return {
        "source": lock,
        "manifest_sha256": hashlib.sha256((ROOT / "source.manifest.json").read_bytes()).hexdigest(),
        "implementation_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted([*ROOT.glob("*.py"), ROOT / "study.json"])
        },
        "packages": {p: importlib.metadata.version(p) for p in ("torch", "warp-lang", "mujoco", "mujoco-warp")},
        "workflow_id": os.environ.get("ISAACLAB_WORKFLOW_ID"),
        "image_tag": "lab",
        "image_digest": os.environ.get("NEIO_IMAGE_DIGEST"),
        "git_revision": os.environ.get("NEIO_GIT_REVISION"),
        "device": torch.cuda.get_device_name(),
        "cuda": torch.version.cuda,
    }


def validate(network, study, device, sweeps, banks=None):
    scores, rows = {}, []
    banks = {} if banks is None else banks
    for case in study["validation"]:
        limits = dict(study["acceptance"][case["family"]])
        if case.get("thread_speed_rad_s"):
            # Multi-turn engagement permits axial travel; the rollout checks
            # travel against the authored pitch, as well as lateral escape.
            limits.pop("m12_axial_displacement_m", None)
        case = {**case, "acceptance": limits}
        if case["name"] not in banks:
            scene = make_scene(case, study["seeds"]["validation"][0], device)
            count = int(np.ceil(case.get("duration", 16 * scene.dt) / scene.dt))
            banks[case["name"]] = scene, reference_states(scene, count, max(1, count // 96))
        scene, states = banks[case["name"]]
        solver = make_solver(scene, "shared", sweeps, network, history=study["training"]["history_at_step_start"])
        case_scores = []
        for index in np.linspace(0, len(states) - 1, 5, dtype=int):
            t, saved = states[index]
            result, _ = probe(scene, solver, saved, t, repeats=1)
            ratios = [
                result[k] for k in ("particle_residual_ratio", "body_residual_ratio") if result.get(k) is not None
            ]
            final = result["neio_objective_gradient"]
            solved = all(
                final[key] is not None and abs(final[key]) <= 1e-7
                for key in ("particle_final_n", "body_final_mixed_units")
            )
            score = np.mean(np.log1p(np.minimum(ratios, 1e6))) if ratios else (0.0 if solved else 1e6)
            if not result["bounded"] or result.get("contact_overflow", False):
                score = 1e6
            case_scores.append(float(score))
            rows.append(
                {
                    "case": case["name"],
                    "family": case["family"],
                    "sample_time_seconds": t,
                    "score": float(score),
                    **result,
                }
            )
        scores.setdefault(case["family"], []).append(float(np.mean(case_scores)))
    return float(np.mean([np.mean(values) for values in scores.values()])), rows


def log_validation(tracking, output, step, score, rows):
    """Log comparable fixed-state measurements, including the untrained baseline."""
    record = {"step": step, "score": score, "cases": rows}
    write_json(output / "validation.json", record)
    with (output / "validation.jsonl").open("a") as log:
        log.write(json.dumps(record, allow_nan=False) + "\n")
    values = {"step": step, "selection_score": score}
    identity_by_family = {}
    for case in sorted({row["case"] for row in rows}):
        measured = [row for row in rows if row["case"] == case]
        values[f"{case}/residual_score"] = float(np.mean([row["score"] for row in measured]))
        # With no correction, an active residual has ratio one. Use the same
        # workload/family weighting as checkpoint selection, including solved states.
        gradients = [row["neio_objective_gradient"] for row in measured]
        active = [max(g["particle_initial_n"] or 0, g["body_initial_mixed_units"] or 0) > 1e-12 for g in gradients]
        identity = float(np.mean(active) * np.log(2))
        identity_by_family.setdefault(measured[0]["family"], []).append(identity)
        values[f"{case}/improvement_over_no_update"] = identity - values[f"{case}/residual_score"]
        for kind in ("particle", "body"):
            ratios = [
                row[f"{kind}_residual_ratio"] for row in measured if row.get(f"{kind}_residual_ratio") is not None
            ]
            if ratios:
                values[f"{case}/{kind}_residual_ratio"] = float(np.mean(ratios))
        values[f"{case}/failure_fraction"] = float(
            np.mean(
                [
                    not row["bounded"]
                    or row.get("contact_overflow", False)
                    or bool(row.get("physical_threshold_failures"))
                    for row in measured
                ]
            )
        )
    for key in (
        "particle_residual_ratio",
        "body_residual_ratio",
        "max_edge_strain",
        "rigid_penetration_m",
        "soft_penetration_m",
        "eager_substep_ms",
    ):
        measured = [row[key] for row in rows if row.get(key) is not None]
        if measured:
            values[key] = float(np.mean(measured) if key.endswith(("_ratio", "_ms")) else max(measured))
    values["failure_fraction"] = float(
        np.mean(
            [
                not row["bounded"] or row.get("contact_overflow", False) or bool(row.get("physical_threshold_failures"))
                for row in rows
            ]
        )
    )
    ratios = [
        row[key]
        for row in rows
        for key in ("particle_residual_ratio", "body_residual_ratio")
        if row.get(key) is not None
    ]
    values["residual_reduction_fraction"] = float(np.mean(np.asarray(ratios) < 1)) if ratios else 0.0
    values["no_update_score"] = float(np.mean([np.mean(v) for v in identity_by_family.values()]))
    values["improvement_over_no_update"] = values["no_update_score"] - score
    values["substantial_residual_reduction_fraction"] = float(np.mean(np.asarray(ratios) < 0.9)) if ratios else 0.0
    tracking.log({f"validation/{key}": value for key, value in values.items()})


class TrainingTrajectory:
    """Own the evolving simulator state between samples of one training workload."""

    def __init__(self, scene, states, solver):
        self.scene, self.states, self.solver = scene, states, solver
        self.a, self.b = scene.model.state(), scene.model.state()
        self.control = scene.model.control()
        self.pipeline = newton.CollisionPipeline(scene.model, contact_matching="latest")
        self.contacts = self.pipeline.contacts()
        self.step = 0

    def prepare(self, seed, horizon):
        if self.step == 0 or self.step >= horizon:
            rng = np.random.default_rng(seed)
            self.time, self.saved = self.states[int(rng.integers(len(self.states)))]
            restore(self.a, self.saved, self.control)
            restore(self.b, self.saved, self.control)
            self.solver.reset(self.a, flags=0)
            self.pipeline.reset_contact_matching()
            self.step = 0
        # Hold the recorded command over the short trajectory. Its positions and
        # velocities after the first timestep come from the learned solver.
        for name in self.saved.keys() & {"body_f", "particle_f"}:
            getattr(self.a, name).assign(self.saved[name])
        self.scene.prepare(self.a, self.time + self.step * self.scene.dt, self.control, sampled=True)
        self.pipeline.collide(self.a, self.contacts)


def train(args, study, network, device, rank, ranks, output, metadata, tracking, checkpoint=None):
    config = study["training"]
    optimizer = torch.optim.AdamW(network.parameters(), lr=config["learning_rate"])
    parameters = list(network.parameters())
    start_step, best_step, best, best_network = 0, 0, float("inf"), None
    if checkpoint:
        network.load_state_dict(checkpoint["network"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step, best = checkpoint["step"], checkpoint["best"]
        best_step = checkpoint["best_step"]
        best_network = checkpoint["best_network"]
        torch.set_rng_state(checkpoint["rng"][rank]["cpu"].cpu())
        torch.cuda.set_rng_state(checkpoint["rng"][rank]["cuda"].cpu(), device)
    families = study["arms"][args.arm]
    batch_size = config["samples_per_step"]
    begin = time.perf_counter()
    nodes, oracle_calls, backward_calls, rejected = 0, 0, 0, 0
    banks, validation_banks = {}, {}
    if start_step == 0 and rank == 0:
        score, validation = validate(network, study, device, args.sweeps, validation_banks)
        best = score
        best_network = {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}
        log_validation(tracking, output, 0, score, validation)
    for step in range(start_step, args.steps):
        horizon = max(length for begin_step, length in config["trajectory_curriculum"] if step >= begin_step)
        accumulated = [torch.zeros_like(parameter) for parameter in parameters]
        bad = torch.zeros((), dtype=torch.int32, device=device)
        sample_metrics, gradient_norms = [], []
        for sample in range(batch_size):
            index = (step * batch_size + sample) * ranks + rank
            family = families[index % len(families)]
            cases = [case for case in study["train"] if case["family"] == family]
            case = cases[(index // len(families)) % len(cases)]
            scene_seed = args.seed + 100000 * rank
            if case["name"] not in banks:
                scene = make_scene(case, scene_seed, device, args.worlds)
                directory = output / "banks" / f"rank-{rank}" / case["name"]
                count = int(np.ceil(case.get("duration", 96 * scene.dt) / scene.dt))
                states = reference_states(scene, count, max(1, count // 96), directory)
                solver = make_solver(
                    scene, "shared", args.sweeps, network, history=config["history_at_step_start"], learn=True
                )
                banks[case["name"]] = TrainingTrajectory(scene, states, solver)
                print(json.dumps({"bank_ready": case["name"], "rank": rank, "samples": len(states)}), flush=True)
            trajectory = banks[case["name"]]
            scene, solver = trajectory.scene, trajectory.solver
            trajectory.prepare(args.seed + 1009 * index, horizon)
            optimizer.zero_grad(set_to_none=True)
            solver.residuals.clear()
            solver.losses.clear()
            solver.record_convergence = (step + 1) % 8 == 0 or step + 1 == args.steps
            solver.convergence_slot = 0
            before = solver.failed_updates
            solver.step(trajectory.a, trajectory.b, trajectory.control, trajectory.contacts, scene.dt)
            valid = solver.failed_updates == before and all(
                parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in parameters
            )
            if valid:
                norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
                gradient_norms.append(norm)
                for total, parameter in zip(accumulated, parameters):
                    total.add_(parameter.grad)
            else:
                bad.fill_(1)
                norm = None
            nodes += scene.model.particle_count + 2 * scene.model.body_count
            oracle_calls += 2 * args.sweeps + int(solver.record_convergence)
            backward_calls += args.sweeps
            if solver.record_convergence:
                convergence = solver.convergence_buffer[0].tolist()
                measured = {
                    "rank": rank,
                    "family": family,
                    "workload": case["name"],
                    "sample_time_seconds": trajectory.time + trajectory.step * scene.dt,
                    "trajectory_step": trajectory.step,
                    "bank_states": len(trajectory.states),
                    "scene_seed": scene_seed,
                    "residual_loss": float(np.mean(solver.losses))
                    if solver.losses and np.isfinite(solver.losses).all()
                    else None,
                    "gradient_norm_before_clipping": norm,
                    "processed_nodes": nodes,
                    "oracle_calls": oracle_calls,
                    "oracle_backward_calls": backward_calls,
                }
                for kind, initial, final in (("particle", *convergence[:2]), ("body", *convergence[2:])):
                    if initial > 1e-12 and np.isfinite(initial) and np.isfinite(final):
                        measured[f"{kind}_residual_ratio"] = final / initial
                for adapter in (solver.particle_optimizer, solver.rigid_optimizer):
                    if adapter is not None and bool(adapter.graph.free.any()):
                        applied = adapter.last_update[adapter.graph.free]
                        measured[f"{adapter.kind}_last_sweep_update_rms_m"] = float(applied.square().mean().sqrt())
                        measured[f"{adapter.kind}_last_sweep_zero_fraction"] = float(
                            (applied == 0).all(-1).float().mean()
                        )
                sample_metrics.append(measured)
            trajectory.a, trajectory.b = trajectory.b, trajectory.a
            trajectory.step = trajectory.step + 1 if valid else 0
        if ranks > 1:
            dist.all_reduce(bad, op=dist.ReduceOp.MAX)
        if not bool(bad):
            for parameter, total in zip(parameters, accumulated):
                if ranks > 1:
                    dist.all_reduce(total)
                parameter.grad = total / (batch_size * ranks)
            optimizer.step()
        else:
            rejected += 1
        if (step + 1) % 8 == 0 or step + 1 == args.steps:
            gathered = [None] * ranks
            if ranks > 1:
                dist.all_gather_object(gathered, sample_metrics)
            else:
                gathered = [sample_metrics]
            if rank == 0:
                rank_metrics = [measured for samples in gathered for measured in samples]
                row = {
                    "step": step + 1,
                    "sampled_timesteps": (step + 1) * batch_size * ranks,
                    "trajectory_horizon": horizon,
                    "rejected_updates": rejected,
                    "loss": float(np.mean([m["residual_loss"] for m in rank_metrics]))
                    if all(m["residual_loss"] is not None for m in rank_metrics)
                    else None,
                    "gradient_norm_before_clipping": float(np.mean(gradient_norms)) if gradient_norms else None,
                    "elapsed_seconds": time.perf_counter() - begin,
                    "rank_metrics": rank_metrics,
                }
                with (output / "training.jsonl").open("a") as log:
                    log.write(json.dumps(row, allow_nan=False) + "\n")
                print(json.dumps(row), flush=True)
                log_training(tracking, row)
        if (step + 1) % config["checkpoint_every"] == 0 or step + 1 == args.steps:
            # Resuming a checkpoint starts the same next reference sample without
            # serializing simulator internals, contact caches, or hidden history.
            for trajectory in banks.values():
                trajectory.step = 0
            rng = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device)}
            all_rng = [None] * ranks
            if ranks > 1:
                dist.all_gather_object(all_rng, rng)
            else:
                all_rng = [rng]
            if rank == 0:
                improved = False
                if (step + 1) % 256 == 0 or step + 1 == args.steps:
                    score, validation = validate(network, study, device, args.sweeps, validation_banks)
                    improved = score < best
                    best = min(best, score)
                    if improved:
                        best_step = step + 1
                        best_network = {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}
                    log_validation(tracking, output, step + 1, score, validation)
                checkpoint = {
                    "network": network.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step + 1,
                    "best": best,
                    "best_step": best_step,
                    "best_network": best_network,
                    "rng": all_rng,
                    "study": study,
                    "configuration": {**vars(args), "ranks": ranks},
                    "provenance": metadata,
                    "wandb": metadata["wandb"],
                }
                torch.save(checkpoint, output / "latest.pt.tmp")
                (output / "latest.pt.tmp").replace(output / "latest.pt")
                torch.save(
                    {"network": best_network, "score": best, "step": best_step, "study": study, "provenance": metadata},
                    output / "best.pt.tmp",
                )
                (output / "best.pt.tmp").replace(output / "best.pt")
                if (step + 1) % 256 == 0 or step + 1 == args.steps:
                    log_checkpoint(tracking, output, step + 1)
            if ranks > 1:
                dist.barrier()
        if rejected > max(8, (step - start_step + 1) // 4):
            raise RuntimeError("More than 25% of training updates have missing or nonfinite gradients")

    if rank == 0 and (output / "banks").exists():
        artifact = wandb.Artifact(f"training-states-{tracking.id}", type="dataset")
        artifact.add_dir(str(output / "banks"))
        tracking.log_artifact(artifact)


def evaluate(args, study, network, device, output, tracking):
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if checkpoint.get("study") != study:
            raise ValueError("Evaluation requires a checkpoint from this study, including its update parameterization")
        network.load_state_dict(checkpoint["network"])
        tracking.summary["evaluation/checkpoint_step"] = checkpoint["step"]
    elif "shared" in args.solvers.split(","):
        raise ValueError("Shared model evaluation requires --checkpoint; random weights are not trained NeIO")
    rows = []
    cases = study[args.split]
    if args.cases:
        selected = set(args.cases.split(","))
        if selected - {c["name"] for c in cases}:
            raise ValueError(f"Unknown {args.split} cases: {selected}")
        cases = [case for case in cases if case["name"] in selected]
    for case in cases:
        limits = dict(study["acceptance"][case["family"]])
        if case.get("thread_speed_rad_s"):
            limits.pop("m12_axial_displacement_m", None)
        case = {**case, "acceptance": limits}
        for seed in study["seeds"][args.split][: args.eval_seeds]:
            try:
                scene = make_scene(case, seed, device)
                count = max(args.probe_steps, int(np.ceil(case.get("duration", 0) / scene.dt)))
                states = reference_states(scene, count, max(1, count // 4), output / "banks" / f"{case['name']}-{seed}")
                states = [states[i] for i in np.linspace(0, len(states) - 1, 5, dtype=int)]
            except Exception as error:
                rows.append({"case": case["name"], "seed": seed, "status": "setup_failed", "error": str(error)})
                write_json(output / "evaluation.json", rows)
                continue
            references, reference_failures = {}, {}
            if not args.quick:
                for reference_name in ("vbd", "mjwarp"):
                    try:
                        lower = make_solver(scene, reference_name, 200)
                        upper = make_solver(scene, reference_name, 400)
                        references[reference_name] = []
                        for t, saved in states:
                            _, state200 = probe(scene, lower, saved, t, 1)
                            _, state400 = probe(scene, upper, saved, t, 1)
                            references[reference_name].append(
                                (state400, state_distance(state200, state400, scene.case.get("length_unit_m", 1)))
                            )
                    except (NotImplementedError, RuntimeError, ValueError) as error:
                        references.pop(reference_name, None)
                        reference_failures[reference_name] = str(error)
            for name in args.solvers.split(","):
                sweep_values = [args.sweeps] if args.quick and name != "pgs" else study["sweeps"][name]
                for sweeps in sweep_values:
                    do_rollout = args.rollout_steps and (args.quick or sweeps in study["rollout_sweeps"][name])
                    row = {
                        "case": case["name"],
                        "seed": seed,
                        "solver": name,
                        "formulation": "compliant_alm"
                        if name == "vbd"
                        else "cubic_penalty"
                        if name in ("shared", "supplied")
                        else "mujoco_constraints",
                        "sweeps": sweeps,
                        "scene": scene.case,
                        "dt": scene.dt,
                        "history": "cold",
                        "rollout_steps": max(args.rollout_steps, count) if do_rollout else 0,
                        "status": "complete",
                        "reference_failures": reference_failures,
                    }
                    try:
                        solver = make_solver(scene, name, sweeps, network)
                        row["probes"] = []
                        for index, (t, saved) in enumerate(states):
                            measurement, final_state = probe(
                                scene, solver, saved, t, 1 if args.quick else 3, capture=not args.quick
                            )
                            measurement["time"] = t
                            measurement["reference_distances"] = {
                                key: {
                                    "distance_to_400": state_distance(
                                        final_state, values[index][0], scene.case.get("length_unit_m", 1)
                                    ),
                                    "change_200_to_400": values[index][1],
                                    "same_formulation": key == name,
                                }
                                for key, values in references.items()
                            }
                            row["probes"].append(measurement)
                        if any(not r["bounded"] or r.get("contact_overflow") for r in row["probes"]):
                            row["status"] = "numerical_failure"
                        row["physical_threshold_pass"] = all(
                            r["bounded"] and not r.get("contact_overflow") and not r.get("physical_threshold_failures")
                            for r in row["probes"]
                        )
                        row["probe_physical_threshold_pass"] = row["physical_threshold_pass"]
                        if do_rollout:
                            rollout_scene = make_scene(case, seed, device)
                            history = study["training"]["history_at_step_start"] if name == "shared" else "carry"
                            solver = make_solver(rollout_scene, name, sweeps, network, history=history)
                            capture = not args.quick and all(p.get("capture_parity_pass", False) for p in row["probes"])
                            row["rollout"] = rollout(
                                rollout_scene, solver, max(args.rollout_steps, count), capture=capture
                            )
                            row["rollout"]["history_at_step_start"] = history
                            row["rollout"]["execution"] = "captured" if capture else "eager"
                            row["physical_threshold_pass"] &= not bool(row["rollout"]["physical_threshold_failures"])
                            if row["rollout"]["failed_at_step"] is not None:
                                row["physical_threshold_pass"] = False
                                row["status"] = "numerical_failure"
                    except NotImplementedError as error:
                        row.update(status="unsupported", error=str(error))
                    except Exception as error:
                        row.update(status="failed", error=str(error))
                    rows.append(row)
                    write_json(output / "evaluation.json", rows)
                    log_evaluation(tracking, row)
                    print(json.dumps({k: row[k] for k in ("case", "solver", "sweeps", "status")}), flush=True)
    if not any(row["status"] == "complete" for row in rows):
        raise RuntimeError("Evaluation produced no completed comparisons; inspect evaluation.json")


def log_evaluation(tracking, row):
    """Expose solver accuracy and cost as sweep curves, retaining failed comparisons."""
    if row["status"] == "unsupported":
        return
    values = {
        "failure": float(row["status"] != "complete"),
        "probe_physical_pass": float(row.get("probe_physical_threshold_pass", False)),
    }
    probes = row.get("probes", [])
    if probes:
        values["capture_parity_pass"] = float(all(p.get("capture_parity_pass", False) for p in probes))
    for key in (
        "particle_residual_ratio",
        "body_residual_ratio",
        "eager_substep_ms",
        "captured_substep_ms",
        "rigid_penetration_m",
        "soft_penetration_m",
        "joint_anchor_gap_m",
        "max_edge_strain",
    ):
        if key == "captured_substep_ms" and not all(p.get("capture_parity_pass", False) for p in probes):
            continue
        measured = [p[key] for p in probes if p.get(key) is not None and np.isfinite(p[key])]
        if measured:
            values[key] = float(np.mean(measured) if key.endswith(("_ms", "_ratio")) else max(measured))
    for reference in ("vbd", "mjwarp"):
        for key in ("particle_q_rms_m", "body_q_rms_m", "body_angle_max_rad"):
            measured = [
                p["reference_distances"][reference]["distance_to_400"][key]
                for p in probes
                if key in p.get("reference_distances", {}).get(reference, {}).get("distance_to_400", {})
            ]
            if measured and all(np.isfinite(measured)):
                values[f"distance_to_{reference}_400/{key}"] = float(max(measured))
    trajectory = row.get("rollout", {})
    values.update(
        {
            f"rollout/{key}": value
            for key, value in (trajectory.get("threading", {}) | trajectory.get("settling", {})).items()
            if value is not None and np.isfinite(value)
        }
    )
    if trajectory:
        values["rollout/failure"] = float(
            trajectory["failed_at_step"] is not None
            or bool(trajectory["physical_threshold_failures"])
            or trajectory.get("native_test_pass") is False
        )
    tracking.log(
        {"eval/sweeps": row["sweeps"]}
        | {f"eval/{row['case']}/{row['solver']}/{key}": value for key, value in values.items()}
    )


def verify_workloads(args, study, device, output, tracking):
    reports = []
    cases = {case["name"]: case for split in ("train", "validation", "test") for case in study[split]}
    selected = set(args.cases.split(",")) if args.cases else set(cases)
    if selected - set(cases):
        raise ValueError(f"Unknown verification cases: {selected - set(cases)}")
    for name in sorted(selected):
        case = cases[name]
        scene = make_scene(case, args.seed, device)
        count = int(np.ceil(case.get("duration", 1) / scene.dt))
        states = reference_states(scene, count, max(1, count // 96), output / "banks" / name)
        reports.append({"name": name, "samples": len(states), "evidence": scene.evidence})
        write_json(output / "verification.json", reports)
        tracking.log({"data/case": name, "data/samples": len(states)})
        print(json.dumps({"verified": name, "samples": len(states)}), flush=True)
    artifact = wandb.Artifact(f"workloads-{tracking.id}", type="dataset")
    artifact.add_dir(str(output / "banks"))
    artifact.add_file(str(output / "verification.json"))
    tracking.log_artifact(artifact).wait()


def load_training_checkpoint(args, study, device, ranks, metadata):
    checkpoint = torch.load(args.resume, map_location=device, weights_only=False) if args.resume else None
    if checkpoint:
        for key in ("arm", "seed", "worlds", "sweeps", "ranks"):
            current = ranks if key == "ranks" else getattr(args, key)
            if checkpoint["configuration"][key] != current:
                raise ValueError(f"Resume changes {key}; start a new run instead")
        if checkpoint["study"] != study or checkpoint["provenance"]["source"] != metadata["source"]:
            raise ValueError("Resume changes the study or source revision")
        if checkpoint["step"] % study["training"]["checkpoint_every"]:
            raise ValueError("Resume requires a completed optimizer checkpoint")
    return checkpoint


def main():
    study = json.loads((ROOT / "study.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "verify", "train", "eval", "upload"), default="preflight")
    parser.add_argument("--study", default=str(ROOT / "study.json"))
    parser.add_argument("--arm", choices=tuple(study["arms"]), default="cloth")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--steps", type=int, default=study["training"]["steps"])
    parser.add_argument("--worlds", type=int, default=study["training"]["worlds"])
    parser.add_argument("--sweeps", type=int, default=study["training"]["sweeps"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--run_name", default="neio-study")
    parser.add_argument("--output", default=os.environ.get("ISAACLAB_OUTPUT_DIR", "logs/neio"))
    parser.add_argument("--image-digest")
    parser.add_argument("--git-revision")
    parser.add_argument("--resume")
    parser.add_argument("--checkpoint")
    parser.add_argument("--solvers", default="shared,supplied,vbd,mjwarp,pgs")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--cases")
    parser.add_argument("--eval-seeds", type=int, default=1)
    parser.add_argument("--probe-steps", type=int, default=48)
    parser.add_argument("--rollout-steps", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--eval-after", action="store_true")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "neio-transfer"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    args = parser.parse_args()
    study = json.loads(Path(args.study).read_text())
    if args.mode in ("train", "eval") and study["status"] != "validated":
        parser.error("The scene suite needs verified replacement workloads before launching training or evaluation.")
    if args.mode in ("train", "eval", "verify") and any(
        (
            "example" not in case
            and case["family"] != "m12"
            and case.get("workload") not in ("rigid_pile", "franka_payload", "anymal_disturbance")
        )
        or case.get("duration", 0) < 0.5
        for split in ("train", "validation", "test")
        for case in study[split]
    ):
        parser.error("Production workloads must use full native, collision-pile, robot, or threaded M12 scenes")
    args.solvers = args.solvers.strip("[]")
    args.cases = args.cases.strip("[]") if args.cases else None
    if set(args.solvers.split(",")) - set(study["sweeps"]):
        parser.error("Unknown solver in --solvers")
    if args.mode == "train" and args.worlds != 1:
        parser.error("Use one complete native scene per GPU process (--worlds=1)")
    if min(args.steps, args.worlds, args.sweeps, args.eval_seeds, args.probe_steps) <= 0 or args.rollout_steps < 0:
        parser.error("Counts must be positive and rollout steps must be nonnegative")
    if args.mode == "train" and args.steps % study["training"]["checkpoint_every"]:
        parser.error("Training steps must end at a checkpoint boundary (multiple of 32)")
    if args.mode == "train" and args.seed not in study["seeds"]["train"]:
        parser.error("Training must use a registered training seed")
    rank, ranks = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    output = Path(args.output) / args.run_name
    if args.mode == "upload":
        if ranks != 1:
            parser.error("Upload existing logs once, without a distributed launcher")
        upload_existing(args, output)
        return
    device = f"cuda:{os.environ['LOCAL_RANK']}" if ranks > 1 else args.device
    torch.cuda.set_device(device)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    wp.init()
    wp.set_device(device)
    torch.manual_seed(args.seed)
    if ranks > 1:
        # A rank can generate a full threaded-contact bank while its peer waits
        # at the next gradient reduction. Keep that deliberate work within timeout.
        dist.init_process_group("nccl", device_id=torch.device(device), timeout=timedelta(minutes=60))
    if args.mode == "train" and (output / "latest.pt").exists() and not args.resume:
        raise ValueError("The run already has a checkpoint; choose a new run name or resume it")
    output.mkdir(parents=True, exist_ok=True)
    metadata = source_provenance()
    metadata.update(image_digest=args.image_digest, git_revision=args.git_revision)
    if args.mode == "preflight":
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--noconftest", str(ROOT / "test_study.py"), "-q"],
            env={**os.environ, "NEIO_TEST_DEVICE": device},
            check=False,
        )
        raise SystemExit(result.returncode)
    network = new_network(device)
    checkpoint = load_training_checkpoint(args, study, device, ranks, metadata)
    tracking, started = None, [None]
    if rank == 0:
        try:
            tracking, identity = start_tracking(
                args, study, metadata, output, checkpoint.get("wandb") if checkpoint else None
            )
            started[0] = {"identity": identity}
        except Exception as error:
            started[0] = {"error": f"W&B initialization failed: {error}"}
    if ranks > 1:
        dist.broadcast_object_list(started)
    if "error" in started[0]:
        if ranks > 1:
            dist.destroy_process_group()
        raise RuntimeError(started[0]["error"])
    metadata["wandb"] = started[0]["identity"]
    if rank == 0:
        write_json(output / "provenance.json", {**metadata, "arguments": vars(args), "study": study})
    exit_code = 1
    try:
        if args.mode == "verify":
            verify_workloads(args, study, device, output, tracking)
        elif args.mode == "train":
            train(args, study, network, device, rank, ranks, output, metadata, tracking, checkpoint)
            if args.eval_after and rank == 0:
                args.checkpoint = str(output / "best.pt")
                evaluate(args, study, network, device, output, tracking)
        elif rank == 0:
            evaluate(args, study, network, device, output, tracking)
        exit_code = 0
    finally:
        if tracking:
            if (output / "evaluation.json").exists():
                artifact = wandb.Artifact(f"evaluation-{tracking.id}", type="evaluation")
                artifact.add_file(str(output / "evaluation.json"), name="evaluation.json")
                tracking.log_artifact(artifact)
            tracking.finish(exit_code=exit_code)
        if ranks > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
