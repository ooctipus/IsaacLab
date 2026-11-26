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
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "_source"))

import newton
import numpy as np
import torch
import torch.distributed as dist
import warp as wp
from benchmark import make_solver, probe, reference_states, rollout, state_distance
from optimizer import StudySolver, new_network
from scenes import make_scene


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


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
        "packages": {p: importlib.metadata.version(p) for p in ("torch", "warp-lang", "mujoco", "mujoco-warp")},
        "workflow_id": os.environ.get("ISAACLAB_WORKFLOW_ID"),
        "image_tag": "lab",
        "image_digest": os.environ.get("NEIO_IMAGE_DIGEST"),
        "git_revision": os.environ.get("NEIO_GIT_REVISION"),
        "device": torch.cuda.get_device_name(),
        "cuda": torch.version.cuda,
    }


def validate(network, study, device, sweeps):
    scores, rows = [], []
    for case in study["validation"]:
        case = {**case, "acceptance": study["acceptance"][case["family"]]}
        scene = make_scene(case, study["seeds"]["validation"][0], device)
        saved = reference_states(scene, 16, 16)[-1]
        result, _ = probe(scene, make_solver(scene, "shared", sweeps, network), saved[1], saved[0], repeats=1)
        ratios = [result[k] for k in ("particle_residual_ratio", "body_residual_ratio") if result.get(k) is not None]
        score = np.mean(np.log1p(np.minimum(ratios, 1e6))) if ratios and result["bounded"] else 1e6
        if case["family"] == "cloth":
            scores.append(float(score))
        rows.append({"case": case["name"], "score": float(score), **result})
    return float(np.mean(scores)), rows


def train(args, study, network, device, rank, ranks, output, metadata):
    config = study["training"]
    optimizer = torch.optim.AdamW(network.parameters(), lr=config["learning_rate"])
    start_step, best, best_network = 0, float("inf"), None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        for key in ("arm", "seed", "worlds", "sweeps", "ranks"):
            current = ranks if key == "ranks" else getattr(args, key)
            if checkpoint["configuration"][key] != current:
                raise ValueError(f"Resume changes {key}; start a new run instead")
        if checkpoint["study"] != study or checkpoint["provenance"]["source"] != metadata["source"]:
            raise ValueError("Resume changes the study or source revision")
        network.load_state_dict(checkpoint["network"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step, best = checkpoint["step"], checkpoint["best"]
        best_network = checkpoint["best_network"]
        torch.set_rng_state(checkpoint["rng"][rank]["cpu"].cpu())
        torch.cuda.set_rng_state(checkpoint["rng"][rank]["cuda"].cpu(), device)
    families = study["arms"][args.arm]
    episode_steps = config["episode_steps"]
    if start_step % episode_steps:
        raise ValueError("Resume requires an episode-boundary checkpoint")
    begin = time.perf_counter()
    nodes, oracle_calls, rejected = 0, 0, 0
    for step in range(start_step, args.steps):
        within = step % episode_steps
        if within == 0:
            episode = step // episode_steps
            family = families[(episode * ranks + rank) % len(families)]
            scene_seed = args.seed + 100000 * rank + 1009 * episode
            rng = np.random.default_rng(scene_seed)
            case = {
                "family": family,
                "drive": family == "articulation",
                "amplitude": float(rng.uniform(0.1, 1)),
                "frequency": float(rng.uniform(1, 6)),
                "obstacle": bool(episode % 2),
                "twist": bool(episode % 3 == 0),
                "dt": float(rng.choice([1 / 240, 1 / 300, 1 / 600])),
            }
            scene = make_scene(case, scene_seed, device, args.worlds)
            model = scene.model
            a, b, control = model.state(), model.state(), model.control()
            solver = StudySolver(model, network, learn=True, history="clear", iterations=args.sweeps, **scene.options)
            pipeline = newton.CollisionPipeline(model)
            contacts = pipeline.contacts()
        scene.prepare(a, within * scene.dt, control)
        pipeline.collide(a, contacts)
        optimizer.zero_grad(set_to_none=True)
        solver.residuals.clear()
        before = solver.failed_updates
        solver.step(a, b, control, contacts, scene.dt)
        bad = torch.tensor(int(solver.failed_updates > before), device=device)
        for parameter in network.parameters():
            if parameter.grad is None:
                raise RuntimeError("A shared network parameter received no learning signal")
            bad = torch.maximum(bad, (~torch.isfinite(parameter.grad).all()).int())
        if ranks > 1:
            dist.all_reduce(bad, op=dist.ReduceOp.MAX)
        if not bool(bad):
            if ranks > 1:
                for parameter in network.parameters():
                    dist.all_reduce(parameter.grad)
                    parameter.grad.div_(ranks)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0))
            optimizer.step()
        else:
            rejected += 1
            grad_norm = None
        a, b = b, a
        nodes += model.particle_count + 2 * model.body_count
        oracle_calls += 2 * args.sweeps
        if rank == 0 and ((step + 1) % 8 == 0 or step + 1 == args.steps):
            row = {
                "step": step + 1,
                "family": family,
                "scene_seed": scene_seed,
                "gradient_norm_before_clipping": grad_norm,
                "post_update_gradient_proxy": solver.residuals[-1] if np.isfinite(solver.residuals[-1]) else None,
                "rejected_updates_rank0": rejected,
                "processed_nodes_rank0": nodes,
                "oracle_calls_rank0": oracle_calls,
                "elapsed_seconds": time.perf_counter() - begin,
            }
            with (output / "training.jsonl").open("a") as log:
                log.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row), flush=True)
        if (step + 1) % episode_steps == 0 or step + 1 == args.steps:
            rng = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device)}
            all_rng = [None] * ranks
            if ranks > 1:
                dist.all_gather_object(all_rng, rng)
            else:
                all_rng = [rng]
            if rank == 0:
                improved = False
                if (step + 1) % 256 == 0 or step + 1 == args.steps:
                    score, validation = validate(network, study, device, args.sweeps)
                    improved = score < best
                    best = min(best, score)
                    if improved:
                        best_network = {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}
                    write_json(output / "validation.json", {"step": step + 1, "score": score, "cases": validation})
                checkpoint = {
                    "network": network.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step + 1,
                    "best": best,
                    "best_network": best_network,
                    "rng": all_rng,
                    "study": study,
                    "configuration": {**vars(args), "ranks": ranks},
                    "provenance": metadata,
                }
                torch.save(checkpoint, output / "latest.pt.tmp")
                (output / "latest.pt.tmp").replace(output / "latest.pt")
                if best_network is not None:
                    torch.save({"network": best_network, "score": best, "provenance": metadata}, output / "best.pt.tmp")
                    (output / "best.pt.tmp").replace(output / "best.pt")
            if ranks > 1:
                dist.barrier()
        if rejected > max(8, (step - start_step + 1) // 4):
            raise RuntimeError("More than 25% of training updates are nonfinite; inspect the recorded failures")


def evaluate(args, study, network, device, output):
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        network.load_state_dict(checkpoint["network"])
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
        case = {**case, "acceptance": study["acceptance"][case["family"]]}
        for seed in study["seeds"][args.split][: args.eval_seeds]:
            try:
                scene = make_scene(case, seed, device)
                count = (
                    max(args.probe_steps, int(np.ceil(0.12 / scene.dt)))
                    if case["family"] == "m12"
                    else args.probe_steps
                )
                states = reference_states(scene, count, max(1, count // 2))
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
                            references[reference_name].append((state400, state_distance(state200, state400)))
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
                                    "distance_to_400": state_distance(final_state, values[index][0]),
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
                        if do_rollout:
                            solver = make_solver(scene, name, sweeps, network, history="carry")
                            row["carried_history_rollout"] = rollout(scene, solver, max(args.rollout_steps, count))
                            row["physical_threshold_pass"] &= not bool(
                                row["carried_history_rollout"]["physical_threshold_failures"]
                            )
                            if row["carried_history_rollout"]["failed_at_step"] is not None:
                                row["physical_threshold_pass"] = False
                                row["status"] = "numerical_failure"
                    except NotImplementedError as error:
                        row.update(status="unsupported", error=str(error))
                    except Exception as error:
                        row.update(status="failed", error=str(error))
                    rows.append(row)
                    write_json(output / "evaluation.json", rows)
                    print(json.dumps({k: row[k] for k in ("case", "solver", "sweeps", "status")}), flush=True)
    if not any(row["status"] == "complete" for row in rows):
        raise RuntimeError("Evaluation produced no completed comparisons; inspect evaluation.json")


def main():
    study = json.loads((ROOT / "study.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "train", "eval"), default="preflight")
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
    args = parser.parse_args()
    args.solvers = args.solvers.strip("[]")
    args.cases = args.cases.strip("[]") if args.cases else None
    if set(args.solvers.split(",")) - set(study["sweeps"]):
        parser.error("Unknown solver in --solvers")
    if min(args.steps, args.worlds, args.sweeps, args.eval_seeds, args.probe_steps) <= 0 or args.rollout_steps < 0:
        parser.error("Counts must be positive and rollout steps must be nonnegative")
    if args.mode == "train" and args.steps % study["training"]["episode_steps"]:
        parser.error("Training steps must end at an episode boundary (multiple of 32)")
    if args.mode == "train" and args.seed not in study["seeds"]["train"]:
        parser.error("Training must use a registered training seed")
    rank, ranks = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = f"cuda:{os.environ['LOCAL_RANK']}" if ranks > 1 else args.device
    torch.cuda.set_device(device)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    wp.init()
    wp.set_device(device)
    torch.manual_seed(args.seed)
    if ranks > 1:
        dist.init_process_group("nccl", device_id=torch.device(device))
    output = Path(args.output) / args.run_name
    if args.mode == "train" and (output / "latest.pt").exists() and not args.resume:
        raise ValueError("The run already has a checkpoint; choose a new run name or resume it")
    output.mkdir(parents=True, exist_ok=True)
    metadata = source_provenance()
    metadata.update(image_digest=args.image_digest, git_revision=args.git_revision)
    if rank == 0:
        write_json(output / "provenance.json", {**metadata, "arguments": vars(args), "study": study})
    if args.mode == "preflight":
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--noconftest", str(ROOT / "test_study.py"), "-q"],
            env={**os.environ, "NEIO_TEST_DEVICE": device},
            check=False,
        )
        raise SystemExit(result.returncode)
    network = new_network(device)
    if args.mode == "train":
        train(args, study, network, device, rank, ranks, output, metadata)
        if args.eval_after and rank == 0:
            args.checkpoint = str(output / "latest.pt")
            evaluate(args, study, network, device, output)
    elif rank == 0:
        evaluate(args, study, network, device, output)
    if ranks > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
