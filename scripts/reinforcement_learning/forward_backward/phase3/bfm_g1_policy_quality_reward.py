# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""bfm-rewardgate-20260806 broad-reward component of the frozen G1 policy-quality gate v1.

Campaign port completing bfm-ab-20260805's blocked measurement: the frozen
gate's 38-task x 10-episode broad-reward component, run exactly as
:mod:`g1_policy_quality_evidence` runs it (same rollout code object, same
frozen reward runtime, protocol pins, and normalization) on one compact
checkpoint, with the same HEAD-staleness patches as the tracking port
(``bfm_lafan_control`` registration token, ``target_artifact_root`` rename,
value-head model construction) plus one more of the same class: HEAD derives
``auxiliary_evidence_names`` from the value-helper terms rather than a replay
config field — this port re-derives them the same way and requires the frozen
8-name order.

The reward-inference dataset is the bfm-rewardgate-20260806 REGENERATED
artifact (the pinned original is unrecoverable on this host); its sha256 is
verified here, and the receipt carries the regenerated-not-original anchor
comparability caveat. The dataset is identical for both A/B arms.

The BFM-Zero source tree is also absent from this host, so the broad-reward
code identity is carried by the frozen parity fixtures
(``fixtures/bfm_reward_source_identity_v1.json`` and
``fixtures/runtime/bfm_reward_gpu_runtime_v1.json``) instead of re-hashing the
source files; the reward equations themselves are the dependency-free
:mod:`bfm_reward_runtime` whose CUDA parity against that source is
receipt-proven.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

from bfm_reward_runtime import (
    BFM_AUXILIARY_COST_COEFFICIENTS,
    BFM_AUXILIARY_EVIDENCE_NAMES,
    BFM_HARD_SAFETY_NAMES,
    BFM_REWARD_TASKS,
    BFM_REWARD_TASKS_SHA256,
    BfmRewardRuntime,
    infer_reward_contexts_from_dataset,
    reward_metric_rows,
)
from g1_policy_quality_evidence import (
    _POLICY_QUALITY_GATE,
    _BFMRewardContextPolicy,
    _broad_reward_rollout,
    _evaluation_history_factory,
    _mujoco_model_source_identity,
    _sha256,
    _source_sha256,
    _statistics,
    _write_csv,
)

_PRESET = "g1_lafan"
_SCHEMA = "bfm_rewardgate_20260806_g1_policy_quality_broad_reward_v1"
_SOURCE_IDENTITY_FIXTURE = Path(__file__).parent / "fixtures" / "bfm_reward_source_identity_v1.json"
_REQUIRED_METRICS = ("return", "safety_violation_rate", "termination_rate", "auxiliary_cost", "action_l2")


def _environment_tokens() -> frozenset[str]:
    """Native gate axes with the training-control LAFAN registration selected."""
    from motion_environment_identity import motion_environment_axes

    return frozenset((motion_environment_axes(_PRESET) - {"lafan"}) | {"bfm_lafan_control"})


def _auxiliary_evidence_names(value_helpers: object) -> tuple[str, ...]:
    """Mirror HEAD's stored-evidence channel derivation from value helpers."""
    names: list[str] = []
    for helper in value_helpers:
        for term in helper["terms"]:
            if term["source"] == "stored_evidence" and term["name"] not in names:
                names.append(term["name"])
    return tuple(names)


def _broad_reward_decision(gate: dict[str, object], measured: dict[str, float]) -> dict[str, object]:
    """Apply the frozen gate's broad-reward acceptance to one arm's means."""
    baseline = gate["phase2_baseline"]["broad_reward"]
    acceptance = gate["acceptance"]["broad_reward"]
    return_ratio = measured["return_mean"] / baseline["return_mean"]
    safety_increase = measured["safety_violation_rate_mean"] - baseline["safety_violation_rate_mean"]
    auxiliary_delta = abs(measured["auxiliary_cost_mean"] - baseline["auxiliary_cost_mean"])
    action_delta = abs(measured["action_l2_mean"] - baseline["action_l2_mean"])
    checks = {
        "return_mean_ratio": {
            "value": return_ratio,
            "minimum": acceptance["return_mean_ratio_min"],
            "passed": return_ratio >= acceptance["return_mean_ratio_min"],
        },
        "safety_violation_rate_mean_increase": {
            "value": safety_increase,
            "maximum": acceptance["safety_violation_rate_mean_increase_max"],
            "passed": safety_increase <= acceptance["safety_violation_rate_mean_increase_max"],
        },
        "termination_rate_mean": {
            "value": measured["termination_rate_mean"],
            "maximum": acceptance["termination_rate_mean_max"],
            "passed": measured["termination_rate_mean"] <= acceptance["termination_rate_mean_max"],
        },
        "auxiliary_cost_mean_absolute_delta": {
            "value": auxiliary_delta,
            "maximum": acceptance["auxiliary_cost_mean_absolute_delta_max"],
            "passed": auxiliary_delta <= acceptance["auxiliary_cost_mean_absolute_delta_max"],
        },
        "action_l2_mean_absolute_delta": {
            "value": action_delta,
            "maximum": acceptance["action_l2_mean_absolute_delta_max"],
            "passed": action_delta <= acceptance["action_l2_mean_absolute_delta_max"],
        },
    }
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
        "anchor_comparability_caveat": (
            "The phase2 baseline broad-reward block was measured with the ORIGINAL reward-inference "
            "dataset; this run uses the bfm-rewardgate-20260806 REGENERATED dataset (original "
            "unrecoverable), so this frozen-anchor comparison carries a dataset-provenance caveat. "
            "The dataset is identical for both A/B arms."
        ),
    }


def _run(args: argparse.Namespace) -> dict[str, object]:
    """Build the native gate environment and measure the broad-reward component."""
    import torch
    from gpu_ownership import exclusive_physical_gpu_snapshot, validate_same_exclusive_gpu
    from motion_environment_identity import motion_runner_axes
    from rsl_rl.algorithms.forward_backward import forward_backward_model_from_config

    from isaaclab.envs import ManagerBasedRLEnv

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from isaaclab_tasks.core.multi_task.kinematics import NewtonKinematics, NewtonKinematicsCfg
    from isaaclab_tasks.core.multi_task.motion.config.agents import MotionForwardBackwardRunnerCfg
    from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionImitationEnvCfg
    from isaaclab_tasks.core.multi_task.rl.rsl_rl.forward_backward_tracking import forward_backward_evaluation_scope
    from isaaclab_tasks.utils import resolve_presets

    gate = json.loads(_POLICY_QUALITY_GATE.read_text(encoding="utf-8"))
    checkpoint = args.checkpoint.expanduser().resolve()
    checkpoint_sha256 = _sha256(checkpoint)
    if checkpoint_sha256 != args.checkpoint_sha256:
        raise ValueError("Checkpoint SHA-256 differs from the requested policy identity.")

    dataset_path = args.reward_dataset.expanduser().resolve()
    dataset_sha256 = _sha256(dataset_path)
    if dataset_sha256 != args.reward_dataset_sha256:
        raise ValueError("Reward-inference dataset SHA-256 differs from the regenerated pin.")
    dataset = torch.load(dataset_path, map_location=args.device, weights_only=True)
    reward_model_path = args.reward_model_entrypoint.expanduser().resolve()
    reward_model_sha256 = _sha256(reward_model_path)
    reward_model_source_identity = _mujoco_model_source_identity(reward_model_path)
    if (
        reward_model_sha256 != args.reward_model_entrypoint_sha256
        or dataset["reward_model_sha256"] != reward_model_sha256
    ):
        raise ValueError("Reward model and reward-inference dataset identities differ.")
    if reward_model_source_identity["bundle_sha256"] != args.reward_model_bundle_sha256:
        raise ValueError("Reward model source bundle SHA-256 differs.")
    if dataset["reward_tasks"] != BFM_REWARD_TASKS:
        raise ValueError("Reward dataset and evaluator task orders differ.")
    source_identity_fixture = json.loads(_SOURCE_IDENTITY_FIXTURE.read_text(encoding="utf-8"))

    runner_selection = _environment_tokens() | motion_runner_axes(_PRESET)
    runner_values = resolve_presets(MotionForwardBackwardRunnerCfg(), selected=runner_selection).to_dict()
    auxiliary_names = _auxiliary_evidence_names(runner_values["value_helpers"])
    if auxiliary_names != BFM_AUXILIARY_EVIDENCE_NAMES:
        raise RuntimeError("HEAD stored-evidence channels differ from the frozen broad-reward order.")

    cfg = resolve_presets(MotionImitationEnvCfg(), selected=_environment_tokens())
    cfg.sim.device = args.device
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.environment_seed
    table_cfg = cfg.commands.motion.task_table
    table_cfg.source_artifact_root = str(args.source_artifact_root.expanduser().resolve())
    table_cfg.target_artifact_root = str(args.target_artifact_root.expanduser().resolve())
    table_cfg.motion_split = "train"

    env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=cfg))
    try:
        table = env.unwrapped.command_manager.get_term("motion").table
        if len(table.clip_ids) != 862 or table.clip_index.total_frames != 258_600:
            raise RuntimeError("Policy-quality motion corpus differs from its frozen profile.")

        observations, _reset_info = env.reset()
        history_factory = _evaluation_history_factory(runner_values["replay"])
        construction_history = history_factory(observations)
        if construction_history is not None:
            observations = construction_history.decorate_current(observations)
        model = forward_backward_model_from_config(
            observations,
            runner_values["obs_groups"],
            env.num_actions,
            runner_values["model"],
            runner_values["value_helpers"],
        ).to(env.device)
        saved = torch.load(checkpoint, map_location=env.device, weights_only=True)
        if set(saved) != {"model_state_dict"}:
            raise ValueError("Policy checkpoint must contain exactly model_state_dict.")
        loaded = model.load_state_dict(saved["model_state_dict"], strict=True, assign=True)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise RuntimeError("Strict checkpoint load returned incompatible state keys.")
        model.eval()

        operators = {
            "reward_context_policy": _BFMRewardContextPolicy,
            "runtime_type": BfmRewardRuntime,
            "auxiliary_names": BFM_AUXILIARY_EVIDENCE_NAMES,
            "auxiliary_coefficients": BFM_AUXILIARY_COST_COEFFICIENTS,
            "hard_safety_names": BFM_HARD_SAFETY_NAMES,
            "tasks": BFM_REWARD_TASKS,
            "tasks_sha256": BFM_REWARD_TASKS_SHA256,
            "infer_contexts": infer_reward_contexts_from_dataset,
            "metric_rows": reward_metric_rows,
            "runtime_sha256": _source_sha256(BfmRewardRuntime),
            "code_identity": {
                "identity_source": "frozen_fixture_bfm_reward_source_identity_v1",
                "reason": "BFM-Zero source tree absent from host; parity receipt-proven equations used",
                "fixture": source_identity_fixture,
            },
        }
        command = env.unwrapped.command_manager.get_term("motion")

        before = exclusive_physical_gpu_snapshot(args.device)
        reward_runtime_started = time.perf_counter()
        reward_kinematics = NewtonKinematics(NewtonKinematicsCfg(mjcf_path=str(reward_model_path), device=env.device))
        reward_runtime = BfmRewardRuntime(
            reward_kinematics,
            env.unwrapped.action_manager.get_term("joint_position").joint_names,
            args.episodes_per_task,
        )
        if torch.device(env.device).type == "cuda":
            torch.cuda.synchronize(env.device)
        reward_runtime_seconds = time.perf_counter() - reward_runtime_started
        reward_rows, broad_reward_timing = _broad_reward_rollout(
            model=model,
            env=env.unwrapped,
            scope_env=env,
            evaluation_scope=forward_backward_evaluation_scope,
            command=command,
            domain_scope=command.payload.sampler.reset_sampling_scope,
            history_factory=history_factory,
            dataset=dataset,
            reward_runtime=reward_runtime,
            runtime_setup_seconds=reward_runtime_seconds,
            operators=operators,
            auxiliary_evidence_names=auxiliary_names,
            episodes_per_task=args.episodes_per_task,
            horizon=args.reward_horizon,
            batch_size=args.inference_batch_size,
            seed=args.evaluation_seed,
        )
        after = exclusive_physical_gpu_snapshot(args.device)
        physical_gpu_uuid = validate_same_exclusive_gpu(before, after)

        by_metric: dict[str, list[float]] = {}
        for row in reward_rows:
            by_metric.setdefault(str(row["metric_name"]), []).append(float(row["metric_value"]))
        broad_summary = {name: _statistics(values) for name, values in sorted(by_metric.items())}
        if any(name not in broad_summary for name in _REQUIRED_METRICS):
            raise ValueError("Broad-reward evidence is missing a frozen quality metric.")
        per_task: dict[str, dict[str, float]] = {}
        for row in reward_rows:
            task_bucket = per_task.setdefault(str(row["task"]), {})
            name = str(row["metric_name"])
            if name in _REQUIRED_METRICS:
                task_bucket[name] = task_bucket.get(name, 0.0) + float(row["metric_value"]) / args.episodes_per_task
        measured_means = {f"{name}_mean": broad_summary[name]["mean"] for name in _REQUIRED_METRICS}
        decision = _broad_reward_decision(gate, measured_means)

        output = args.output.expanduser().resolve()
        csv_path = output.with_name(output.stem + "_rows.csv")
        _write_csv(
            csv_path,
            reward_rows,
            {
                "arm": args.arm,
                "preset": _PRESET,
                "training_seed": 4728,
                "evaluation_seed": args.evaluation_seed,
                "checkpoint_transition": 9_600_000,
            },
        )
        return {
            "schema": _SCHEMA,
            "status": "measured_broad_reward",
            "arm": args.arm,
            "preset": _PRESET,
            "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha256},
            "protocol": {
                "preset": _PRESET,
                "motion_split": table_cfg.motion_split,
                "clip_count": len(table.clip_ids),
                "frame_count": table.clip_index.total_frames,
                "environment_seed": args.environment_seed,
                "evaluation_seed": args.evaluation_seed,
                "num_envs": args.num_envs,
                "reward_task_count": len(BFM_REWARD_TASKS),
                "episodes_per_task": args.episodes_per_task,
                "reward_horizon": args.reward_horizon,
                "inference_batch_size": args.inference_batch_size,
                "domain_randomization": all(
                    getattr(cfg.events, name) is not None
                    for name in ("robot_material", "body_mass", "torso_com", "push")
                ),
                "observation_noise": bool(cfg.observations.joint_position.enable_corruption),
                "source_registration": "bfm_lafan_control",
            },
            "broad_reward": {
                "dataset_provenance": "regenerated_not_original",
                "inference_dataset_path": str(dataset_path),
                "inference_dataset_sha256": dataset_sha256,
                "reward_tasks_sha256": BFM_REWARD_TASKS_SHA256,
                "reward_model_entrypoint_path": str(reward_model_path),
                "reward_model_entrypoint_sha256": reward_model_sha256,
                "reward_model_source_identity": reward_model_source_identity,
                "row_count": len(reward_rows),
                "rows_artifact": csv_path.name,
                "rows_sha256": _sha256(csv_path),
                "stage_durations_seconds": broad_reward_timing,
                "metrics": broad_summary,
                "per_task_means": per_task,
                "code_identity": operators["code_identity"],
            },
            "gate_reference": {
                "fixture": _POLICY_QUALITY_GATE.name,
                "fixture_sha256": _sha256(_POLICY_QUALITY_GATE),
                "phase2_baseline_broad_reward": gate["phase2_baseline"]["broad_reward"],
                "acceptance_broad_reward": gate["acceptance"]["broad_reward"],
            },
            "decision": decision,
            "gpu_ownership": {"physical_gpu_uuid": physical_gpu_uuid},
        }
    finally:
        env.close()


def main() -> None:
    """Measure the frozen gate's broad-reward component for one compact checkpoint."""
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("B1", "B2"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint_sha256", required=True)
    parser.add_argument("--source_artifact_root", type=Path, required=True)
    parser.add_argument("--target_artifact_root", type=Path, required=True)
    parser.add_argument("--reward_dataset", type=Path, required=True)
    parser.add_argument("--reward_dataset_sha256", required=True)
    parser.add_argument("--reward_model_entrypoint", type=Path, required=True)
    parser.add_argument(
        "--reward_model_entrypoint_sha256",
        default="321160a15b5aebc45eec59779bf3c75a7674d64a5b5e6e0092de918b04c858d4",
    )
    parser.add_argument(
        "--reward_model_bundle_sha256",
        default="8efbbb74cee40c33a491ddae5ab9da46a1b907ca92fc8ca05a9bd4f4988726d9",
    )
    parser.add_argument("--num_envs", type=int, default=380)
    parser.add_argument("--episodes_per_task", type=int, default=10)
    parser.add_argument("--reward_horizon", type=int, default=500)
    parser.add_argument("--inference_batch_size", type=int, default=1024)
    parser.add_argument("--environment_seed", type=int, default=4728)
    parser.add_argument("--evaluation_seed", type=int, default=4728)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs != len(BFM_REWARD_TASKS) * args.episodes_per_task:
        raise ValueError("num_envs must equal 38 broad tasks times episodes_per_task.")
    if min(args.episodes_per_task, args.reward_horizon, args.inference_batch_size) < 1:
        raise ValueError("Policy-quality evaluation counts must be positive.")
    if args.environment_seed != args.evaluation_seed:
        raise ValueError("Policy-quality evaluation uses one frozen seed for environment and evaluation transactions.")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Policy-quality broad-reward output already exists: {output}.")

    launcher = AppLauncher(args)
    simulation_app = launcher.app
    try:
        report = _run(args)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, output)
        print(json.dumps({"schema": _SCHEMA, "output": str(output)}, indent=2, sort_keys=True))
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
