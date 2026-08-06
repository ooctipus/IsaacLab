# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""bfm-abdiag-20260805 OWN-REFERENCE variant of :mod:`bfm_g1_policy_quality_tracking`.

Eval-side diagnosis script (F ledger row bfm-abdiag-20260805): identical native
stochastic tracking protocol (same evaluator, projections, seeds, num_envs), but
the tracking references come from OUR campaign v5-dump deployment instead of the
released joblib corpus: preset ``g1_lafan_retarget`` (token ``lafan_retarget``,
identifier ``lafan_g1_retarget_dump_v5``), train split = 843 accepted clips /
252,900 frames. Together with the archived original-reference G5 numbers this
fills the (train-domain x eval-domain) 2x2. The broad-reward gate component
remains BLOCKED on this host (missing frozen reward_dataset.pt) exactly as in
the parent script. Nothing tuned; production untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import traceback
from pathlib import Path

from g1_policy_quality_evidence import (
    _POLICY_CORPORA,
    _POLICY_QUALITY_GATE,
    _evaluation_history_factory,
    _policy_motion_split,
    _sha256,
    _statistics,
)

_PRESET = "g1_lafan_retarget"
_SCHEMA = "bfm_abdiag_20260805_g1_policy_quality_tracking_ourref_v1"

# bfm-abdiag-20260805: frozen corpus pins for the campaign v5-dump train split
# (dumps/MANIFEST.json accepted_full accounting; registration retarget_dump_v5.py).
_POLICY_CORPORA["g1_lafan_retarget"] = {"split": "train", "clip_count": 843, "frame_count": 252_900}


def _environment_tokens() -> frozenset[str]:
    """Native gate axes with the campaign v5-dump registration selected."""
    from motion_environment_identity import motion_environment_axes

    return frozenset(motion_environment_axes(_PRESET))


def _run(args: argparse.Namespace) -> dict[str, object]:
    """Build the native gate environment and measure the tracking component."""
    import torch
    from gpu_ownership import exclusive_physical_gpu_snapshot, validate_same_exclusive_gpu
    from motion_environment_identity import motion_runner_axes
    from motion_tracking_records import motion_tracking_metrics_to_dict
    from rsl_rl.algorithms.forward_backward import forward_backward_model_from_config

    from isaaclab.envs import ManagerBasedRLEnv

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from isaaclab_tasks.core.multi_task.motion.config.agents import MotionForwardBackwardRunnerCfg
    from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionImitationEnvCfg
    from isaaclab_tasks.core.multi_task.rl.rsl_rl.forward_backward_expert import forward_backward_expert_buffer
    from isaaclab_tasks.core.multi_task.rl.rsl_rl.forward_backward_tracking import (
        forward_backward_evaluation_scope,
        forward_backward_tracking_evaluator,
    )
    from isaaclab_tasks.utils import resolve_presets

    gate = json.loads(_POLICY_QUALITY_GATE.read_text(encoding="utf-8"))
    checkpoint = args.checkpoint.expanduser().resolve()
    checkpoint_sha256 = _sha256(checkpoint)
    if checkpoint_sha256 != args.checkpoint_sha256:
        raise ValueError("Checkpoint SHA-256 differs from the requested policy identity.")

    runner_selection = _environment_tokens() | motion_runner_axes(_PRESET)
    runner_values = resolve_presets(MotionForwardBackwardRunnerCfg(), selected=runner_selection).to_dict()

    cfg = resolve_presets(MotionImitationEnvCfg(), selected=_environment_tokens())
    cfg.sim.device = args.device
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.environment_seed
    table_cfg = cfg.commands.motion.task_table
    table_cfg.source_artifact_root = str(args.source_artifact_root.expanduser().resolve())
    table_cfg.target_artifact_root = str(args.target_artifact_root.expanduser().resolve())
    table_cfg.motion_split = _policy_motion_split(_PRESET)

    env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=cfg))
    try:
        corpus = _POLICY_CORPORA[_PRESET]
        table = env.unwrapped.command_manager.get_term("motion").table
        if len(table.clip_ids) != corpus["clip_count"] or table.clip_index.total_frames != corpus["frame_count"]:
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

        before_tracking = exclusive_physical_gpu_snapshot(args.device)
        expert_cfg = runner_values["expert"]
        expert = forward_backward_expert_buffer(
            env,
            model.observation_schema,
            env.device,
            source_bind=expert_cfg["source_bind"],
            priorities_bind=expert_cfg["priorities_bind"],
            clock=expert_cfg["clock"],
            target_projection=expert_cfg["target_projection"],
            target_projection_binds=tuple(expert_cfg["target_projection_binds"]),
            window_lengths=tuple(expert_cfg["window_lengths"]),
            seed=runner_values["seed"],
        )
        command = env.unwrapped.command_manager.get_term("motion")
        with forward_backward_evaluation_scope(
            env,
            command,
            command.payload.sampler.reset_sampling_scope,
            args.evaluation_seed,
            reset_source_name="reference",
        ):
            tracking_cfg = runner_values["tracking_curriculum"]
            tracking = forward_backward_tracking_evaluator(
                model,
                env,
                expert,
                expert.clip_ids,
                command=command,
                history_factory=history_factory,
                sequence_start_rows=table.clip_start_rows,
                projections=tuple(tracking_cfg["projections"]),
                context_window_length=tracking_cfg["context_window_length"],
                include_reset_frame=tracking_cfg["include_reset_frame"],
                allow_horizon_truncation=tracking_cfg["allow_horizon_truncation"],
                shuffle_assignments=tracking_cfg["shuffle_assignments"],
                assignment_rng=random.Random(args.evaluation_seed),
            )
        after_tracking = exclusive_physical_gpu_snapshot(args.device)
        physical_gpu_uuid = validate_same_exclusive_gpu(before_tracking, after_tracking)

        tracking_metrics = motion_tracking_metrics_to_dict(tracking)
        tracking_emd = [row["emd"] for row in tracking_metrics.values()]
        return {
            "schema": _SCHEMA,
            "status": "measured_tracking_only",
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
                "domain_randomization": all(
                    getattr(cfg.events, name) is not None
                    for name in ("robot_material", "body_mass", "torso_com", "push")
                ),
                "observation_noise": bool(cfg.observations.joint_position.enable_corruption),
                "source_registration": "lafan_retarget",
            },
            "tracking": {
                "clip_ids": list(tracking.sequence_ids),
                "metrics": tracking_metrics,
                "emd_statistics": _statistics(tracking_emd),
                "obs_state_emd_statistics": _statistics(
                    [row["obs_state_emd"] for row in tracking_metrics.values()]
                ),
                "coverage_statistics": _statistics(
                    [row["coverage_fraction"] for row in tracking_metrics.values()]
                ),
                "duration_seconds": tracking.duration_seconds,
            },
            "broad_reward": {
                "status": "blocked_missing_frozen_artifact",
                "missing_artifact": "reward_dataset.pt",
                "missing_artifact_sha256": "ae76737bee3b780074770dce56e1673e60e8b5954659ee4a3db255e169a1e151",
                "reason": (
                    "The frozen reward-inference dataset lived under the deleted "
                    "/home/isaaclab/octi/forward_backward tree and no generator exists on this host; "
                    "the 38-task x 10-episode broad-reward component cannot run without inventing an "
                    "unregistered protocol."
                ),
            },
            "gate_reference": {
                "fixture": _POLICY_QUALITY_GATE.name,
                "fixture_sha256": _sha256(_POLICY_QUALITY_GATE),
                "phase2_baseline_tracking": gate["phase2_baseline"]["tracking"],
                "acceptance_tracking": gate["acceptance"]["tracking"],
            },
            "gpu_ownership": {"physical_gpu_uuid": physical_gpu_uuid},
        }
    finally:
        env.close()


def main() -> None:
    """Measure the frozen gate's tracking component for one compact checkpoint."""
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint_sha256", required=True)
    parser.add_argument("--source_artifact_root", type=Path, required=True)
    parser.add_argument("--target_artifact_root", type=Path, required=True)
    parser.add_argument("--num_envs", type=int, default=380)
    parser.add_argument("--environment_seed", type=int, default=4728)
    parser.add_argument("--evaluation_seed", type=int, default=4728)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Policy-quality tracking output already exists: {output}.")

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
