# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G2 gate of bfm-converter-20260805: deterministic one-update learner canary on v5-dump corpora.

Port of the :mod:`motion_learning_evidence` pattern to the campaign data arms and the
HEAD config schema (``target_artifact_root``; skeletons owned by decoder modules).
Expert-corpus content hashes for the retarget corpora are RECORDED (first freeze),
while clip/frame counts are asserted against the campaign registration pins.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import motion_learning_evidence as mle
import numpy as np
import torch
from motion_environment_identity import motion_environment_axes, motion_runner_axes

from isaaclab_tasks.core.multi_task.kinematics import NewtonKinematics, NewtonKinematicsCfg
from isaaclab_tasks.core.multi_task.motion.config.agents import MotionForwardBackwardRunnerCfg
from isaaclab_tasks.core.multi_task.motion.data.sources.retarget_dump_v5 import _route_skeleton
from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table import build_motion_task_table
from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionImitationEnvCfg
from isaaclab_tasks.utils.hydra import resolve_presets

_PRESETS = {
    "smpl_cmu_retarget": ("smpl_cmu", "cmu", "cmu_retarget"),
    "g1_lafan_retarget": ("g1_lafan", "lafan", "lafan_retarget"),
}


def _expert_buffer(base: str, env: SimpleNamespace, runner: dict[str, object]):
    """HEAD port of :func:`motion_learning_evidence._expert_buffer` (expert clock moved under ``clock``)."""
    from isaaclab_tasks.core.multi_task.rl.rsl_rl.forward_backward_expert import forward_backward_expert_buffer

    schema = mle._expert_schema(base, runner)
    expert_cfg = runner["expert"]
    return forward_backward_expert_buffer(
        env,
        schema,
        str(env.unwrapped.command_manager.get_term("motion").payload.table.device),
        source_bind=expert_cfg["source_bind"],
        priorities_bind=expert_cfg["priorities_bind"],
        clock=expert_cfg["clock"],
        target_projection=expert_cfg["target_projection"],
        target_projection_binds=tuple(expert_cfg["target_projection_binds"]),
        window_lengths=tuple(expert_cfg["window_lengths"]),
        seed=int(runner["seed"]),
    )


def _canary_config_head(runner: dict[str, object], expert) -> dict[str, object]:
    """HEAD port of :func:`motion_learning_evidence._canary_config` (capacity under ``replay.policy``)."""
    import copy

    cfg = copy.deepcopy(runner)
    cfg["replay"]["policy"]["capacity_transitions"] = mle.CANARY_REPLAY_TRANSITIONS
    cfg["algorithm"]["batch_size"] = mle.CANARY_BATCH_SIZE
    # Execution-scale shrink for the pinned learner's warm-up boundary
    # (collected > random + num_envs): the frozen two-env trace cannot cover the
    # production random warm-up, and the canary claims integration, not schedule.
    cfg["algorithm"]["exploration"]["random_action_transitions"] = 0
    cfg["torch_compile_mode"] = None
    expert_seed = int(cfg["seed"])

    def provider(_env, _observation_schema, _device, **_kwargs):
        return mle._clone_expert(expert, expert_seed)

    cfg["expert"]["provider"] = provider
    return cfg


# The pattern module's canary path resolves _canary_config at call time; install the HEAD port.
mle._canary_config = _canary_config_head


def _live_axes(base: str, cfg: MotionImitationEnvCfg, device: torch.device) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the exact axes resolved by each native simulator articulation (HEAD port)."""
    if base == "smpl_cmu":
        asset = NewtonKinematics(
            NewtonKinematicsCfg(
                mjcf_path=cfg.scene.robot.spawn.asset_path,
                device=str(device),
                collapse_fixed_joints=False,
            )
        )
        body_names = tuple(asset.body_names)
        joint_names = tuple(f"{joint}:{component}" for joint in asset.joint_names[1:] for component in range(3))
        if len(body_names) != 24 or len(joint_names) != 69:
            raise ValueError("Packaged SMPL asset does not expose the expected ball-joint coordinate count.")
        return joint_names, body_names
    actuator = next(iter(cfg.scene.robot.actuators.values()))
    joint_names = tuple(actuator.joint_names_expr)
    skeleton = _route_skeleton("lafan_g1")
    if len(joint_names) != skeleton.num_joints:
        raise ValueError("Resolved G1 preset does not define one complete articulated tree.")
    joint_by_name = {name: index for index, name in enumerate(skeleton.joint_names)}
    child_body_names = tuple(
        skeleton.body_names[skeleton.joint_child_body_indices[joint_by_name[name]]] for name in joint_names
    )
    body_names = (skeleton.body_names[0], *child_body_names)
    if len(set(body_names)) != skeleton.num_bodies:
        raise ValueError("Resolved G1 action axis does not map one-to-one onto physical bodies.")
    return joint_names, body_names


def _construction_env(
    preset: str,
    source_artifact_root: Path,
    target_artifact_root: Path,
    device: torch.device,
) -> SimpleNamespace:
    """Resolve one campaign data arm into the minimum table-construction environment."""
    base, control_token, retarget_token = _PRESETS[preset]
    axes = frozenset(token if token != control_token else retarget_token for token in motion_environment_axes(base))
    cfg = resolve_presets(MotionImitationEnvCfg(), selected=axes)
    cfg.seed = 0
    cfg.sim.device = str(device)
    table_cfg = cfg.commands.motion.task_table
    table_cfg.source_artifact_root = str(source_artifact_root.expanduser().resolve())
    table_cfg.target_artifact_root = str(target_artifact_root.expanduser().resolve())
    table_cfg.motion_split = "train"
    joint_names, body_names = _live_axes(base, cfg, device)
    return SimpleNamespace(
        cfg=cfg,
        device=device,
        scene={"robot": SimpleNamespace(joint_names=joint_names, body_names=body_names)},
    )


def _expert_audit(table, expert, split_cfg, sampling_mode: str, sampling_step_seconds: float | None):
    """Verify pinned corpus cardinality and freeze the new expert content identities."""
    sampled = table.sample(sampling_mode, sampling_step_seconds)
    offsets = sampled.clip_offsets
    counts = tuple(end - start for start, end in zip(offsets[:-1], offsets[1:], strict=True))
    if len(counts) != split_cfg.clip_count or table.clip_index.total_frames != split_cfg.frame_count:
        raise RuntimeError("Resolved retarget expert cardinality differs from the campaign registration pins.")
    if tuple(expert.clip_offsets.tolist()) != offsets or expert.clip_ids != table.clip_ids:
        raise RuntimeError("Expert offsets or clip ids differ from the command-owned table clock.")
    window_counts = {
        str(length): sum(max(0, count - length) for count in counts) for length in expert.schema.window_lengths
    }
    return {
        "clip_count": len(counts),
        "source_frame_count": table.clip_index.total_frames,
        "expert_frame_count": expert.schema.num_frames,
        "expert_feature_width": expert.schema.expert_feature_width,
        "sample_grid": {"mode": sampling_mode, "step_seconds": sampling_step_seconds},
        "window_lengths": list(expert.schema.window_lengths),
        "window_counts": window_counts,
        "clip_offsets_sha256": mle._tensor_sha256(expert.clip_offsets),
        "expert_frames_sha256": mle._tensor_sha256(expert.frames),
        "expert_schema_sha256": expert.schema.schema_hash,
        "expert_data_sha256": expert.schema.data_hash,
    }


def measure(args: argparse.Namespace) -> dict[str, object]:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    base, _control_token, _retarget_token = _PRESETS[args.preset]
    construction = _construction_env(args.preset, args.source_artifact_root, args.target_artifact_root, device)
    table = build_motion_task_table(construction.cfg.commands.motion, construction.cfg.scene, str(device))
    if table.family_name != "exact":
        raise RuntimeError(f"RISK-4 VIOLATION: resolved family {table.family_name!r} != 'exact'.")
    runner_selection = motion_environment_axes(base) | motion_runner_axes(base)
    runner_cfg = resolve_presets(MotionForwardBackwardRunnerCfg(), selected=runner_selection)
    runner = runner_cfg.to_dict()
    env = mle._learner_env(table, construction.cfg, base)
    # HEAD port: the expert priorities bind resolves through the payload sampler;
    # expose the production sampler's initial state (table base priorities).
    command = env.unwrapped.command_manager.get_term("motion")
    command.payload.sampler = SimpleNamespace(clip_priorities=table.base_priorities.clone())
    expert = _expert_buffer(base, env, runner)
    clock = runner["expert"]["clock"]
    split = construction.cfg.commands.motion.task_table.source.train
    audit = _expert_audit(table, expert, split, clock["sampling_mode"], clock["sampling_step_seconds"])
    canary = mle._first_update_canary(base, env, runner, expert, mle.TRACE_PATHS[base])
    return {
        "schema": "bfm_converter_20260805_motion_learning_evidence_v1",
        "status": "measured",
        "preset": args.preset,
        "base_profile": base,
        "claim_scope": {
            "expert_attachment": "exact_v5_dump_train_corpus",
            "first_update": "deterministic_integration_canary",
            "convergence_non_inferiority": "not_evaluated",
        },
        "source": {
            "identifier": construction.cfg.commands.motion.task_table.source.identifier,
            "split": split.name,
            "artifact": split.artifact,
            "artifact_sha256": split.artifact_sha256,
            "source_content_sha256": table.clip_index.source_content_sha256,
        },
        "task_table": {
            "identity_sha256": table.cache_identity,
            "family_name": table.family_name,
            "construction_version": table.construction_version,
            "joint_names_count": len(table.joint_names),
        },
        "learner": {
            "runner_config_sha256": mle._normalized_runner_identity(runner),
            "observation_schema_sha256": mle._expert_schema(base, runner).schema_hash,
            "routes": runner["obs_groups"],
        },
        "expert": audit,
        "first_update_canary": canary,
        "code_identity": {
            "evidence_sha256": mle._sha256(Path(__file__).resolve()),
            "pattern_module_sha256": mle._sha256(Path(mle.__file__).resolve()),
            "native_trace_sha256": mle._sha256(mle.TRACE_PATHS[base]),
        },
        "runtime": {
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(_PRESETS), required=True)
    parser.add_argument("--source_artifact_root", type=Path, required=True)
    parser.add_argument("--target_artifact_root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = measure(args)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
