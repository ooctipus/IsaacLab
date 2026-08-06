# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""bfm-rewardgate-20260806 regeneration of the BFM reward-inference dataset.

The frozen gate's pinned ``reward_dataset.pt`` (sha256
ae76737bee3b780074770dce56e1673e60e8b5954659ee4a3db255e169a1e151) lived only
under the deleted ``/home/isaaclab/octi/forward_backward`` tree and is
unrecoverable on this host. This generator rebuilds a schema-identical
``bfm_reward_inference_dataset_v2`` artifact from frozen components only:

* states: the FULL ``bfm_lafan_control`` train corpus (862 clips / 258,600
  frames) — the deterministic, parameter-free choice (the original sampling
  law is unrecorded; no subsampling or RNG is invented here);
* observations: the frozen training-side expert projection
  :func:`g1_bfm_expert_target` (byte-identical backward-map inputs — state 64
  and privileged_state 463) evaluated on the table reference frames;
* reward labels: the frozen 38-task reward equations
  (:mod:`bfm_reward_runtime`, GPU parity receipt-proven against the BFM-Zero
  source) on reference qpos/qvel in the released MuJoCo conventions;
* ``motion_id``: the clip index of each frame.

The regenerated artifact is REGENERATED-NOT-ORIGINAL: it is pinned by its own
sha256 for both gate runs, and the frozen-gate anchor comparability carries a
caveat because the phase2 baseline broad-reward block was measured with the
original dataset. The dataset is identical for both A/B arms, so the A/B read
is unaffected.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

from bfm_reward_runtime import (
    BFM_REWARD_INFERENCE_DATASET_SCHEMA,
    BFM_REWARD_TASKS,
    BFM_REWARD_TASKS_SHA256,
    BfmRewardRuntime,
    validate_reward_inference_dataset,
)
from g1_policy_quality_evidence import _canonical_sha256, _mujoco_model_source_identity, _sha256

_PRESET = "g1_lafan"
_SCHEMA = "bfm_rewardgate_20260806_g1_reward_dataset_regen_v1"
_ORIGINAL_DATASET_SHA256 = "ae76737bee3b780074770dce56e1673e60e8b5954659ee4a3db255e169a1e151"
_TRAIN_SPLIT_ARTIFACT = "humanoidverse/data/lafan_29dof_10s-clipped.pkl"
_TRAIN_SPLIT_SHA256 = "7f5aa36957808ee2e972472b18add8510533742710ba312d8b8c6e6014f1c010"


def _environment_tokens() -> frozenset[str]:
    """Native gate axes with the training-control LAFAN registration selected."""
    from motion_environment_identity import motion_environment_axes

    return frozenset((motion_environment_axes(_PRESET) - {"lafan"}) | {"bfm_lafan_control"})


def _run(args: argparse.Namespace) -> dict[str, object]:
    """Build the gate environment's motion table and regenerate the dataset."""
    import torch
    from gpu_ownership import exclusive_physical_gpu_snapshot, validate_same_exclusive_gpu

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.math import quat_apply_inverse

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from isaaclab_tasks.core.multi_task.kinematics import NewtonKinematics, NewtonKinematicsCfg
    from isaaclab_tasks.core.multi_task.motion.robots.g1.observations import g1_bfm_expert_target
    from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionImitationEnvCfg
    from isaaclab_tasks.utils import resolve_presets

    reward_model_path = args.reward_model_entrypoint.expanduser().resolve()
    reward_model_sha256 = _sha256(reward_model_path)
    if reward_model_sha256 != args.reward_model_entrypoint_sha256:
        raise ValueError("Reward model entrypoint SHA-256 differs from the frozen pin.")
    reward_model_source_identity = _mujoco_model_source_identity(reward_model_path)
    if reward_model_source_identity["bundle_sha256"] != args.reward_model_bundle_sha256:
        raise ValueError("Reward model source bundle SHA-256 differs from the frozen pin.")

    source_root = args.source_artifact_root.expanduser().resolve()
    # Deployment roots link their split artifacts (fb-current layout) — hash the resolved regular file.
    split_artifact = (source_root / _TRAIN_SPLIT_ARTIFACT).resolve()
    split_sha256 = _sha256(split_artifact)
    if split_sha256 != _TRAIN_SPLIT_SHA256:
        raise ValueError("Control train split artifact SHA-256 differs from its registration pin.")

    cfg = resolve_presets(MotionImitationEnvCfg(), selected=_environment_tokens())
    cfg.sim.device = args.device
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.environment_seed
    table_cfg = cfg.commands.motion.task_table
    table_cfg.source_artifact_root = str(source_root)
    table_cfg.target_artifact_root = str(args.target_artifact_root.expanduser().resolve())
    table_cfg.motion_split = "train"

    env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=cfg))
    try:
        before = exclusive_physical_gpu_snapshot(args.device)
        command = env.unwrapped.command_manager.get_term("motion")
        table = command.table
        if len(table.clip_ids) != 862 or table.clip_index.total_frames != 258_600:
            raise RuntimeError("Reward-dataset motion corpus differs from the frozen gate profile.")
        robot = env.unwrapped.scene["robot"]
        action = env.unwrapped.action_manager.get_term("joint_position")
        device = torch.device(env.device)

        expert_fields, projection_identity = g1_bfm_expert_target(robot, action, table, table.field)
        state = torch.cat(
            (
                expert_fields["joint_position"],
                expert_fields["joint_velocity"],
                expert_fields["projected_gravity"],
                expert_fields["base_angular_velocity"],
            ),
            dim=-1,
        )
        privileged_state = expert_fields["privileged_state"]
        total_frames = table.clip_index.total_frames
        if state.shape != (total_frames, 64) or privileged_state.shape != (total_frames, 463):
            raise RuntimeError("Expert projection widths differ from the frozen dataset schema.")

        behavior_joint_names = tuple(action.joint_names)
        joint_indices = torch.tensor(
            [table.joint_names.index(name) for name in behavior_joint_names], dtype=torch.int64, device=device
        )
        pelvis = table.reference_frame_names.index("pelvis")
        body_rotation = table.field("body_rotation")[:, pelvis]
        qpos = torch.empty(total_frames, 36, dtype=torch.float32, device=device)
        qvel = torch.empty(total_frames, 35, dtype=torch.float32, device=device)
        qpos[:, :3] = table.field("body_position")[:, pelvis]
        qpos[:, 3] = body_rotation[:, 3]
        qpos[:, 4:7] = body_rotation[:, :3]
        qpos[:, 7:] = table.field("joint_position").index_select(1, joint_indices)
        qvel[:, :3] = table.field("body_linear_velocity")[:, pelvis]
        qvel[:, 3:6] = quat_apply_inverse(body_rotation, table.field("body_angular_velocity")[:, pelvis])
        qvel[:, 6:] = table.field("joint_velocity").index_select(1, joint_indices)

        labels_started = time.perf_counter()
        kinematics = NewtonKinematics(NewtonKinematicsCfg(mjcf_path=str(reward_model_path), device=str(device)))
        runtime = BfmRewardRuntime(kinematics, behavior_joint_names, args.chunk_frames)
        task_count = len(BFM_REWARD_TASKS)
        reward_labels = torch.empty(total_frames, task_count, dtype=torch.float32, device=device)
        for start in range(0, total_frames, args.chunk_frames):
            stop = min(start + args.chunk_frames, total_frames)
            rows = torch.arange(start, stop, device=device)
            if rows.shape[0] < args.chunk_frames:
                rows = torch.cat((rows, rows[-1:].expand(args.chunk_frames - rows.shape[0])))
            tiled = rows.repeat(task_count)
            chunk = runtime.evaluate(qpos.index_select(0, tiled), qvel.index_select(0, tiled))
            reward_labels[start:stop] = chunk.T[: stop - start]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        labels_seconds = time.perf_counter() - labels_started
        if not torch.isfinite(reward_labels).all():
            raise RuntimeError("Regenerated reward labels are not finite.")

        clip_lengths = torch.diff(
            torch.cat(
                (
                    table.clip_start_rows.to(dtype=torch.int64),
                    torch.tensor([total_frames], dtype=torch.int64, device=table.clip_start_rows.device),
                )
            )
        )
        motion_id = torch.repeat_interleave(
            torch.arange(len(table.clip_ids), dtype=torch.int64, device=clip_lengths.device), clip_lengths
        )
        if motion_id.shape != (total_frames,):
            raise RuntimeError("Motion-id rows differ from the frame count.")

        reference_config = {
            "schema": _SCHEMA,
            "preset": _PRESET,
            "source_registration": "bfm_lafan_control",
            "motion_split": "train",
            "clip_count": len(table.clip_ids),
            "frame_count": total_frames,
            "split_artifact": _TRAIN_SPLIT_ARTIFACT,
            "split_artifact_sha256": split_sha256,
            "table_identity": table.cache_identity,
            "projection_identity": {
                "version": projection_identity["version"],
                "joint_names": list(projection_identity["joint_names"]),
                "body_names": list(projection_identity["body_names"]),
                "joint_default_position": projection_identity["joint_default_position"],
            },
            "reward_tasks_sha256": BFM_REWARD_TASKS_SHA256,
            "reward_model_sha256": reward_model_sha256,
            "reward_model_bundle_sha256": reward_model_source_identity["bundle_sha256"],
            "chunk_frames": args.chunk_frames,
            "environment_seed": args.environment_seed,
        }
        dataset = {
            "schema": BFM_REWARD_INFERENCE_DATASET_SCHEMA,
            "reward_tasks": BFM_REWARD_TASKS,
            "reference_config_sha256": _canonical_sha256(reference_config),
            "data_sha256": split_sha256,
            "reward_model_sha256": reward_model_sha256,
            "observation": {
                "state": state.detach().to("cpu", torch.float32).contiguous(),
                "privileged_state": privileged_state.detach().to("cpu", torch.float32).contiguous(),
            },
            "reward_labels": reward_labels.detach().to("cpu", torch.float32).contiguous(),
            "motion_id": motion_id.detach().to("cpu", torch.int64).contiguous(),
        }
        validate_reward_inference_dataset(dataset)

        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        torch.save(dataset, temporary)
        os.replace(temporary, output)
        reloaded = torch.load(output, map_location="cpu", weights_only=True)
        validate_reward_inference_dataset(reloaded)
        if reloaded["reference_config_sha256"] != dataset["reference_config_sha256"]:
            raise RuntimeError("Regenerated dataset did not round-trip its reference-config identity.")
        dataset_sha256 = _sha256(output)
        after = exclusive_physical_gpu_snapshot(args.device)
        physical_gpu_uuid = validate_same_exclusive_gpu(before, after)

        labels_cpu = dataset["reward_labels"]
        return {
            "schema": _SCHEMA,
            "status": "regenerated_not_original",
            "provenance": {
                "original_dataset_sha256": _ORIGINAL_DATASET_SHA256,
                "original_dataset_state": (
                    "unrecoverable: lived only under the deleted /home/isaaclab/octi/forward_backward tree; "
                    "host searched exhaustively before regeneration (F ledger bfm-rewardgate-20260806)"
                ),
                "anchor_comparability_caveat": (
                    "The frozen gate's phase2 baseline broad-reward block was measured with the ORIGINAL "
                    "dataset whose sampling law is unrecorded; this regenerated dataset uses the full "
                    "train corpus deterministically. Identical for both A/B arms - the A/B read is unaffected."
                ),
            },
            "dataset": {
                "path": str(output),
                "sha256": dataset_sha256,
                "sample_count": int(labels_cpu.shape[0]),
                "task_count": int(labels_cpu.shape[1]),
                "reference_config": reference_config,
                "reference_config_sha256": dataset["reference_config_sha256"],
                "data_sha256": split_sha256,
                "reward_model_sha256": reward_model_sha256,
                "label_statistics": {
                    "min": float(labels_cpu.min()),
                    "max": float(labels_cpu.max()),
                    "mean": float(labels_cpu.mean()),
                    "per_task_max_min": float(labels_cpu.max(dim=0).values.min()),
                },
                "label_generation_seconds": labels_seconds,
            },
            "reward_model_source_identity": reward_model_source_identity,
            "gpu_ownership": {"physical_gpu_uuid": physical_gpu_uuid},
        }
    finally:
        env.close()


def main() -> None:
    """Regenerate the reward-inference dataset and publish its receipt."""
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_artifact_root", type=Path, required=True)
    parser.add_argument("--target_artifact_root", type=Path, required=True)
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
    parser.add_argument("--environment_seed", type=int, default=4728)
    parser.add_argument("--chunk_frames", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.chunk_frames < 1:
        raise ValueError("chunk_frames must be positive.")
    receipt = args.receipt.expanduser().resolve()
    if receipt.exists() or args.output.expanduser().resolve().exists():
        raise FileExistsError("Reward-dataset regeneration outputs already exist.")

    launcher = AppLauncher(args)
    simulation_app = launcher.app
    try:
        report = _run(args)
        receipt.parent.mkdir(parents=True, exist_ok=True)
        temporary = receipt.with_name(f".{receipt.name}.{os.getpid()}.tmp")
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, receipt)
        print(json.dumps({"schema": _SCHEMA, "receipt": str(receipt), "dataset_sha256": report["dataset"]["sha256"]}))
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
