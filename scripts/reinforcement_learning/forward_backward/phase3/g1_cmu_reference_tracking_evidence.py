# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure G1-CMU retarget and production-simulator errors without claiming policy quality."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

ROW_LIFECYCLE = "retire_after_done_or_reference_tail_exhaustion"
_REFERENCE_FIELDS = (
    "joint_position",
    "joint_velocity",
    "body_position",
    "body_rotation",
    "body_linear_velocity",
    "body_angular_velocity",
)


def _statistics(values: torch.Tensor) -> dict[str, float | int]:
    """Return exact finite scalar statistics for one nonempty tensor."""
    values = values.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if values.numel() == 0:
        raise ValueError("Tracking evidence cannot summarize an empty tensor.")
    if not torch.all(torch.isfinite(values)):
        raise ValueError("Tracking evidence contains non-finite measurements.")
    quantiles = torch.quantile(values, torch.tensor((0.5, 0.95, 0.99), dtype=torch.float64))
    return {
        "count": values.numel(),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "q50": float(quantiles[0]),
        "q95": float(quantiles[1]),
        "q99": float(quantiles[2]),
        "max": float(values.max()),
    }


def _quaternion_geodesic(actual_xyzw: torch.Tensor, expected_xyzw: torch.Tensor) -> torch.Tensor:
    """Return sign-invariant unit-quaternion geodesic error [rad]."""
    actual = torch.nn.functional.normalize(actual_xyzw, dim=-1)
    expected = torch.nn.functional.normalize(expected_xyzw, dim=-1)
    dot = torch.sum(actual * expected, dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def _reference_pd_behavior_action(
    reference_joint_position: torch.Tensor,
    reference_joint_velocity: torch.Tensor,
    default_joint_offset: torch.Tensor,
    action: Any,
    position_lower_limit: torch.Tensor,
    position_upper_limit: torch.Tensor,
    step_seconds: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return lookahead, limit-bounded, achievable targets and normalized action."""
    lookahead = reference_joint_position + step_seconds * reference_joint_velocity
    bounded = torch.maximum(lookahead, position_lower_limit)
    bounded = torch.minimum(bounded, position_upper_limit)
    processed = (
        (bounded - action.joint_default_position - default_joint_offset)
        * action.joint_stiffness
        / (action.cfg.action_scale * action.joint_effort_limit)
    )
    processed.clamp_(-action.cfg.action_clip, action.cfg.action_clip)
    behavior = processed / action.cfg.normalize_to
    achievable = action.joint_default_position + default_joint_offset + processed * action.joint_target_gain
    return lookahead, bounded, achievable, behavior


def _finish_active_rows(
    alive: torch.Tensor,
    active_before_step: torch.Tensor,
    done: torch.Tensor,
    reached_tail_valid: torch.Tensor,
) -> torch.Tensor:
    """Return reached rows and permanently retire done or exhausted environments."""
    reached_active = active_before_step & ~done
    alive.logical_and_(~done)
    alive.logical_and_(reached_tail_valid)
    return reached_active


def _policy_error_layer() -> dict[str, object]:
    """Reject the oracle reference controller as evidence about a learned policy."""
    return {
        "status": "not_measured_no_real_checkpoint_supplied",
        "reference_controller_is_policy_evidence": False,
        "required_evaluator": "g1_motion_tracking_evaluator",
        "required_metrics": ["evaluation_emd", "broad_reward", "safety_violations"],
    }


def _failed_report(args: argparse.Namespace, error: BaseException) -> dict[str, object]:
    """Return a durable failure record without promoting an unmeasured layer."""
    return {
        "schema": "forward_backward_phase3g_g1_cmu_reference_tracking_evidence_v3",
        "status": "failed",
        "request": {
            "motion_split": args.motion_split,
            "num_clips": args.num_clips,
            "num_steps": args.num_steps,
            "seed": args.seed,
            "row_lifecycle": ROW_LIFECYCLE,
        },
        "failure": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "error_layers": {
            "retarget_fit": {"status": "not_reported_run_failed"},
            "reference_controller_simulator": {"status": "failed"},
            "policy": _policy_error_layer(),
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha256(value: object) -> str:
    """Hash the file defining one measured runtime boundary."""
    path = inspect.getsourcefile(value)
    if path is None:
        raise RuntimeError(f"Cannot locate source for {value!r}.")
    return _sha256(Path(path))


def _write_report(path: Path, report: dict[str, object]) -> None:
    """Atomically persist either measured evidence or a failed-run record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _retarget_error_layer(
    path: Path,
    source_split_name: str,
) -> tuple[dict[str, object], set[str], Mapping[str, object]]:
    """Validate and summarize the companion source-to-robot measurement."""
    report = json.loads(path.read_text())
    if report.get("schema") != "forward_backward_phase3g_g1_cmu_composition_evidence_v3":
        raise ValueError("Retarget evidence does not use the G1-CMU composition schema.")
    code_identity = report.get("code_identity")
    if not isinstance(code_identity, Mapping):
        raise ValueError("Retarget evidence must declare code identity.")
    composition_dependency_identity = code_identity.get("composition_dependency_identity")
    if not isinstance(composition_dependency_identity, Mapping):
        raise ValueError("Retarget evidence must declare composition dependency identity.")
    composition = report.get("composition", {})
    source = report.get("source", {})
    retarget = report.get("error_layers", {}).get("retarget_fit", {})
    if composition.get("selected") != "g1_cmu" or composition.get("source") != "smpl_cmu":
        raise ValueError("Retarget evidence does not describe the resolved G1-CMU axes.")
    if source.get("split") != source_split_name or retarget.get("status") != "measured":
        raise ValueError("Retarget evidence split or measurement status differs from this probe.")
    clip_ids = source.get("selected_clip_ids")
    if not isinstance(clip_ids, list) or not clip_ids or not all(isinstance(value, str) for value in clip_ids):
        raise ValueError("Retarget evidence must declare measured stable clip ids.")
    return (
        {
            "status": "measured_by_companion_source_composition_probe",
            "evidence_path": str(path),
            "evidence_sha256": _sha256(path),
            "frame_builder_version": composition["frame_builder_version"],
            "frame_builder_identity_sha256": composition["frame_builder_identity_sha256"],
            "joint_names": composition["joint_names"],
            "reference_frame_names": composition["reference_frame_names"],
            "all_groups_rad": retarget["all_groups_rad"],
            "reference_ground_feasibility": retarget["reference_ground_feasibility"],
        },
        set(clip_ids),
        composition_dependency_identity,
    )


def _append_tracking_errors(
    errors: dict[str, list[torch.Tensor]],
    robot: Any,
    reference: Any,
    env_origins: torch.Tensor,
    active: torch.Tensor,
) -> None:
    """Append equal-time physical-body and joint errors for active clip rows."""
    physical_body_count = len(robot.body_names)
    simulated_joint_position = robot.data.joint_pos.torch
    simulated_joint_velocity = robot.data.joint_vel.torch
    simulated_body_position = robot.data.body_link_pos_w.torch
    simulated_body_position = simulated_body_position - env_origins[:, None]
    reference_body_position = reference["body_position"][:, :physical_body_count]
    simulated_body_rotation = robot.data.body_link_quat_w.torch
    reference_body_rotation = reference["body_rotation"][:, :physical_body_count]
    simulated_body_linear_velocity = robot.data.body_link_lin_vel_w.torch
    reference_body_linear_velocity = reference["body_linear_velocity"][:, :physical_body_count]
    simulated_body_angular_velocity = robot.data.body_link_ang_vel_w.torch
    reference_body_angular_velocity = reference["body_angular_velocity"][:, :physical_body_count]
    errors["joint_position_abs_rad"].append((simulated_joint_position - reference["joint_position"])[active].abs())
    errors["joint_velocity_abs_rad_s"].append((simulated_joint_velocity - reference["joint_velocity"])[active].abs())
    errors["body_position_l2_m"].append(
        torch.linalg.vector_norm(simulated_body_position - reference_body_position, dim=-1)[active]
    )
    errors["root_position_l2_m"].append(
        torch.linalg.vector_norm(simulated_body_position[:, 0] - reference_body_position[:, 0], dim=-1)[active]
    )
    errors["body_root_relative_position_l2_m"].append(
        torch.linalg.vector_norm(
            simulated_body_position
            - simulated_body_position[:, :1]
            - reference_body_position
            + reference_body_position[:, :1],
            dim=-1,
        )[active]
    )
    errors["body_rotation_geodesic_rad"].append(
        _quaternion_geodesic(simulated_body_rotation, reference_body_rotation)[active]
    )
    errors["root_rotation_geodesic_rad"].append(
        _quaternion_geodesic(simulated_body_rotation[:, 0], reference_body_rotation[:, 0])[active]
    )
    errors["body_linear_velocity_l2_m_s"].append(
        torch.linalg.vector_norm(
            simulated_body_linear_velocity - reference_body_linear_velocity,
            dim=-1,
        )[active]
    )
    errors["root_linear_velocity_l2_m_s"].append(
        torch.linalg.vector_norm(simulated_body_linear_velocity[:, 0] - reference_body_linear_velocity[:, 0], dim=-1)[
            active
        ]
    )
    errors["body_angular_velocity_l2_rad_s"].append(
        torch.linalg.vector_norm(
            simulated_body_angular_velocity - reference_body_angular_velocity,
            dim=-1,
        )[active]
    )
    errors["root_angular_velocity_l2_rad_s"].append(
        torch.linalg.vector_norm(simulated_body_angular_velocity[:, 0] - reference_body_angular_velocity[:, 0], dim=-1)[
            active
        ]
    )


def _reference(table: Any, payload: Any) -> dict[str, torch.Tensor]:
    """Resolve the current equal-time trajectory fields directly from the command table."""
    view = table.reference_view(payload.clip_indices, payload.reference_time_seconds)
    return {name: view.field(name) for name in _REFERENCE_FIELDS}


def _run(args: argparse.Namespace) -> dict[str, object]:
    """Run the explicit oracle reference controller through the production preset."""
    from motion_environment_identity import (
        motion_composition_dependency_identity,
        motion_composition_semantic_sha256,
        motion_environment_dependency_identity,
    )

    from isaaclab_tasks.core.multi_task.motion.data.importers import HumEnvHdf5Clips
    from isaaclab_tasks.core.multi_task.motion.mdp.actions import MotionJointPositionAction
    from isaaclab_tasks.core.multi_task.motion.mdp.commands import MotionStatePayload
    from isaaclab_tasks.core.multi_task.motion.trajectory.g1_smpl import G1SmplHumEnvFrameBuilder
    from isaaclab_tasks.core.multi_task.motion_env import MotionImitationEnv
    from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionImitationEnvCfg
    from isaaclab_tasks.utils import resolve_presets

    cfg = resolve_presets(MotionImitationEnvCfg(), selected={"g1_cmu"})
    table_cfg = cfg.commands.motion.task_table
    source_cfg = table_cfg.source
    source_split = source_cfg.train if args.motion_split == "train" else source_cfg.evaluation
    retarget_layer, retarget_clip_ids, retarget_dependency_identity = _retarget_error_layer(
        args.retarget_evidence.expanduser().resolve(), source_split.name
    )
    cfg.sim.device = args.device
    table_cfg.source_artifact_root = str(args.source_artifact_root.expanduser().resolve())
    table_cfg.reference_artifact_root = str(args.reference_artifact_root.expanduser().resolve())
    table_cfg.motion_split = args.motion_split
    cfg.scene.num_envs = args.num_clips
    cfg.seed = args.seed
    dependency_identity = motion_environment_dependency_identity(
        preset="g1_cmu",
        cfg=cfg,
        importer_type=HumEnvHdf5Clips,
        frame_builder_type=G1SmplHumEnvFrameBuilder,
        reference_artifact_root=table_cfg.reference_artifact_root,
    )

    env = MotionImitationEnv(cfg=cfg)
    try:
        command = env.command_manager.get_term("motion")
        table = command.table
        payload = command.payload
        action = env.action_manager.get_term("joint_position")
        if not isinstance(payload, MotionStatePayload) or not isinstance(action, MotionJointPositionAction):
            raise TypeError("G1-CMU tracking requires the production payload and joint-position action.")
        if payload.table is not table:
            raise RuntimeError("The motion command payload must consume its command-owned task table.")
        composition_dependency_identity = motion_composition_dependency_identity(
            preset="g1_cmu",
            cfg=cfg,
            importer_type=HumEnvHdf5Clips,
            frame_builder_type=G1SmplHumEnvFrameBuilder,
            frame_builder_identity_sha256=table.frame_builder_identity_sha256,
            reference_artifact_root=table_cfg.reference_artifact_root,
        )
        composition_semantic_sha256 = motion_composition_semantic_sha256(composition_dependency_identity)
        retarget_semantic_sha256 = motion_composition_semantic_sha256(retarget_dependency_identity)
        if retarget_semantic_sha256 != composition_semantic_sha256:
            raise ValueError("Retarget evidence and simulator source-to-target semantics differ.")
        retarget_layer["composition_semantic_sha256"] = retarget_semantic_sha256
        if table.frame_builder_identity_sha256 != retarget_layer["frame_builder_identity_sha256"]:
            raise ValueError("Retarget evidence and simulator trajectory builders differ.")
        if list(table.joint_names) != retarget_layer["joint_names"]:
            raise ValueError("Retarget evidence and simulator joint axes differ.")
        if list(table.reference_frame_names) != retarget_layer["reference_frame_names"]:
            raise ValueError("Retarget evidence and simulator reference-frame axes differ.")
        valid_clips = torch.arange(len(table.source_clip_ids), device=env.device)[table.clip_valid]
        if valid_clips.shape[0] < args.num_clips:
            raise ValueError("num_clips exceeds the valid G1-CMU motion-bank clip count.")
        selected = valid_clips[: args.num_clips]
        selected_clip_ids = tuple(table.source_clip_ids[index] for index in selected.cpu().tolist())
        missing_retarget = set(selected_clip_ids) - retarget_clip_ids
        if missing_retarget:
            raise ValueError(f"Simulator clips lack retarget evidence: {sorted(missing_retarget)}")

        env.reset_motion_clips(selected)
        robot = payload.robot
        if table.joint_names != tuple(robot.joint_names):
            raise ValueError("The trajectory and articulation must share one physical joint axis.")
        action_joint_ids = action.joint_ids
        if action_joint_ids.shape != (len(table.joint_names),) or action_joint_ids.dtype is not torch.int64:
            raise ValueError("The behavior action must map every physical joint exactly once.")
        behavior_to_physical = tuple(int(index) for index in action_joint_ids.cpu())
        if sorted(behavior_to_physical) != list(range(len(table.joint_names))):
            raise ValueError("The behavior-to-physical joint map must be a permutation.")
        resolved_behavior_names = tuple(table.joint_names[index] for index in behavior_to_physical)
        if action.joint_names != resolved_behavior_names:
            raise ValueError("The action joint names must declare the behavior-to-physical permutation.")
        physical_body_names = tuple(robot.body_names)
        if table.reference_frame_names[: len(physical_body_names)] != physical_body_names:
            raise ValueError("The trajectory references must begin with the live articulation body axis.")
        position_limits = robot.data.joint_pos_limits.torch[0]
        position_lower_limit = position_limits[:, 0]
        position_upper_limit = position_limits[:, 1]
        behavior_position_lower_limit = position_lower_limit.index_select(0, action_joint_ids)
        behavior_position_upper_limit = position_upper_limit.index_select(0, action_joint_ids)

        reset_errors = {
            name: []
            for name in (
                "joint_position_abs_rad",
                "joint_velocity_abs_rad_s",
                "body_position_l2_m",
                "root_position_l2_m",
                "body_root_relative_position_l2_m",
                "body_rotation_geodesic_rad",
                "root_rotation_geodesic_rad",
                "body_linear_velocity_l2_m_s",
                "root_linear_velocity_l2_m_s",
                "body_angular_velocity_l2_rad_s",
                "root_angular_velocity_l2_rad_s",
            )
        }
        active = payload.motion_facts["tail_valid"].bool()
        _append_tracking_errors(
            reset_errors,
            robot,
            _reference(table, payload),
            env.scene.env_origins,
            active,
        )

        errors = {name: [] for name in reset_errors}
        limit_projection: list[torch.Tensor] = []
        action_projection: list[torch.Tensor] = []
        simulated_limit_exceedance: list[torch.Tensor] = []
        active_rows = torch.zeros((), dtype=torch.int64, device=env.device)
        alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        position_limit_clipped = torch.zeros((), dtype=torch.int64, device=env.device)
        action_limit_clipped = torch.zeros((), dtype=torch.int64, device=env.device)
        unexpected_done = torch.zeros((), dtype=torch.int64, device=env.device)

        started = time.perf_counter()
        for _ in range(args.num_steps):
            active_before_step = alive & payload.motion_facts["tail_valid"].bool()
            current_reference = _reference(table, payload)
            behavior_reference_position = current_reference["joint_position"].index_select(1, action_joint_ids)
            behavior_reference_velocity = current_reference["joint_velocity"].index_select(1, action_joint_ids)
            lookahead, bounded, achievable, behavior = _reference_pd_behavior_action(
                behavior_reference_position,
                behavior_reference_velocity,
                action.default_joint_offset,
                action,
                behavior_position_lower_limit,
                behavior_position_upper_limit,
                env.step_dt,
            )
            _, _, terminated, truncated, _ = env.step(behavior)
            applied_error = torch.max(torch.abs(action.joint_position_target - achievable))
            torch._assert_async(applied_error <= 2.0e-6, "Production action differs from reference inversion.")
            done = terminated | truncated
            reached_active = _finish_active_rows(
                alive,
                active_before_step,
                done,
                payload.motion_facts["tail_valid"].bool(),
            )
            active_rows.add_(reached_active.sum())
            unexpected_done.add_((active_before_step & done).sum())
            _append_tracking_errors(
                errors,
                robot,
                _reference(table, payload),
                env.scene.env_origins,
                reached_active,
            )
            limit_projection.append((bounded - lookahead)[reached_active].abs())
            action_projection.append((achievable - bounded)[reached_active].abs())
            simulated_limit_exceedance.append(
                torch.maximum(
                    position_lower_limit - robot.data.joint_pos.torch,
                    robot.data.joint_pos.torch - position_upper_limit,
                )[reached_active].clamp_min(0.0)
            )
            position_limit_clipped.add_(((bounded - lookahead).abs() > 1.0e-7)[reached_active].sum())
            action_limit_clipped.add_(((achievable - bounded).abs() > 2.0e-6)[reached_active].sum())
        if str(env.device).startswith("cuda"):
            torch.cuda.synchronize(torch.device(env.device))
        duration_seconds = time.perf_counter() - started
        active_row_count = int(active_rows)
        unexpected_done_count = int(unexpected_done)
        position_limit_clipped_count = int(position_limit_clipped)
        action_limit_clipped_count = int(action_limit_clipped)
        if active_row_count == 0:
            raise RuntimeError("The selected probe window contains no active reached reference rows.")
        simulated_limit_exceedance_value = torch.cat(simulated_limit_exceedance)
        active_coordinate_count = active_row_count * len(table.joint_names)

        return {
            "schema": "forward_backward_phase3g_g1_cmu_reference_tracking_evidence_v3",
            "status": "measured",
            "code_identity": {
                "probe_sha256": _source_sha256(_run),
                "dependency_identity": dependency_identity,
                "composition_dependency_identity": composition_dependency_identity,
            },
            "composition": {
                "selected": "g1_cmu",
                "source": source_cfg.identifier,
                "scene_robot": "g1_29dof",
                "frame_builder_version": table.frame_builder_version,
                "frame_builder_identity_sha256": table.frame_builder_identity_sha256,
                "table_identity_sha256": table.cache_identity,
                "joint_names": list(table.joint_names),
                "reference_frame_names": list(table.reference_frame_names),
                "resolved_environment_axes_unmodified": True,
            },
            "execution": {
                "source_artifact_root": table_cfg.source_artifact_root,
                "reference_artifact_root": table_cfg.reference_artifact_root,
                "seed": args.seed,
                "device": str(env.device),
                "physics_dt_seconds": cfg.sim.dt,
                "control_decimation": cfg.decimation,
                "control_dt_seconds": env.step_dt,
                "observation_corruption_enabled": cfg.observations.state.enable_corruption,
                "default_joint_offset_range_rad": list(cfg.actions.joint_position.default_joint_offset_range),
            },
            "selection": {
                "split": args.motion_split,
                "clip_ids": selected_clip_ids,
                "num_clips": len(selected_clip_ids),
                "requested_steps": args.num_steps,
                "active_reached_rows": active_row_count,
                "unexpected_done_rows": unexpected_done_count,
                "row_lifecycle": ROW_LIFECYCLE,
            },
            "reference_controller": {
                "kind": "one_step_joint_reference_lookahead_through_production_position_pd",
                "oracle_reference_access": True,
                "trajectory_and_simulator_share_physical_axis": True,
                "behavior_joint_names": list(action.joint_names),
                "behavior_to_physical_joint_indices": list(behavior_to_physical),
                "step_seconds": env.step_dt,
                "position_limits_owned_by": "ArticulationData.joint_pos_limits",
                "controlled_coordinates": "29 joints; floating root unactuated",
                "active_coordinate_count": active_coordinate_count,
                "position_limit_clipped_coordinates": position_limit_clipped_count,
                "position_limit_clipped_fraction": position_limit_clipped_count / active_coordinate_count,
                "action_limit_clipped_coordinates": action_limit_clipped_count,
                "action_limit_clipped_fraction": action_limit_clipped_count / active_coordinate_count,
                "position_limit_projection_abs_rad": _statistics(torch.cat(limit_projection)),
                "action_limit_projection_abs_rad": _statistics(torch.cat(action_projection)),
            },
            "error_layers": {
                "retarget_fit": retarget_layer,
                "reference_controller_simulator": {
                    "status": "measured",
                    "claim": "simulated_physical_robot_state_minus_materialized_reference_at_equal_time",
                    "reset_alignment": {name: _statistics(torch.cat(chunks)) for name, chunks in reset_errors.items()},
                    "tracking": {name: _statistics(torch.cat(chunks)) for name, chunks in errors.items()},
                    "simulated_joint_limit_feasibility": {
                        "exceedance_rad": _statistics(simulated_limit_exceedance_value),
                        "violating_coordinate_count": int(torch.count_nonzero(simulated_limit_exceedance_value)),
                        "violating_reached_row_count": int(
                            torch.count_nonzero(torch.any(simulated_limit_exceedance_value > 0.0, dim=1))
                        ),
                    },
                    "duration_seconds": duration_seconds,
                    "simulated_transitions_per_second": active_row_count / duration_seconds,
                    "physical_body_scope": "all live articulation bodies; derived head reference excluded",
                },
                "policy": _policy_error_layer(),
            },
        }
    finally:
        env.close()


def main() -> None:
    """Launch Isaac Sim, run one bounded production-preset probe, and write evidence."""
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_artifact_root", type=Path, required=True)
    parser.add_argument("--reference_artifact_root", type=Path, required=True)
    parser.add_argument("--retarget_evidence", type=Path, required=True)
    parser.add_argument("--motion_split", choices=("train", "evaluation"), default="evaluation")
    parser.add_argument("--num_clips", type=int, required=True)
    parser.add_argument("--num_steps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_clips < 1 or args.num_steps < 1:
        raise ValueError("num_clips and num_steps must be positive.")
    if args.num_steps >= 501:
        raise ValueError("The reference probe must stop before the native G1 timeout edge.")

    app_launcher = None
    try:
        app_launcher = AppLauncher(args)
        report = _run(args)
        _write_report(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
    except BaseException as error:
        _write_report(args.output, _failed_report(args, error))
        raise
    finally:
        if app_launcher is not None:
            app_launcher.app.close()


if __name__ == "__main__":
    main()
