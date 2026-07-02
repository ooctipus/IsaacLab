# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay frozen BFM G1 edges through the shared motion environment.

The source trace closes every exposed transition input: randomized rigid-body
mass, material, center-of-mass pose, reached contact force, root/joint state,
controller offset, actor facts, history, and behavior action.  The controlled
candidate disables new randomness and injects those source facts.  Each
edge starts from the source root/joint state and receives the source default
joint offset, current actor facts, history, and behavior action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
import types
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source_artifact_root", type=Path, required=True)
parser.add_argument("--reference_artifact_root", type=Path, required=True)
parser.add_argument("--oracle", type=Path, required=True)
parser.add_argument("--oracle_metadata", type=Path, required=True)
parser.add_argument("--evidence_contract", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import warp as wp
from motion_environment_identity import (
    motion_environment_dependency_identity,
    motion_environment_semantic_sha256,
)

from isaaclab.utils.math import quat_apply, quat_apply_inverse

from isaaclab_tasks.core.multi_task.motion.config.robots import G1_MOTION_ARTICULATION_CFG
from isaaclab_tasks.core.multi_task.motion.config.source_skeletons import g1_lafan_source_skeleton
from isaaclab_tasks.core.multi_task.motion.data.importers import BfmG1JoblibClips
from isaaclab_tasks.core.multi_task.motion.mdp.actions import MotionJointPositionAction
from isaaclab_tasks.core.multi_task.motion.mdp.commands import MotionStatePayload
from isaaclab_tasks.core.multi_task.motion.mdp.observations import g1_privileged_observation
from isaaclab_tasks.core.multi_task.motion.trajectory.g1 import G1LafanFrameBuilder
from isaaclab_tasks.core.multi_task.motion_env import MotionImitationEnv
from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionImitationEnvCfg
from isaaclab_tasks.utils import resolve_presets

_HISTORY_FIELDS = (
    ("processed_action", 29, slice(0, 29)),
    ("base_angular_velocity", 3, slice(61, 64)),
    ("joint_position", 29, slice(0, 29)),
    ("joint_velocity", 29, slice(29, 58)),
    ("projected_gravity", 3, slice(58, 61)),
)
_SUBSTEP_FIELDS = (
    "qpos",
    "qvel",
    "body_position",
    "body_rotation_xyzw",
    "body_linear_velocity",
    "body_angular_velocity",
    "contact_force",
    "applied_pd_torque",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    difference = np.abs(np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64))
    return {
        "max_abs": float(difference.max(initial=0.0)),
        "mean_abs": float(difference.mean()) if difference.size else 0.0,
        "rms": float(np.sqrt(np.mean(np.square(difference)))) if difference.size else 0.0,
    }


def _within(actual: np.ndarray, expected: np.ndarray, *, atol: float, rtol: float = 0.0) -> dict[str, object]:
    result = _metric(actual, expected)
    result.update({"atol": atol, "rtol": rtol, "passed": bool(np.allclose(actual, expected, atol=atol, rtol=rtol))})
    return result


def _quaternion_geodesic(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    actual64 = np.asarray(actual, dtype=np.float64)
    expected64 = np.asarray(expected, dtype=np.float64)
    actual64 = actual64 / np.linalg.norm(actual64, axis=-1, keepdims=True).clip(min=1.0e-12)
    expected64 = expected64 / np.linalg.norm(expected64, axis=-1, keepdims=True).clip(min=1.0e-12)
    dot = np.abs(np.sum(actual64 * expected64, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def _rotation_metric(actual: np.ndarray, expected: np.ndarray) -> dict[str, float | str]:
    angle = _quaternion_geodesic(actual, expected)
    return {
        "unit": "rad",
        "max": float(angle.max(initial=0.0)),
        "mean": float(angle.mean()) if angle.size else 0.0,
        "rms": float(np.sqrt(np.mean(np.square(angle)))) if angle.size else 0.0,
    }


def _rotation_within(actual: np.ndarray, expected: np.ndarray, *, atol: float) -> dict[str, object]:
    result: dict[str, object] = _rotation_metric(actual, expected)
    result.update({"atol": atol, "passed": bool(np.all(_quaternion_geodesic(actual, expected) <= atol))})
    return result


def _history_successor(
    current_history: np.ndarray,
    current_state: np.ndarray,
    current_last_action: np.ndarray,
    prior_edge_applied: np.ndarray,
) -> np.ndarray:
    expected = current_history.copy()
    offset = 0
    for name, width, state_slice in _HISTORY_FIELDS:
        end = offset + 4 * width
        view = expected[:, offset:end].reshape(-1, 4, width)
        source = current_last_action if name == "processed_action" else current_state[:, state_slice]
        view[prior_edge_applied, 1:] = view[prior_edge_applied, :-1].copy()
        view[prior_edge_applied, 0] = source[prior_edge_applied]
        offset = end
    return expected


def _controlled_environment_cfg() -> MotionImitationEnvCfg:
    """Resolve G1-LAFAN and disable randomness before exact source-fact injection."""
    cfg = resolve_presets(MotionImitationEnvCfg(), selected={"g1_lafan"})
    action = cfg.actions.joint_position
    action.default_joint_offset_range = (0.0, 0.0)
    cfg.observations.state.enable_corruption = False
    assert cfg.events.robot_material is not None
    assert cfg.events.body_mass is not None
    assert cfg.events.torso_com is not None
    cfg.events.robot_material.params["static_friction_range"] = (1.0, 1.0)
    cfg.events.robot_material.params["dynamic_friction_range"] = (1.0, 1.0)
    cfg.events.robot_material.params["num_buckets"] = 1
    cfg.events.body_mass.params["mass_distribution_params"] = (1.0, 1.0)
    cfg.events.torso_com.params["com_range"] = {axis: (0.0, 0.0) for axis in "xyz"}
    cfg.events.push = None
    return cfg


def _flatten(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(value.reshape(-1, *value.shape[2:]))


def _root_xyzw(qpos: torch.Tensor) -> torch.Tensor:
    return torch.cat((qpos[:, 4:7], qpos[:, 3:4]), dim=-1)


def _candidate_qpos_qvel(robot, joint_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    root_xyzw = robot.data.root_link_quat_w.torch
    root_wxyz = torch.cat((root_xyzw[:, 3:4], root_xyzw[:, :3]), dim=-1)
    angular_body = quat_apply_inverse(root_xyzw, robot.data.root_com_ang_vel_w.torch)
    joint_position = robot.data.joint_pos.torch.index_select(1, joint_ids)
    joint_velocity = robot.data.joint_vel.torch.index_select(1, joint_ids)
    qpos = torch.cat((robot.data.root_link_pos_w.torch, root_wxyz, joint_position), dim=-1)
    qvel = torch.cat((robot.data.root_com_lin_vel_w.torch, angular_body, joint_velocity), dim=-1)
    return qpos, qvel


def _candidate_body(robot, body_ids) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    position = robot.data.body_link_pos_w.torch[:, body_ids]
    rotation = robot.data.body_link_quat_w.torch[:, body_ids]
    linear_velocity = robot.data.body_com_lin_vel_w.torch[:, body_ids]
    angular_velocity = robot.data.body_com_ang_vel_w.torch[:, body_ids]
    torso = 15
    offset = position.new_tensor((0.0, 0.0, 0.35)).expand(position.shape[0], 3)
    synthetic_position = position[:, torso] + quat_apply(rotation[:, torso], offset)
    synthetic_rotation = rotation[:, torso]
    synthetic_angular = angular_velocity[:, torso]
    synthetic_linear = linear_velocity[:, torso] + torch.cross(synthetic_angular, offset, dim=-1)
    return (
        torch.cat((position, synthetic_position[:, None]), dim=1),
        torch.cat((rotation, synthetic_rotation[:, None]), dim=1),
        torch.cat((linear_velocity, synthetic_linear[:, None]), dim=1),
        torch.cat((angular_velocity, synthetic_angular[:, None]), dim=1),
    )


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


def _substep_measurement(
    candidate: Mapping[str, torch.Tensor],
    oracle: Mapping[str, np.ndarray],
    body_names: tuple[str, ...],
) -> dict[str, object]:
    """Localize cross-simulator residuals after each physics substep."""
    reports: list[dict[str, object]] = []
    for index in range(4):
        actual_qpos = _numpy(candidate["qpos"][:, index])
        expected_qpos = oracle["substep_qpos"][:, index]
        actual_qvel = _numpy(candidate["qvel"][:, index])
        expected_qvel = oracle["substep_qvel"][:, index]
        actual_body_rotation = _numpy(candidate["body_rotation_xyzw"][:, index])
        expected_body_rotation = oracle["substep_body_rotation_xyzw"][:, index]
        reports.append(
            {
                "index_zero_based": index,
                "qpos": {
                    "root_position": _metric(actual_qpos[:, :3], expected_qpos[:, :3]),
                    "root_rotation": _rotation_metric(actual_qpos[:, 3:7], expected_qpos[:, 3:7]),
                    "joint_position": _metric(actual_qpos[:, 7:], expected_qpos[:, 7:]),
                },
                "qvel": {
                    "root_linear_velocity": _metric(actual_qvel[:, :3], expected_qvel[:, :3]),
                    "root_angular_velocity": _metric(actual_qvel[:, 3:6], expected_qvel[:, 3:6]),
                    "joint_velocity": _metric(actual_qvel[:, 6:], expected_qvel[:, 6:]),
                },
                "body_position": _metric(
                    _numpy(candidate["body_position"][:, index]), oracle["substep_body_position"][:, index]
                ),
                "body_rotation_geodesic": _rotation_metric(actual_body_rotation, expected_body_rotation),
                "body_linear_velocity": _metric(
                    _numpy(candidate["body_linear_velocity"][:, index]),
                    oracle["substep_body_linear_velocity"][:, index],
                ),
                "body_angular_velocity": _metric(
                    _numpy(candidate["body_angular_velocity"][:, index]),
                    oracle["substep_body_angular_velocity"][:, index],
                ),
                "contact_force": _metric(
                    _numpy(candidate["contact_force"][:, index]), oracle["substep_contact_force"][:, index]
                ),
                "applied_pd_torque": _metric(
                    _numpy(candidate["applied_pd_torque"][:, index]),
                    oracle["substep_applied_pd_torque"][:, index],
                ),
                "max_abs_by_edge": {
                    name: np.max(
                        np.abs(_numpy(candidate[name][:, index]) - oracle[f"substep_{name}"][:, index]),
                        axis=tuple(range(1, candidate[name][:, index].ndim)),
                    ).tolist()
                    for name in ("qpos", "qvel", "contact_force", "applied_pd_torque")
                },
            }
        )

    first_body_position = np.abs(_numpy(candidate["body_position"][:, 0]) - oracle["substep_body_position"][:, 0])
    first_body_rotation = _quaternion_geodesic(
        _numpy(candidate["body_rotation_xyzw"][:, 0]), oracle["substep_body_rotation_xyzw"][:, 0]
    )
    return {
        "claim": "measured_cross_simulator_physics_not_elementwise_parity",
        "substeps": reports,
        "first_substep_body_max": {
            name: {
                "position_m": float(first_body_position[:, body_index].max()),
                "rotation_rad": float(first_body_rotation[:, body_index].max()),
            }
            for body_index, name in enumerate(body_names)
        },
    }


def _offline_exact(oracle: Mapping[str, np.ndarray], evidence: Mapping[str, object]) -> dict[str, object]:
    source_skeleton = g1_lafan_source_skeleton()
    default_joint_position = torch.tensor(
        [G1_MOTION_ARTICULATION_CFG.init_state.joint_pos[name] for name in source_skeleton.joint_names],
        dtype=torch.float32,
    )
    behavior = torch.from_numpy(oracle["behavior_action"])
    processed = torch.clamp(behavior * 5.0, -5.0, 5.0)
    target = (
        default_joint_position
        + torch.from_numpy(oracle["current_default_joint_offset"])
        + processed
        * 0.25
        * torch.from_numpy(oracle["current_joint_effort_limit"])
        / torch.from_numpy(oracle["current_joint_stiffness"])
    )

    privileged: dict[str, object] = {}
    for prefix in ("current", "returned", "final"):
        values = tuple(
            torch.from_numpy(oracle[f"{prefix}_{name}"]).reshape(-1, *oracle[f"{prefix}_{name}"].shape[2:])
            for name in ("body_position", "body_rotation_xyzw", "body_linear_velocity", "body_angular_velocity")
        )
        actual = g1_privileged_observation(*values).reshape(*oracle[f"{prefix}_privileged_state"].shape)
        expected = torch.from_numpy(oracle[f"{prefix}_privileged_state"])
        valid = (
            torch.from_numpy(oracle["final_observation_valid"])
            if prefix == "final"
            else torch.ones_like(torch.from_numpy(oracle["truncated"]))
        )
        privileged[prefix] = _within(_numpy(actual[valid]), _numpy(expected[valid]), atol=2.0e-5, rtol=2.0e-6)

    current_state = _flatten(oracle["current_state"])
    current_history = _flatten(oracle["current_history_actor"])
    current_last_action = _flatten(oracle["current_last_action"])
    prior = _flatten(oracle["current_episode_step"]) > 0
    post_autoreset_seed = np.zeros_like(oracle["terminated"], dtype=np.bool_)
    post_autoreset_seed[1:] = (oracle["terminated"] | oracle["truncated"])[:-1]
    source_history_eligible = prior | _flatten(post_autoreset_seed)
    history_expected = _history_successor(current_history, current_state, current_last_action, source_history_eligible)
    done = _flatten(oracle["truncated"] | oracle["terminated"])
    history_observed = _flatten(oracle["returned_history_actor"]).copy()
    history_observed[done] = _flatten(oracle["final_history_actor"])[done]

    source_qpos = torch.from_numpy(_flatten(oracle["current_qpos"]))
    source_qvel = torch.from_numpy(_flatten(oracle["current_qvel"]))
    source_qpos_root_xyzw = _root_xyzw(source_qpos)
    source_body_root_xyzw = torch.from_numpy(_flatten(oracle["current_body_rotation_xyzw"]))[:, 0]
    gravity = torch.zeros_like(source_qvel[:, :3])
    gravity[:, 2] = -1.0
    noise_free = torch.cat(
        (
            source_qpos[:, 7:]
            - default_joint_position
            - torch.from_numpy(_flatten(oracle["current_default_joint_offset"])),
            source_qvel[:, 6:],
            quat_apply_inverse(source_body_root_xyzw, gravity),
            source_qvel[:, 3:6] * 0.25,
        ),
        dim=-1,
    )
    residual = torch.from_numpy(current_state) - noise_free
    noise_bounds = torch.tensor(
        (*(0.01 for _ in range(29)), *(0.5 for _ in range(29)), *(0.05 for _ in range(3)), *(0.05 for _ in range(3)))
    )

    source_evidence_names = tuple(evidence["environment_raw_evidence"])
    auxiliary_names = tuple(evidence["learner_raw_evidence"])
    auxiliary_indices = [source_evidence_names.index(name) for name in auxiliary_names]
    selected_auxiliary = oracle["environment_raw_evidence"][..., auxiliary_indices]
    exact = {
        "processed_action": _within(_numpy(processed), oracle["processed_action"], atol=0.0),
        "controller_target_joint_position": _within(
            _numpy(target), oracle["controller_target_joint_position"], atol=0.0
        ),
        "privileged_body_algebra": privileged,
        "history_recurrence": _within(history_expected, history_observed, atol=0.0),
        "source_reward_recomposition": _within(
            oracle["environment_reward_recomposed"], oracle["environment_reward"], atol=1.0e-5, rtol=1.0e-5
        ),
        "source_auxiliary_selection": _within(selected_auxiliary, oracle["learner_auxiliary_raw_evidence"], atol=0.0),
    }
    noise = {
        "max_fraction_of_declared_half_range": float(torch.max(torch.abs(residual) / noise_bounds).item()),
        "passed": bool(torch.all(torch.abs(residual) <= noise_bounds + 2.0e-6)),
    }
    clock_angle = _quaternion_geodesic(_numpy(source_qpos_root_xyzw), _numpy(source_body_root_xyzw))
    return {
        "exact": exact,
        "actor_noise_envelope": noise,
        "history_reset_seed_normalization": {
            "source_post_autoreset_seed_rows": int(post_autoreset_seed.sum()),
            "source_law": "post_reset_observation_is_appended_to_history",
            "unified_law": "only_nodes_reached_by_applied_actions_enter_history",
        },
        "source_current_clock_skew": {
            **_rotation_metric(_numpy(source_qpos_root_xyzw), _numpy(source_body_root_xyzw)),
            "skewed_rows": int(np.count_nonzero(clock_angle > 1.0e-6)),
            "total_rows": int(clock_angle.size),
        },
    }


def _run_candidate(
    oracle: Mapping[str, np.ndarray],
    evidence: Mapping[str, object],
    source_body_names: tuple[str, ...],
    source_shape_count: int,
) -> tuple[dict[str, object], dict[str, object]]:
    flat = {name: _flatten(value) for name, value in oracle.items()}
    num_edges = flat["behavior_action"].shape[0]
    cfg = _controlled_environment_cfg()
    table_cfg = cfg.commands.motion.task_table
    table_cfg.source_artifact_root = str(args.source_artifact_root.expanduser().resolve())
    table_cfg.reference_artifact_root = str(args.reference_artifact_root.expanduser().resolve())
    table_cfg.motion_split = "evaluation"
    cfg.scene.num_envs = num_edges
    cfg.seed = 0
    cfg.sim.device = args.device
    dependency_identity = motion_environment_dependency_identity(
        preset="g1_lafan",
        cfg=cfg,
        importer_type=BfmG1JoblibClips,
        frame_builder_type=G1LafanFrameBuilder,
        reference_artifact_root=table_cfg.reference_artifact_root,
    )
    env = MotionImitationEnv(cfg=cfg)

    try:
        env.reset()
        robot = env.scene["robot"]
        action = env.action_manager.get_term("joint_position")
        command = env.command_manager.get_term("motion")
        table = command.table
        payload = command.payload
        if not isinstance(action, MotionJointPositionAction) or not isinstance(payload, MotionStatePayload):
            raise TypeError("Controlled G1 replay requires the final motion action and payload types.")
        source_joint_names = g1_lafan_source_skeleton().joint_names
        simulator_joint_names = tuple(robot.joint_names)
        if table.joint_names != simulator_joint_names:
            raise ValueError("The trajectory table must retain the live simulator joint axis.")
        if action.joint_names != source_joint_names:
            raise ValueError("The action term must retain the frozen behavior joint axis.")
        if set(source_joint_names) != set(simulator_joint_names):
            raise ValueError("Frozen BFM and live G1 joint names differ.")
        device = torch.device(env.device)
        expected_joint_ids = torch.tensor(
            tuple(simulator_joint_names.index(name) for name in source_joint_names),
            dtype=torch.int64,
            device=device,
        )
        joint_ids = action.joint_ids
        if not torch.equal(joint_ids, expected_joint_ids):
            raise ValueError("The action term's behavior-to-simulator joint map is incorrect.")
        simulator_joint_ids = joint_ids.to(dtype=torch.int32)
        env_ids = torch.arange(num_edges, dtype=torch.int64, device=device)
        body_ids, observed_body_names = robot.find_bodies(list(source_body_names), preserve_order=True)
        if tuple(observed_body_names) != source_body_names:
            raise ValueError("Candidate articulation cannot reproduce the source physical-body order.")
        contact_sensor = env._motion_runtime.contact_sensor
        sensor_ids, observed_sensor_names = contact_sensor.find_sensors(list(source_body_names), preserve_order=True)
        if tuple(observed_sensor_names) != source_body_names:
            raise ValueError("Candidate contact sensor cannot reproduce the source physical-body order.")
        if robot.root_view.max_shapes != source_shape_count:
            raise ValueError("Candidate collision-shape count differs from the native source.")

        source_mass = torch.from_numpy(flat["current_body_mass"]).to(device)
        source_com = torch.from_numpy(flat["current_body_com_pose_xyzw"]).to(device)
        source_inertia = torch.from_numpy(flat["current_body_inertia"]).to(device)
        source_material = torch.from_numpy(flat["current_shape_material"])
        robot.set_masses_index(masses=source_mass, body_ids=body_ids, env_ids=env_ids)
        robot.set_coms_index(coms=source_com, body_ids=body_ids, env_ids=env_ids)
        robot.set_inertias_index(inertias=source_inertia, body_ids=body_ids, env_ids=env_ids)

        material_full = source_material
        cpu_env_ids = torch.arange(num_edges, dtype=torch.int32)
        robot.root_view.set_material_properties(
            wp.from_torch(material_full, dtype=wp.float32),
            wp.from_torch(cpu_env_ids, dtype=wp.int32),
        )
        source_qpos = torch.from_numpy(flat["current_qpos"]).to(device)
        source_qvel = torch.from_numpy(flat["current_qvel"]).to(device)
        root_xyzw = _root_xyzw(source_qpos)
        root_velocity = torch.cat((source_qvel[:, :3], quat_apply(root_xyzw, source_qvel[:, 3:6])), dim=-1)
        robot.write_root_link_pose_to_sim_index(
            root_pose=torch.cat((source_qpos[:, :3], root_xyzw), dim=-1), env_ids=env_ids
        )
        robot.write_root_com_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)
        robot.write_joint_position_to_sim_index(
            position=source_qpos[:, 7:], joint_ids=simulator_joint_ids, env_ids=env_ids
        )
        robot.write_joint_velocity_to_sim_index(
            velocity=source_qvel[:, 6:], joint_ids=simulator_joint_ids, env_ids=env_ids
        )
        source_offset = torch.from_numpy(flat["current_default_joint_offset"]).to(device)
        source_last_action = torch.from_numpy(flat["current_last_action"]).to(device)
        action.default_joint_offset.copy_(source_offset)
        action._processed_actions.copy_(source_last_action)
        action._raw_actions.copy_(action._processed_actions / action.cfg.normalize_to)
        action.joint_position_target.copy_(action.joint_default_position + action.default_joint_offset)
        source_history = torch.from_numpy(flat["current_history_actor"]).to(device)
        payload.history_value.copy_(source_history)
        payload.prior_edge_applied.copy_(torch.from_numpy(flat["current_episode_step"] > 0).to(device))
        env.episode_length_buf.zero_()
        expected_done = torch.from_numpy(flat["terminated"] | flat["truncated"]).to(device)
        timeout_count = env.cfg.terminations.time_out.params["applied_actions_before_timeout"]
        env.episode_length_buf[expected_done] = timeout_count - 1

        env.scene.write_data_to_sim()
        env.sim.forward()
        current_qpos, current_qvel = _candidate_qpos_qvel(robot, joint_ids)
        injected_mass = robot.data.body_mass.torch[:, body_ids].clone()
        injected_com = robot.data.body_com_pose_b.torch[:, body_ids].clone()
        injected_material = wp.to_torch(robot.root_view.get_material_properties()).clone()
        injected_inertia = robot.data.body_inertia.torch[:, body_ids].clone()
        drive_values = {
            "stiffness": robot.data.joint_stiffness.torch.index_select(1, joint_ids),
            "damping": robot.data.joint_damping.torch.index_select(1, joint_ids),
            "armature": robot.data.joint_armature.torch.index_select(1, joint_ids),
            "friction": robot.data.joint_friction_coeff.torch.index_select(1, joint_ids),
            "effort_limit": robot.data.joint_effort_limits.torch.index_select(1, joint_ids),
            "velocity_limit": robot.data.joint_vel_limits.torch.index_select(1, joint_ids),
            "position_limit": robot.data.joint_pos_limits.torch.index_select(1, joint_ids),
        }
        candidate_current = {name: value.clone() for name, value in env.observation_manager.compute().items()}

        # Feed the exact frozen actor facts to the history recurrence.  The
        # separately retained candidate_current remains the actual noise-free
        # observation measured from the injected physical state.
        env.obs_buf = {name: value.clone() for name, value in candidate_current.items()}
        source_state = torch.from_numpy(flat["current_state"]).to(device)
        env.obs_buf["state"].copy_(source_state)
        env.obs_buf["last_action"].copy_(source_last_action)
        env.obs_buf["history_actor"].copy_(payload.history_value)
        substep_values: dict[str, list[torch.Tensor]] = {name: [] for name in _SUBSTEP_FIELDS}
        original_scene_update = env.scene.update

        def update_with_capture(_instance, *call_args, **call_kwargs):
            result = original_scene_update(*call_args, **call_kwargs)
            qpos, qvel = _candidate_qpos_qvel(robot, joint_ids)
            body = _candidate_body(robot, body_ids)
            values = {
                "qpos": qpos,
                "qvel": qvel,
                "body_position": body[0],
                "body_rotation_xyzw": body[1],
                "body_linear_velocity": body[2],
                "body_angular_velocity": body[3],
                "contact_force": contact_sensor.data.net_forces_w.torch[:, sensor_ids],
                "applied_pd_torque": action.applied_torque,
            }
            for name, value in values.items():
                substep_values[name].append(value.clone())
            return result

        reached_qpos = torch.empty_like(source_qpos)
        reached_qvel = torch.empty_like(source_qvel)
        reached_body = tuple(torch.empty((num_edges, 31, width), device=device) for width in (3, 4, 3, 3))
        reached_target = torch.empty((num_edges, 29), device=device)
        reached_valid = torch.zeros(num_edges, dtype=torch.bool, device=device)
        reached_contact = torch.empty((num_edges, len(source_body_names), 3), device=device)
        original_reset = env._reset_idx

        def capture_reset(_instance, reset_ids):
            reset_ids = torch.as_tensor(reset_ids, dtype=torch.int64, device=device)
            qpos, qvel = _candidate_qpos_qvel(robot, joint_ids)
            reached_qpos.index_copy_(0, reset_ids, qpos.index_select(0, reset_ids))
            reached_qvel.index_copy_(0, reset_ids, qvel.index_select(0, reset_ids))
            for destination, source in zip(reached_body, _candidate_body(robot, body_ids), strict=True):
                destination.index_copy_(0, reset_ids, source.index_select(0, reset_ids))
            source_target = action.joint_position_target
            reached_target.index_copy_(0, reset_ids, source_target.index_select(0, reset_ids))
            contact = contact_sensor.data.net_forces_w.torch[:, sensor_ids]
            reached_contact.index_copy_(0, reset_ids, contact.index_select(0, reset_ids))
            reached_valid.index_fill_(0, reset_ids, True)
            return original_reset(reset_ids)

        env._reset_idx = types.MethodType(capture_reset, env)
        env.scene.update = types.MethodType(update_with_capture, env.scene)
        behavior = torch.from_numpy(flat["behavior_action"]).to(device)
        try:
            returned, reward, terminated, truncated, extras = env.step(behavior)
        finally:
            env._reset_idx = original_reset
            env.scene.update = original_scene_update
        if any(len(values) != 4 for values in substep_values.values()):
            raise RuntimeError("Candidate G1 replay must capture exactly four physics substeps per action.")
        candidate_substeps = {name: torch.stack(values, dim=1) for name, values in substep_values.items()}

        done = terminated | truncated
        if not torch.equal(done, expected_done):
            raise RuntimeError("Candidate G1 timeout rows differ from the frozen native edge lifecycle.")
        not_done = ~done
        qpos, qvel = _candidate_qpos_qvel(robot, joint_ids)
        reached_qpos[not_done] = qpos[not_done]
        reached_qvel[not_done] = qvel[not_done]
        for destination, source in zip(reached_body, _candidate_body(robot, body_ids), strict=True):
            destination[not_done] = source[not_done]
        reached_target[not_done] = action.joint_position_target[not_done]
        reached_valid[not_done] = True
        reached_contact[not_done] = contact_sensor.data.net_forces_w.torch[not_done][:, sensor_ids]

        final = extras.get("final_obs")
        if final is None:
            raise RuntimeError("Candidate G1 done rows require exact pre-reset final observations.")
        history_actual = returned["history_actor"].clone()
        history_actual[expected_done] = final["history_actor"][expected_done]
        history_expected = _history_successor(
            _numpy(source_history),
            _numpy(source_state),
            _numpy(source_last_action),
            flat["current_episode_step"] > 0,
        )

        raw_names = tuple(evidence["environment_raw_evidence"])
        raw = torch.cat(tuple(payload.raw_evidence[name] for name in raw_names), dim=-1)
        auxiliary_names = tuple(payload.auxiliary_evidence_names)
        if set(auxiliary_names) != set(evidence["learner_raw_evidence"]):
            raise ValueError("Runtime auxiliary evidence names differ from the frozen learner contract.")
        auxiliary_indices = [raw_names.index(name) for name in auxiliary_names]
        composed_auxiliary = raw[:, auxiliary_indices]

        oracle_reached_qpos = np.where(
            flat["final_observation_valid"][:, None], flat["final_qpos"], flat["returned_qpos"]
        )
        oracle_reached_qvel = np.where(
            flat["final_observation_valid"][:, None], flat["final_qvel"], flat["returned_qvel"]
        )
        oracle_reached_body = tuple(
            np.where(
                flat["final_observation_valid"][:, None, None],
                flat[f"final_body_{name}"],
                flat[f"returned_body_{name}"],
            )
            for name in ("position", "rotation_xyzw", "linear_velocity", "angular_velocity")
        )
        current_state = _numpy(candidate_current["state"])
        oracle_reached_contact = np.where(
            flat["final_observation_valid"][:, None, None],
            flat["final_contact_force"],
            flat["returned_contact_force"],
        )
        source_state = _numpy(source_state)
        actual_qpos = _numpy(current_qpos)
        expected_qpos = flat["current_qpos"]
        root_position = _within(actual_qpos[:, :3], expected_qpos[:, :3], atol=2.0e-6)
        root_rotation = _rotation_within(actual_qpos[:, 3:7], expected_qpos[:, 3:7], atol=2.0e-6)
        joint_position = _within(actual_qpos[:, 7:], expected_qpos[:, 7:], atol=2.0e-6)

        mass_readback = _within(_numpy(injected_mass), flat["current_body_mass"], atol=0.0)
        inertia_readback = _within(_numpy(injected_inertia), flat["current_body_inertia"], atol=1.0e-7, rtol=1.0e-6)
        material_readback = _within(_numpy(injected_material), flat["current_shape_material"], atol=0.0)
        com_readback = _within(_numpy(injected_com), flat["current_body_com_pose_xyzw"], atol=0.0)
        drive_readback = {
            name: _within(_numpy(value), flat[f"current_joint_{name}"], atol=0.0)
            for name, value in drive_values.items()
        }
        drive_readback["passed"] = all(bool(value["passed"]) for value in drive_readback.values())
        physics_readback = {
            "body_mass": mass_readback,
            "shape_material": material_readback,
            "body_com_pose": com_readback,
            "body_inertia": inertia_readback,
            "passed": bool(
                mass_readback["passed"]
                and inertia_readback["passed"]
                and material_readback["passed"]
                and com_readback["passed"]
            ),
        }
        exact = {
            "injected_qpos_readback": {
                "root_position": root_position,
                "root_rotation": root_rotation,
                "joint_position": joint_position,
                "passed": bool(root_position["passed"] and root_rotation["passed"] and joint_position["passed"]),
            },
            "injected_qvel_readback": _within(_numpy(current_qvel), flat["current_qvel"], atol=2.0e-5),
            "controller_target_joint_position": _within(
                _numpy(reached_target), flat["controller_target_joint_position"], atol=2.0e-6
            ),
            "injected_physics_fact_readback": physics_readback,
            "joint_drive_readback": drive_readback,
            "history_recurrence": _within(_numpy(history_actual), history_expected, atol=0.0),
            "done_mask": {
                "passed": bool(torch.equal(terminated | truncated, expected_done)),
                "expected_done_rows": int(expected_done.sum()),
                "observed_done_rows": int((terminated | truncated).sum()),
            },
            "final_observation_valid": {
                "passed": bool(torch.equal(extras["final_obs_valid"], expected_done)),
                "expected_rows": int(expected_done.sum()),
                "observed_rows": int(extras["final_obs_valid"].sum()),
            },
            "auxiliary_evidence_selection": _within(
                _numpy(extras["auxiliary_reward_evidence"]), _numpy(composed_auxiliary), atol=0.0
            ),
            "environment_reward_from_raw_evidence": _within(
                _numpy(reward), _numpy(env._motion_runtime.environment_reward * env.step_dt), atol=2.0e-6
            ),
            "all_reached_rows_captured": {"passed": bool(reached_valid.all())},
        }
        current = {
            "comparable": {
                "joint_position": _within(current_state[:, :29], source_state[:, :29], atol=0.01002),
                "joint_velocity": _within(current_state[:, 29:58], source_state[:, 29:58], atol=0.50002),
                "base_angular_velocity": _within(current_state[:, 61:64], source_state[:, 61:64], atol=0.05002),
            },
            "source_clock_or_simulator_residual": {
                "projected_gravity": _metric(current_state[:, 58:61], source_state[:, 58:61]),
                "privileged_state": _metric(
                    _numpy(candidate_current["privileged_state"]), flat["current_privileged_state"]
                ),
            },
        }
        reached_transition = {
            "claim": "measurement_after_all_exposed_source_transition_facts_injected",
            "qpos": _metric(_numpy(reached_qpos), oracle_reached_qpos),
            "qvel": _metric(_numpy(reached_qvel), oracle_reached_qvel),
            "body_position": _metric(_numpy(reached_body[0]), oracle_reached_body[0]),
            "body_rotation_geodesic": _rotation_metric(_numpy(reached_body[1]), oracle_reached_body[1]),
            "body_linear_velocity": _metric(_numpy(reached_body[2]), oracle_reached_body[2]),
            "body_angular_velocity": _metric(_numpy(reached_body[3]), oracle_reached_body[3]),
            "environment_raw_evidence_by_channel": {
                name: _metric(_numpy(raw[:, index]), flat["environment_raw_evidence"][:, index])
                for index, name in enumerate(raw_names)
            },
            "contact_force": _metric(_numpy(reached_contact), oracle_reached_contact),
            "applied_pd_torque": _metric(
                _numpy(candidate_substeps["applied_pd_torque"][:, -1]), flat["estimated_pd_torque"]
            ),
        }
        substep_measurement = _substep_measurement(candidate_substeps, flat, (*source_body_names, "head_link"))
        return {
            "exact": exact,
            "current": current,
            "reached_transition": reached_transition,
            "substep_transition": substep_measurement,
        }, dependency_identity
    finally:
        env.close()


def main() -> None:
    """Run the controlled replay and atomically write its scientific contract."""
    with np.load(args.oracle, allow_pickle=False) as loaded:
        oracle = {name: loaded[name].copy() for name in loaded.files}
    metadata = json.loads(args.oracle_metadata.read_text())
    evidence_root = json.loads(args.evidence_contract.read_text())
    evidence = evidence_root["profiles"]["g1_lafan_50hz"]
    required_facts = {
        "current_body_mass",
        "current_shape_material",
        "current_body_inertia",
        "current_body_com_pose_xyzw",
        "returned_contact_force",
        "final_contact_force",
        "current_joint_stiffness",
        "current_joint_damping",
        "current_joint_armature",
        "current_joint_friction",
        "current_joint_effort_limit",
        "current_joint_velocity_limit",
        "current_joint_position_limit",
        *(f"substep_{name}" for name in _SUBSTEP_FIELDS),
    }
    if missing := required_facts.difference(oracle):
        raise ValueError(f"Native oracle is missing transition facts: {sorted(missing)}")
    offline = _offline_exact(oracle, evidence)
    source_body_names = tuple(metadata["source"]["physical_body_names"])
    candidate, dependency_identity = _run_candidate(
        oracle, evidence, source_body_names, metadata["source"]["physical_shape_count"]
    )

    exact_results = [
        offline["exact"]["processed_action"],
        offline["exact"]["controller_target_joint_position"],
        *offline["exact"]["privileged_body_algebra"].values(),
        offline["exact"]["history_recurrence"],
        offline["exact"]["source_reward_recomposition"],
        offline["exact"]["source_auxiliary_selection"],
        *candidate["exact"].values(),
    ]
    code_identity = {
        "probe_sha256": _sha256(Path(__file__).resolve()),
        "dependency_identity": dependency_identity,
        "environment_semantic_sha256": motion_environment_semantic_sha256(dependency_identity),
    }
    exact_passed = all(bool(result["passed"]) for result in exact_results)
    current_passed = bool(offline["actor_noise_envelope"]["passed"]) and all(
        bool(result["passed"]) for result in candidate["current"]["comparable"].values()
    )
    report = {
        "schema": "forward_backward_phase3e_g1_native_edge_replay_v3",
        "profile": "g1_lafan",
        "code_identity": code_identity,
        "oracle": {
            "path": str(args.oracle.resolve()),
            "sha256": _sha256(args.oracle),
            "source_trace_generator": metadata["source"]["trace_generator"],
            "physical_shape_count": metadata["source"]["physical_shape_count"],
            "metadata_sha256": _sha256(args.oracle_metadata),
            "source_revision": metadata["source"]["repository_revision"],
            "edges": int(oracle["behavior_action"].shape[0] * oracle["behavior_action"].shape[1]),
        },
        "controlled_candidate": {
            "source_artifact_root": str(args.source_artifact_root.expanduser().resolve()),
            "reference_artifact_root": str(args.reference_artifact_root.expanduser().resolve()),
            "observation_noise": "disabled",
            "frozen_source_joint_axis_mapped_once_to_live_axis": True,
            "physics_facts": "source_body_mass_inertia_com_and_shape_material_per_edge",
            "pushes": "disabled",
            "edge_seed": "exact_frozen_source_transition_facts",
        },
        "offline_oracle_checks": offline,
        "candidate_replay": candidate,
        "decision": {
            "exact_contract_passed": exact_passed,
            "current_observation_contract_passed": current_passed,
            "simulator_elementwise_parity_claimed": False,
            "passed": exact_passed and current_passed,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["decision"]["passed"]:
        raise RuntimeError(f"G1 native edge contract failed: {json.dumps(report['decision'], sort_keys=True)}")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
