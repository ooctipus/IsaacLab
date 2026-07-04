# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay applied HumEnv NEXT_STEP edges through the shared SMPL environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source_artifact_root", type=Path, required=True)
parser.add_argument("--oracle", type=Path, required=True)
parser.add_argument("--oracle_metadata", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import warp as wp
from motion_environment_identity import motion_environment_axes, motion_environment_dependency_identity

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from isaaclab_tasks.core.multi_task.mdp.native_mujoco_action import NativeMujocoControlAction
from isaaclab_tasks.core.multi_task.motion.data.sources import CmuHumEnvSmplClips
from isaaclab_tasks.core.multi_task.motion.robots.smpl.frames import smpl_live_joint_source_names
from isaaclab_tasks.core.multi_task.motion.robots.smpl.reference import SmplGeneralizedCoordinateFrameBuilder
from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionImitationEnvCfg
from isaaclab_tasks.utils import resolve_presets

from isaaclab_assets.robots.smpl.smpl_constants import (
    MUJOCO_BODY_NAMES,
    MUJOCO_JOINT_NAMES,
    SMPL_HUMENV_MJCF_PATH,
)

_OBSERVATION_SLICES = {
    "root_height": slice(0, 1),
    "heading_local_body_position": slice(1, 70),
    "heading_local_body_rotation_tangent_normal": slice(70, 214),
    "heading_local_body_linear_velocity": slice(214, 286),
    "heading_local_body_angular_velocity": slice(286, 358),
}


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _metric(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    difference = np.abs(actual - expected)
    result = {
        "max_abs": float(difference.max(initial=0.0)),
        "mean_abs": float(difference.mean()) if difference.size else 0.0,
        "rms": float(np.sqrt(np.mean(np.square(difference)))) if difference.size else 0.0,
    }
    if difference.size:
        index = np.unravel_index(int(difference.argmax()), difference.shape)
        result.update(
            {
                "max_index": [int(component) for component in index],
                "actual_at_max": float(actual[index]),
                "expected_at_max": float(expected[index]),
            }
        )
    return result


def _within(actual: np.ndarray, expected: np.ndarray, *, atol: float, rtol: float = 0.0) -> dict[str, object]:
    result: dict[str, object] = _metric(actual, expected)
    result.update({"atol": atol, "rtol": rtol, "passed": bool(np.allclose(actual, expected, atol=atol, rtol=rtol))})
    return result


def _quaternion_geodesic(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    actual = actual / np.linalg.norm(actual, axis=-1, keepdims=True).clip(min=1.0e-12)
    expected = expected / np.linalg.norm(expected, axis=-1, keepdims=True).clip(min=1.0e-12)
    dot = np.abs(np.sum(actual * expected, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def _rotation_within(actual: np.ndarray, expected: np.ndarray, *, atol: float) -> dict[str, object]:
    angle = _quaternion_geodesic(actual, expected)
    result = {
        "unit": "rad",
        "max": float(angle.max(initial=0.0)),
        "mean": float(angle.mean()) if angle.size else 0.0,
        "rms": float(np.sqrt(np.mean(np.square(angle)))) if angle.size else 0.0,
        "atol": atol,
        "passed": bool(np.all(angle <= atol)),
    }
    if angle.size:
        index = np.unravel_index(int(angle.argmax()), angle.shape)
        result.update(
            {
                "max_index": [int(component) for component in index],
                "actual_at_max": actual[index].tolist(),
                "expected_at_max": expected[index].tolist(),
            }
        )
    return result


def _per_edge_metrics(actual: np.ndarray, expected: np.ndarray, *, atol: float) -> list[dict[str, object]]:
    """Return scalar error summaries for each applied source edge."""
    return [
        {
            **_metric(actual[index], expected[index]),
            "atol": atol,
            "passed": bool(np.allclose(actual[index], expected[index], atol=atol, rtol=0.0)),
        }
        for index in range(actual.shape[0])
    ]


def _per_edge_rotation_metrics(actual: np.ndarray, expected: np.ndarray, *, atol: float) -> list[dict[str, object]]:
    """Return quaternion-geodesic error summaries for each applied source edge."""
    results = []
    for index in range(actual.shape[0]):
        angle = _quaternion_geodesic(actual[index], expected[index])
        results.append(
            {
                "unit": "rad",
                "max": float(angle.max(initial=0.0)),
                "mean": float(angle.mean()) if angle.size else 0.0,
                "rms": float(np.sqrt(np.mean(np.square(angle)))) if angle.size else 0.0,
                "atol": atol,
                "passed": bool(np.all(angle <= atol)),
            }
        )
    return results


def _quaternion_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Return rotation matrices for normalized ``wxyz`` quaternions."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True).clip(min=1.0e-12)
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def _inertia_tensor(principal: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """Reconstruct body-frame inertia tensors from principal-axis encodings."""
    rotation = _quaternion_matrix_wxyz(orientation)
    diagonal = np.zeros((*principal.shape[:-1], 3, 3), dtype=np.float64)
    diagonal[..., np.arange(3), np.arange(3)] = principal
    return rotation @ diagonal @ np.swapaxes(rotation, -1, -2)


def _collision_graph(
    contype: np.ndarray,
    conaffinity: np.ndarray,
    *,
    body_ids: np.ndarray | None = None,
    excluded_body_pairs: tuple[tuple[int, int], ...] = (),
) -> np.ndarray:
    """Return the pairwise collision permission represented by MuJoCo bit masks."""
    contype = np.asarray(contype, dtype=np.int64)
    conaffinity = np.asarray(conaffinity, dtype=np.int64)
    graph = ((contype[:, None] & conaffinity[None, :]) != 0) | ((contype[None, :] & conaffinity[:, None]) != 0)
    np.fill_diagonal(graph, False)
    if body_ids is not None:
        for first, second in excluded_body_pairs:
            blocked = ((body_ids[:, None] == first) & (body_ids[None, :] == second)) | (
                (body_ids[:, None] == second) & (body_ids[None, :] == first)
            )
            graph[blocked] = False
    return graph


def _source_contact_exclusions() -> tuple[tuple[int, int], ...]:
    """Read the source MJCF's named body-pair exclusions in model index space."""
    body_id = {name: index + 1 for index, name in enumerate(MUJOCO_BODY_NAMES)}
    root = ET.parse(SMPL_HUMENV_MJCF_PATH).getroot()
    exclusions = []
    for element in root.findall("./contact/exclude"):
        first = element.get("body1")
        second = element.get("body2")
        if first not in body_id or second not in body_id:
            raise ValueError(f"Source contact exclusion names unknown SMPL bodies: {first!r}, {second!r}.")
        exclusions.append(tuple(sorted((body_id[first], body_id[second]))))
    return tuple(sorted(exclusions))


def _candidate_contact_exclusions(model, *, rows: int) -> tuple[tuple[int, int], ...]:
    """Read solver-owned MuJoCo exclusion signatures in source body-index space."""
    signatures = _model_rows(model.exclude_signature, rows=rows)
    return tuple(sorted((int(signature) // 65_536, int(signature) % 65_536) for signature in signatures))


def _geometry_orientation_metrics(
    geom_type: np.ndarray,
    actual_quaternion: np.ndarray,
    expected_quaternion: np.ndarray,
    actual_size: np.ndarray,
    expected_size: np.ndarray,
) -> dict[str, dict[str, object]]:
    """Compare shape orientations modulo each primitive's exact symmetries."""
    unsupported = set(np.unique(geom_type).tolist()).difference({0, 3, 6})
    if unsupported:
        raise ValueError(f"SMPL geometry comparison does not define primitive types {sorted(unsupported)}.")
    actual_rotation = _quaternion_matrix_wxyz(actual_quaternion)
    expected_rotation = _quaternion_matrix_wxyz(expected_quaternion)
    metrics: dict[str, dict[str, object]] = {}
    plane = geom_type == 0
    if plane.any():
        metrics["plane_normal"] = _within(actual_rotation[plane, :, 2], expected_rotation[plane, :, 2], atol=2.0e-6)
    capsule = geom_type == 3
    if capsule.any():
        actual_axis = actual_rotation[capsule, :, 2]
        expected_axis = expected_rotation[capsule, :, 2]
        metrics["capsule_axis_projector"] = _within(
            actual_axis[..., :, None] * actual_axis[..., None, :],
            expected_axis[..., :, None] * expected_axis[..., None, :],
            atol=2.0e-6,
        )
    box = geom_type == 6
    if box.any():
        metrics["box_extent_tensor"] = _within(
            _inertia_tensor(np.square(actual_size[box]), actual_quaternion[box]),
            _inertia_tensor(np.square(expected_size[box]), expected_quaternion[box]),
            atol=2.0e-6,
        )
    return metrics


def _observation_metrics(actual: np.ndarray, expected: np.ndarray, *, atol: float) -> dict[str, object]:
    fields = {
        name: _within(actual[:, field], expected[:, field], atol=atol) for name, field in _OBSERVATION_SLICES.items()
    }
    return {"fields": fields, "passed": all(bool(value["passed"]) for value in fields.values())}


def _flatten_applied(tensors: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    applied = tensors["action_applied"].reshape(-1)
    flat = {
        name: value.reshape(-1, *value.shape[2:])[applied]
        for name, value in tensors.items()
        if value.ndim >= 2 and value.shape[:2] == tensors["action_applied"].shape
    }
    return flat, applied


def _root_xyzw(qpos: torch.Tensor) -> torch.Tensor:
    return torch.cat((qpos[:, 4:7], qpos[:, 3:4]), dim=-1)


def _candidate_state(robot, joint_ids: list[int], source_indices: torch.Tensor, origins: torch.Tensor):
    root_xyzw = robot.data.root_link_quat_w.torch
    root_wxyz = torch.cat((root_xyzw[:, 3:4], root_xyzw[:, :3]), dim=-1)
    root_position = robot.data.root_link_pos_w.torch - origins
    angular_body = quat_apply_inverse(root_xyzw, robot.data.root_link_ang_vel_w.torch)
    joint_position_sim = robot.data.joint_pos.torch[:, joint_ids]
    joint_velocity_sim = robot.data.joint_vel.torch[:, joint_ids]
    joint_position = torch.empty_like(joint_position_sim)
    joint_velocity = torch.empty_like(joint_velocity_sim)
    joint_position[:, source_indices] = joint_position_sim
    joint_velocity[:, source_indices] = joint_velocity_sim
    qpos = torch.cat((root_position, root_wxyz, joint_position), dim=-1)
    qvel = torch.cat((robot.data.root_link_lin_vel_w.torch, angular_body, joint_velocity), dim=-1)
    return qpos, qvel


def _candidate_body(robot, body_ids: list[int], origins: torch.Tensor):
    position = robot.data.body_link_pos_w.torch[:, body_ids] - origins[:, None]
    rotation_xyzw = robot.data.body_link_quat_w.torch[:, body_ids]
    rotation_wxyz = torch.cat((rotation_xyzw[..., 3:4], rotation_xyzw[..., :3]), dim=-1)
    return (
        position,
        rotation_wxyz,
        robot.data.body_link_lin_vel_w.torch[:, body_ids],
        robot.data.body_link_ang_vel_w.torch[:, body_ids],
    )


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


def _solver_tensor(data, name: str, rows: int) -> torch.Tensor:
    value = wp.to_torch(getattr(data, name))
    if value.shape[0] != rows:
        raise ValueError(f"MuJoCo-Warp {name} has shape {tuple(value.shape)}, expected first dimension {rows}.")
    return value


def _source_to_sim_generalized(value: torch.Tensor, source_indices: torch.Tensor) -> torch.Tensor:
    """Map source qvel-order generalized rows to simulator qvel order."""
    return torch.cat((value[:, :6], value[:, 6:].index_select(1, source_indices)), dim=-1)


def _sim_to_source_generalized(value: torch.Tensor, source_indices: torch.Tensor) -> torch.Tensor:
    """Map simulator qvel-order generalized rows back to source order."""
    result = torch.empty_like(value)
    result[:, :6] = value[:, :6]
    result[:, 6:][:, source_indices] = value[:, 6:]
    return result


def _control_tensor(action: NativeMujocoControlAction, rows: int) -> torch.Tensor:
    """Return the native control destination as environment-major rows."""
    return wp.to_torch(action._control_destination).view(rows, action.action_dim)


def _model_array(value, *, rows: int, source_shape: tuple[int, ...]) -> np.ndarray:
    """Return one candidate-world model field with its source shape."""
    if isinstance(value, (bool, int, float, np.generic)):
        array = np.asarray(value)
    else:
        array = _numpy(wp.to_torch(value))
    if source_shape == () and array.shape == (1,):
        return array[0]
    repeated_shape = (rows, *source_shape)
    if array.shape == repeated_shape:
        return array[0]
    if array.shape != source_shape:
        raise ValueError(f"Candidate model field has shape {array.shape}; expected {source_shape} or {repeated_shape}.")
    return array


def _model_rows(value, *, rows: int) -> np.ndarray:
    """Return one candidate-world model array without assuming source cardinality."""
    if isinstance(value, (bool, int, float, np.generic)):
        array = np.asarray(value)
    else:
        array = _numpy(wp.to_torch(value))
    if array.ndim > 1 and array.shape[0] == rows:
        return array[0]
    return array


def _unique_model_values(value) -> list[object]:
    """Serialize unique scalar or row values from one Newton model array."""
    array = _numpy(wp.to_torch(value))
    unique = np.unique(array, axis=0) if array.ndim > 1 else np.unique(array)
    return unique.tolist()


def _unique_model_attribute(namespace, name: str) -> list[object] | None:
    """Serialize one optional Newton custom attribute when it survived finalization."""
    value = getattr(namespace, name, None)
    return None if value is None else _unique_model_values(value)


def _geometry_row(
    fields: dict[str, np.ndarray],
    body_ids: np.ndarray,
    index: int,
) -> dict[str, object]:
    """Serialize one geometry row for source/candidate provenance."""
    row: dict[str, object] = {"index": int(index), "body_id": int(body_ids[index])}
    for name, values in fields.items():
        value = values[index]
        row[name] = value.item() if np.asarray(value).ndim == 0 else value.tolist()
    return row


def _reorder_candidate_joints(value: np.ndarray, source_indices: np.ndarray, *, root_width: int) -> np.ndarray:
    """Return one joint/qpos/qvel model field in source order."""
    candidate_for_source = np.argsort(source_indices)
    order = np.concatenate((np.arange(root_width), root_width + candidate_for_source))
    return value[order]


def _fixed_model_metrics(
    tensors: dict[str, np.ndarray],
    model,
    *,
    rows: int,
    body_ids: list[int],
    source_indices: torch.Tensor,
    first_env_spawn_translation: np.ndarray,
    expected_option_disableflags: int,
) -> dict[str, object]:
    """Compare physical model facts after semantic and representation normalization."""
    source = {name.removeprefix("model_"): value for name, value in tensors.items() if name.startswith("model_")}
    source_joint_indices = _numpy(source_indices)
    body_order = np.asarray((0, *(body_id + 1 for body_id in body_ids)), dtype=np.int64)
    body_to_source = np.full(int(body_order.max()) + 1, -1, dtype=np.int64)
    body_to_source[body_order] = np.arange(body_order.size)

    fields: dict[str, dict[str, object]] = {}
    representation_provenance: dict[str, dict[str, object]] = {}

    def compare(
        name: str,
        actual: np.ndarray,
        *,
        expected: np.ndarray | None = None,
        atol: float = 2.0e-6,
    ) -> None:
        fields[name] = _within(actual, source[name] if expected is None else expected, atol=atol)

    dynamic_body_order = body_order[1:]
    body_quat = _model_array(model.body_quat, rows=rows, source_shape=source["body_quat"].shape)
    body_position = _model_array(model.body_pos, rows=rows, source_shape=source["body_pos"].shape)[
        dynamic_body_order
    ].copy()
    body_position[0] -= first_env_spawn_translation
    body_position[0] = _quaternion_matrix_wxyz(body_quat[dynamic_body_order[0]]).T @ body_position[0]
    compare("body_pos", body_position, expected=source["body_pos"][1:])
    for name in ("body_mass", "body_ipos"):
        actual = _model_array(getattr(model, name), rows=rows, source_shape=source[name].shape)
        compare(name, actual[dynamic_body_order], expected=source[name][1:])

    body_inertia = _model_array(model.body_inertia, rows=rows, source_shape=source["body_inertia"].shape)
    body_iquat = _model_array(model.body_iquat, rows=rows, source_shape=source["body_iquat"].shape)
    fields["body_inertia_tensor"] = _within(
        _inertia_tensor(body_inertia[dynamic_body_order], body_iquat[dynamic_body_order]),
        _inertia_tensor(source["body_inertia"][1:], source["body_iquat"][1:]),
        atol=2.0e-6,
    )
    representation_provenance["body_inertia_principal"] = _within(
        body_inertia[dynamic_body_order], source["body_inertia"][1:], atol=2.0e-6
    )
    representation_provenance["body_inertia_orientation"] = _rotation_within(
        body_iquat[dynamic_body_order], source["body_iquat"][1:], atol=2.0e-6
    )
    representation_provenance["body_frame_orientation"] = _rotation_within(
        body_quat[dynamic_body_order], source["body_quat"][1:], atol=2.0e-6
    )
    body_parent = _model_array(model.body_parentid, rows=rows, source_shape=source["body_parentid"].shape)[
        dynamic_body_order
    ]
    compare("body_parentid", body_to_source[body_parent], expected=source["body_parentid"][1:], atol=0.0)

    joint_fields = {
        "jnt_type": 0.0,
        "jnt_pos": 2.0e-6,
        "jnt_axis": 2.0e-6,
        "jnt_stiffness": 2.0e-5,
    }
    for name, atol in joint_fields.items():
        actual = _model_array(getattr(model, name), rows=rows, source_shape=source[name].shape)
        compare(name, _reorder_candidate_joints(actual, source_joint_indices, root_width=1), atol=atol)
    joint_body = _model_array(model.jnt_bodyid, rows=rows, source_shape=source["jnt_bodyid"].shape)
    joint_body = _reorder_candidate_joints(joint_body, source_joint_indices, root_width=1)
    compare("jnt_bodyid", body_to_source[joint_body], atol=0.0)

    for name, root_width, atol in (
        ("dof_armature", 6, 2.0e-7),
        ("dof_damping", 6, 2.0e-5),
        ("qpos_spring", 7, 2.0e-6),
    ):
        actual = _model_array(getattr(model, name), rows=rows, source_shape=source[name].shape)
        actual = _reorder_candidate_joints(actual, source_joint_indices, root_width=root_width)
        if name == "qpos_spring":
            representation_provenance["qpos_spring_free_root"] = _within(actual[:7], source[name][:7], atol=atol)
            fields["qpos_spring_actuated"] = _within(actual[7:], source[name][7:], atol=atol)
            continue
        compare(name, actual, atol=atol)

    for name in (
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_gear",
        "actuator_ctrlrange",
        "actuator_forcerange",
    ):
        actual = _model_array(getattr(model, name), rows=rows, source_shape=source[name].shape)
        compare(name, actual, atol=2.0e-5)
    actuator_target = _model_array(model.actuator_trnid, rows=rows, source_shape=source["actuator_trnid"].shape).copy()
    target = actuator_target[:, 0]
    target[target >= 1] = 1 + source_joint_indices[target[target >= 1] - 1]
    compare("actuator_trnid", actuator_target, atol=0.0)

    geom_fields = (
        "geom_type",
        "geom_pos",
        "geom_quat",
        "geom_size",
        "geom_contype",
        "geom_conaffinity",
        "geom_condim",
        "geom_friction",
        "geom_margin",
        "geom_gap",
        "geom_priority",
        "geom_solmix",
        "geom_solimp",
        "geom_solref",
    )
    candidate_geom = {name: _model_rows(getattr(model, name), rows=rows) for name in geom_fields}
    candidate_geom_body = _model_rows(model.geom_bodyid, rows=rows)
    normalized_geom_body = body_to_source[candidate_geom_body]
    source_active = (source["geom_contype"] != 0) | (source["geom_conaffinity"] != 0)
    candidate_active = (candidate_geom["geom_contype"] != 0) | (candidate_geom["geom_conaffinity"] != 0)
    source_active_indices = np.flatnonzero(source_active)
    candidate_active_indices = np.flatnonzero(candidate_active)
    unused = set(candidate_active_indices.tolist())
    geom_order: list[int] = []
    missing: list[int] = []
    for source_index in source_active_indices:
        source_body = source["geom_bodyid"][source_index]
        candidates = [index for index in unused if normalized_geom_body[index] == source_body]
        if not candidates:
            missing.append(int(source_index))
            continue
        chosen = min(
            candidates,
            key=lambda index: (
                int(candidate_geom["geom_type"][index] != source["geom_type"][source_index]),
                float(np.linalg.vector_norm(candidate_geom["geom_size"][index] - source["geom_size"][source_index])),
                float(np.linalg.vector_norm(candidate_geom["geom_pos"][index] - source["geom_pos"][source_index])),
            ),
        )
        geom_order.append(chosen)
        unused.remove(chosen)
    geom_order = np.asarray(geom_order, dtype=np.int64)
    geometry_fields: dict[str, dict[str, object]] = {}
    geometry_provenance: dict[str, dict[str, object]] = {}
    if not missing:
        geometry_fields["geom_bodyid"] = _within(
            normalized_geom_body[geom_order], source["geom_bodyid"][source_active_indices], atol=0.0
        )
        for name in (
            "geom_type",
            "geom_pos",
            "geom_condim",
            "geom_friction",
            "geom_margin",
            "geom_gap",
            "geom_priority",
            "geom_solmix",
            "geom_solimp",
            "geom_solref",
        ):
            actual = candidate_geom[name][geom_order]
            expected = source[name][source_active_indices]
            geometry_fields[name] = _within(
                actual,
                expected,
                atol=0.0 if np.issubdtype(source[name].dtype, np.integer) else 2.0e-6,
            )
        expected_type = source["geom_type"][source_active_indices]
        nonplane = expected_type != 0
        geometry_fields["geom_size_nonplane"] = _within(
            candidate_geom["geom_size"][geom_order][nonplane],
            source["geom_size"][source_active_indices][nonplane],
            atol=2.0e-6,
        )
        orientation = _geometry_orientation_metrics(
            expected_type,
            candidate_geom["geom_quat"][geom_order],
            source["geom_quat"][source_active_indices],
            candidate_geom["geom_size"][geom_order],
            source["geom_size"][source_active_indices],
        )
        geometry_fields.update({f"geom_orientation_{name}": value for name, value in orientation.items()})
        source_exclusions = _source_contact_exclusions()
        candidate_exclusions = _candidate_contact_exclusions(model, rows=rows)
        source_parent = source["body_parentid"]
        parent_exclusions = tuple(
            (int(parent), int(child)) for child, parent in enumerate(source_parent) if child > 0 and parent > 0
        )
        expected_effective_exclusions = tuple(sorted({*source_exclusions, *parent_exclusions}))
        geometry_fields["geom_contact_exclusions"] = {
            "source_explicit_body_pairs": [list(pair) for pair in source_exclusions],
            "source_implicit_parent_body_pairs": [list(pair) for pair in parent_exclusions],
            "candidate_effective_body_pairs": [list(pair) for pair in candidate_exclusions],
            "passed": candidate_exclusions == expected_effective_exclusions,
        }
        source_active_body = source["geom_bodyid"][source_active_indices]
        actual_collision_graph = _collision_graph(
            candidate_geom["geom_contype"][geom_order],
            candidate_geom["geom_conaffinity"][geom_order],
            body_ids=source_active_body,
            excluded_body_pairs=(*parent_exclusions, *candidate_exclusions),
        )
        expected_collision_graph = _collision_graph(
            source["geom_contype"][source_active_indices],
            source["geom_conaffinity"][source_active_indices],
            body_ids=source_active_body,
            excluded_body_pairs=(*parent_exclusions, *source_exclusions),
        )
        collision_graph = _within(actual_collision_graph, expected_collision_graph, atol=0.0)
        collision_graph["source_parent_body_pairs"] = [list(pair) for pair in parent_exclusions]
        collision_graph["source_excluded_body_pairs"] = [list(pair) for pair in source_exclusions]
        collision_graph["candidate_excluded_body_pairs"] = [list(pair) for pair in candidate_exclusions]
        collision_graph["source_permitted_candidate_blocked"] = [
            {
                "source_geom_indices": [int(source_active_indices[first]), int(source_active_indices[second])],
                "source_body_ids": [int(source_active_body[first]), int(source_active_body[second])],
            }
            for first, second in np.argwhere(expected_collision_graph & ~actual_collision_graph)
            if first < second
        ]
        geometry_fields["geom_collision_graph"] = collision_graph
        geometry_provenance["geom_quaternion_encoding"] = _rotation_within(
            candidate_geom["geom_quat"][geom_order], source["geom_quat"][source_active_indices], atol=2.0e-6
        )
        geometry_provenance["geom_plane_visual_extent"] = _within(
            candidate_geom["geom_size"][geom_order][~nonplane],
            source["geom_size"][source_active_indices][~nonplane],
            atol=2.0e-6,
        )
        for name in ("geom_contype", "geom_conaffinity"):
            geometry_provenance[f"{name}_encoding"] = _within(
                candidate_geom[name][geom_order], source[name][source_active_indices], atol=0.0
            )
    geometry = {
        "contract": "Only collision-active geometry belongs to the physical model; inactive decoration is provenance.",
        "active": {
            "source_count": int(source_active.sum()),
            "candidate_count": int(candidate_active.sum()),
            "source_indices": source_active_indices.tolist(),
            "candidate_indices": candidate_active_indices.tolist(),
            "matched_candidate_indices": geom_order.tolist(),
            "missing_source_indices": missing,
            "extra_candidate_indices": sorted(unused),
            "fields": geometry_fields,
            "representation_provenance": geometry_provenance,
            "passed": (
                not missing
                and not unused
                and int(source_active.sum()) == int(candidate_active.sum())
                and all(bool(value["passed"]) for value in geometry_fields.values())
            ),
        },
        "inactive_provenance": {
            "source": [
                _geometry_row(
                    {name: source[name] for name in geom_fields},
                    source["geom_bodyid"],
                    int(index),
                )
                for index in np.flatnonzero(~source_active)
            ],
            "candidate": [
                _geometry_row(candidate_geom, normalized_geom_body, int(index))
                for index in np.flatnonzero(~candidate_active)
            ],
            "candidate_omission_allowed": True,
        },
    }

    option = model.opt
    direct_options = (
        "solver",
        "integrator",
        "cone",
        "iterations",
        "ls_iterations",
        "sdf_iterations",
        "tolerance",
        "ls_tolerance",
        "sdf_initpoints",
        "disableflags",
        "enableflags",
        "gravity",
        "timestep",
    )
    for name in direct_options:
        source_name = f"option_{name}"
        actual = _model_array(getattr(option, name), rows=rows, source_shape=source[source_name].shape)
        if name == "gravity":
            quantized_source = np.asarray(source[source_name], dtype=np.float32).astype(np.float64)
            compare(source_name, actual, expected=quantized_source, atol=0.0)
            fields[source_name]["comparison_contract"] = "exact source-to-float32 representation"
            continue
        if name == "tolerance":
            representation_provenance["option_tolerance_source_exact"] = _within(
                actual, source[source_name], atol=2.0e-7
            )
            compare(source_name, actual, expected=np.maximum(source[source_name], 1.0e-6), atol=2.0e-7)
            fields[source_name]["comparison_contract"] = "MuJoCo-Warp IO floor of 1e-6"
            continue
        if name == "disableflags":
            expected = np.full_like(source[source_name], expected_option_disableflags)
            compare(source_name, actual, expected=expected, atol=0.0)
            fields[source_name].update(
                {
                    "source_value": int(source[source_name]),
                    "expected_value": expected_option_disableflags,
                    "bridge_added_bits": expected_option_disableflags & ~int(source[source_name]),
                    "comparison_contract": ("source flags plus explicitly configured MJWarp CCD disable bits"),
                }
            )
            continue
        compare(
            source_name,
            actual,
            atol=0.0 if np.issubdtype(source[source_name].dtype, np.integer) else 2.0e-7,
        )
    compare("option_is_sparse", np.asarray(model.is_sparse), atol=0.0)
    impratio = _model_array(option.impratio_invsqrt, rows=rows, source_shape=source["option_impratio"].shape)
    compare("option_impratio", np.reciprocal(np.square(impratio)), atol=2.0e-6)
    fields["option_jacobian"] = {
        "source_enum": int(source["option_jacobian"]),
        "resolved_by": "option_is_sparse",
        "passed": int(source["option_jacobian"]) == 2,
    }
    fields["option_noslip"] = {
        "source_iterations": int(source["option_noslip_iterations"]),
        "source_tolerance": float(source["option_noslip_tolerance"]),
        "candidate_support": "absent",
        "passed": int(source["option_noslip_iterations"]) == 0,
    }
    return {
        "fields": fields,
        "representation_provenance": representation_provenance,
        "geometry": geometry,
        "passed": all(bool(value["passed"]) for value in fields.values()) and bool(geometry["active"]["passed"]),
    }


def _source_contract(tensors: dict[str, np.ndarray], metadata: dict[str, object]) -> dict[str, object]:
    flat, applied_flat = _flatten_applied(tensors)
    done = flat["terminated"] | flat["truncated"]
    source_control = _within(flat["returned_control"], flat["actions"], atol=0.0)
    reward = _within(flat["reward"], np.zeros_like(flat["reward"]), atol=0.0)
    qfrc = _within(flat["current_qfrc_applied"], np.zeros_like(flat["current_qfrc_applied"]), atol=0.0)
    xfrc = _within(flat["current_xfrc_applied"], np.zeros_like(flat["current_xfrc_applied"]), atol=0.0)
    return {
        "applied_edges": int(applied_flat.sum()),
        "reset_only_rows_excluded": int((~applied_flat).sum()),
        "done_applied_edges": int(done.sum()),
        "action_equals_applied_control": source_control,
        "zero_reward": reward,
        "zero_qfrc_applied": qfrc,
        "zero_xfrc_applied": xfrc,
        "next_step_final_lifecycle": {
            "source_reached_observation_location": "step_return",
            "source_reset_only_row_follows_done": True,
            "unified_reached_observation_location": "extras.final_obs_on_done_else_step_return",
            "passed": bool(
                metadata["mdp"]["autoreset_mode"] == "next_step"
                and np.all(tensors["action_applied"][3] == 0)
                and np.all(tensors["truncated"][2])
            ),
        },
    }


def _run_candidate(
    tensors: dict[str, np.ndarray], metadata: dict[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    source, _ = _flatten_applied(tensors)
    num_edges = source["actions"].shape[0]
    cfg = resolve_presets(MotionImitationEnvCfg(), selected=motion_environment_axes("smpl_cmu"))
    table_cfg = cfg.commands.motion.task_table
    table_cfg.source_artifact_root = str(args.source_artifact_root.expanduser().resolve())
    table_cfg.motion_split = "evaluation"
    cfg.scene.num_envs = num_edges
    cfg.seed = 0
    cfg.sim.device = args.device
    dependency_identity = motion_environment_dependency_identity(
        preset="smpl_cmu",
        cfg=cfg,
        importer_type=CmuHumEnvSmplClips,
        frame_builder_type=SmplGeneralizedCoordinateFrameBuilder,
    )
    env = ManagerBasedRLEnv(cfg=cfg)
    try:
        env.reset()
        device = torch.device(env.device)
        env_ids = torch.arange(num_edges, dtype=torch.int64, device=device)
        robot = env.scene["robot"]
        action = env.action_manager.get_term("control")
        if not isinstance(action, NativeMujocoControlAction):
            raise TypeError("SMPL edge replay requires the native MuJoCo control action.")
        table = env.command_manager.get_term("motion").table
        simulator_names = tuple(robot.joint_names)
        if table.joint_names != simulator_names:
            raise ValueError("The SMPL trajectory table and live articulation must share one joint axis.")
        motion_names = smpl_live_joint_source_names(simulator_names)
        joint_ids, observed_joint_names = robot.find_joints(list(simulator_names), preserve_order=True)
        if tuple(observed_joint_names) != simulator_names:
            raise ValueError("SMPL simulator joint order differs from its preset layout.")
        source_index = {name: index for index, name in enumerate(MUJOCO_JOINT_NAMES)}
        source_indices = torch.tensor([source_index[name] for name in motion_names], dtype=torch.int64, device=device)
        body_ids, observed_body_names = robot.find_bodies(list(MUJOCO_BODY_NAMES), preserve_order=True)
        if tuple(observed_body_names) != MUJOCO_BODY_NAMES:
            raise ValueError("SMPL physical body order differs from the native source order.")
        solver = env.sim.physics_manager._solver
        solver_model = solver.mjw_model

        source_qpos = torch.from_numpy(source["current_qpos"]).to(device=device, dtype=torch.float32)
        source_qvel = torch.from_numpy(source["current_qvel"]).to(device=device, dtype=torch.float32)
        root_xyzw = _root_xyzw(source_qpos)
        root_pose = torch.cat((source_qpos[:, :3] + env.scene.env_origins, root_xyzw), dim=-1)
        root_velocity = torch.cat((source_qvel[:, :3], quat_apply(root_xyzw, source_qvel[:, 3:6])), dim=-1)
        simulator_joint_position = source_qpos[:, 7:].index_select(1, source_indices)
        simulator_joint_velocity = source_qvel[:, 6:].index_select(1, source_indices)
        robot.write_root_link_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        robot.write_root_link_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)
        robot.write_joint_position_to_sim_index(position=simulator_joint_position, joint_ids=joint_ids, env_ids=env_ids)
        robot.write_joint_velocity_to_sim_index(velocity=simulator_joint_velocity, joint_ids=joint_ids, env_ids=env_ids)
        env.scene.write_data_to_sim()
        env.sim.forward()

        solver_data = env.sim.physics_manager._solver.mjw_data
        source_warmstart = torch.from_numpy(source["current_qacc_warmstart"]).to(device=device, dtype=torch.float32)
        _solver_tensor(solver_data, "qacc_warmstart", num_edges).copy_(
            _source_to_sim_generalized(source_warmstart, source_indices)
        )
        _solver_tensor(solver_data, "qfrc_applied", num_edges).copy_(
            torch.from_numpy(source["current_qfrc_applied"]).to(device=device, dtype=torch.float32)
        )
        _solver_tensor(solver_data, "xfrc_applied", num_edges).copy_(
            torch.from_numpy(source["current_xfrc_applied"]).to(device=device, dtype=torch.float32)
        )
        _solver_tensor(solver_data, "time", num_edges).copy_(
            torch.from_numpy(source["current_simulation_time_seconds"]).to(device=device, dtype=torch.float32)
        )
        action._actions.copy_(torch.from_numpy(source["current_control"]).to(device=device, dtype=torch.float32))
        wp.copy(action._control_destination, action._control_source)

        current_qpos, current_qvel = _candidate_state(robot, joint_ids, source_indices, env.scene.env_origins)
        current_observation = env.observation_manager.compute()["policy"].clone()
        current_control = _control_tensor(action, num_edges).clone()
        current_warmstart = _sim_to_source_generalized(
            _solver_tensor(solver_data, "qacc_warmstart", num_edges), source_indices
        ).clone()
        current_body = tuple(value.clone() for value in _candidate_body(robot, body_ids, env.scene.env_origins))

        expected_done = torch.from_numpy(source["terminated"] | source["truncated"]).to(device)
        env.episode_length_buf.zero_()
        env.episode_length_buf[expected_done] = env.max_episode_length - 1
        reached_qpos = torch.empty_like(source_qpos)
        reached_qvel = torch.empty_like(source_qvel)
        reached_observation = torch.empty(num_edges, 358, device=device)
        reached_body = tuple(torch.empty(num_edges, 24, width, device=device) for width in (3, 4, 3, 3))
        reached_control = torch.empty(num_edges, 69, device=device)
        reached_valid = torch.zeros(num_edges, dtype=torch.bool, device=device)
        original_reset = env._reset_idx

        def capture_reset(_instance, reset_ids):
            reset_ids = torch.as_tensor(reset_ids, dtype=torch.int64, device=device)
            qpos, qvel = _candidate_state(robot, joint_ids, source_indices, env.scene.env_origins)
            observation = env.observation_manager.compute()["policy"]
            reached_qpos.index_copy_(0, reset_ids, qpos.index_select(0, reset_ids))
            reached_qvel.index_copy_(0, reset_ids, qvel.index_select(0, reset_ids))
            reached_observation.index_copy_(0, reset_ids, observation.index_select(0, reset_ids))
            for destination, value in zip(
                reached_body, _candidate_body(robot, body_ids, env.scene.env_origins), strict=True
            ):
                destination.index_copy_(0, reset_ids, value.index_select(0, reset_ids))
            control = _control_tensor(action, num_edges)
            reached_control.index_copy_(0, reset_ids, control.index_select(0, reset_ids))
            reached_valid.index_fill_(0, reset_ids, True)
            return original_reset(reset_ids)

        env._reset_idx = types.MethodType(capture_reset, env)
        returned, reward, terminated, truncated, extras = env.step(
            torch.from_numpy(source["actions"]).to(device=device, dtype=torch.float32)
        )
        env._reset_idx = original_reset
        not_done = ~(terminated | truncated)
        qpos, qvel = _candidate_state(robot, joint_ids, source_indices, env.scene.env_origins)
        reached_qpos[not_done] = qpos[not_done]
        reached_qvel[not_done] = qvel[not_done]
        reached_observation[not_done] = returned["policy"][not_done]
        for destination, value in zip(
            reached_body, _candidate_body(robot, body_ids, env.scene.env_origins), strict=True
        ):
            destination[not_done] = value[not_done]
        reached_control[not_done] = _control_tensor(action, num_edges)[not_done]
        reached_valid[not_done] = True

        source_disableflags = int(tensors["model_option_disableflags"])
        expected_disableflags = source_disableflags
        if not cfg.sim.physics.solver_cfg.enable_native_ccd:
            expected_disableflags |= int(solver._mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
        if not cfg.sim.physics.solver_cfg.enable_multiccd:
            expected_disableflags |= int(solver._mujoco.mjtDisableBit.mjDSBL_MULTICCD)
        fixed_model = _fixed_model_metrics(
            tensors,
            solver_model,
            rows=num_edges,
            body_ids=body_ids,
            source_indices=source_indices,
            first_env_spawn_translation=(
                _numpy(env.scene.env_origins[0]) + np.asarray(cfg.scene.robot.init_state.pos, dtype=np.float64)
            ),
            expected_option_disableflags=expected_disableflags,
        )
        source_actuator_rows = int(tensors["model_actuator_gainprm"].shape[0])
        finalized_actuator_rows = int(
            _model_array(
                solver_model.actuator_gainprm,
                rows=num_edges,
                source_shape=tensors["model_actuator_gainprm"].shape,
            ).shape[0]
        )
        actuator_ownership = {
            "articulation_config_actuator_groups": len(cfg.scene.robot.actuators),
            "native_action_width": action.action_dim,
            "source_model_actuator_rows": source_actuator_rows,
            "finalized_model_actuator_rows_per_world": finalized_actuator_rows,
            "passed": (
                len(cfg.scene.robot.actuators) == 0
                and action.action_dim == source_actuator_rows == finalized_actuator_rows == 69
            ),
        }
        fixed_model["native_actuator_ownership"] = actuator_ownership
        fixed_model["passed"] = bool(fixed_model["passed"] and actuator_ownership["passed"])
        newton_model = env.sim.physics_manager._model
        newton_mujoco = newton_model.mujoco
        fixed_model["bridge_provenance"] = {
            "configured_tolerance": float(env.cfg.sim.physics.solver_cfg.tolerance),
            "configured_enable_multiccd": bool(env.cfg.sim.physics.solver_cfg.enable_multiccd),
            "configured_enable_native_ccd": bool(env.cfg.sim.physics.solver_cfg.enable_native_ccd),
            "source_option_disableflags": source_disableflags,
            "effective_option_disableflags": int(solver_model.opt.disableflags),
            "shape_margin_unique": _unique_model_values(newton_model.shape_margin),
            "shape_solref_unique": _unique_model_attribute(newton_mujoco, "solref"),
            "shape_solref_mode_unique": _unique_model_attribute(newton_mujoco, "solref_mode"),
            "shape_solimp_unique": _unique_model_attribute(newton_mujoco, "geom_solimp"),
        }

        final = extras.get("final_obs")
        if final is None:
            raise RuntimeError("Candidate SMPL done rows require exact pre-reset final observations.")
        final_valid = terminated | truncated
        normalized_return = returned["policy"].clone()
        normalized_return[expected_done] = final["policy"][expected_done]
        current = {
            "qpos_root_position": _within(_numpy(current_qpos[:, :3]), source["current_qpos"][:, :3], atol=2.0e-6),
            "qpos_root_rotation": _rotation_within(
                _numpy(current_qpos[:, 3:7]), source["current_qpos"][:, 3:7], atol=2.0e-6
            ),
            "qpos_joint_position": _within(_numpy(current_qpos[:, 7:]), source["current_qpos"][:, 7:], atol=2.0e-6),
            "qvel": _within(_numpy(current_qvel), source["current_qvel"], atol=2.0e-5),
            "body_position": _within(_numpy(current_body[0]), source["current_body_pos"], atol=2.0e-5),
            "body_rotation": _rotation_within(_numpy(current_body[1]), source["current_body_quat"], atol=2.0e-5),
            "body_linear_velocity": _within(_numpy(current_body[2]), source["current_body_lin_vel"], atol=2.0e-5),
            "body_angular_velocity": _within(_numpy(current_body[3]), source["current_body_ang_vel"], atol=2.0e-5),
            "observation": _observation_metrics(
                _numpy(current_observation), source["current_observation"], atol=1.0e-4
            ),
            "control": _within(_numpy(current_control), source["current_control"], atol=1.0e-7),
            "qacc_warmstart": _within(_numpy(current_warmstart), source["current_qacc_warmstart"], atol=5.0e-5),
        }
        reached = {
            "qpos_root_position": _within(_numpy(reached_qpos[:, :3]), source["returned_qpos"][:, :3], atol=2.0e-4),
            "qpos_root_rotation": _rotation_within(
                _numpy(reached_qpos[:, 3:7]), source["returned_qpos"][:, 3:7], atol=2.0e-4
            ),
            "qpos_joint_position": _within(_numpy(reached_qpos[:, 7:]), source["returned_qpos"][:, 7:], atol=2.0e-4),
            "qvel": _within(_numpy(reached_qvel), source["returned_qvel"], atol=5.0e-3),
            "body_position": _within(_numpy(reached_body[0]), source["returned_body_pos"], atol=2.0e-4),
            "body_rotation": _rotation_within(_numpy(reached_body[1]), source["returned_body_quat"], atol=2.0e-4),
            "body_linear_velocity": _within(_numpy(reached_body[2]), source["returned_body_lin_vel"], atol=5.0e-3),
            "body_angular_velocity": _within(_numpy(reached_body[3]), source["returned_body_ang_vel"], atol=5.0e-3),
            "observation": _observation_metrics(
                _numpy(reached_observation), source["returned_observation"], atol=1.0e-3
            ),
            "control": _within(_numpy(reached_control), source["returned_control"], atol=1.0e-7),
        }
        edge_coordinates = np.argwhere(tensors["action_applied"])
        reached_qpos_numpy = _numpy(reached_qpos)
        reached_qvel_numpy = _numpy(reached_qvel)
        reached_body_numpy = tuple(_numpy(value) for value in reached_body)
        reached_by_edge = {
            "coordinates": edge_coordinates.tolist(),
            "qpos_root_position": _per_edge_metrics(
                reached_qpos_numpy[:, :3], source["returned_qpos"][:, :3], atol=2.0e-4
            ),
            "qpos_root_rotation": _per_edge_rotation_metrics(
                reached_qpos_numpy[:, 3:7], source["returned_qpos"][:, 3:7], atol=2.0e-4
            ),
            "qpos_joint_position": _per_edge_metrics(
                reached_qpos_numpy[:, 7:], source["returned_qpos"][:, 7:], atol=2.0e-4
            ),
            "qvel": _per_edge_metrics(reached_qvel_numpy, source["returned_qvel"], atol=5.0e-3),
            "body_position": _per_edge_metrics(reached_body_numpy[0], source["returned_body_pos"], atol=2.0e-4),
            "body_rotation": _per_edge_rotation_metrics(
                reached_body_numpy[1], source["returned_body_quat"], atol=2.0e-4
            ),
            "body_linear_velocity": _per_edge_metrics(
                reached_body_numpy[2], source["returned_body_lin_vel"], atol=5.0e-3
            ),
            "body_angular_velocity": _per_edge_metrics(
                reached_body_numpy[3], source["returned_body_ang_vel"], atol=5.0e-3
            ),
        }
        exact = {
            # Same-Step applies every submitted action before any row is autoreset.
            "action_applied": {"passed": True},
            "done_mask": {
                "passed": bool(torch.equal(terminated | truncated, expected_done)),
                "expected_done_rows": int(expected_done.sum()),
                "observed_done_rows": int((terminated | truncated).sum()),
            },
            "final_observation_valid": {
                "passed": bool(torch.equal(final_valid, expected_done)),
                "expected_rows": int(expected_done.sum()),
                "observed_rows": int(final_valid.sum()),
            },
            "normalized_reached_observation": _within(_numpy(normalized_return), _numpy(reached_observation), atol=0.0),
            "environment_reward": _within(_numpy(reward), source["reward"], atol=0.0),
            "all_reached_rows_captured": {"passed": bool(reached_valid.all())},
        }
        current_passed = all(bool(value["passed"]) for value in current.values())
        reached_passed = all(bool(value["passed"]) for value in reached.values())
        exact_passed = all(bool(value["passed"]) for value in exact.values())
        candidate = {
            "current": current,
            "reached": reached,
            "reached_by_edge": reached_by_edge,
            "exact_lifecycle": exact,
            "fixed_model": fixed_model,
            "decision": {
                "current_passed": current_passed,
                "reached_passed": reached_passed,
                "exact_lifecycle_passed": exact_passed,
                "fixed_model_passed": fixed_model["passed"],
                "passed": (current_passed and reached_passed and exact_passed and fixed_model["passed"]),
            },
        }
        return candidate, dependency_identity
    finally:
        env.close()


def main() -> None:
    """Run the controlled replay and atomically persist its scientific contract."""
    with np.load(args.oracle, allow_pickle=False) as loaded:
        tensors = {name: loaded[name].copy() for name in loaded.files}
    metadata = json.loads(args.oracle_metadata.read_text())
    if metadata["schema"] != "forward_backward_phase3_meta_humenv_trace_v2":
        raise ValueError("SMPL edge replay requires the control-complete HumEnv v2 trace.")
    required = {
        "actions",
        "action_applied",
        "current_control",
        "current_body_ang_vel",
        "current_body_lin_vel",
        "current_body_pos",
        "current_body_quat",
        "current_observation",
        "current_qacc_warmstart",
        "current_qfrc_applied",
        "current_qpos",
        "current_qvel",
        "current_xfrc_applied",
        "returned_body_ang_vel",
        "returned_body_lin_vel",
        "returned_body_pos",
        "returned_body_quat",
        "returned_control",
        "returned_observation",
        "returned_qpos",
        "returned_qvel",
        "reward",
        "terminated",
        "truncated",
    }
    if missing := required.difference(tensors):
        raise ValueError(f"HumEnv edge oracle is missing source facts: {sorted(missing)}")
    source = _source_contract(tensors, metadata)
    candidate, dependency_identity = _run_candidate(tensors, metadata)
    source_passed = all(
        bool(source[name]["passed"])
        for name in (
            "action_equals_applied_control",
            "zero_reward",
            "zero_qfrc_applied",
            "zero_xfrc_applied",
            "next_step_final_lifecycle",
        )
    )
    report = {
        "schema": "forward_backward_phase3e_smpl_native_edge_parity_v2",
        "profile": "smpl_cmu",
        "oracle": {
            "path": str(args.oracle.resolve()),
            "sha256": _sha256(args.oracle),
            "metadata_sha256": _sha256(args.oracle_metadata),
            "source_revision": metadata["source"]["revision"],
            "source_robot_sha256": metadata["source"]["files"]["humenv/assets/robot.xml"],
        },
        "code_identity": {
            "generator_sha256": _sha256(Path(__file__).resolve()),
            "environment_dependencies": dependency_identity,
        },
        "normalization": {
            "source_autoreset_mode": "next_step",
            "candidate_autoreset_mode": "same_step_with_exact_final_obs",
            "reset_only_rows_are_not_edges": True,
        },
        "source_contract": source,
        "candidate_replay": candidate,
        "decision": {
            "source_contract_passed": source_passed,
            "current_passed": candidate["decision"]["current_passed"],
            "reached_passed": candidate["decision"]["reached_passed"],
            "exact_lifecycle_passed": candidate["decision"]["exact_lifecycle_passed"],
            "fixed_model_passed": candidate["decision"]["fixed_model_passed"],
            "passed": source_passed and candidate["decision"]["passed"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["decision"]["passed"]:
        raise RuntimeError(f"SMPL native edge parity failed: {json.dumps(report['decision'], sort_keys=True)}")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
