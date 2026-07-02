# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dependency-free runtime for the frozen BFM broad-reward equations."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch
from rsl_rl.modules.forward_backward import reward_context

from isaaclab.utils.math import quat_apply, quat_apply_inverse

BFM_REWARD_TASKS = (
    "move-ego-0-0",
    "move-ego-low0.5-0-0",
    "move-ego-0-0.7",
    "move-ego-0-0.3",
    "move-ego-90-0.3",
    "move-ego-180-0.3",
    "move-ego--90-0.3",
    "rotate-z-5-0.5",
    "rotate-z--5-0.5",
    "raisearms-l-l",
    "raisearms-l-m",
    "raisearms-m-l",
    "raisearms-m-m",
    "move-arms-0-0.7-m-m",
    "move-arms-90-0.7-m-m",
    "move-arms-180-0.4-m-m",
    "move-arms--90-0.7-m-m",
    "move-arms-0-0.7-l-m",
    "move-arms-90-0.7-l-m",
    "move-arms-180-0.4-l-m",
    "move-arms--90-0.7-l-m",
    "move-arms-0-0.7-m-l",
    "move-arms-90-0.7-m-l",
    "move-arms-180-0.4-m-l",
    "move-arms--90-0.7-m-l",
    "move-arms-0-0.7-l-l",
    "move-arms-90-0.7-l-l",
    "move-arms-180-0.4-l-l",
    "move-arms--90-0.7-l-l",
    "spin-arms-5-l-l",
    "spin-arms--5-l-l",
    "spin-arms-5-l-m",
    "spin-arms--5-l-m",
    "spin-arms-5-m-l",
    "spin-arms--5-m-l",
    "crouch-0",
    "crouch-0.25",
    "sitonground",
)
BFM_REWARD_TASKS_SHA256 = hashlib.sha256("\0".join(BFM_REWARD_TASKS).encode()).hexdigest()
BFM_AUXILIARY_EVIDENCE_NAMES = (
    "penalty_torques",
    "penalty_action_rate",
    "limits_dof_pos",
    "limits_torque",
    "penalty_undesired_contact",
    "penalty_feet_ori",
    "penalty_ankle_roll",
    "penalty_slippage",
)
BFM_AUXILIARY_COST_COEFFICIENTS = torch.tensor((0.0, 0.1, 10.0, 0.0, 1.0, 0.4, 4.0, 2.0))
BFM_HARD_SAFETY_NAMES = ("limits_dof_pos", "limits_torque", "penalty_undesired_contact")
BFM_REWARD_INFERENCE_DATASET_SCHEMA = "bfm_reward_inference_dataset_v2"
BFM_REWARD_OBSERVATION_WIDTHS = {"state": 64, "privileged_state": 463}
BFM_QPOS_DIM = 36
BFM_QVEL_DIM = 35
BFM_ACTION_DIM = 29
BFM_REWARD_FEATURE_NAMES = (
    "pelvis_height",
    "pelvis_yaw",
    "pelvis_up_z",
    "torso_up_x",
    "torso_up_y",
    "torso_up_z",
    "torso_angular_velocity_x",
    "torso_angular_velocity_y",
    "torso_angular_velocity_z",
    "torso_subtree_linear_velocity_x",
    "torso_subtree_linear_velocity_y",
    "torso_subtree_linear_velocity_z",
    "left_wrist_height",
    "right_wrist_height",
    "left_knee_height",
    "right_knee_height",
)
BFM_REWARD_FEATURE_WIDTH = len(BFM_REWARD_FEATURE_NAMES)

_REFERENCE_FILES = (
    "phase2_adapter/environment.py",
    "phase2_adapter/policy.py",
    "phase2_adapter/reward_evaluation.py",
    "humanoidverse/envs/g1_env_helper/bench/reward_eval_hv.py",
    "humanoidverse/envs/g1_env_helper/rewards.py",
)


def _sha256(path: Path) -> str:
    """Return the digest of one required regular frozen source file."""
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"BFM reward source must be a regular non-symbolic file: {path}.")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bfm_reward_source_identity(root: str | Path) -> dict[str, str]:
    """Return the immutable source identity used to derive the GPU equations."""
    root = Path(root).expanduser().resolve()
    return {name: _sha256(root / name) for name in _REFERENCE_FILES}


def _sigmoid(value: torch.Tensor, value_at_margin: float, kind: str) -> torch.Tensor:
    """Evaluate the ``dm_control`` soft-boundary functions used by BFM."""
    if kind in ("cosine", "linear", "quadratic"):
        if not 0.0 <= value_at_margin < 1.0:
            raise ValueError("value_at_margin must lie in [0, 1) for compact sigmoids.")
    elif not 0.0 < value_at_margin < 1.0:
        raise ValueError("value_at_margin must lie in (0, 1) for non-compact sigmoids.")

    margin_value = value.new_tensor(value_at_margin)
    if kind == "gaussian":
        scale = torch.sqrt(-2.0 * torch.log(margin_value))
        return torch.exp(-0.5 * torch.square(value * scale))
    if kind == "linear":
        scaled = value * (1.0 - value_at_margin)
        return torch.where(torch.abs(scaled) < 1.0, 1.0 - scaled, 0.0)
    if kind == "quadratic":
        scaled = value * torch.sqrt(1.0 - margin_value)
        return torch.where(torch.abs(scaled) < 1.0, 1.0 - torch.square(scaled), 0.0)
    raise ValueError(f"Unknown sigmoid kind: {kind!r}.")


def tolerance(
    value: torch.Tensor,
    bounds: tuple[float | torch.Tensor, float | torch.Tensor] = (0.0, 0.0),
    margin: float = 0.0,
    sigmoid: str = "gaussian",
    value_at_margin: float = 0.1,
) -> torch.Tensor:
    """Return the exact tensor soft interval consumed by released BFM tasks."""
    lower, upper = bounds
    if isinstance(lower, (int, float)) and isinstance(upper, (int, float)) and lower > upper:
        raise ValueError("Lower bound must not exceed upper bound.")
    if margin < 0.0:
        raise ValueError("margin must be non-negative.")
    in_bounds = (lower <= value) & (value <= upper)
    if margin == 0.0:
        return in_bounds.to(value.dtype)
    distance = torch.where(value < lower, lower - value, value - upper) / margin
    return torch.where(in_bounds, 1.0, _sigmoid(distance, value_at_margin, sigmoid))


def _orientation_reward(torso_up: torch.Tensor) -> torch.Tensor:
    target = torso_up.new_tensor((0.073, 0.0, 1.0))
    error = torch.sum(torch.square(torso_up - target), dim=-1)
    return tolerance(error, bounds=(0.0, 0.1), margin=3.0, sigmoid="linear", value_at_margin=0.0)


def _standing_reward(pelvis_height: torch.Tensor) -> torch.Tensor:
    return tolerance(
        pelvis_height,
        bounds=(0.5, float("inf")),
        margin=0.5,
        sigmoid="linear",
        value_at_margin=0.01,
    )


def _arm_reward(height: torch.Tensor, *, low: bool) -> torch.Tensor:
    bounds = (0.6, 0.8) if low else (1.0, float("inf"))
    margin = 0.2 if low else 0.1
    reward = tolerance(height, bounds=bounds, margin=margin, sigmoid="linear", value_at_margin=0.0)
    return (4.0 * reward + 1.0) / 5.0


def _still_reward(
    linear_velocity: torch.Tensor,
    angular_velocity: torch.Tensor,
    *,
    linear_margin: float,
) -> torch.Tensor:
    dont_move = tolerance(linear_velocity, margin=linear_margin).mean(dim=-1)
    dont_rotate = tolerance(angular_velocity, margin=0.1).mean(dim=-1)
    return dont_move * dont_rotate


def _moving_reward(
    linear_velocity: torch.Tensor,
    pelvis_yaw: torch.Tensor,
    angles_degrees: torch.Tensor,
    speeds: torch.Tensor,
) -> torch.Tensor:
    horizontal = linear_velocity[..., :2]
    speed = torch.linalg.vector_norm(horizontal, dim=-1)
    # ``margin`` is speed-dependent in the release. Scaling the distance before
    # the Gaussian is algebraically identical while retaining vectorized tasks.
    in_bounds = (0.9 * speeds <= speed) & (speed <= 1.1 * speeds)
    distance = torch.where(speed < 0.9 * speeds, 0.9 * speeds - speed, speed - 1.1 * speeds)
    distance = distance / (0.5 * speeds)
    move = torch.where(in_bounds, 1.0, _sigmoid(distance, 0.5, "gaussian"))
    move = (5.0 * move + 1.0) / 6.0

    angle = pelvis_yaw + torch.deg2rad(angles_degrees)
    target = torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
    direction = horizontal / (speed.unsqueeze(-1) + 1.0e-6)
    angle_reward = (torch.sum(target * direction, dim=-1) + 1.0) / 2.0
    return move * torch.where(torch.isclose(speed, torch.zeros((), device=speed.device)), 1.0, angle_reward)


def _spin_reward(
    pelvis_height: torch.Tensor,
    pelvis_up_z: torch.Tensor,
    angular_velocity_z: torch.Tensor,
    *,
    direction: float,
) -> torch.Tensor:
    height = _standing_reward(pelvis_height)
    move = tolerance(
        direction * angular_velocity_z,
        bounds=(5.0, 10.0),
        margin=2.5,
        sigmoid="linear",
        value_at_margin=0.0,
    )
    aligned = tolerance(
        pelvis_up_z,
        bounds=(0.9, float("inf")),
        margin=0.9,
        sigmoid="linear",
        value_at_margin=0.0,
    )
    return height * move * aligned


def bfm_reward(features: torch.Tensor) -> torch.Tensor:
    """Evaluate one assigned released task per row entirely on the input device.

    Args:
        features: Reward features in frozen task-major layout with shape
            ``[38, episodes, 16]``.

    Returns:
        Reward values with shape ``[38, episodes]``.
    """
    if (
        features.ndim != 3
        or features.shape[0] != len(BFM_REWARD_TASKS)
        or features.shape[2] != BFM_REWARD_FEATURE_WIDTH
    ):
        raise ValueError("BFM reward features must have shape [38, episodes, 16].")
    pelvis_height = features[..., 0]
    pelvis_yaw = features[..., 1]
    pelvis_up_z = features[..., 2]
    torso_up = features[..., 3:6]
    angular_velocity = features[..., 6:9]
    linear_velocity = features[..., 9:12]
    left_wrist_height = features[..., 12]
    right_wrist_height = features[..., 13]
    left_knee_height = features[..., 14]
    right_knee_height = features[..., 15]

    result = torch.empty(features.shape[:2], dtype=features.dtype, device=features.device)
    orientation = _orientation_reward(torso_up)
    standing = _standing_reward(pelvis_height)

    locomotion_stand = standing[:7].clone()
    locomotion_stand[1] = tolerance(
        pelvis_height[1],
        bounds=(0.475, 0.525),
        margin=0.25,
        sigmoid="linear",
        value_at_margin=0.01,
    )
    locomotion_stand.mul_(orientation[:7])
    result[:2] = locomotion_stand[:2] * _still_reward(
        linear_velocity[:2, ..., :2], angular_velocity[:2], linear_margin=0.2
    )
    locomotion_angles = features.new_tensor((0.0, 0.0, 90.0, 180.0, -90.0)).unsqueeze(-1)
    locomotion_speeds = features.new_tensor((0.7, 0.3, 0.3, 0.3, 0.3)).unsqueeze(-1)
    result[2:7] = locomotion_stand[2:7] * _moving_reward(
        linear_velocity[2:7], pelvis_yaw[2:7], locomotion_angles, locomotion_speeds
    )

    result[7] = _spin_reward(pelvis_height[7], pelvis_up_z[7], angular_velocity[7, ..., 2], direction=1.0)
    result[8] = _spin_reward(pelvis_height[8], pelvis_up_z[8], angular_velocity[8, ..., 2], direction=-1.0)

    arm_base = standing[9:13] * orientation[9:13]
    arm_base *= _still_reward(linear_velocity[9:13], angular_velocity[9:13], linear_margin=0.2)
    left_arm = torch.cat(
        (_arm_reward(left_wrist_height[9:11], low=True), _arm_reward(left_wrist_height[11:13], low=False))
    )
    right_arm = torch.stack(
        (
            _arm_reward(right_wrist_height[9], low=True),
            _arm_reward(right_wrist_height[10], low=False),
            _arm_reward(right_wrist_height[11], low=True),
            _arm_reward(right_wrist_height[12], low=False),
        )
    )
    result[9:13] = arm_base * left_arm * right_arm

    move_angles = features.new_tensor((0.0, 90.0, 180.0, -90.0) * 4).unsqueeze(-1)
    move_speeds = features.new_tensor((0.7, 0.7, 0.4, 0.7) * 4).unsqueeze(-1)
    move_base = standing[13:29] * orientation[13:29]
    move_base *= _moving_reward(linear_velocity[13:29], pelvis_yaw[13:29], move_angles, move_speeds)
    left_arm = torch.cat(
        (
            _arm_reward(left_wrist_height[13:17], low=False),
            _arm_reward(left_wrist_height[17:21], low=True),
            _arm_reward(left_wrist_height[21:25], low=False),
            _arm_reward(left_wrist_height[25:29], low=True),
        )
    )
    right_arm = torch.cat(
        (_arm_reward(right_wrist_height[13:21], low=False), _arm_reward(right_wrist_height[21:29], low=True))
    )
    result[13:29] = move_base * left_arm * right_arm

    spin_base = torch.empty_like(result[29:35])
    spin_base[0::2] = _spin_reward(
        pelvis_height[29:35:2], pelvis_up_z[29:35:2], angular_velocity[29:35:2, ..., 2], direction=1.0
    )
    spin_base[1::2] = _spin_reward(
        pelvis_height[30:35:2], pelvis_up_z[30:35:2], angular_velocity[30:35:2, ..., 2], direction=-1.0
    )
    left_arm = torch.cat(
        (_arm_reward(left_wrist_height[29:33], low=True), _arm_reward(left_wrist_height[33:35], low=False))
    )
    right_arm = torch.cat(
        (
            _arm_reward(right_wrist_height[29:31], low=True),
            _arm_reward(right_wrist_height[31:33], low=False),
            _arm_reward(right_wrist_height[33:35], low=True),
        )
    )
    result[29:35] = spin_base * left_arm * right_arm

    sit_orientation = orientation[35:38]
    sit_still = _still_reward(linear_velocity[35:38], angular_velocity[35:38], linear_margin=0.5)
    thresholds = features.new_tensor((0.0, 0.25, 0.0)).unsqueeze(-1)
    pelvis_reward = tolerance(
        pelvis_height[35:38],
        bounds=(thresholds, thresholds + 0.1),
        margin=0.7,
        sigmoid="linear",
        value_at_margin=0.0,
    )
    crouch_knees = tolerance(
        left_knee_height[35:37], bounds=(0.2, 1.0), margin=0.1, sigmoid="linear", value_at_margin=0.0
    )
    crouch_knees *= tolerance(
        right_knee_height[35:37], bounds=(0.2, 1.0), margin=0.1, sigmoid="linear", value_at_margin=0.0
    )
    ground_knees = tolerance(left_knee_height[37], bounds=(0.0, 0.1), margin=0.7, sigmoid="linear", value_at_margin=0.0)
    ground_knees *= tolerance(
        right_knee_height[37], bounds=(0.0, 0.1), margin=0.7, sigmoid="linear", value_at_margin=0.0
    )
    knee_reward = torch.cat((crouch_knees, ground_knees.unsqueeze(0)))
    result[35:38] = sit_orientation * sit_still * pelvis_reward * (2.0 * knee_reward + 1.0) / 3.0
    return result


class BfmRewardRuntime:
    """Own caller-sized GPU FK buffers for released BFM reward evaluation."""

    def __init__(self, reference_kinematics: Any, live_joint_names: Sequence[str], episodes_per_task: int):
        if episodes_per_task < 1:
            raise ValueError("episodes_per_task must be positive.")
        reference_joint_names = tuple(reference_kinematics.joint_q_names[7:])
        if tuple(live_joint_names) != reference_joint_names:
            raise ValueError("BFM reward FK requires the released semantic G1 joint order.")
        if reference_kinematics.model.joint_coord_count != BFM_QPOS_DIM:
            raise ValueError("BFM reward FK requires 36 generalized coordinates.")
        if reference_kinematics.model.joint_dof_count != BFM_QVEL_DIM:
            raise ValueError("BFM reward FK requires 35 generalized velocities.")

        self.kinematics = reference_kinematics
        self.episodes_per_task = episodes_per_task
        self.num_envs = len(BFM_REWARD_TASKS) * episodes_per_task
        self.device = torch.device(reference_kinematics.device)
        body_names = tuple(reference_kinematics.body_names)
        required_bodies = (
            "pelvis",
            "torso_link",
            "left_wrist_roll_link",
            "right_wrist_roll_link",
            "left_knee_link",
            "right_knee_link",
        )
        if any(body_names.count(name) != 1 for name in required_bodies):
            raise ValueError("BFM reward FK is missing a unique required body.")
        self._pelvis = body_names.index("pelvis")
        self._torso = body_names.index("torso_link")
        self._left_wrist = body_names.index("left_wrist_roll_link")
        self._right_wrist = body_names.index("right_wrist_roll_link")
        self._left_knee = body_names.index("left_knee_link")
        self._right_knee = body_names.index("right_knee_link")

        parents = reference_kinematics.model.joint_parent.numpy().tolist()
        children = reference_kinematics.model.joint_child.numpy().tolist()
        descendants = {self._torso}
        changed = True
        while changed:
            changed = False
            for parent, child in zip(parents, children, strict=True):
                if parent in descendants and child not in descendants:
                    descendants.add(child)
                    changed = True
        subtree = sorted(descendants)
        masses = torch.tensor(reference_kinematics.model.body_mass.numpy().tolist(), device=self.device)
        subtree_mass = masses[subtree]
        self._subtree_indices = torch.tensor(subtree, dtype=torch.long, device=self.device)
        self._subtree_weights = (subtree_mass / subtree_mass.sum()).view(1, -1, 1)
        root_com = reference_kinematics.model.body_com.numpy()[self._pelvis].tolist()
        self._root_com = torch.tensor(root_com, dtype=torch.float32, device=self.device).expand(self.num_envs, -1)
        self._x_axis = torch.zeros(self.num_envs, 3, device=self.device)
        self._x_axis[:, 0] = 1.0
        self._z_axis = torch.zeros(self.num_envs, 3, device=self.device)
        self._z_axis[:, 2] = 1.0

        body_count = reference_kinematics.model.body_count
        self._joint_q = torch.empty(self.num_envs, BFM_QPOS_DIM, device=self.device)
        self._joint_qd = torch.empty(self.num_envs, BFM_QVEL_DIM, device=self.device)
        self._body_q = torch.empty(self.num_envs, body_count, 7, device=self.device)
        self._body_qd = torch.empty(self.num_envs, body_count, 6, device=self.device)
        self._features = torch.empty(self.num_envs, BFM_REWARD_FEATURE_WIDTH, device=self.device)
        self._reward = bfm_reward
        if self.device.type == "cuda":
            self._reward = torch.compile(bfm_reward, fullgraph=True, dynamic=False)
            self._features.zero_()
            self._reward(self._features.view(len(BFM_REWARD_TASKS), self.episodes_per_task, -1))

    @property
    def features(self) -> torch.Tensor:
        """Last evaluated frozen reward features in task-major environment order."""
        return self._features

    def evaluate(self, qpos: torch.Tensor, qvel: torch.Tensor) -> torch.Tensor:
        """Evaluate assigned BFM tasks from live source-compatible state rows."""
        expected = (
            (qpos, (self.num_envs, BFM_QPOS_DIM), "qpos"),
            (qvel, (self.num_envs, BFM_QVEL_DIM), "qvel"),
        )
        for value, shape, name in expected:
            if value.shape != shape or value.dtype != torch.float32 or value.device != self.device:
                raise ValueError(f"{name} must be float32 on {self.device} with shape {shape}.")

        self._joint_q[:, :3].copy_(qpos[:, :3])
        self._joint_q[:, 3:6].copy_(qpos[:, 4:7])
        self._joint_q[:, 6].copy_(qpos[:, 3])
        self._joint_q[:, 7:].copy_(qpos[:, 7:])
        root_quaternion = self._joint_q[:, 3:7]
        root_angular_velocity = quat_apply(root_quaternion, qvel[:, 3:6])
        root_com_offset = quat_apply(root_quaternion, self._root_com)
        self._joint_qd[:, :3].copy_(qvel[:, :3])
        self._joint_qd[:, :3].add_(torch.cross(root_angular_velocity, root_com_offset, dim=-1))
        self._joint_qd[:, 3:6].copy_(root_angular_velocity)
        self._joint_qd[:, 6:].copy_(qvel[:, 6:])
        self.kinematics.eval_fk_batched_torch(self._joint_q, self._joint_qd, self._body_q, self._body_qd)

        pelvis_quaternion = self._body_q[:, self._pelvis, 3:7]
        pelvis_forward = quat_apply(pelvis_quaternion, self._x_axis)
        pelvis_up = quat_apply(pelvis_quaternion, self._z_axis)
        torso_quaternion = self._body_q[:, self._torso, 3:7]
        torso_up = quat_apply(torso_quaternion, self._z_axis)
        torso_angular_velocity = quat_apply_inverse(torso_quaternion, self._body_qd[:, self._torso, 3:])
        subtree_velocity = torch.sum(
            self._body_qd.index_select(1, self._subtree_indices)[..., :3] * self._subtree_weights,
            dim=1,
        )

        self._features[:, 0].copy_(self._body_q[:, self._pelvis, 2])
        self._features[:, 1].copy_(torch.atan2(pelvis_forward[:, 1], pelvis_forward[:, 0]))
        self._features[:, 2].copy_(pelvis_up[:, 2])
        self._features[:, 3:6].copy_(torso_up)
        self._features[:, 6:9].copy_(torso_angular_velocity)
        self._features[:, 9:12].copy_(subtree_velocity)
        self._features[:, 12].copy_(self._body_q[:, self._left_wrist, 2])
        self._features[:, 13].copy_(self._body_q[:, self._right_wrist, 2])
        self._features[:, 14].copy_(self._body_q[:, self._left_knee, 2])
        self._features[:, 15].copy_(self._body_q[:, self._right_knee, 2])
        return self._reward(self._features.view(len(BFM_REWARD_TASKS), self.episodes_per_task, -1))


def _validate_sha256(name: str, value: object) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")


def validate_reward_inference_dataset(dataset: Mapping[str, object]) -> None:
    """Validate the immutable policy-independent BFM reward labels."""
    required = {
        "schema",
        "reward_tasks",
        "reference_config_sha256",
        "data_sha256",
        "reward_model_sha256",
        "observation",
        "reward_labels",
        "motion_id",
    }
    if set(dataset) != required or dataset["schema"] != BFM_REWARD_INFERENCE_DATASET_SCHEMA:
        raise ValueError("Reward-inference dataset fields or schema differ.")
    if dataset["reward_tasks"] != BFM_REWARD_TASKS:
        raise ValueError("Reward-inference task order differs from frozen BFM.")
    for name in ("reference_config_sha256", "data_sha256", "reward_model_sha256"):
        _validate_sha256(name, dataset[name])
    observations = dataset["observation"]
    if not isinstance(observations, Mapping) or set(observations) != set(BFM_REWARD_OBSERVATION_WIDTHS):
        raise ValueError("Reward-inference observations must contain state and privileged_state.")
    labels = dataset["reward_labels"]
    motion_id = dataset["motion_id"]
    if not isinstance(labels, torch.Tensor) or labels.ndim != 2:
        raise ValueError("reward_labels must be a tensor with shape [samples, tasks].")
    sample_count = labels.shape[0]
    if labels.shape != (sample_count, len(BFM_REWARD_TASKS)) or sample_count < 1:
        raise ValueError("reward_labels have the wrong sample/task shape.")
    if labels.dtype != torch.float32 or not labels.is_contiguous():
        raise ValueError("reward_labels must be contiguous float32.")
    if not torch.isfinite(labels).all():
        raise ValueError("reward_labels must be finite.")
    if (
        not isinstance(motion_id, torch.Tensor)
        or motion_id.shape != (sample_count,)
        or motion_id.device != labels.device
        or motion_id.dtype != torch.long
        or not motion_id.is_contiguous()
        or torch.any(motion_id < 0)
    ):
        raise ValueError("motion_id must be contiguous non-negative int64 on the label device.")
    for name, width in BFM_REWARD_OBSERVATION_WIDTHS.items():
        value = observations[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (sample_count, width)
            or value.device != labels.device
            or value.dtype != torch.float32
            or not value.is_contiguous()
            or not torch.isfinite(value).all()
        ):
            raise ValueError(
                f"observation[{name!r}] must be finite contiguous float32 "
                f"[{sample_count}, {width}] on the label device."
            )


@torch.no_grad()
def infer_reward_contexts_from_dataset(
    policy: Any,
    dataset: Mapping[str, object],
    *,
    batch_size: int,
    reference_config_sha256: str,
    data_sha256: str,
    reward_model_sha256: str,
) -> torch.Tensor:
    """Infer all reward contexts from the frozen policy-independent labels."""
    validate_reward_inference_dataset(dataset)
    expected_hashes = {
        "reference_config_sha256": reference_config_sha256,
        "data_sha256": data_sha256,
        "reward_model_sha256": reward_model_sha256,
    }
    for name, expected in expected_hashes.items():
        if dataset[name] != expected:
            raise ValueError(f"Reward-inference labels have a different {name.removesuffix('_sha256')} identity.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    observations = cast(Mapping[str, torch.Tensor], dataset["observation"])
    rewards = cast(torch.Tensor, dataset["reward_labels"])
    device = torch.device(policy.device)
    backward_chunks = []
    for start in range(0, rewards.shape[0], batch_size):
        stop = min(start + batch_size, rewards.shape[0])
        batch = {name: value[start:stop].to(device) for name, value in observations.items()}
        backward_chunks.append(policy.backward_map(batch).float())
    backward = torch.cat(backward_chunks)
    task_rewards = rewards.to(device=device, dtype=backward.dtype)
    weights = torch.softmax(10.0 * task_rewards, dim=0)
    return policy.project_z(reward_context(backward, task_rewards, weights))


def reward_metric_rows(
    task_names: Sequence[str],
    task_returns: torch.Tensor,
    auxiliary_evidence_sum: torch.Tensor,
    auxiliary_evidence_active_count: torch.Tensor,
    auxiliary_cost_sum: torch.Tensor,
    safety_violation_count: torch.Tensor,
    termination_count: torch.Tensor,
    action_l2_sum: torch.Tensor,
    *,
    step_count: int,
) -> list[dict[str, object]]:
    """Transfer final reduced scalars to CPU and serialize frozen metric rows."""
    task_count, episode_count, evidence_count = auxiliary_evidence_sum.shape
    if task_count != len(task_names) or evidence_count != len(BFM_AUXILIARY_EVIDENCE_NAMES):
        raise ValueError("Auxiliary evidence does not match the task or reward schema.")
    if task_returns.shape != (task_count, episode_count):
        raise ValueError("task_returns must have shape [tasks, episodes].")
    expected_matrix = (task_count, episode_count)
    if auxiliary_evidence_active_count.shape != auxiliary_evidence_sum.shape:
        raise ValueError("Auxiliary active counts must align with evidence sums.")
    for name, value in (
        ("auxiliary_cost_sum", auxiliary_cost_sum),
        ("safety_violation_count", safety_violation_count),
        ("termination_count", termination_count),
        ("action_l2_sum", action_l2_sum),
    ):
        if value.shape != expected_matrix:
            raise ValueError(f"{name} must have shape {expected_matrix}.")
    if step_count < 1:
        raise ValueError("step_count must be positive.")

    names = ["return", "auxiliary_cost", "safety_violation_rate", "termination_rate", "action_l2"]
    values = [
        task_returns,
        auxiliary_cost_sum / step_count,
        safety_violation_count / step_count,
        termination_count / step_count,
        action_l2_sum / step_count,
    ]
    for column, name in enumerate(BFM_AUXILIARY_EVIDENCE_NAMES):
        names.extend((f"{name}_mean", f"{name}_active_fraction"))
        values.extend(
            (
                auxiliary_evidence_sum[..., column] / step_count,
                auxiliary_evidence_active_count[..., column] / step_count,
            )
        )
    serialized = torch.stack(values, dim=-1).detach().cpu()
    rows = []
    for task_index, task_name in enumerate(task_names):
        for episode in range(episode_count):
            rows.extend(
                {
                    "task": task_name,
                    "episode": episode,
                    "metric_name": name,
                    "metric_value": float(serialized[task_index, episode, column]),
                }
                for column, name in enumerate(names)
            )
    return rows


__all__ = [
    "BFM_AUXILIARY_EVIDENCE_NAMES",
    "BFM_HARD_SAFETY_NAMES",
    "BFM_REWARD_FEATURE_NAMES",
    "BFM_REWARD_TASKS",
    "BFM_REWARD_TASKS_SHA256",
    "BfmRewardRuntime",
    "bfm_reward",
    "bfm_reward_source_identity",
    "infer_reward_contexts_from_dataset",
    "reward_metric_rows",
    "tolerance",
    "validate_reward_inference_dataset",
]
