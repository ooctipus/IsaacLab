# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compact SMPL linear-blend-skinning mechanics shared by sources and targets."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from isaaclab.utils.math import matrix_from_quat, quat_from_rotation_vector

from ..identity import file_sha256

SMPL_BODY_COUNT = 24
SMPL_PARENT_INDICES = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)
SMPL_BODY_NAMES = (
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Torso",
    "L_Knee",
    "R_Knee",
    "Spine",
    "L_Ankle",
    "R_Ankle",
    "Chest",
    "L_Toe",
    "R_Toe",
    "Neck",
    "L_Thorax",
    "R_Thorax",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hand",
    "R_Hand",
)
SMPL_LBS_FORMAT = "smpl_lbs_v1"
SMPL_COMPATIBLE_POSE_PROFILE_SHA256 = "9a8dd90a36ddfc094d783dc0872a96a04b1ce1a186ffc2b988e1dc75a508a151"
"""SMPL-H body-core pose, shape, units, and surface-mechanics profile."""

_MODEL_FIELDS = (
    "format_version",
    "gender",
    "source_sha256",
    "vertex_template_m",
    "shape_blend_directions_m",
    "pose_blend_directions_m",
    "joint_regressor",
    "skinning_weights",
    "parent_indices",
)


@dataclass(frozen=True, slots=True)
class SmplLbsModel:
    """Compact SMPL linear-blend-skinning mechanics on one Torch device.

    Attributes:
        gender: SMPL body-model gender.
        source_sha256: SHA-256 digest of the licensed source model.
        artifact_sha256: Verified SHA-256 digest of the compact mechanics archive.
        vertex_template_m: Neutral template vertices [m], shape [vertex_count, 3].
        shape_blend_directions_m: First ten shape directions [m], shape [vertex_count, 3, 10].
        pose_blend_directions_m: Pose directions [m], shape [vertex_count, 3, 207].
        joint_regressor: Vertex-to-joint regression weights, shape [24, vertex_count].
        skinning_weights: Vertex skinning weights, shape [vertex_count, 24].
        parent_indices: SMPL parent body per body, shape [24].
    """

    gender: str
    source_sha256: str
    artifact_sha256: str
    vertex_template_m: torch.Tensor
    shape_blend_directions_m: torch.Tensor
    pose_blend_directions_m: torch.Tensor
    joint_regressor: torch.Tensor
    skinning_weights: torch.Tensor
    parent_indices: torch.Tensor

    def __post_init__(self) -> None:
        """Validate the compact mechanics boundary once at load time."""
        vertex_count = self.vertex_template_m.shape[0]
        tensors = (
            self.vertex_template_m,
            self.shape_blend_directions_m,
            self.pose_blend_directions_m,
            self.joint_regressor,
            self.skinning_weights,
        )
        if self.gender not in ("female", "male", "neutral"):
            raise ValueError("Compact SMPL gender must be female, male, or neutral.")
        for name, digest in (("source", self.source_sha256), ("artifact", self.artifact_sha256)):
            if len(digest) != 64:
                raise ValueError(f"Compact SMPL {name} digest must be one SHA-256 hex string.")
            try:
                int(digest, 16)
            except ValueError as error:
                raise ValueError(f"Compact SMPL {name} digest must contain hexadecimal digits.") from error
        if (
            self.vertex_template_m.shape != (vertex_count, 3)
            or self.shape_blend_directions_m.shape != (vertex_count, 3, 10)
            or self.pose_blend_directions_m.shape != (vertex_count, 3, 207)
            or self.joint_regressor.shape != (SMPL_BODY_COUNT, vertex_count)
            or self.skinning_weights.shape != (vertex_count, SMPL_BODY_COUNT)
            or self.parent_indices.shape != (SMPL_BODY_COUNT,)
        ):
            raise ValueError("Compact SMPL mechanics have incompatible tensor shapes.")
        if vertex_count < 1 or any(value.dtype != torch.float32 for value in tensors):
            raise ValueError("Compact SMPL floating-point mechanics must use float32.")
        if any(not value.is_contiguous() or not bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("Compact SMPL floating-point mechanics must be contiguous and finite.")
        if self.parent_indices.dtype != torch.int64:
            raise ValueError("Compact SMPL parent indices must use int64.")
        if any(value.device != self.vertex_template_m.device for value in (*tensors, self.parent_indices)):
            raise ValueError("Compact SMPL mechanics must share one device.")
        expected_parents = torch.tensor(
            SMPL_PARENT_INDICES,
            dtype=torch.int64,
            device=self.parent_indices.device,
        )
        if not torch.equal(self.parent_indices, expected_parents):
            raise ValueError("Compact SMPL model must use the canonical 24-body topology.")

    @property
    def device(self) -> torch.device:
        """Device containing the mechanics tensors."""
        return self.vertex_template_m.device

    def to(self, device: str | torch.device) -> SmplLbsModel:
        """Return this immutable model on the requested Torch device."""
        target = torch.device(device)
        if target == self.device:
            return self
        return replace(
            self,
            vertex_template_m=self.vertex_template_m.to(target),
            shape_blend_directions_m=self.shape_blend_directions_m.to(target),
            pose_blend_directions_m=self.pose_blend_directions_m.to(target),
            joint_regressor=self.joint_regressor.to(target),
            skinning_weights=self.skinning_weights.to(target),
            parent_indices=self.parent_indices.to(target),
        )

    def shaped_joints(self, betas: torch.Tensor) -> torch.Tensor:
        """Return shaped SMPL joints [m], shape [batch_size, 24, 3].

        Args:
            betas: First ten SMPL shape coefficients, shape [batch_size, 10].

        Returns:
            Shaped joint positions before pose articulation [m].
        """
        return torch.einsum("jv,bvk->bjk", self.joint_regressor, self._shaped_vertices(betas))

    def vertices(
        self,
        local_axis_angle_rad: torch.Tensor,
        betas: torch.Tensor,
        root_translation_m: torch.Tensor,
    ) -> torch.Tensor:
        """Return posed SMPL vertices [m] with linear blend skinning.

        Args:
            local_axis_angle_rad: World-root then parent-local rotation vectors [rad], shape [batch_size, 24, 3].
            betas: First ten shape coefficients, shape [batch_size, 10] or [1, 10].
            root_translation_m: World root translations [m], shape [batch_size, 3].

        Returns:
            Posed world vertices [m], shape [batch_size, vertex_count, 3].
        """
        batch_size = local_axis_angle_rad.shape[0]
        if (
            local_axis_angle_rad.shape != (batch_size, SMPL_BODY_COUNT, 3)
            or root_translation_m.shape != (batch_size, 3)
            or betas.ndim != 2
            or betas.shape[1] != 10
            or betas.shape[0] not in (1, batch_size)
        ):
            raise ValueError("SMPL pose, shape, and translation batch shapes are incompatible.")
        if any(
            value.dtype != torch.float32 or value.device != self.device
            for value in (local_axis_angle_rad, betas, root_translation_m)
        ):
            raise ValueError("SMPL pose, shape, and translation must use model-owned float32 storage.")
        if betas.shape[0] == 1 and batch_size != 1:
            betas = betas.expand(batch_size, -1)
        shaped = self._shaped_vertices(betas)
        joints = torch.einsum("jv,bvk->bjk", self.joint_regressor, shaped)
        rotation = matrix_from_quat(quat_from_rotation_vector(local_axis_angle_rad))
        identity = torch.eye(3, dtype=torch.float32, device=self.device)
        pose_feature = (rotation[:, 1:] - identity).reshape(batch_size, 207)
        posed = shaped + torch.einsum("bp,vkp->bvk", pose_feature, self.pose_blend_directions_m)
        transforms = _smpl_joint_transforms(rotation, joints)
        vertex_transforms = torch.einsum("vj,bjkl->bvkl", self.skinning_weights, transforms)
        homogeneous = torch.cat((posed, torch.ones_like(posed[..., :1])), dim=-1)
        vertices = torch.matmul(vertex_transforms, homogeneous.unsqueeze(-1))[..., :3, 0]
        return vertices + root_translation_m[:, None]

    def _shaped_vertices(self, betas: torch.Tensor) -> torch.Tensor:
        """Apply the first ten SMPL shape directions [m]."""
        return self.vertex_template_m + torch.einsum("bl,vkl->bvk", betas, self.shape_blend_directions_m)


def load_smpl_lbs_model(
    path: str | Path,
    *,
    artifact_sha256: str,
    device: str | torch.device = "cpu",
) -> SmplLbsModel:
    """Load one dependency-free compact SMPL mechanics archive.

    Args:
        path: Compact smpl_lbs_v1 NPZ archive.
        artifact_sha256: Expected SHA-256 digest of the compact mechanics archive.
        device: Torch device receiving the mechanics tensors.

    Returns:
        Validated SMPL mechanics.
    """
    if len(artifact_sha256) != 64 or any(character not in "0123456789abcdef" for character in artifact_sha256):
        raise ValueError("Compact SMPL artifact_sha256 must be a lowercase SHA-256 digest.")
    model_path = Path(path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    target_device = torch.device(device)
    if target_device.type == "cuda" and target_device.index is None:
        target_device = torch.device("cuda", torch.cuda.current_device())
    return _load_smpl_lbs_model_cached(str(model_path), artifact_sha256, str(target_device))


@lru_cache(maxsize=12)
def _load_smpl_lbs_model_cached(
    model_path: str,
    artifact_sha256: str,
    device: str,
) -> SmplLbsModel:
    """Load one verified immutable compact model once per artifact path and device."""
    actual_sha256 = file_sha256(model_path)
    if actual_sha256 != artifact_sha256:
        raise ValueError(
            f"Compact SMPL artifact hash differs for {model_path}: expected {artifact_sha256}, got {actual_sha256}."
        )
    with np.load(model_path, allow_pickle=False) as archive:
        if tuple(archive.files) != _MODEL_FIELDS:
            raise ValueError(f"Compact SMPL fields differ from {_MODEL_FIELDS}: {archive.files}.")
        if str(archive["format_version"].item()) != SMPL_LBS_FORMAT:
            raise ValueError("Unsupported compact SMPL model format.")
        gender = str(archive["gender"].item())
        source_sha256 = str(archive["source_sha256"].item())
        arrays = {name: np.ascontiguousarray(archive[name]) for name in _MODEL_FIELDS[3:]}
    return SmplLbsModel(
        gender=gender,
        source_sha256=source_sha256,
        artifact_sha256=artifact_sha256,
        vertex_template_m=torch.as_tensor(arrays["vertex_template_m"], device=device),
        shape_blend_directions_m=torch.as_tensor(arrays["shape_blend_directions_m"], device=device),
        pose_blend_directions_m=torch.as_tensor(arrays["pose_blend_directions_m"], device=device),
        joint_regressor=torch.as_tensor(arrays["joint_regressor"], device=device),
        skinning_weights=torch.as_tensor(arrays["skinning_weights"], device=device),
        parent_indices=torch.as_tensor(arrays["parent_indices"], device=device),
    )


def _smpl_joint_transforms(
    local_rotation: torch.Tensor,
    shaped_joints: torch.Tensor,
) -> torch.Tensor:
    """Return SMPL rest-relative skinning transforms."""
    batch_size, joint_count = shaped_joints.shape[:2]
    relative_translation = shaped_joints.clone()
    relative_translation[:, 1:] -= shaped_joints[:, SMPL_PARENT_INDICES[1:]]
    local = torch.zeros((batch_size, joint_count, 4, 4), dtype=torch.float32, device=shaped_joints.device)
    local[:, :, :3, :3] = local_rotation
    local[:, :, :3, 3] = relative_translation
    local[:, :, 3, 3] = 1.0
    absolute = [local[:, 0]]
    for joint_index in range(1, joint_count):
        absolute.append(torch.matmul(absolute[SMPL_PARENT_INDICES[joint_index]], local[:, joint_index]))
    absolute = torch.stack(absolute, dim=1)
    rest_homogeneous = torch.cat((shaped_joints, torch.zeros_like(shaped_joints[..., :1])), dim=-1)
    correction = torch.zeros_like(absolute)
    correction[:, :, :, 3] = torch.matmul(absolute, rest_homogeneous.unsqueeze(-1)).squeeze(-1)
    return absolute - correction
