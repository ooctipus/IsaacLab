# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exact-MJCF SMPL trajectory construction for native HumEnv rows."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.utils.math import convert_quat, quat_apply

from ..data import MotionSkeleton
from ..data._identity import canonical_sha256
from ..mdp.commands.motion_task_table import MotionTaskTable

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from ...kinematics import NewtonKinematics

_SOURCE_FIELDS = ("motion_id", "observation", "qpos", "qvel", "terminated", "truncated")
_ROOT_STATE_POLICY = "humenv_qpos_wxyz_qvel_root_local_angular_to_world_v1"


def _sha256(path: str) -> str:
    """Hash one exact reference artifact without retaining its contents."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def smpl_live_joint_source_names(live_joint_names: tuple[str, ...]) -> tuple[str, ...]:
    """Resolve native combined three-axis joint labels to HumEnv coordinates."""
    source_names: list[str] = []
    for name in live_joint_names:
        joint_name, separator, component = name.rpartition(":")
        match = re.fullmatch(r"(.+)_x_\1_y_\1_z", joint_name)
        if not separator or match is None or component not in ("0", "1", "2"):
            raise ValueError("SMPL live joints must use native Body_x_Body_y_Body_z:0/1/2 coordinate labels.")
        source_names.append(f"{match.group(1)}_{'xyz'[int(component)]}")
    resolved = tuple(source_names)
    if len(set(resolved)) != len(resolved):
        raise ValueError("SMPL live joints do not resolve to unique source coordinates.")
    return resolved


@dataclass(frozen=True, slots=True)
class SmplHumEnvFrameBuilder:
    """Build simulator-ordered SMPL frames from native HumEnv qpos/qvel rows."""

    source_skeleton: MotionSkeleton
    reference_kinematics: NewtonKinematics
    live_joint_names: tuple[str, ...]
    live_body_names: tuple[str, ...]
    version: str = "smpl_humenv_exact_mjcf_v1"
    construction_identity_sha256: str = field(init=False)
    reference_mjcf_sha256: str = field(init=False)
    _live_from_source_indices: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Resolve live/source ordering and freeze complete construction provenance."""
        reference = self.reference_kinematics
        reference_body_names = tuple(reference.body_names)
        if reference_body_names != self.source_skeleton.body_names:
            raise ValueError("The SMPL source-body order differs from the exact reference MJCF.")
        if (
            len(self.live_body_names) != len(reference_body_names)
            or self.live_body_names[0] != reference_body_names[0]
            or set(self.live_body_names) != set(reference_body_names)
        ):
            raise ValueError("The live SMPL bodies differ from the exact reference MJCF.")
        if reference.model.joint_coord_count != 7 + self.source_skeleton.num_joints:
            raise ValueError("The exact SMPL MJCF generalized-position width differs from the source schema.")
        if reference.model.joint_dof_count != 6 + self.source_skeleton.num_joints:
            raise ValueError("The exact SMPL MJCF generalized-velocity width differs from the source schema.")

        live_source_names = smpl_live_joint_source_names(self.live_joint_names)
        if len(live_source_names) != self.source_skeleton.num_joints or set(live_source_names) != set(
            self.source_skeleton.joint_names
        ):
            raise ValueError("The live SMPL joint coordinates differ from the HumEnv source schema.")
        live_from_source = tuple(self.source_skeleton.joint_names.index(name) for name in live_source_names)
        reference_sha256 = _sha256(reference.mjcf_path)
        identity = canonical_sha256(
            {
                "math_version": self.version,
                "source_skeleton_sha256": self.source_skeleton.identity_sha256,
                "reference_mjcf_sha256": reference_sha256,
                "reference_body_names": reference_body_names,
                "live_body_names": self.live_body_names,
                "live_joint_names": self.live_joint_names,
                "live_source_joint_names": live_source_names,
                "live_from_source": live_from_source,
                "root_state_policy": _ROOT_STATE_POLICY,
                "observation_policy": "native_humenv_358_left_sample_v1",
            }
        )
        object.__setattr__(
            self,
            "_live_from_source_indices",
            torch.tensor(live_from_source, dtype=torch.int64, device=reference.device),
        )
        object.__setattr__(self, "reference_mjcf_sha256", reference_sha256)
        object.__setattr__(self, "construction_identity_sha256", identity)

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Live-articulation order of the output joint axis."""
        return self.live_joint_names

    @property
    def reference_frame_names(self) -> tuple[str, ...]:
        """Return empty because native HumEnv stores a projected observation."""
        return ()

    def allocate(self, frame_count: int, *, device: str | torch.device) -> MotionTaskTable.Frames:
        """Allocate exact-capacity SMPL trajectory columns in live simulator order."""
        joint_count = len(self.live_joint_names)
        return MotionTaskTable.Frames(
            root_position=torch.empty(frame_count, 3, dtype=torch.float32, device=device),
            root_rotation=torch.empty(frame_count, 4, dtype=torch.float32, device=device),
            root_linear_velocity=torch.empty(frame_count, 3, dtype=torch.float32, device=device),
            root_angular_velocity=torch.empty(frame_count, 3, dtype=torch.float32, device=device),
            joint_position=torch.empty(frame_count, joint_count, dtype=torch.float32, device=device),
            joint_velocity=torch.empty(frame_count, joint_count, dtype=torch.float32, device=device),
            observation=torch.empty(frame_count, 358, dtype=torch.float32, device=device),
        )

    def build_frames(
        self,
        fields: Mapping[str, object],
        *,
        device: str | torch.device,
    ) -> MotionTaskTable.Frames:
        """Build one native HumEnv clip and cast its observation into table storage."""
        if tuple(fields) != _SOURCE_FIELDS:
            raise ValueError("HumEnv trajectory fields differ from the native ordered contract.")
        arrays = (fields["qpos"], fields["qvel"], fields["observation"])
        if not all(isinstance(value, np.ndarray) for value in arrays):
            raise ValueError("HumEnv qpos, qvel, and observation must be NumPy arrays.")
        qpos_array, qvel_array, observation_array = arrays
        if qpos_array.dtype != np.float32 or qvel_array.dtype != np.float32:
            raise ValueError("Native HumEnv qpos and qvel must be float32 NumPy arrays.")
        if observation_array.dtype != np.float64:
            raise ValueError("Native HumEnv observation must be a float64 NumPy array.")
        qpos = torch.as_tensor(qpos_array, device=device)
        qvel = torch.as_tensor(qvel_array, device=device)
        observation = torch.as_tensor(observation_array, dtype=torch.float32, device=device)
        frame_count = qpos.shape[0]
        if (
            qpos.shape != (frame_count, 7 + self.source_skeleton.num_joints)
            or qvel.shape != (frame_count, 6 + self.source_skeleton.num_joints)
            or observation.shape != (frame_count, 358)
        ):
            raise ValueError("HumEnv trajectory widths differ from the declared SMPL source contract.")
        if self._live_from_source_indices.device != qpos.device:
            raise ValueError("SMPL trajectory tensors must use the reference-kinematics device.")

        root_rotation = convert_quat(qpos[:, 3:7], to="xyzw").contiguous()
        return MotionTaskTable.Frames(
            root_position=qpos[:, :3].contiguous(),
            root_rotation=root_rotation,
            root_linear_velocity=qvel[:, :3].contiguous(),
            root_angular_velocity=quat_apply(root_rotation, qvel[:, 3:6]).contiguous(),
            joint_position=qpos[:, 7:].index_select(1, self._live_from_source_indices),
            joint_velocity=qvel[:, 6:].index_select(1, self._live_from_source_indices),
            observation=observation.contiguous(),
        )


def smpl_humenv_frame_builder(env: ManagerBasedRLEnv) -> SmplHumEnvFrameBuilder:
    """Build native SMPL trajectory policy from the live articulation and exact MJCF."""
    from ...kinematics import NewtonKinematics

    table_cfg = env.cfg.commands.motion.task_table
    source_skeleton = table_cfg.source.build_skeleton()
    reference = table_cfg.reference_kinematics_factory(env)
    if not isinstance(reference, NewtonKinematics) or not reference.mjcf_path:
        raise TypeError("SMPL frame construction requires an MJCF-backed NewtonKinematics reference.")
    robot = env.scene["robot"]
    return SmplHumEnvFrameBuilder(
        source_skeleton=source_skeleton,
        reference_kinematics=reference,
        live_joint_names=tuple(robot.joint_names),
        live_body_names=tuple(robot.body_names),
    )


__all__ = [
    "SmplHumEnvFrameBuilder",
    "smpl_humenv_frame_builder",
    "smpl_live_joint_source_names",
]
