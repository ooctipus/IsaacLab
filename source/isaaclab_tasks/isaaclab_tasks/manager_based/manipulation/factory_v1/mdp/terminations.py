# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import warp as wp
from typing import TYPE_CHECKING

from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers import TerminationTermCfg as DoneTermCfg
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject, Articulation
    from isaaclab.envs import ManagerBasedRLEnv

from ..assembly_keypoints import Offset
from ..utils import AssemblyProfile, AssemblyProfileCfg


def out_of_bound(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    in_bound_range: dict[str, tuple[float, float]] = {},
) -> torch.Tensor:
    """Termination condition for the object falls out of bound.

    Args:
        env: The environment.
        asset_cfg: The object configuration. Defaults to SceneEntityCfg("object").
        in_bound_range: The range in x, y, z such that the object is considered in range
    """
    object: RigidObject = env.scene[asset_cfg.name]
    range_list = [in_bound_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=env.device)

    object_pos_local = wp.to_torch(object.data.root_pos_w) - env.scene.env_origins
    outside_bounds = ((object_pos_local < ranges[:, 0]) | (object_pos_local > ranges[:, 1])).any(dim=1)
    return outside_bounds


def abnormal_robot_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return (wp.to_torch(robot.data.joint_vel).abs() > (wp.to_torch(robot.data.joint_vel_limits) * 2)).any(dim=1)


class progress_context(ManagerTermBase):
    def __init__(self, cfg: DoneTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.held_asset: Articulation | RigidObject = env.scene[cfg.params.get("held_asset_cfg").name]  # type: ignore
        self.fixed_asset: Articulation | RigidObject = env.scene[cfg.params.get("fixed_asset_cfg").name]  # type: ignore
        self.held_asset_offset: Offset = cfg.params.get("held_asset_offset")  # type: ignore
        profile_cfg: AssemblyProfileCfg = cfg.params.get("assembly_profile")  # type: ignore
        self.profile: AssemblyProfile = profile_cfg.class_type(profile_cfg)
        self.success_threshold: float = cfg.params.get("success_threshold")  # type: ignore

        self.orientation_aligned = torch.zeros((env.num_envs), dtype=torch.bool, device=env.device)
        self.position_centered = torch.zeros((env.num_envs), dtype=torch.bool, device=env.device)
        self.z_distance_reached = torch.zeros((env.num_envs), dtype=torch.bool, device=env.device)
        self.is_success = torch.zeros((env.num_envs), dtype=torch.bool, device=env.device)
        self.euler_xy_diff = torch.zeros((env.num_envs), device=env.device)
        self.xy_distance = torch.zeros((env.num_envs), device=env.device)
        self.z_distance = torch.zeros((env.num_envs), device=env.device)
        self.dummy_false_tensor = torch.zeros((env.num_envs), dtype=torch.bool, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        success_threshold: float,
        held_asset_cfg: SceneEntityCfg,
        fixed_asset_cfg: SceneEntityCfg,
        held_asset_offset: Offset,
        assembly_profile: AssemblyProfileCfg,
    ) -> torch.Tensor:
        held_asset_alignment_pos_w, held_asset_alignment_quat_w = self.held_asset_offset.apply(self.held_asset)
        fixed_asset_alignment_pos_w, fixed_asset_alignment_quat_w = self.profile.assembled_offset.apply(self.fixed_asset)
        held_asset_in_fixed_asset_frame_pos, held_asset_in_fixed_asset_frame_quat = (
            math_utils.subtract_frame_transforms(
                fixed_asset_alignment_pos_w,
                fixed_asset_alignment_quat_w,
                held_asset_alignment_pos_w,
                held_asset_alignment_quat_w,
            )
        )

        e_x, e_y, _ = math_utils.euler_xyz_from_quat(held_asset_in_fixed_asset_frame_quat)
        self.euler_xy_diff[:] = math_utils.wrap_to_pi(e_x).abs() + math_utils.wrap_to_pi(e_y).abs()
        self.xy_distance[:] = torch.norm(held_asset_in_fixed_asset_frame_pos[:, 0:2], dim=1)
        self.z_distance[:] = held_asset_in_fixed_asset_frame_pos[:, 2]

        self.orientation_aligned[:] = self.euler_xy_diff < 0.025
        self.position_centered[:] = self.xy_distance < 0.0025
        self.z_distance_reached[:] = self.z_distance < self.success_threshold
        self.is_success[:] = self.orientation_aligned & self.position_centered & self.z_distance_reached
        env.extras["successes"] = self.is_success

        return self.dummy_false_tensor


def success_termination(env: ManagerBasedRLEnv, context: str = "progress_context") -> torch.Tensor:
    return env.termination_manager.get_term_cfg(context).func.is_success


class _PredictorTruncationBase(ManagerTermBase):
    """Shared base for predictor-driven truncation terms.

    Subclasses share a single prediction tensor via :meth:`bind`. Each subclass
    decides its own truncation condition.
    """

    _shared_predictions: torch.Tensor | None = None

    def __init__(self, cfg: DoneTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    @classmethod
    def bind(cls, predictions: torch.Tensor) -> None:
        """Bind a shared prediction tensor from the algorithm.

        Args:
            predictions: A ``(num_envs,)`` tensor the algorithm writes into each step.
        """
        cls._shared_predictions = predictions


class predictor_success_truncation(_PredictorTruncationBase):
    """Truncate envs where the success estimator predicts high success probability.

    Only applied to the first ``truncation_ratio`` fraction of envs (the truncatable
    group). The remaining envs (exam group) always run to natural completion.

    The success estimator bootstraps through these truncations via
    ``extras["predictor_truncations"]``. The success monitor should **ignore**
    these episodes (outcome unknown).
    """

    def __init__(self, cfg: DoneTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.threshold: float = cfg.params.get("threshold", 0.98)  # type: ignore
        truncation_ratio: float = cfg.params.get("truncation_ratio", 0.5)  # type: ignore
        self.n_truncatable = int(env.num_envs * truncation_ratio)

    def __call__(self, env: ManagerBasedRLEnv, threshold: float = 0.98, truncation_ratio: float = 0.5) -> torch.Tensor:
        result = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        if self._shared_predictions is not None and self.n_truncatable > 0:
            probs = torch.sigmoid(self._shared_predictions[:self.n_truncatable])
            result[:self.n_truncatable] = probs > self.threshold
        env.extras["predictor_truncations"] = env.extras.get("predictor_truncations", torch.zeros(env.num_envs, device=env.device))
        env.extras["predictor_truncations"] += result.float()
        return result


class predictor_failure_truncation(_PredictorTruncationBase):
    """Truncate envs where the success estimator predicts near-certain failure.

    Applied to **all** envs. These are logged as failures by the success monitor.
    """

    def __init__(self, cfg: DoneTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.failure_threshold: float = cfg.params.get("failure_threshold", 0.02)  # type: ignore

    def __call__(self, env: ManagerBasedRLEnv, failure_threshold: float = 0.02) -> torch.Tensor:
        result = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        if self._shared_predictions is not None:
            probs = torch.sigmoid(self._shared_predictions)
            result[:] = probs < self.failure_threshold
        env.extras["predictor_truncations"] = env.extras.get("predictor_truncations", torch.zeros(env.num_envs, device=env.device))
        env.extras["predictor_truncations"] += result.float()
        return result
