# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTermCfg
from isaaclab.managers.manager_base import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv

from ..assembly_keypoints import Offset
from ..assembly_profile import AssemblyProfile
from ..assembly_profile_cfg import AssemblyProfileCfg


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
        fixed_asset_alignment_pos_w, fixed_asset_alignment_quat_w = self.profile.assembled_offset.apply(
            self.fixed_asset
        )
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


def split_time_out(
    env: ManagerBasedRLEnv,
    short_episode_length_s: float = 2.0,
    split_ratio: float = 0.5,
    exclude_from_estimator: bool = False,
) -> torch.Tensor:
    """Timeout with a shorter episode length for the first ``split_ratio`` fraction of envs.

    The first ``split_ratio * num_envs`` envs use ``short_episode_length_s`` as their
    timeout. The remaining envs use the environment's default ``max_episode_length``.
    Short-horizon timeouts are treated as regular failures by the success estimator
    (outcome=0), not bootstrapped.

    Args:
        short_episode_length_s: Episode length [s] for the short-horizon group.
        split_ratio: Fraction of envs in the short-horizon group.
        exclude_from_estimator: If True, write ``extras["success_train_mask"]``
            to exclude short-horizon envs from success estimator training entirely.
    """
    n_short = int(env.num_envs * split_ratio)
    short_max_length = int(short_episode_length_s / env.step_dt)
    result = env.episode_length_buf >= env.max_episode_length
    result[:n_short] = env.episode_length_buf[:n_short] >= short_max_length

    if exclude_from_estimator:
        mask = torch.ones(env.num_envs, device=env.device)
        mask[:n_short] = 0.0
        env.extras["success_train_mask"] = mask

    return result


class _PredictorTruncationBase(ManagerTermBase):
    """Shared base for predictor-driven truncation terms.

    Subclasses share a single prediction tensor via :meth:`bind`. Each subclass
    decides its own truncation condition.
    """

    _shared_predictions: torch.Tensor | None = None
    _shared_loss: float = float("inf")

    def __init__(self, cfg: DoneTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    @classmethod
    def bind(cls, predictions: torch.Tensor) -> None:
        """Bind a shared prediction tensor from the algorithm.

        Args:
            predictions: A ``(num_envs,)`` tensor the algorithm writes into each step.
        """
        cls._shared_predictions = predictions

    @classmethod
    def update_loss(cls, loss: float) -> None:
        """Update the shared success estimator loss.

        Args:
            loss: The current mean success estimator loss from the latest update.
        """
        cls._shared_loss = loss


class predictor_truncation(_PredictorTruncationBase):
    """Truncate envs based on success estimator predictions.

    Only applied to the first ``truncation_ratio`` fraction of envs (the truncatable
    group). The remaining envs (exam group) always run to natural completion.

    Supports two truncation modes controlled independently:

    - **Failure truncation** (``truncate_on_failure``): truncate after
      ``consecutive_steps`` consecutive predictions below ``failure_threshold``.
    - **Success truncation** (``truncate_on_success``): truncate after
      ``consecutive_steps`` consecutive predictions above ``success_threshold``.

    Both modes require the estimator loss to drop below ``min_loss`` before activating.

    When ``exclude_from_estimator`` is True, envs in the truncatable group are masked
    out of success estimator training via ``extras["success_train_mask"]`` so the
    estimator never trains on outcomes it influenced.
    """

    def __init__(self, cfg: DoneTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.success_threshold: float = cfg.params.get("success_threshold", 0.98)  # type: ignore
        self.failure_threshold: float = cfg.params.get("failure_threshold", 0.02)  # type: ignore
        self.consecutive_steps: int = cfg.params.get("consecutive_steps", 10)  # type: ignore
        self.min_loss: float = cfg.params.get("min_loss", 0.1)  # type: ignore
        self.truncate_on_success: bool = cfg.params.get("truncate_on_success", False)  # type: ignore
        self.truncate_on_failure: bool = cfg.params.get("truncate_on_failure", True)  # type: ignore
        truncation_ratio: float = cfg.params.get("truncation_ratio", 0.5)  # type: ignore
        self.n_truncatable = int(env.num_envs * truncation_ratio)
        self.exclude_from_estimator: bool = cfg.params.get("exclude_from_estimator", False)  # type: ignore
        self._failure_consecutive_count = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
        self._success_consecutive_count = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
        self._progress_context: progress_context | None = None
        if self.exclude_from_estimator:
            self._estimator_mask = torch.ones(env.num_envs, device=env.device)
            self._estimator_mask[: self.n_truncatable] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        success_threshold: float = 0.98,
        failure_threshold: float = 0.02,
        consecutive_steps: int = 10,
        truncation_ratio: float = 0.5,
        min_loss: float = 0.1,
        truncate_on_success: bool = False,
        truncate_on_failure: bool = True,
        exclude_from_estimator: bool = False,
    ) -> torch.Tensor:
        if self._progress_context is None:
            self._progress_context = env.termination_manager.get_term_cfg("progress_context").func  # type: ignore[assignment]

        result = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        n_success_truncated = 0
        n_failure_truncated = 0
        if self._shared_predictions is not None and self.n_truncatable > 0 and self._shared_loss < self.min_loss:
            probs = torch.sigmoid(self._shared_predictions[: self.n_truncatable])

            if self.truncate_on_failure:
                below = probs < self.failure_threshold
                self._failure_consecutive_count[: self.n_truncatable] = torch.where(
                    below, self._failure_consecutive_count[: self.n_truncatable] + 1, 0
                )
                failure_fired = self._failure_consecutive_count[: self.n_truncatable] >= self.consecutive_steps
                result[: self.n_truncatable] = failure_fired
                n_failure_truncated = failure_fired.sum().item()

            if self.truncate_on_success:
                above = probs > self.success_threshold
                self._success_consecutive_count[: self.n_truncatable] = torch.where(
                    above, self._success_consecutive_count[: self.n_truncatable] + 1, 0
                )
                success_fired = self._success_consecutive_count[: self.n_truncatable] >= self.consecutive_steps
                result[: self.n_truncatable] |= success_fired
                self._progress_context.is_success[: self.n_truncatable] |= success_fired
                n_success_truncated = success_fired.sum().item()

        reset = (env.episode_length_buf == 0) | result
        self._failure_consecutive_count[reset] = 0
        self._success_consecutive_count[reset] = 0

        if self.exclude_from_estimator:
            env.extras["success_train_mask"] = self._estimator_mask

        n_total = n_success_truncated + n_failure_truncated
        ratio = n_success_truncated / n_total if n_total > 0 else 0.0
        env.extras.setdefault("log", {})["Episode_Termination/termination_ratio_success_over_failure"] = ratio

        return result
