# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Domain-agnostic termination terms shared across terrain and factory tasks.

The generic watchdog here:

- :func:`out_of_bound` — env-origin-relative AABB containment check on a rigid
  asset's root position. Replaces the absolute-z ``root_height_below_minimum``
  used by terrain (which doesn't generalize to non-zero spawn heights) and
  generalizes the manipulation-side held-asset bounds check. :func:`in_bound`
  is its reset-acceptance counterpart, shaped for the acceptance-condition
  contract of :class:`~isaaclab_tasks.contrib.nistv2.utils.reset_accumulator`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTermCfg

from .assembly_variants import assembly_variant_context

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def out_of_bound(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    in_bound_range: dict[str, tuple[float, float]] = {},
) -> torch.Tensor:
    """Fire when the asset's env-relative root position leaves the AABB.

    Args:
        env: The environment.
        asset_cfg: The asset to track. Defaults to the ``"robot"`` scene entity.
        in_bound_range: Per-axis ``(min, max)`` bounds in env-local frame. Axes
            absent from the dict default to ``(0.0, 0.0)`` — i.e. nothing
            allowed — so callers should specify every axis they care about.

    Note: env-origin-relative, not absolute-world. For terrain envs whose
    spawn z varies with the terrain mesh, this remains correct because the
    env origin tracks the spawn cell.
    """
    object: RigidObject = env.scene[asset_cfg.name]
    range_list = [in_bound_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=env.device)

    object_pos_local = wp.to_torch(object.data.root_pos_w) - env.scene.env_origins
    return ((object_pos_local < ranges[:, 0]) | (object_pos_local > ranges[:, 1])).any(dim=1)


def in_bound(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    in_bound_range: dict[str, tuple[float, float]] = {},
) -> torch.Tensor:
    """Negation of :func:`out_of_bound` over ``env_ids``, shaped for reset acceptance."""
    return ~out_of_bound(env, asset_cfg, in_bound_range)[env_ids]


class progress_context(ManagerTermBase):
    def __init__(self, cfg: DoneTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.held_asset: Articulation | RigidObject = env.scene[cfg.params.get("held_asset_cfg").name]  # type: ignore
        self.fixed_asset: Articulation | RigidObject = env.scene[cfg.params.get("fixed_asset_cfg").name]  # type: ignore
        self.variants = assembly_variant_context(env, cfg.params.get("variant_context", "assembly_variants"))
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
        variant_context: str = "assembly_variants",
    ) -> torch.Tensor:
        held_asset_alignment_pos_w, held_asset_alignment_quat_w = self.variants.apply("held_align", self.held_asset)
        fixed_asset_alignment_pos_w, fixed_asset_alignment_quat_w = self.variants.apply("assembled", self.fixed_asset)
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


def assembly_contact_force(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("assembly_contact"),
) -> torch.Tensor:
    """Terminate when the held asset drives the fixed asset harder than the contact can take.

    Reads the filtered pair force rather than the net force on the held asset: the
    net force is dominated by the gripper holding it, which says nothing about the
    thread. The threshold should sit below the load at which the contact settings
    in use let the held asset pass through -- screened with the nut/bolt tunneling
    example in Newton.

    This bounds force-driven penetration only. Tunneling that comes from too coarse
    a collision rate happens because the contact is never generated, so the force
    reads low exactly when it fails; pair this with a check that the held asset has
    not descended further than its screw phase allows.

    Args:
        threshold: Contact force magnitude that ends the episode [N].
        sensor_cfg: The filtered contact sensor on the held asset.

    Returns:
        Per-environment termination flags, shape ``(num_envs,)``.
    """
    # (N, sensors, filters, 3) -> largest pair force per environment
    forces = env.scene.sensors[sensor_cfg.name].data.force_matrix_w.torch
    return torch.linalg.norm(forces, dim=-1).flatten(start_dim=1).max(dim=1)[0] > threshold
