# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Full-board reset-state generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager
from tqdm import tqdm

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import math as math_utils

from isaaclab_tasks.contrib.nist.assembly_profile import AssemblyProfile
from isaaclab_tasks.contrib.nist.assembly_variants import ASSEMBLY_VARIANTS
from isaaclab_tasks.contrib.nist.utils import Sampler, SamplerCfg, StateLayout
from isaaclab_tasks.contrib.nistv2.board_layout import (
    FIXED_ASSET_NAME_BY_VARIANT,
    HELD_ASSET_NAMES,
    NUM_ASSEMBLIES,
)
from isaaclab_tasks.utils import SuccessMonitor, SuccessMonitorCfg

from ..newton_selection import NewtonBodySelectorCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction

    from isaaclab_tasks.contrib.nist.assembly_keypoints import Offset


RESET_LABELS = (
    "start_random",
    "start_near_grasped",
    "start_pick",
    "start_grasped",
    "grasped_near_goal",
    "start_near_assembled",
    "start_assembled",
)
COARSE_STATE_NAMES = ("fallen", "workspace_random", "partial_assembly")

FALLEN = 0
WORKSPACE_RANDOM = 1
PARTIAL_ASSEMBLY = 2
ASSEMBLED = 3
TARGET = 4


@dataclass(slots=True)
class ResetPlan:
    """Discrete choices for one reset-generation batch."""

    unfinished_count: torch.Tensor
    unfinished: torch.Tensor
    target_slot: torch.Tensor
    target_label: torch.Tensor
    slot_state: torch.Tensor


class BalancedResetPlanner:
    """Sample reset choices while softly correcting marginal deficits."""

    def __init__(self, device: str | torch.device):
        self.device = torch.device(device)
        self.unfinished_counts = torch.zeros(NUM_ASSEMBLIES, dtype=torch.long, device=device)
        self.label_counts = torch.zeros(len(RESET_LABELS), dtype=torch.long, device=device)

    def sample(self, count: int) -> ResetPlan:
        unfinished_count = self._sample_marginal(self.unfinished_counts, count) + 1
        target_label = self._sample_marginal(self.label_counts, count)

        order = torch.rand((count, NUM_ASSEMBLIES), device=self.device).argsort(dim=1)
        ranks = order.argsort(dim=1)
        unfinished = ranks < unfinished_count[:, None]
        target_slot = torch.multinomial(unfinished.float(), 1).squeeze(1)

        slot_state = torch.randint(0, len(COARSE_STATE_NAMES), (count, NUM_ASSEMBLIES), device=self.device)
        slot_state[~unfinished] = ASSEMBLED
        slot_state[torch.arange(count, device=self.device), target_slot] = TARGET
        return ResetPlan(unfinished_count, unfinished, target_slot, target_label, slot_state.to(torch.uint8))

    def accept(self, plan: ResetPlan, indices: torch.Tensor) -> None:
        """Update balancing counts from states that entered the bank."""
        self.unfinished_counts.add_(
            torch.bincount(plan.unfinished_count[indices] - 1, minlength=self.unfinished_counts.numel())
        )
        self.label_counts.add_(torch.bincount(plan.target_label[indices], minlength=self.label_counts.numel()))

    @staticmethod
    def _sample_marginal(counts: torch.Tensor, count: int) -> torch.Tensor:
        probabilities = counts.float().add(1.0).reciprocal()
        return torch.multinomial(probabilities, count, replacement=True)


class board_reset(ManagerTermBase):
    """Build and sample reset states for the complete NIST board."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._held: tuple[RigidObject, ...] = tuple(env.scene[name] for name in HELD_ASSET_NAMES)
        self._fixed: tuple[RigidObject, ...] = tuple(env.scene[name] for name in FIXED_ASSET_NAME_BY_VARIANT)
        self._robot: Articulation = env.scene[cfg.params["robot_ik_cfg"].name]
        self._robot_ik_cfg: SceneEntityCfg = cfg.params["robot_ik_cfg"]
        self._gripper_joint_ids = cfg.params["robot_gripper_cfg"].joint_ids
        self._gripper_offset: Offset = cfg.params["gripper_grasp_offset"]

        for slot, asset in enumerate(self._held):
            if asset.num_mesh_variants != NUM_ASSEMBLIES:
                raise ValueError(f"{HELD_ASSET_NAMES[slot]} has {asset.num_mesh_variants} mesh variants, expected 20.")

        held_bodies = NewtonBodySelectorCfg(path=tuple(rf".*/{name}(?:/.*)?" for name in HELD_ASSET_NAMES)).resolve(
            NewtonManager.get_model()
        )
        self._held_body_ids = wp.array(held_bodies.ids, dtype=wp.int32, device=env.device)

        self._profiles = tuple(AssemblyProfile(variant.profile) for variant in ASSEMBLY_VARIANTS)
        offsets = torch.tensor(
            [
                [
                    variant.held_align.pose,
                    variant.held_grasp_point.pose,
                    variant.held_grasp_middle.pose,
                    variant.fixed_tip.pose,
                ]
                for variant in ASSEMBLY_VARIANTS
            ],
            device=env.device,
        )
        self._offsets = offsets
        pos, quat = offsets[..., :3], offsets[..., 3:]
        inv_quat = math_utils.quat_inv(quat.reshape(-1, 4)).view_as(quat)
        inv_pos = -math_utils.quat_apply(inv_quat.reshape(-1, 4), pos.reshape(-1, 3)).view_as(pos)
        self._inverse_offsets = torch.cat((inv_pos, inv_quat), dim=-1)
        self._grasp_diameters = torch.tensor(
            [variant.held_grasp_diameter for variant in ASSEMBLY_VARIANTS], device=env.device
        )
        self._grasp_ranges = torch.tensor(
            [
                [
                    [variant.grasped_pose_range[axis], variant.grasped_pose_range_centered[axis]]
                    for axis in ("x", "y", "z", "roll", "pitch", "yaw")
                ]
                for variant in ASSEMBLY_VARIANTS
            ],
            device=env.device,
        ).permute(0, 2, 1, 3)

        self._capacity = int(cfg.params["state_table_size"])
        self._fallen_capacity = int(cfg.params.get("fallen_state_table_size", self._capacity))
        self._settle_steps = int(cfg.params.get("settle_steps", 20))
        monitor_cfg: SuccessMonitorCfg = cfg.params["success_monitor_cfg"]
        self.success_monitor: SuccessMonitor = monitor_cfg.class_type(monitor_cfg, 1, self._capacity, env.device)
        self._sampling_cfg: SamplerCfg = cfg.params["sampling"]
        self._planner = BalancedResetPlanner(env.device)
        self._solver: DifferentialInverseKinematicsAction | None = None

        self._held_pose = torch.empty((self._capacity, NUM_ASSEMBLIES, 7), device=env.device)
        self._variant_ids = torch.empty((self._capacity, NUM_ASSEMBLIES), dtype=torch.uint8, device=env.device)
        self._robot_joint_pos = torch.empty((self._capacity, self._robot.num_joints), device=env.device)
        self._unfinished_count = torch.empty(self._capacity, dtype=torch.uint8, device=env.device)
        self._target_slot = torch.empty_like(self._unfinished_count)
        self._target_label = torch.empty_like(self._unfinished_count)
        self._slot_state = torch.empty((self._capacity, NUM_ASSEMBLIES), dtype=torch.uint8, device=env.device)
        self._ready = False

        self.sampled_state = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self.unfinished_count = torch.zeros(env.num_envs, dtype=torch.uint8, device=env.device)
        self.target_slot = torch.zeros_like(self.unfinished_count)
        self.target_label = torch.zeros_like(self.unfinished_count)
        self.variant_ids = (
            torch.arange(NUM_ASSEMBLIES, dtype=torch.uint8, device=env.device).expand(env.num_envs, -1).clone()
        )
        self.slot_state = torch.zeros_like(self.variant_ids)
        self._slot_asleep = torch.zeros_like(self.slot_state, dtype=torch.bool)
        self._slot_asleep_warp = wp.from_torch(self._slot_asleep)
        self.initial_unfinished = torch.zeros_like(self.variant_ids, dtype=torch.bool)
        self.sample_counts = torch.zeros(NUM_ASSEMBLIES, dtype=torch.long, device=env.device)
        self.sample_total = torch.zeros((), dtype=torch.long, device=env.device)
        self.revision = 0

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        robot_ik_cfg: SceneEntityCfg,
        robot_gripper_cfg: SceneEntityCfg,
        gripper_grasp_offset: Offset,
        state_table_size: int,
        success_monitor_cfg: SuccessMonitorCfg,
        sampling: SamplerCfg,
        fallen_state_table_size: int | None = None,
        settle_steps: int = 20,
        report: bool = True,
    ) -> None:
        if not self._ready:
            self._build_bank(report)
        if env_ids.numel() == 0:
            return

        played = env_ids[self.sampled_state[env_ids] >= 0]
        if played.numel() > 0:
            progress = env.termination_manager.get_term_cfg("progress_context").func
            self.success_monitor.success_update(self.sampled_state[played], progress.is_success[played])

        _, state_ids = self._sampler.probabilities_and_sample(len(env_ids))
        self.sampled_state[env_ids] = state_ids
        self.unfinished_count[env_ids] = self._unfinished_count[state_ids]
        self.target_slot[env_ids] = self._target_slot[state_ids]
        self.target_label[env_ids] = self._target_label[state_ids]
        self.variant_ids[env_ids] = self._variant_ids[state_ids]
        self.slot_state[env_ids] = self._slot_state[state_ids]
        self.initial_unfinished[env_ids.long()[:, None], self.variant_ids[env_ids].long()] = (
            self.slot_state[env_ids] != ASSEMBLED
        )
        self.sample_counts.add_(torch.bincount(self.unfinished_count[env_ids].long() - 1, minlength=NUM_ASSEMBLIES))
        self.sample_total.add_(env_ids.numel())
        self._write_state(env_ids, state_ids)
        self._slot_asleep[env_ids] = self.slot_state[env_ids] == ASSEMBLED
        NewtonManager.set_body_sleep_state(
            self._held_body_ids,
            self._slot_asleep_warp,
            wp.from_torch(env_ids.to(dtype=torch.int32)),
        )
        self.revision += 1
        env.extras.setdefault("diagnostics", {}).update(
            {
                "factory_board_reset_state": self.sampled_state,
                "factory_board_unfinished_count": self.unfinished_count,
                "factory_board_target_slot": self.target_slot,
                "factory_board_target_label": self.target_label,
                "factory_board_variant_ids": self.variant_ids,
                "factory_board_slot_state": self.slot_state,
                "factory_board_reset_labels": RESET_LABELS,
                "factory_board_coarse_states": COARSE_STATE_NAMES,
            }
        )

    def _build_bank(self, report: bool) -> None:
        fallen_pose, fallen_variants = self._precollect_fallen()
        size = 0
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        pbar = tqdm(total=self._capacity, desc="factory_board_reset")
        while size < self._capacity:
            count = min(self.num_envs, self._capacity - size)
            env_ids = all_env_ids[:count]
            fallen_ids = torch.randint(self._fallen_capacity, (count,), device=self.device)
            variants = fallen_variants[fallen_ids]
            poses = fallen_pose[fallen_ids].clone()
            plan = self._planner.sample(count)

            self._compose_held_poses(env_ids, poses, variants, plan)
            self._reset_robot(env_ids)
            valid = self._reset_target(env_ids, poses, variants, plan)
            valid &= torch.isfinite(poses).all(dim=(1, 2))
            valid &= (poses[..., 2] > -0.1).all(dim=1) & (poses[..., 2] < 2.0).all(dim=1)
            accepted = valid.nonzero().flatten()
            if accepted.numel() == 0:
                continue

            accepted = accepted[: self._capacity - size]
            self._planner.accept(plan, accepted)
            end = size + len(accepted)
            self._held_pose[size:end] = poses[accepted]
            self._variant_ids[size:end] = variants[accepted]
            self._robot_joint_pos[size:end] = self._robot.data.joint_pos.torch[env_ids[accepted]]
            self._unfinished_count[size:end] = plan.unfinished_count[accepted]
            self._target_slot[size:end] = plan.target_slot[accepted]
            self._target_label[size:end] = plan.target_label[accepted]
            self._slot_state[size:end] = plan.slot_state[accepted]
            pbar.update(end - size)
            size = end
        pbar.close()

        del fallen_pose, fallen_variants
        if torch.device(self.device).type == "cuda":
            torch.cuda.empty_cache()
        rows = torch.arange(self._capacity, device=self.device)
        layout = StateLayout(coords=self._held_pose[:, 0, :3], spawn_index=rows)
        self._sampler: Sampler = self._sampling_cfg.class_type(
            self._sampling_cfg, layout, env=self._env, success_rates=self.success_monitor.success_rate
        )
        self._ready = True
        if report:
            self._report_distribution()

    def _precollect_fallen(self) -> tuple[torch.Tensor, torch.Tensor]:
        poses = torch.empty((self._fallen_capacity, NUM_ASSEMBLIES, 7), device=self.device)
        variants = torch.empty((self._fallen_capacity, NUM_ASSEMBLIES), dtype=torch.uint8, device=self.device)
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        size = 0
        pbar = tqdm(total=self._fallen_capacity, desc="factory_board_fallen")

        self._park_robot(all_env_ids)
        substeps = 1 if self._env.sim.physics_manager.handles_decimation() else self._env.cfg.decimation
        while size < self._fallen_capacity:
            permutation = torch.rand((self.num_envs, NUM_ASSEMBLIES), device=self.device).argsort(dim=1)
            self._drop_held_assets(all_env_ids, permutation)
            for _ in range(self._settle_steps * substeps):
                self._env.sim.step(render=False)
            self._env.scene.update(dt=self._env.step_dt)

            finite = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            batch_pose = torch.empty((self.num_envs, NUM_ASSEMBLIES, 7), device=self.device)
            for slot, asset in enumerate(self._held):
                pose = asset.data.root_link_pose_w.torch.clone()
                pose[:, :3] -= self._env.scene.env_origins
                batch_pose[:, slot] = pose
                finite &= torch.isfinite(pose).all(dim=1)
            valid = finite & (batch_pose[..., 2] > -0.1).all(dim=1) & (batch_pose[..., 2] < 1.0).all(dim=1)
            accepted = valid.nonzero().flatten()[: self._fallen_capacity - size]
            if accepted.numel() == 0:
                continue
            end = size + len(accepted)
            poses[size:end] = batch_pose[accepted]
            variants[size:end] = permutation[accepted].to(torch.uint8)
            pbar.update(end - size)
            size = end
        pbar.close()
        return poses, variants

    def _drop_held_assets(self, env_ids: torch.Tensor, variants: torch.Tensor) -> None:
        count = len(env_ids)
        yaw = torch.rand((count, NUM_ASSEMBLIES), device=self.device) * 6.283185307179586 - 3.141592653589793
        roll = torch.rand_like(yaw) * 3.141592653589793 - 1.5707963267948966
        pitch = torch.rand_like(yaw) * 3.141592653589793 - 1.5707963267948966
        quat = math_utils.quat_from_euler_xyz(roll.flatten(), pitch.flatten(), yaw.flatten()).view(
            count, NUM_ASSEMBLIES, 4
        )
        for slot, asset in enumerate(self._held):
            x = 0.10 + 0.09 * (slot % 5) + (torch.rand(count, device=self.device) - 0.5) * 0.01
            y = -0.15 + 0.10 * (slot // 5) + (torch.rand(count, device=self.device) - 0.5) * 0.01
            z = 0.10 + 0.04 * torch.rand(count, device=self.device)
            pose = torch.cat(
                (torch.stack((x, y, z), dim=1) + self._env.scene.env_origins[env_ids], quat[:, slot]), dim=1
            )
            asset.write_mesh_variant_to_sim(variants[:, slot].to(torch.int32), env_ids)
            asset.write_root_link_pose_to_sim(pose, env_ids=env_ids)
            asset.write_root_com_velocity_to_sim(torch.zeros((count, 6), device=self.device), env_ids=env_ids)

    def _compose_held_poses(
        self, env_ids: torch.Tensor, poses: torch.Tensor, variants: torch.Tensor, plan: ResetPlan
    ) -> None:
        self._place_profile_state(env_ids, poses, variants, plan.slot_state == ASSEMBLED, (0.0, 0.0))
        self._place_profile_state(env_ids, poses, variants, plan.slot_state == PARTIAL_ASSEMBLY, (1.05, 1.5))

        random_mask = plan.slot_state == WORKSPACE_RANDOM
        rows, slots = random_mask.nonzero(as_tuple=True)
        if rows.numel() > 0:
            x = 0.08 + 0.10 * (slots % 5).float()
            y = torch.where(slots < 10, -0.42 + 0.12 * (slots // 5).float(), 0.30 + 0.12 * ((slots - 10) // 5).float())
            z = 0.08 + 0.12 * torch.rand(len(rows), device=self.device)
            angles = torch.rand((len(rows), 3), device=self.device)
            angles[:, :2] = angles[:, :2] * 3.141592653589793 - 1.5707963267948966
            angles[:, 2] = angles[:, 2] * 6.283185307179586 - 3.141592653589793
            poses[rows, slots, :3] = torch.stack((x, y, z), dim=1)
            poses[rows, slots, 3:] = math_utils.quat_from_euler_xyz(angles[:, 0], angles[:, 1], angles[:, 2])

    def _place_profile_state(
        self,
        env_ids: torch.Tensor,
        poses: torch.Tensor,
        variants: torch.Tensor,
        mask: torch.Tensor,
        fraction_range: tuple[float, float],
    ) -> None:
        for variant_id, profile in enumerate(self._profiles):
            rows, slots = (mask & (variants == variant_id)).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            offset_pos, offset_quat = profile.sample(fraction_range, len(rows), self.device)
            fixed = self._fixed[variant_id]
            fixed_pos = fixed.data.root_pos_w.torch[env_ids[rows]]
            fixed_quat = fixed.data.root_quat_w.torch[env_ids[rows]]
            align_pos, align_quat = math_utils.combine_frame_transforms(fixed_pos, fixed_quat, offset_pos, offset_quat)
            inverse = self._inverse_offsets[variant_id, 0].expand(len(rows), -1)
            root_pos, root_quat = math_utils.combine_frame_transforms(
                align_pos, align_quat, inverse[:, :3], inverse[:, 3:]
            )
            poses[rows, slots, :3] = root_pos - self._env.scene.env_origins[env_ids[rows]]
            poses[rows, slots, 3:] = root_quat

    def _reset_target(
        self, env_ids: torch.Tensor, poses: torch.Tensor, variants: torch.Tensor, plan: ResetPlan
    ) -> torch.Tensor:
        valid = torch.ones(len(env_ids), dtype=torch.bool, device=self.device)
        for label in range(len(RESET_LABELS)):
            rows = (plan.target_label == label).nonzero().flatten()
            if rows.numel() == 0:
                continue
            label_name = RESET_LABELS[label]
            slots = plan.target_slot[rows]
            variant_ids = variants[rows, slots].long()
            selected_env_ids = env_ids[rows]

            if label_name == "start_random":
                self._place_workspace_target(poses, rows, slots)
            elif label_name == "start_assembled":
                self._place_target_profile(selected_env_ids, poses, rows, slots, variant_ids, (0.0, 1.0))
            elif label_name == "start_near_assembled":
                self._place_target_profile(selected_env_ids, poses, rows, slots, variant_ids, (1.0, 1.1), True)
            elif label_name == "start_pick":
                pass

            if label_name in ("start_near_grasped", "start_grasped", "grasped_near_goal"):
                reference_pos, reference_quat = self._fixed_keypoint(selected_env_ids, variant_ids, 3)
                range_index = 1 if label_name == "grasped_near_goal" else 0
                ranges = self._grasp_ranges[variant_ids, range_index]
                reached = self._solve_end_effector(
                    selected_env_ids, reference_pos, reference_quat, ranges, False, (10, 20), None
                )
                self._place_target_in_gripper(selected_env_ids, poses, rows, slots, variant_ids)
            else:
                pose = poses[rows, slots].clone()
                pose[:, :3] += self._env.scene.env_origins[selected_env_ids]
                reference_pos, reference_quat = self._apply_offset(pose, variant_ids, 2)
                ranges = self._held_target_ranges(len(rows), label_name)
                reached = self._solve_end_effector(
                    selected_env_ids,
                    reference_pos,
                    reference_quat,
                    ranges,
                    label_name in ("start_random", "start_pick"),
                    (25, 35) if label_name in ("start_assembled", "start_near_assembled", "start_pick") else (1, 30),
                    (0.001, 0.05) if label_name in ("start_assembled", "start_near_assembled", "start_pick") else None,
                )
            valid[rows] &= reached
            self._set_gripper(
                selected_env_ids,
                variant_ids,
                flexible=label_name in ("start_random", "start_near_grasped", "start_assembled"),
            )
        return valid

    def _place_workspace_target(self, poses: torch.Tensor, rows: torch.Tensor, slots: torch.Tensor) -> None:
        samples = torch.rand((len(rows), 6), device=self.device)
        samples[:, 0] = samples[:, 0] * 0.5
        samples[:, 1] = samples[:, 1] - 0.5
        samples[:, 2] = samples[:, 2] * 0.185 + 0.015
        samples[:, 3:5] = samples[:, 3:5] * 3.14 - 1.57
        samples[:, 5] = samples[:, 5] * 6.28 - 3.14
        poses[rows, slots, :3] = samples[:, :3]
        poses[rows, slots, 3:] = math_utils.quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5])

    def _place_target_profile(
        self,
        env_ids: torch.Tensor,
        poses: torch.Tensor,
        rows: torch.Tensor,
        slots: torch.Tensor,
        variant_ids: torch.Tensor,
        fraction_range: tuple[float, float],
        add_noise: bool = False,
    ) -> None:
        for variant_id, profile in enumerate(self._profiles):
            local = (variant_ids == variant_id).nonzero().flatten()
            if local.numel() == 0:
                continue
            pos, quat = profile.sample(fraction_range, len(local), self.device)
            if add_noise:
                pos[:, :2] += (torch.rand((len(local), 2), device=self.device) - 0.5) * 0.004
                pos[:, 2] += torch.rand(len(local), device=self.device) * 0.01
                angles = torch.rand((len(local), 3), device=self.device)
                angles[:, :2] = angles[:, :2] * 0.6 - 0.3
                angles[:, 2] = angles[:, 2] - 0.5
                noise = math_utils.quat_from_euler_xyz(angles[:, 0], angles[:, 1], angles[:, 2])
                quat = math_utils.quat_mul(noise, quat)
            fixed = self._fixed[variant_id]
            fixed_pos = fixed.data.root_pos_w.torch[env_ids[local]]
            fixed_quat = fixed.data.root_quat_w.torch[env_ids[local]]
            align_pos, align_quat = math_utils.combine_frame_transforms(fixed_pos, fixed_quat, pos, quat)
            inverse = self._inverse_offsets[variant_id, 0].expand(len(local), -1)
            root_pos, root_quat = math_utils.combine_frame_transforms(
                align_pos, align_quat, inverse[:, :3], inverse[:, 3:]
            )
            target_rows, target_slots = rows[local], slots[local]
            poses[target_rows, target_slots, :3] = root_pos - self._env.scene.env_origins[env_ids[local]]
            poses[target_rows, target_slots, 3:] = root_quat

    def _fixed_keypoint(
        self, env_ids: torch.Tensor, variant_ids: torch.Tensor, offset_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos = torch.empty((len(env_ids), 3), device=self.device)
        quat = torch.empty((len(env_ids), 4), device=self.device)
        for variant_id, fixed in enumerate(self._fixed):
            rows = (variant_ids == variant_id).nonzero().flatten()
            if rows.numel() == 0:
                continue
            pose = torch.cat(
                (fixed.data.root_pos_w.torch[env_ids[rows]], fixed.data.root_quat_w.torch[env_ids[rows]]), 1
            )
            pos[rows], quat[rows] = self._apply_offset(pose, variant_ids[rows], offset_index)
        return pos, quat

    def _apply_offset(
        self, pose: torch.Tensor, variant_ids: torch.Tensor, offset_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        offset = self._offsets[variant_ids, offset_index]
        return math_utils.combine_frame_transforms(pose[:, :3], pose[:, 3:], offset[:, :3], offset[:, 3:])

    def _place_target_in_gripper(
        self,
        env_ids: torch.Tensor,
        poses: torch.Tensor,
        rows: torch.Tensor,
        slots: torch.Tensor,
        variant_ids: torch.Tensor,
    ) -> None:
        hand_pos = self._robot.data.body_link_pos_w.torch[env_ids, self._robot_ik_cfg.body_ids].view(-1, 3)
        hand_quat = self._robot.data.body_link_quat_w.torch[env_ids, self._robot_ik_cfg.body_ids].view(-1, 4)
        grasp_pos, grasp_quat = self._gripper_offset.combine(hand_pos, hand_quat)
        noise = torch.rand((len(env_ids), 6), device=self.device)
        noise[:, :3] = (noise[:, :3] - 0.5) * 0.01
        noise[:, 3] = 0.0
        noise[:, 4] = noise[:, 4] * 4.0 - 2.0
        noise[:, 5] = 0.0
        grasp_pos, grasp_quat = math_utils.combine_frame_transforms(
            grasp_pos,
            grasp_quat,
            noise[:, :3],
            math_utils.quat_from_euler_xyz(noise[:, 3], noise[:, 4], noise[:, 5]),
        )
        inverse = self._inverse_offsets[variant_ids, 1]
        root_pos, root_quat = math_utils.combine_frame_transforms(grasp_pos, grasp_quat, inverse[:, :3], inverse[:, 3:])
        poses[rows, slots, :3] = root_pos - self._env.scene.env_origins[env_ids]
        poses[rows, slots, 3:] = root_quat

    def _solve_end_effector(
        self,
        env_ids: torch.Tensor,
        reference_pos: torch.Tensor,
        reference_quat: torch.Tensor,
        ranges: torch.Tensor,
        upright: bool,
        iterations: tuple[int, int],
        pose_tolerance: tuple[float, float] | None,
    ) -> torch.Tensor:
        if self._solver is None:
            solver_cfg = DifferentialInverseKinematicsActionCfg(
                asset_name=self._robot_ik_cfg.name,
                joint_names=self._robot_ik_cfg.joint_names,
                body_name=self._robot_ik_cfg.body_names,
                controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
                scale=1.0,
            )
            self._solver = solver_cfg.class_type(solver_cfg, self._env)

        samples = math_utils.sample_uniform(ranges[..., 0], ranges[..., 1], (len(env_ids), 6), device=self.device)
        root_pos = self._robot.data.root_link_pos_w.torch[env_ids]
        root_quat = self._robot.data.root_link_quat_w.torch[env_ids]
        if upright:
            reference_b = math_utils.quat_mul(math_utils.quat_inv(root_quat), reference_quat)
            axis = math_utils.quat_apply(
                reference_b, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(len(env_ids), 3)
            )
            yaw = torch.atan2(axis[:, 1], axis[:, 0])
            zero = torch.zeros_like(yaw)
            reference_quat = math_utils.quat_mul(root_quat, math_utils.quat_from_euler_xyz(zero, zero, yaw))

        grasp_pos, grasp_quat = math_utils.combine_frame_transforms(
            reference_pos,
            reference_quat,
            samples[:, :3],
            math_utils.quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5]),
        )
        body_pos, body_quat = self._gripper_offset.subtract(grasp_pos, grasp_quat)
        command_pos, command_quat = self._solver._compute_frame_pose()
        command_pos[env_ids], command_quat[env_ids] = math_utils.subtract_frame_transforms(
            root_pos, root_quat, body_pos, body_quat
        )
        self._solver.process_actions(torch.cat((command_pos, command_quat), dim=1))

        limits = self._robot.data.joint_pos_limits.torch[env_ids][:, self._robot_ik_cfg.joint_ids]
        steps = int(torch.randint(iterations[0], iterations[1] + 1, (1,)).item())
        for _ in range(steps):
            self._solver.apply_actions()
            current = self._robot.data.joint_pos.torch[env_ids]
            target = self._robot.data.joint_pos_target.torch[env_ids]
            joints = (current + 0.25 * (target - current))[:, self._robot_ik_cfg.joint_ids]
            self._robot.write_joint_position_to_sim(
                torch.clamp(joints, limits[..., 0], limits[..., 1]), self._robot_ik_cfg.joint_ids, env_ids
            )

        if pose_tolerance is None:
            return torch.ones(len(env_ids), dtype=torch.bool, device=self.device)
        reached_pos, reached_quat = self._solver._compute_frame_pose()
        pos_error = torch.norm(reached_pos[env_ids] - command_pos[env_ids], dim=1)
        rot_error = math_utils.quat_error_magnitude(reached_quat[env_ids], command_quat[env_ids])
        return (pos_error < pose_tolerance[0]) & (rot_error < pose_tolerance[1])

    def _held_target_ranges(self, count: int, label: str) -> torch.Tensor:
        ranges = torch.zeros((count, 6, 2), device=self.device)
        ranges[:, 0, :] = torch.tensor([-0.005, 0.005], device=self.device)
        ranges[:, 1, :] = torch.tensor([-0.005, 0.005], device=self.device)
        ranges[:, 2, :] = torch.tensor([0.0, 0.01] if label == "start_pick" else [-0.015, 0.025], device=self.device)
        ranges[:, 3, :] = torch.tensor([-0.1, 0.1], device=self.device)
        ranges[:, 4, :] = torch.tensor(
            [-1.0, 1.0] if label in ("start_assembled", "start_near_assembled") else [-0.5, 0.5],
            device=self.device,
        )
        ranges[:, 5, :] = torch.tensor([-0.3, 0.3] if label == "start_pick" else [-2.09, 2.09], device=self.device)
        return ranges

    def _set_gripper(self, env_ids: torch.Tensor, variant_ids: torch.Tensor, flexible: bool) -> None:
        position = self._robot.data.joint_pos.torch[env_ids][:, self._gripper_joint_ids].clone()
        minimum = self._grasp_diameters[variant_ids] * 0.525
        if flexible:
            maximum = self._robot.data.joint_pos_limits.torch[env_ids, self._gripper_joint_ids[0], 1]
            angle = minimum + torch.rand_like(minimum) * (maximum - minimum)
        else:
            angle = minimum
        position[:] = angle[:, None]
        self._robot.write_joint_position_to_sim(position, self._gripper_joint_ids, env_ids)

    def _reset_robot(self, env_ids: torch.Tensor) -> None:
        root = self._robot.data.default_root_state.torch[env_ids].clone()
        root[:, :3] += self._env.scene.env_origins[env_ids]
        joints = self._robot.data.default_joint_pos.torch[env_ids]
        self._robot.write_root_state_to_sim(root, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joints, torch.zeros_like(joints), env_ids=env_ids)

    def _park_robot(self, env_ids: torch.Tensor) -> None:
        root = self._robot.data.default_root_state.torch[env_ids].clone()
        root[:, :3] += self._env.scene.env_origins[env_ids]
        root[:, 2] += 5.0
        self._robot.write_root_state_to_sim(root, env_ids=env_ids)
        joints = self._robot.data.default_joint_pos.torch[env_ids]
        self._robot.write_joint_state_to_sim(joints, torch.zeros_like(joints), env_ids=env_ids)

    def _write_state(self, env_ids: torch.Tensor, state_ids: torch.Tensor) -> None:
        variants = self._variant_ids[state_ids]
        poses = self._held_pose[state_ids]
        for slot, asset in enumerate(self._held):
            pose = poses[:, slot].clone()
            pose[:, :3] += self._env.scene.env_origins[env_ids]
            asset.write_mesh_variant_to_sim(variants[:, slot].to(torch.int32), env_ids)
            asset.write_root_link_pose_to_sim(pose, env_ids=env_ids)
            asset.write_root_com_velocity_to_sim(torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids)
        root = self._robot.data.default_root_state.torch[env_ids].clone()
        root[:, :3] += self._env.scene.env_origins[env_ids]
        joints = self._robot_joint_pos[state_ids]
        self._robot.write_root_state_to_sim(root, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joints, torch.zeros_like(joints), env_ids=env_ids)

    def _report_distribution(self) -> None:
        n_counts = torch.bincount(self._unfinished_count.long(), minlength=NUM_ASSEMBLIES + 1)[1:]
        label_counts = torch.bincount(self._target_label.long(), minlength=len(RESET_LABELS))
        joint_ids = (self._unfinished_count.long() - 1) * len(RESET_LABELS) + self._target_label.long()
        joint_counts = torch.bincount(joint_ids, minlength=NUM_ASSEMBLIES * len(RESET_LABELS)).view(
            NUM_ASSEMBLIES, len(RESET_LABELS)
        )
        target_variants = self._variant_ids[torch.arange(self._capacity, device=self.device), self._target_slot.long()]
        variant_counts = torch.bincount(target_variants.long(), minlength=NUM_ASSEMBLIES)
        coarse = self._slot_state.flatten()
        coarse_counts = torch.bincount(coarse[coarse < ASSEMBLED].long(), minlength=len(COARSE_STATE_NAMES))
        print(f"[factory_board_reset] unfinished: {n_counts.tolist()}")
        print(f"[factory_board_reset] target labels: {dict(zip(RESET_LABELS, label_counts.tolist()))}")
        for unfinished, counts in enumerate(joint_counts.tolist(), 1):
            print(f"[factory_board_reset] unfinished={unfinished:02d}: {dict(zip(RESET_LABELS, counts))}")
        print(f"[factory_board_reset] target variants: {variant_counts.tolist()}")
        print(f"[factory_board_reset] coarse states: {dict(zip(COARSE_STATE_NAMES, coarse_counts.tolist()))}")
