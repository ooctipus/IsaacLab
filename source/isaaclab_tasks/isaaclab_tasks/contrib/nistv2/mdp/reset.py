# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Full-board reset-state generation."""

from __future__ import annotations

import math
from collections.abc import Sequence
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
from isaaclab_tasks.contrib.nist.utils import Sampler, SamplerCfg, StateLayout
from isaaclab_tasks.utils import SuccessMonitor, SuccessMonitorCfg

from ..board_layout import board_layout
from ..newton_selection import NewtonBodySelectorCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
    from isaaclab.managers import ManagerTermBaseCfg

    from isaaclab_tasks.contrib.nist.assembly_keypoints import Offset


RESET_LABELS = (
    "start_random",
    "start_near_preassembled",
    "start_near_grasped",
    "start_pick",
    "start_grasped",
    "grasped_near_goal",
    "start_near_assembled",
    "start_assembled",
)
_NEAR_PREASSEMBLED_LABEL = RESET_LABELS.index("start_near_preassembled")
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
    required_assembly_gain: torch.Tensor
    unfinished: torch.Tensor
    focus_slot: torch.Tensor
    label: torch.Tensor
    slot_state: torch.Tensor


class BalancedResetPlanner:
    """Sample reset choices while softly correcting marginal deficits."""

    def __init__(
        self,
        num_slots: int,
        device: str | torch.device,
        unfinished_count: int | None = None,
        num_variants: int = 1,
        progress_goal: bool = False,
    ):
        if num_slots < 1:
            raise ValueError(f"num_slots must be positive, got {num_slots}.")
        if num_variants < 1:
            raise ValueError(f"num_variants must be positive, got {num_variants}.")
        if unfinished_count is not None and not 1 <= unfinished_count <= num_slots:
            raise ValueError(f"unfinished_count must be between 1 and {num_slots}, got {unfinished_count}.")
        self.num_slots = num_slots
        self.num_variants = num_variants
        self.device = torch.device(device)
        self._unfinished_count = unfinished_count
        self._progress_goal = progress_goal
        self.unfinished_counts = torch.zeros(num_slots, dtype=torch.long, device=device)
        self.progress_goal_counts = torch.zeros((num_slots, num_slots), dtype=torch.long, device=device)
        self.focus_slot_counts = torch.zeros(num_slots, dtype=torch.long, device=device)
        self.unfinished_label_counts = torch.zeros((num_slots, len(RESET_LABELS)), dtype=torch.long, device=device)
        self.cell_counts = torch.zeros((len(RESET_LABELS), num_variants), dtype=torch.long, device=device)

    def sample(self, variants: torch.Tensor) -> ResetPlan:
        if variants.ndim != 2 or variants.shape[1] != self.num_slots:
            raise ValueError(f"variants must have shape [N, {self.num_slots}], got {tuple(variants.shape)}.")
        count = len(variants)
        if self._unfinished_count is None:
            unfinished_count = self._sample_marginal(self.unfinished_counts, count) + 1
        else:
            unfinished_count = torch.full((count,), self._unfinished_count, dtype=torch.long, device=self.device)

        if self._progress_goal:
            goal_weights = self.progress_goal_counts[unfinished_count - 1].float().add_(1.0).reciprocal_()
            goal_weights.mul_(torch.arange(self.num_slots, device=self.device) < unfinished_count[:, None])
            required_assembly_gain = torch.multinomial(goal_weights, 1).squeeze(1) + 1
        else:
            required_assembly_gain = unfinished_count

        ranks = torch.rand((count, self.num_slots), device=self.device).argsort(dim=1).argsort(dim=1)
        unfinished = ranks < unfinished_count[:, None]

        cell_counts = self.cell_counts[:, variants.long()].permute(1, 0, 2)
        weights = cell_counts.float().add_(1.0).reciprocal_()
        weights.mul_(self.focus_slot_counts.float().add(1.0).reciprocal()[None, None, :])
        weights[:, :_NEAR_PREASSEMBLED_LABEL].mul_(unfinished[:, None, :])
        weights[:, _NEAR_PREASSEMBLED_LABEL].mul_(~unfinished)
        weights[:, _NEAR_PREASSEMBLED_LABEL + 1 :].mul_(unfinished[:, None, :])
        label_weights = weights.sum(dim=2, keepdim=True)
        weights.div_(label_weights.clamp_min_(torch.finfo(weights.dtype).tiny))
        joint_counts = self.unfinished_label_counts[unfinished_count - 1]
        weights.mul_(joint_counts.float().add_(1.0).reciprocal_().unsqueeze(2))
        choice = torch.multinomial(weights.flatten(1), 1).squeeze(1)
        label = torch.div(choice, self.num_slots, rounding_mode="floor")
        focus_slot = choice % self.num_slots

        slot_state = torch.randint(0, len(COARSE_STATE_NAMES), (count, self.num_slots), device=self.device)
        slot_state[~unfinished] = ASSEMBLED
        rows = torch.arange(count, device=self.device)
        unfinished_focus = label != _NEAR_PREASSEMBLED_LABEL
        slot_state[rows[unfinished_focus], focus_slot[unfinished_focus]] = TARGET
        return ResetPlan(
            unfinished_count,
            required_assembly_gain,
            unfinished,
            focus_slot,
            label,
            slot_state.to(torch.uint8),
        )

    def accept(self, plan: ResetPlan, variants: torch.Tensor, indices: torch.Tensor) -> None:
        """Update balancing counts from states that entered the bank."""
        self.unfinished_counts.add_(
            torch.bincount(plan.unfinished_count[indices] - 1, minlength=self.unfinished_counts.numel())
        )
        goal_ids = (plan.unfinished_count[indices] - 1) * self.num_slots + plan.required_assembly_gain[indices] - 1
        self.progress_goal_counts.add_(
            torch.bincount(goal_ids, minlength=self.progress_goal_counts.numel()).view_as(self.progress_goal_counts)
        )
        self.focus_slot_counts.add_(torch.bincount(plan.focus_slot[indices], minlength=self.focus_slot_counts.numel()))
        joint_ids = (plan.unfinished_count[indices] - 1) * len(RESET_LABELS) + plan.label[indices]
        self.unfinished_label_counts.add_(
            torch.bincount(joint_ids, minlength=self.unfinished_label_counts.numel()).view_as(
                self.unfinished_label_counts
            )
        )
        focus_variants = variants[indices, plan.focus_slot[indices]].long()
        cells = plan.label[indices] * self.num_variants + focus_variants
        self.cell_counts.add_(torch.bincount(cells, minlength=self.cell_counts.numel()).view_as(self.cell_counts))

    @staticmethod
    def _sample_marginal(counts: torch.Tensor, count: int) -> torch.Tensor:
        probabilities = counts.float().add(1.0).reciprocal()
        return torch.multinomial(probabilities, count, replacement=True)


class board_reset(ManagerTermBase):
    """Build and sample reset states for the configured NIST assemblies."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.layout = board_layout(
            cfg.params["variant_names"],
            num_slots=cfg.params["num_slots"],
            spawn_all_sockets=cfg.params["spawn_all_sockets"],
        )
        self.num_variants = self.layout.num_variants
        self.num_slots = self.layout.num_slots
        self._sleep_assembled = self.num_slots > 1
        variants = self.layout.variants
        self._board: RigidObject = env.scene["nistboard"]
        self._held: tuple[RigidObject, ...] = tuple(env.scene[name] for name in self.layout.held_asset_names)
        self._fixed: tuple[RigidObject, ...] = tuple(env.scene[name] for name in self.layout.fixed_asset_names)
        self._robot: Articulation = env.scene[cfg.params["robot_ik_cfg"].name]
        self._robot_ik_cfg: SceneEntityCfg = cfg.params["robot_ik_cfg"]
        self._gripper_joint_ids = cfg.params["robot_gripper_cfg"].joint_ids
        self._gripper_offset: Offset = cfg.params["gripper_grasp_offset"]

        for slot, asset in enumerate(self._held):
            if asset.num_mesh_variants != self.num_variants:
                raise ValueError(
                    f"{self.layout.held_asset_names[slot]} has {asset.num_mesh_variants} mesh variants, "
                    f"expected {self.num_variants}."
                )
        if self.layout.fixed_assets_are_variant_banks:
            for slot, asset in enumerate(self._fixed):
                if asset.num_mesh_variants != self.num_variants:
                    raise ValueError(
                        f"{self.layout.fixed_asset_names[slot]} has {asset.num_mesh_variants} mesh variants, "
                        f"expected {self.num_variants}."
                    )

        held_bodies = NewtonBodySelectorCfg(
            path=tuple(rf".*/{name}(?:/.*)?" for name in self.layout.held_asset_names)
        ).resolve(NewtonManager.get_model())
        self._held_body_ids = wp.array(held_bodies.ids, dtype=wp.int32, device=env.device)

        self._profiles = tuple(AssemblyProfile(variant.profile) for variant in variants)
        self._board_default_pose = self._board.data.default_root_pose.torch[0].clone()
        self._board_offsets = torch.tensor([variant.board_offset.pose for variant in variants], device=env.device)
        self._fixture_variant_indices = torch.tensor(
            self.layout.fixture_variant_indices, dtype=torch.long, device=env.device
        )
        self._fixture_index_by_variant = torch.tensor(
            self.layout.fixture_index_by_variant, dtype=torch.long, device=env.device
        )
        offset_pos, offset_quat = self._board_offsets[:, :3], self._board_offsets[:, 3:]
        inverse_quat = math_utils.quat_inv(offset_quat)
        inverse_pos = -math_utils.quat_apply(inverse_quat, offset_pos)
        self._inverse_board_offsets = torch.cat((inverse_pos, inverse_quat), dim=1)
        pose_range = cfg.params["fixed_asset_pose_range"]
        self._fixed_asset_pose_range = torch.tensor(
            [pose_range.get(axis, (0.0, 0.0)) for axis in ("x", "y", "z", "roll", "pitch", "yaw")],
            device=env.device,
        )
        offsets = torch.tensor(
            [
                [
                    variant.held_align.pose,
                    variant.held_grasp_point.pose,
                    variant.held_grasp_middle.pose,
                    variant.fixed_tip.pose,
                ]
                for variant in variants
            ],
            device=env.device,
        )
        self._offsets = offsets
        pos, quat = offsets[..., :3], offsets[..., 3:]
        inv_quat = math_utils.quat_inv(quat.reshape(-1, 4)).view_as(quat)
        inv_pos = -math_utils.quat_apply(inv_quat.reshape(-1, 4), pos.reshape(-1, 3)).view_as(pos)
        self._inverse_offsets = torch.cat((inv_pos, inv_quat), dim=-1)
        self._grasp_diameters = torch.tensor([variant.held_grasp_diameter for variant in variants], device=env.device)
        self._grasp_ranges = torch.tensor(
            [
                [
                    [variant.grasped_pose_range[axis], variant.grasped_pose_range_centered[axis]]
                    for axis in ("x", "y", "z", "roll", "pitch", "yaw")
                ]
                for variant in variants
            ],
            device=env.device,
        ).permute(0, 2, 1, 3)

        self._capacity = int(cfg.params["state_table_size"])
        fallen_capacity = cfg.params.get("fallen_state_table_size")
        if fallen_capacity is None:
            fallen_capacity = max(env.num_envs // self.num_variants, 1) * self.num_variants
        self._fallen_capacity = int(fallen_capacity)
        self._settle_steps = int(cfg.params.get("settle_steps", 20))
        bound_range = cfg.params["held_asset_in_bound_range"]
        self._held_asset_in_bound_range = torch.tensor(
            [bound_range[axis] for axis in ("x", "y", "z")], device=env.device
        )
        self._acceptance_conditions = cfg.params["acceptance_conditions"]
        monitor_cfg: SuccessMonitorCfg | None = cfg.params["success_monitor_cfg"]
        self.success_monitor: SuccessMonitor | None = (
            None if monitor_cfg is None else monitor_cfg.class_type(monitor_cfg, 1, self._capacity, env.device)
        )
        success_monitor_env_count = cfg.params.get("success_monitor_env_count")
        self._success_monitor_env_count = (
            env.num_envs if success_monitor_env_count is None else int(success_monitor_env_count)
        )
        if not 0 <= self._success_monitor_env_count <= env.num_envs:
            raise ValueError(
                f"success_monitor_env_count must be between 0 and {env.num_envs}, "
                f"got {self._success_monitor_env_count}."
            )
        self._sampling_cfg: SamplerCfg = cfg.params["sampling"]
        self._progress_goal = bool(cfg.params.get("progress_goal", False))
        self._planner = BalancedResetPlanner(
            self.num_slots,
            env.device,
            cfg.params["unfinished_count"],
            num_variants=self.num_variants,
            progress_goal=self._progress_goal,
        )
        self._solver: DifferentialInverseKinematicsAction | None = None

        self._board_pose = torch.empty((self._capacity, 7), device=env.device)
        self._held_pose = torch.empty((self._capacity, self.num_slots, 7), device=env.device)
        self._variant_ids = torch.empty((self._capacity, self.num_slots), dtype=torch.uint8, device=env.device)
        self._robot_joint_pos = torch.empty((self._capacity, self._robot.num_joints), device=env.device)
        self._unfinished_count = torch.empty(self._capacity, dtype=torch.uint8, device=env.device)
        self._required_assembly_gain = torch.empty_like(self._unfinished_count)
        self._focus_slot = torch.empty_like(self._unfinished_count)
        self._reset_label = torch.empty_like(self._unfinished_count)
        self._slot_state = torch.empty((self._capacity, self.num_slots), dtype=torch.uint8, device=env.device)
        self._state_cell_indices = torch.empty(self._capacity, dtype=torch.long, device=env.device)
        self._bank_unfinished_index = torch.empty(self._capacity, dtype=torch.long, device=env.device)
        self.state_features: torch.Tensor | None = None
        self.estimated_success_rate: torch.Tensor | None = None
        state_feature_dim = (
            7
            + 7 * self.num_slots
            + self._robot.num_joints
            + self.num_slots * self.num_variants
            + int(self._progress_goal)
        )
        if monitor_cfg is None:
            self.state_features = torch.empty((self._capacity, state_feature_dim), device=env.device)
            self.estimated_success_rate = torch.full((self._capacity,), 0.5, device=env.device)
        self.value_shift = torch.zeros(self._capacity, device=env.device)
        self.cell_probabilities = torch.empty((len(RESET_LABELS), self.num_variants), device=env.device)
        self.state_probabilities = torch.empty(self._capacity, device=env.device)
        self.reset_probability_mass = torch.empty(self.num_slots, device=env.device)
        self.reset_probability_total = torch.empty((), device=env.device)
        self.reset_success_sum = torch.empty(self.num_slots, device=env.device)
        self.reset_state_count = torch.empty(self.num_slots, dtype=torch.long, device=env.device)
        metric_shape = (self.num_slots, self.num_variants)
        self.asset_unassembled_sum = torch.empty(metric_shape, device=env.device)
        self.asset_unfinished_count = torch.empty(metric_shape, dtype=torch.long, device=env.device)
        self._raw_cell_probabilities = torch.empty(self.cell_probabilities.numel(), device=env.device)
        self._cell_scale = torch.empty_like(self._raw_cell_probabilities)
        self._failure_probability = torch.empty(self._capacity, device=env.device)
        self._ready = False

        self.sampled_state = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self.unfinished_count = torch.zeros(env.num_envs, dtype=torch.uint8, device=env.device)
        self.required_assembly_gain = torch.zeros_like(self.unfinished_count)
        self.focus_slot = torch.zeros_like(self.unfinished_count)
        self.reset_label = torch.zeros_like(self.unfinished_count)
        self.variant_ids = (
            torch.arange(self.num_slots, dtype=torch.uint8, device=env.device).expand(env.num_envs, -1).clone()
        )
        self.slot_state = torch.zeros_like(self.variant_ids)
        self._slot_asleep = torch.zeros_like(self.slot_state, dtype=torch.bool)
        self._slot_asleep_warp = wp.from_torch(self._slot_asleep)
        self._zero_root_velocity = torch.zeros((env.num_envs, 6), device=env.device)
        self.fixed_kind_by_slot = torch.full(
            (env.num_envs, self.layout.num_fixed_slots), -1, dtype=torch.int32, device=env.device
        )
        self.outcome_state_ids = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        outcome_feature_dim = state_feature_dim if self.state_features is not None else 0
        self.outcome_next_features = torch.zeros((env.num_envs, outcome_feature_dim), device=env.device)
        self.outcome_hard_targets = torch.zeros(env.num_envs, device=env.device)
        self.outcome_grounded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.mean_estimated_success_rate = torch.full((), torch.nan, device=env.device)
        self.success_target_grounded_fraction = torch.full((), torch.nan, device=env.device)
        self.revision = 0

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Publish the active curriculum's success metric."""
        log = self._env.extras.setdefault("log", {})
        if self.success_monitor is None:
            log["Metrics/success_rate"] = self.mean_estimated_success_rate
            log["Info/SuccessTargetGroundedFraction"] = self.success_target_grounded_fraction
        else:
            log["Metrics/success_rate"] = self.success_monitor.get_mean_success_rate()

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        robot_ik_cfg: SceneEntityCfg,
        robot_gripper_cfg: SceneEntityCfg,
        gripper_grasp_offset: Offset,
        variant_names: tuple[str, ...],
        num_slots: int,
        state_table_size: int,
        unfinished_count: int | None,
        success_monitor_cfg: SuccessMonitorCfg | None,
        sampling: SamplerCfg,
        fixed_asset_pose_range: dict[str, tuple[float, float]],
        held_asset_in_bound_range: dict[str, tuple[float, float]],
        acceptance_conditions: dict[str, object],
        progress_goal: bool = False,
        success_monitor_env_count: int | None = None,
        fallen_state_table_size: int | None = None,
        settle_steps: int = 20,
        report: bool = True,
        spawn_all_sockets: bool = False,
    ) -> None:
        if not self._ready:
            self._build_bank(report)
        if env_ids.numel() == 0:
            return

        played = env_ids[self.sampled_state[env_ids] >= 0]
        if played.numel() > 0:
            progress = env.termination_manager.get_term_cfg("progress_context").func
            solver_reset = env.termination_manager.get_term("solver_reset_required")[played]
            terminated = env.termination_manager.terminated[played]
            timed_out = env.termination_manager.time_outs[played]
            pending = played[(terminated | timed_out) & ~solver_reset]
            if self.success_monitor is not None:
                if self._success_monitor_env_count < self.sampled_state.shape[0]:
                    pending = pending[pending < self._success_monitor_env_count]
                if pending.numel() > 0:
                    self.success_monitor.success_update(self.sampled_state[pending], progress.is_success[pending])
                    self.refresh_state_curriculum()
            else:
                self.outcome_state_ids[played] = -1
                if pending.numel() > 0:
                    self.outcome_state_ids[pending] = self.sampled_state[pending]
                    self.outcome_hard_targets[pending] = progress.is_success[pending].float()
                    self.outcome_grounded[pending] = env.termination_manager.terminated[pending]
                    self._capture_outcome_features(pending)

        _, state_ids = self._sample_marginally_balanced(len(env_ids))
        self.sampled_state[env_ids] = state_ids
        self.unfinished_count[env_ids] = self._unfinished_count[state_ids]
        self.required_assembly_gain[env_ids] = self._required_assembly_gain[state_ids]
        self.focus_slot[env_ids] = self._focus_slot[state_ids]
        self.reset_label[env_ids] = self._reset_label[state_ids]
        self.variant_ids[env_ids] = self._variant_ids[state_ids]
        self.slot_state[env_ids] = self._slot_state[state_ids]
        self._write_state(env_ids, state_ids)
        self._slot_asleep[env_ids] = self.slot_state[env_ids] == ASSEMBLED
        if self._sleep_assembled:
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
                "factory_board_required_assembly_gain": self.required_assembly_gain,
                "factory_board_focus_slot": self.focus_slot,
                "factory_board_reset_label": self.reset_label,
                "factory_board_variant_ids": self.variant_ids,
                "factory_board_fixed_kind_by_slot": self.fixed_kind_by_slot,
                "factory_board_slot_state": self.slot_state,
                "factory_board_reset_labels": RESET_LABELS,
                "factory_board_coarse_states": COARSE_STATE_NAMES,
            }
        )

    def refresh_state_curriculum(self) -> None:
        """Refresh cached sampling probabilities and reset-bank metrics."""
        self._refresh_state_probabilities()
        unfinished_index = self._bank_unfinished_index
        if self.success_monitor is None:
            assert self.estimated_success_rate is not None
            success_rates = self.estimated_success_rate
        else:
            success_rates = self.success_monitor.success_rate

        self.reset_probability_mass.zero_().scatter_add_(0, unfinished_index, self.state_probabilities)
        self.reset_probability_total.copy_(self.state_probabilities.sum())
        self.reset_success_sum.zero_().scatter_add_(0, unfinished_index, success_rates)
        torch.sub(1.0, success_rates, out=self._failure_probability)
        asset_sum = self.asset_unassembled_sum.flatten().zero_()
        for slot in range(self.num_slots):
            unfinished = self._slot_state[:, slot] != ASSEMBLED
            cells = unfinished_index[unfinished] * self.num_variants
            cells += self._variant_ids[unfinished, slot].long()
            asset_sum.scatter_add_(0, cells, self._failure_probability[unfinished])

    def _build_bank(self, report: bool) -> None:
        fallen_pose, fallen_variants = self._precollect_fallen()
        fallen_row_lookup = None
        fallen_row_counts = None
        if self.num_slots > 1:
            fallen_row_counts = torch.stack(
                [
                    torch.bincount(fallen_variants[:, slot].long(), minlength=self.num_variants)
                    for slot in range(self.num_slots)
                ]
            )
            if (fallen_row_counts == 0).any():
                raise RuntimeError("The fallen-pose bank must contain every mesh variant in every slot.")
            max_rows = int(fallen_row_counts.max())
            fallen_row_lookup = torch.empty(
                (self.num_slots, self.num_variants, max_rows), dtype=torch.long, device=self.device
            )
            ranks = torch.arange(max_rows, device=self.device).expand(self.num_variants, -1)
            for slot in range(self.num_slots):
                order = fallen_variants[:, slot].argsort()
                starts = fallen_row_counts[slot].cumsum(0) - fallen_row_counts[slot]
                valid = ranks < fallen_row_counts[slot, :, None]
                fallen_row_lookup[slot][valid] = order[(starts[:, None] + ranks)[valid]]

        size = 0
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        pbar = tqdm(total=self._capacity, desc="factory_board_reset")
        while size < self._capacity:
            count = min(self.num_envs, self._capacity - size)
            env_ids = all_env_ids[:count]
            slots = torch.arange(self.num_slots, device=self.device).expand(count, -1)
            if self.num_slots == 1:
                fallen_ids = torch.randint(self._fallen_capacity, (count, 1), device=self.device)
                variants = fallen_variants[fallen_ids, slots]
            else:
                variants = torch.rand((count, self.num_variants), device=self.device).argsort(dim=1)[
                    :, : self.num_slots
                ]
                assert fallen_row_counts is not None and fallen_row_lookup is not None
                counts = fallen_row_counts[slots, variants]
                occurrences = (torch.rand((count, self.num_slots), device=self.device) * counts).long()
                fallen_ids = fallen_row_lookup[slots, variants, occurrences]
            plan = self._planner.sample(variants)
            rows = torch.arange(count, device=self.device)
            focus_variants = variants[rows, plan.focus_slot].long()
            board_pose = self._sample_board_pose(focus_variants)
            poses = fallen_pose[fallen_ids, slots].clone()

            self._write_board_and_fixtures(env_ids, board_pose, variants)
            self._compose_held_poses(env_ids, poses, variants, plan, board_pose)
            self._reset_robot(env_ids)
            valid = self._reset_focus(env_ids, poses, variants, plan)
            valid = self._validate_candidates(env_ids, poses, variants, valid)
            accepted = valid.nonzero().flatten()
            if accepted.numel() == 0:
                continue

            accepted = accepted[: self._capacity - size]
            self._planner.accept(plan, variants, accepted)
            end = size + len(accepted)
            self._board_pose[size:end] = board_pose[accepted]
            self._held_pose[size:end] = poses[accepted]
            self._variant_ids[size:end] = variants[accepted]
            self._robot_joint_pos[size:end] = self._robot.data.joint_pos.torch[env_ids[accepted]]
            self._unfinished_count[size:end] = plan.unfinished_count[accepted]
            self._required_assembly_gain[size:end] = plan.required_assembly_gain[accepted]
            self._focus_slot[size:end] = plan.focus_slot[accepted]
            self._reset_label[size:end] = plan.label[accepted]
            self._slot_state[size:end] = plan.slot_state[accepted]
            accepted_variants = variants[accepted, plan.focus_slot[accepted]].long()
            self._state_cell_indices[size:end] = plan.label[accepted] * self.num_variants + accepted_variants
            pbar.update(end - size)
            size = end
        pbar.close()

        del fallen_pose, fallen_variants
        if torch.device(self.device).type == "cuda":
            torch.cuda.empty_cache()
        if self.state_features is not None:
            self._build_state_features()
        rows = torch.arange(self._capacity, device=self.device)
        layout = StateLayout(coords=self._held_pose[:, 0, :3], spawn_index=rows)
        if self.success_monitor is None:
            assert self.estimated_success_rate is not None
            success_rates = self.estimated_success_rate
        else:
            success_rates = self.success_monitor.success_rate
        self._sampler: Sampler = self._sampling_cfg.class_type(
            self._sampling_cfg,
            layout,
            env=self._env,
            success_rates=success_rates,
            value_shifts=self.value_shift,
        )
        self._initialize_bank_metrics()
        self.refresh_state_curriculum()
        self._ready = True
        if report:
            self._report_distribution()

    def _build_state_features(self) -> None:
        """Pack physical features in bank-row order."""
        features = self.state_features
        if features is None:
            raise RuntimeError("Reset-state features were not enabled for this environment.")
        self._pack_state_features(
            features,
            self._board_pose,
            self._held_pose,
            self._robot_joint_pos,
            self._variant_ids,
            self.num_slots - self._unfinished_count + self._required_assembly_gain,
        )

    def _capture_outcome_features(self, env_ids: torch.Tensor) -> None:
        """Capture the live physical endpoint before reset writes the next state."""
        origins = self._env.scene.env_origins[env_ids]
        board_pose = self._board.data.root_link_pose_w.torch[env_ids].clone()
        board_pose[:, :3] -= origins
        held_pose = torch.stack([asset.data.root_link_pose_w.torch[env_ids] for asset in self._held], dim=1)
        held_pose[..., :3] -= origins[:, None, :]
        features = torch.empty((len(env_ids), self.outcome_next_features.shape[1]), device=self.device)
        self._pack_state_features(
            features,
            board_pose,
            held_pose,
            self._robot.data.joint_pos.torch[env_ids],
            self.variant_ids[env_ids],
            self.num_slots - self.unfinished_count[env_ids] + self.required_assembly_gain[env_ids],
        )
        self.outcome_next_features[env_ids] = features

    def _pack_state_features(
        self,
        features: torch.Tensor,
        board_pose: torch.Tensor,
        held_pose: torch.Tensor,
        joint_pos: torch.Tensor,
        variant_ids: torch.Tensor,
        target_assembled_count: torch.Tensor,
    ) -> None:
        """Pack the curriculum state shared by reset rows and live endpoints."""
        features.zero_()
        column = 0

        features[:, column : column + 7].copy_(board_pose)
        column += 7

        width = 7 * self.num_slots
        features[:, column : column + width].copy_(held_pose.flatten(1))
        column += width

        width = self._robot.num_joints
        features[:, column : column + width].copy_(joint_pos)
        column += width

        width = self.num_slots * self.num_variants
        variant_one_hot = features[:, column : column + width].view(len(features), self.num_slots, self.num_variants)
        variant_one_hot.scatter_(2, variant_ids.long().unsqueeze(-1), 1.0)
        if self._progress_goal:
            features[:, column + width].copy_(target_assembled_count)

    def _precollect_fallen(self) -> tuple[torch.Tensor, torch.Tensor]:
        poses = torch.empty((self._fallen_capacity, self.num_slots, 7), device=self.device)
        variants = torch.empty((self._fallen_capacity, self.num_slots), dtype=torch.uint8, device=self.device)
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        size = 0
        pbar = tqdm(total=self._fallen_capacity, desc="factory_board_fallen")

        self._park_robot(all_env_ids)
        board_pose = self._board_default_pose.expand(self.num_envs, -1).clone()
        board_pose[:, :3] += self._env.scene.env_origins
        self._board.write_root_link_pose_to_sim_index(root_pose=board_pose, env_ids=all_env_ids)
        for asset in self._fixed:
            pose = asset.data.root_link_pose_w.torch.clone()
            pose[:, 2] = self._env.scene.env_origins[:, 2] + 5.0
            asset.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=all_env_ids)
        self.fixed_kind_by_slot.fill_(-1)
        substeps = 1 if self._env.sim.physics_manager.handles_decimation() else self._env.cfg.decimation
        while size < self._fallen_capacity:
            selected_variants = (
                torch.arange(self.num_envs, device=self.device)[:, None]
                + torch.arange(self.num_slots, device=self.device)[None, :]
                + size
            ) % self.num_variants
            self._drop_held_assets(all_env_ids, selected_variants)
            for _ in range(self._settle_steps * substeps):
                self._env.sim.step(render=False)
            self._env.scene.update(dt=self._env.step_dt)

            finite = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            batch_pose = torch.empty((self.num_envs, self.num_slots, 7), device=self.device)
            for slot, asset in enumerate(self._held):
                pose = asset.data.root_link_pose_w.torch.clone()
                finite &= torch.isfinite(pose).all(dim=1)
                pose[:, :3] -= self._env.scene.env_origins
                pos_b, quat_b = math_utils.subtract_frame_transforms(
                    self._board_default_pose[:3].expand(self.num_envs, -1),
                    self._board_default_pose[3:].expand(self.num_envs, -1),
                    pose[:, :3],
                    pose[:, 3:],
                )
                batch_pose[:, slot] = torch.cat((pos_b, quat_b), dim=1)
                finite &= (pose[:, 2] > -0.1) & (pose[:, 2] < 1.0)
            valid = finite
            accepted = valid.nonzero().flatten()[: self._fallen_capacity - size]
            if accepted.numel() == 0:
                continue
            end = size + len(accepted)
            poses[size:end] = batch_pose[accepted]
            variants[size:end] = selected_variants[accepted].to(torch.uint8)
            pbar.update(end - size)
            size = end
        pbar.close()
        return poses, variants

    def _sample_board_pose(self, focus_variants: torch.Tensor) -> torch.Tensor:
        count = len(focus_variants)
        board = self._board_default_pose.expand(count, -1)
        focus_offsets = self._board_offsets[focus_variants]
        fixed_pos, fixed_quat = math_utils.combine_frame_transforms(
            board[:, :3], board[:, 3:], focus_offsets[:, :3], focus_offsets[:, 3:]
        )
        samples = math_utils.sample_uniform(
            self._fixed_asset_pose_range[:, 0],
            self._fixed_asset_pose_range[:, 1],
            (count, 6),
            device=self.device,
        )
        fixed_pos += samples[:, :3]
        fixed_quat = math_utils.quat_mul(
            fixed_quat, math_utils.quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5])
        )
        inverse = self._inverse_board_offsets[focus_variants]
        board_pos, board_quat = math_utils.combine_frame_transforms(
            fixed_pos, fixed_quat, inverse[:, :3], inverse[:, 3:]
        )
        return torch.cat((board_pos, board_quat), dim=1)

    def _write_board_and_fixtures(
        self, env_ids: torch.Tensor, board_pose: torch.Tensor, variants: torch.Tensor
    ) -> None:
        board_pose_w = board_pose.clone()
        board_pose_w[:, :3] += self._env.scene.env_origins[env_ids]
        self._board.write_root_link_pose_to_sim_index(root_pose=board_pose_w, env_ids=env_ids)

        if self.layout.fixed_assets_are_variant_banks:
            kinds = self._fixture_index_by_variant[variants.long()].sort(dim=1).values
            first = torch.ones_like(kinds, dtype=torch.bool)
            first[:, 1:] = kinds[:, 1:] != kinds[:, :-1]
            slots = first.cumsum(dim=1) - 1
            selected = torch.full(
                (len(env_ids), self.layout.num_fixed_slots), -1, dtype=torch.int32, device=self.device
            )
            rows, columns = first.nonzero(as_tuple=True)
            selected[rows, slots[rows, columns]] = kinds[rows, columns].to(torch.int32)
            self.fixed_kind_by_slot[env_ids] = selected
        else:
            self.fixed_kind_by_slot[env_ids] = torch.arange(
                self.layout.num_fixed_slots, dtype=torch.int32, device=self.device
            )

        for slot, asset in enumerate(self._fixed):
            fixture_kind = self.fixed_kind_by_slot[env_ids, slot]
            active = fixture_kind >= 0
            safe_kind = fixture_kind.clamp_min(0).long()
            representative_variant = self._fixture_variant_indices[safe_kind]
            if self.layout.fixed_assets_are_variant_banks:
                asset.write_mesh_variant_to_sim(representative_variant.to(torch.int32), env_ids)
            offset = self._board_offsets[representative_variant]
            fixed_pos, fixed_quat = math_utils.combine_frame_transforms(
                board_pose_w[:, :3], board_pose_w[:, 3:], offset[:, :3], offset[:, 3:]
            )
            fixed_pos[:, 2] = torch.where(active, fixed_pos[:, 2], board_pose_w[:, 2] + 5.0 + 0.1 * slot)
            asset.write_root_link_pose_to_sim_index(
                root_pose=torch.cat((fixed_pos, fixed_quat), dim=1), env_ids=env_ids
            )

    def _drop_held_assets(self, env_ids: torch.Tensor, variants: torch.Tensor) -> None:
        count = len(env_ids)
        yaw = torch.rand((count, self.num_slots), device=self.device) * 6.28 - 3.14
        roll = torch.rand_like(yaw) * 3.14 - 1.57
        pitch = torch.rand_like(yaw) * 3.14 - 1.57
        quat = math_utils.quat_from_euler_xyz(roll.flatten(), pitch.flatten(), yaw.flatten()).view(
            count, self.num_slots, 4
        )
        for slot, asset in enumerate(self._held):
            x = 0.5 * torch.rand(count, device=self.device)
            y = torch.rand(count, device=self.device) - 0.5
            z = 0.12 + 0.02 * torch.rand(count, device=self.device)
            pose = torch.cat(
                (torch.stack((x, y, z), dim=1) + self._env.scene.env_origins[env_ids], quat[:, slot]), dim=1
            )
            asset.write_mesh_variant_to_sim(variants[:, slot].to(torch.int32), env_ids)
            asset.write_root_link_pose_to_sim(pose, env_ids=env_ids)
            asset.write_root_com_velocity_to_sim(self._zero_root_velocity[:count], env_ids=env_ids)

    def _validate_candidates(
        self, env_ids: torch.Tensor, poses: torch.Tensor, variants: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """Write candidate states and evaluate every configured acceptance condition."""
        self._write_held_assets(env_ids, poses, variants)
        valid &= torch.isfinite(poses).all(dim=(1, 2))
        positions = poses[..., :3]
        valid &= (
            (positions >= self._held_asset_in_bound_range[:, 0]) & (positions <= self._held_asset_in_bound_range[:, 1])
        ).all(dim=(1, 2))
        for condition in self._acceptance_conditions.values():
            condition_func = condition if callable(condition) else condition.func
            valid &= condition_func(self._env, env_ids)
        return valid

    def _write_held_assets(self, env_ids: torch.Tensor, poses: torch.Tensor, variants: torch.Tensor) -> None:
        for slot, asset in enumerate(self._held):
            pose = poses[:, slot].clone()
            pose[:, :3] += self._env.scene.env_origins[env_ids]
            asset.write_mesh_variant_to_sim(variants[:, slot].to(torch.int32), env_ids)
            asset.write_root_link_pose_to_sim(pose, env_ids=env_ids)
            asset.write_root_com_velocity_to_sim(self._zero_root_velocity[: len(env_ids)], env_ids=env_ids)

    def _compose_held_poses(
        self,
        env_ids: torch.Tensor,
        poses: torch.Tensor,
        variants: torch.Tensor,
        plan: ResetPlan,
        board_pose: torch.Tensor,
    ) -> None:
        board_pos = board_pose[:, None, :3].expand(-1, self.num_slots, -1).reshape(-1, 3)
        board_quat = board_pose[:, None, 3:].expand(-1, self.num_slots, -1).reshape(-1, 4)
        pos, quat = math_utils.combine_frame_transforms(
            board_pos, board_quat, poses[..., :3].reshape(-1, 3), poses[..., 3:].reshape(-1, 4)
        )
        poses.copy_(torch.cat((pos, quat), dim=1).view_as(poses))

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
            fixed_pos, fixed_quat = self._fixed_root_pose(
                env_ids[rows], torch.full_like(rows, variant_id, dtype=torch.long)
            )
            align_pos, align_quat = math_utils.combine_frame_transforms(fixed_pos, fixed_quat, offset_pos, offset_quat)
            inverse = self._inverse_offsets[variant_id, 0].expand(len(rows), -1)
            root_pos, root_quat = math_utils.combine_frame_transforms(
                align_pos, align_quat, inverse[:, :3], inverse[:, 3:]
            )
            poses[rows, slots, :3] = root_pos - self._env.scene.env_origins[env_ids[rows]]
            poses[rows, slots, 3:] = root_quat

    def _reset_focus(
        self, env_ids: torch.Tensor, poses: torch.Tensor, variants: torch.Tensor, plan: ResetPlan
    ) -> torch.Tensor:
        valid = torch.ones(len(env_ids), dtype=torch.bool, device=self.device)
        for label in range(len(RESET_LABELS)):
            rows = (plan.label == label).nonzero().flatten()
            if rows.numel() == 0:
                continue
            label_name = RESET_LABELS[label]
            slots = plan.focus_slot[rows]
            variant_ids = variants[rows, slots].long()
            selected_env_ids = env_ids[rows]

            if label_name == "start_random":
                self._place_workspace_target(poses, rows, slots)
            elif label_name == "start_assembled":
                self._place_target_profile(selected_env_ids, poses, rows, slots, variant_ids, (0.0, 1.0))
            elif label_name == "start_near_assembled":
                self._place_target_profile(selected_env_ids, poses, rows, slots, variant_ids, (1.0, 1.1), True)

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
                strict_pose = label_name in (
                    "start_near_preassembled",
                    "start_pick",
                    "start_near_assembled",
                    "start_assembled",
                )
                reached = self._solve_end_effector(
                    selected_env_ids,
                    reference_pos,
                    reference_quat,
                    ranges,
                    label_name in ("start_random", "start_pick"),
                    (25, 35) if strict_pose else (1, 30),
                    (0.001, 0.05) if strict_pose else None,
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
            fixed_pos, fixed_quat = self._fixed_root_pose(
                env_ids[local], torch.full_like(local, variant_id, dtype=torch.long)
            )
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
        fixed_pos, fixed_quat = self._fixed_root_pose(env_ids, variant_ids)
        return self._apply_offset(torch.cat((fixed_pos, fixed_quat), dim=1), variant_ids, offset_index)

    def _fixed_root_pose(self, env_ids: torch.Tensor, variant_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        offset = self._board_offsets[variant_ids]
        return math_utils.combine_frame_transforms(
            self._board.data.root_pos_w.torch[env_ids],
            self._board.data.root_quat_w.torch[env_ids],
            offset[:, :3],
            offset[:, 3:],
        )

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
            [-1.0, 1.0]
            if label in ("start_near_preassembled", "start_near_assembled", "start_assembled")
            else [-0.5, 0.5],
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

    def _initialize_bank_metrics(self) -> None:
        self._bank_unfinished_index.copy_(self._unfinished_count).sub_(1)
        self.reset_state_count.copy_(torch.bincount(self._bank_unfinished_index, minlength=self.num_slots))
        asset_count = self.asset_unfinished_count.flatten().zero_()
        for slot in range(self.num_slots):
            unfinished = self._slot_state[:, slot] != ASSEMBLED
            cells = self._bank_unfinished_index[unfinished] * self.num_variants
            cells += self._variant_ids[unfinished, slot].long()
            asset_count.add_(torch.bincount(cells, minlength=asset_count.numel()))

    def _refresh_state_probabilities(self) -> None:
        probabilities = self._sampler.probabilities()
        raw_cell_probabilities = self._raw_cell_probabilities.zero_()
        raw_cell_probabilities.scatter_add_(0, self._state_cell_indices, probabilities)
        cell_probabilities = self.cell_probabilities
        cell_probabilities.copy_(raw_cell_probabilities.view_as(cell_probabilities))
        tiny = torch.finfo(probabilities.dtype).tiny
        for _ in range(4):
            row_sum = cell_probabilities.sum(dim=1, keepdim=True).clamp_min_(tiny)
            cell_probabilities.div_(row_sum * cell_probabilities.shape[0])
            column_sum = cell_probabilities.sum(dim=0, keepdim=True).clamp_min_(tiny)
            cell_probabilities.div_(column_sum * cell_probabilities.shape[1])
        cell_probabilities.div_(cell_probabilities.sum().clamp_min_(tiny))

        cell_scale = self._cell_scale.zero_()
        populated = raw_cell_probabilities > 0.0
        cell_scale[populated] = cell_probabilities.flatten()[populated] / raw_cell_probabilities[populated]
        self.state_probabilities.copy_(probabilities)
        self.state_probabilities.mul_(cell_scale[self._state_cell_indices])
        self.state_probabilities.div_(self.state_probabilities.sum().clamp_min_(tiny))

    def _sample_marginally_balanced(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the cached full-bank curriculum distribution."""
        return self.state_probabilities, self._sampler.sample(self.state_probabilities, num_samples)

    def _write_state(self, env_ids: torch.Tensor, state_ids: torch.Tensor) -> None:
        variants = self._variant_ids[state_ids]
        self._write_board_and_fixtures(env_ids, self._board_pose[state_ids], variants)
        self._write_held_assets(env_ids, self._held_pose[state_ids], variants)
        root = self._robot.data.default_root_state.torch[env_ids].clone()
        root[:, :3] += self._env.scene.env_origins[env_ids]
        joints = self._robot_joint_pos[state_ids]
        self._robot.write_root_state_to_sim(root, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joints, torch.zeros_like(joints), env_ids=env_ids)

    def _report_distribution(self) -> None:
        n_counts = torch.bincount(self._unfinished_count.long(), minlength=self.num_slots + 1)[1:]
        goal_ids = (self._unfinished_count.long() - 1) * self.num_slots + self._required_assembly_gain.long() - 1
        goal_counts = torch.bincount(goal_ids, minlength=self.num_slots**2).view(self.num_slots, self.num_slots)
        label_counts = torch.bincount(self._reset_label.long(), minlength=len(RESET_LABELS))
        joint_ids = (self._unfinished_count.long() - 1) * len(RESET_LABELS) + self._reset_label.long()
        joint_counts = torch.bincount(joint_ids, minlength=self.num_slots * len(RESET_LABELS)).view(
            self.num_slots, len(RESET_LABELS)
        )
        focus_variants = self._variant_ids[torch.arange(self._capacity, device=self.device), self._focus_slot.long()]
        variant_counts = torch.bincount(focus_variants.long(), minlength=self.num_variants)
        coarse = self._slot_state.flatten()
        coarse_counts = torch.bincount(coarse[coarse < ASSEMBLED].long(), minlength=len(COARSE_STATE_NAMES))
        print(f"[factory_board_reset] unfinished: {n_counts.tolist()}")
        if self._progress_goal:
            for unfinished, counts in enumerate(goal_counts.tolist(), 1):
                print(f"[factory_board_reset] unfinished={unfinished:02d} progress goals: {counts[:unfinished]}")
        print(f"[factory_board_reset] reset labels: {dict(zip(RESET_LABELS, label_counts.tolist()))}")
        for unfinished, counts in enumerate(joint_counts.tolist(), 1):
            print(f"[factory_board_reset] unfinished={unfinished:02d}: {dict(zip(RESET_LABELS, counts))}")
        print(f"[factory_board_reset] focus variants: {variant_counts.tolist()}")
        print(f"[factory_board_reset] coarse states: {dict(zip(COARSE_STATE_NAMES, coarse_counts.tolist()))}")


class initial_unfinished_time_out(ManagerTermBase):
    """End an episode at a fixed horizon or after time scaled by its initial unfinished count."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        reset = env.event_manager.get_term_cfg("reset_board").func
        if not isinstance(reset, board_reset):
            raise TypeError("initial_unfinished_time_out requires the resolved board reset term.")
        self._reset = reset
        self._enabled = bool(cfg.params.get("enabled", True))
        self._disabled = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._dynamic = bool(cfg.params.get("dynamic", True))
        seconds_per_asset = float(cfg.params.get("seconds_per_asset", 14.0))
        if not math.isfinite(seconds_per_asset) or seconds_per_asset <= 0.0:
            raise ValueError(f"seconds_per_asset must be positive and finite, got {seconds_per_asset}.")
        self._steps_per_asset = math.ceil(seconds_per_asset / env.step_dt)
        fixed_horizon_s = cfg.params.get("fixed_horizon_s")
        if fixed_horizon_s is not None:
            fixed_horizon_s = float(fixed_horizon_s)
            if not math.isfinite(fixed_horizon_s) or fixed_horizon_s <= 0.0:
                raise ValueError(f"fixed_horizon_s must be positive and finite, got {fixed_horizon_s}.")
        self._fixed_steps = None if fixed_horizon_s is None else math.ceil(fixed_horizon_s / env.step_dt)
        dynamic_env_count = cfg.params.get("dynamic_env_count")
        self._dynamic_env_mask: torch.Tensor | None = None
        if dynamic_env_count is not None:
            dynamic_env_count = int(dynamic_env_count)
            if not 0 <= dynamic_env_count <= env.num_envs:
                raise ValueError(f"dynamic_env_count must be between 0 and {env.num_envs}, got {dynamic_env_count}.")
            if self._fixed_steps is None:
                raise ValueError("dynamic_env_count requires fixed_horizon_s.")
            if not self._dynamic:
                raise ValueError("dynamic_env_count requires dynamic=True.")
            self._dynamic_env_mask = torch.arange(env.num_envs, device=env.device) < dynamic_env_count
        initial_steps = self._fixed_steps or self._steps_per_asset * reset.num_slots
        self.episode_limit_steps = torch.full((env.num_envs,), initial_steps, dtype=torch.long, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Snapshot the new episode's unfinished-asset count."""
        if not self._enabled:
            return
        if env_ids is None:
            env_ids = slice(None)
        if self._fixed_steps is not None and self._dynamic_env_mask is not None:
            dynamic_steps = self._reset.unfinished_count[env_ids].long() * self._steps_per_asset
            self.episode_limit_steps[env_ids] = torch.where(
                self._dynamic_env_mask[env_ids], dynamic_steps, self._fixed_steps
            )
        elif self._fixed_steps is not None:
            self.episode_limit_steps[env_ids] = self._fixed_steps
        elif self._dynamic:
            self.episode_limit_steps[env_ids] = self._reset.unfinished_count[env_ids].long() * self._steps_per_asset
        else:
            self.episode_limit_steps[env_ids] = self._reset.num_slots * self._steps_per_asset

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        enabled: bool = True,
        seconds_per_asset: float = 14.0,
        dynamic: bool = True,
        fixed_horizon_s: float | None = None,
        dynamic_env_count: int | None = None,
    ) -> torch.Tensor:
        """Return environments whose individual episode limit has elapsed."""
        del enabled, seconds_per_asset, dynamic, fixed_horizon_s, dynamic_env_count
        if not self._enabled:
            return self._disabled
        return env.episode_length_buf >= self.episode_limit_steps
