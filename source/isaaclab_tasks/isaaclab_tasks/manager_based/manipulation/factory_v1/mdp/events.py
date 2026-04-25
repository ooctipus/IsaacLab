# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Generator
from typing import TYPE_CHECKING, Literal

import warp as wp
from tqdm import tqdm
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import math as math_utils

from ..assembly_keypoints import NIST_BOARD_CFG
from ..utils import AssemblyProfile, AssemblyProfileCfg
from .util import SuccessMonitorCfg, StateBuffer, StateBufferCfg, beta_sampling_probs, tagged_report
from .util import state_ops as factory_utils
from .util.sampling_cfg import UniformSamplingCfg, BetaSamplingCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
    from .util import SuccessMonitor
    from ..assembly_keypoints import Offset


def reset_fixed_assets(env: ManagerBasedRLEnv, env_ids: torch.tensor, asset_map: dict[str, str]):
    """Reset fixed assets to their positions on the NIST board.

    Args:
        env: The environment instance.
        env_ids: Environment indices to reset.
        asset_map: Mapping from scene entity key to :class:`KeyPointsNistBoard` attribute name.
    """
    nistboard: RigidObject = env.scene["nistboard"]
    for scene_key, keypoint_attr in asset_map.items():
        asset: Articulation | RigidObject = env.scene[scene_key]
        asset_offset_on_nist_board: Offset = getattr(NIST_BOARD_CFG, keypoint_attr)
        asset_on_board_pos, asset_on_board_quat = asset_offset_on_nist_board.apply(nistboard)
        root_pose = torch.cat((asset_on_board_pos, asset_on_board_quat), dim=1)[env_ids]
        asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim(torch.zeros_like(wp.to_torch(asset.data.root_vel_w)[env_ids]), env_ids=env_ids)


_PROFILE_CACHE: dict[int, AssemblyProfile] = {}


def _sweep_assembly_fraction(
    lo: float, hi: float, step: float = 0.001
) -> Generator[tuple[float, float]]:
    """Yield ``(frac, frac)`` pairs that bounce between *lo* and *hi*."""
    frac = lo
    increasing = True
    while True:
        frac += step if increasing else -step
        if frac >= hi or frac <= lo:
            increasing = not increasing
        yield (frac, frac)


def reset_held_asset_on_fixed_asset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    assembly_profile: AssemblyProfileCfg,
    held_asset_align_offset: Offset,
    assembly_fraction_range: tuple[float, float],
    fixed_asset_cfg: SceneEntityCfg,
    held_asset_cfg: SceneEntityCfg,
    debug_term: bool = False,
):
    profile = _PROFILE_CACHE.get(id(assembly_profile))
    if profile is None:
        profile = assembly_profile.class_type(assembly_profile)
        _PROFILE_CACHE[id(assembly_profile)] = profile

    fixed_asset: RigidObject = env.scene[fixed_asset_cfg.name]
    held_asset: Articulation = env.scene[held_asset_cfg.name]

    fractions = _sweep_assembly_fraction(*assembly_fraction_range) if debug_term else iter([assembly_fraction_range])
    for frac_range in fractions:
        pos_offset, quat_offset = profile.sample(frac_range, len(env_ids), env.device)
        fixed_root_pos_w = wp.to_torch(fixed_asset.data.root_pos_w)
        fixed_root_quat_w = wp.to_torch(fixed_asset.data.root_quat_w)
        pos, quat = math_utils.combine_frame_transforms(
            fixed_root_pos_w[env_ids], fixed_root_quat_w[env_ids], pos_offset, quat_offset
        )
        pose = torch.cat(held_asset_align_offset.subtract(pos, quat), dim=1)
        vel = wp.to_torch(held_asset.data.default_root_state)[env_ids, 7:]
        held_asset.write_root_pose_to_sim(pose, env_ids=env_ids)
        held_asset.write_root_com_velocity_to_sim(vel, env_ids=env_ids)
        if debug_term:
            env.sim.render()


def reset_held_asset_in_gripper(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    holding_body_cfg: SceneEntityCfg,
    held_asset_cfg: SceneEntityCfg,
    held_asset_graspable_offset: Offset,
    held_asset_inhand_range: dict[str, tuple[float, float]],
    gripper_grasp_offset: Offset,
):
    robot: Articulation = env.scene[holding_body_cfg.name]
    held_asset: Articulation = env.scene[held_asset_cfg.name]

    end_effector_quat_w = wp.to_torch(robot.data.body_link_quat_w)[env_ids, holding_body_cfg.body_ids].view(-1, 4)
    end_effector_pos_w = wp.to_torch(robot.data.body_link_pos_w)[env_ids, holding_body_cfg.body_ids].view(-1, 3)
    grasp_quat = gripper_grasp_offset.quat_t(env.device).expand(len(env_ids), -1)
    translated_held_asset_pos, translated_held_asset_quat = held_asset_graspable_offset.subtract(
        end_effector_pos_w,
        math_utils.quat_mul(end_effector_quat_w, grasp_quat),
    )

    # Add randomization
    range_list = [held_asset_inhand_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)
    new_pos_w = translated_held_asset_pos + samples[:, 0:3]
    quat_b = math_utils.quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5])
    new_quat_w = math_utils.quat_mul(translated_held_asset_quat, quat_b)

    held_asset.write_root_link_pose_to_sim(torch.cat([new_pos_w, new_quat_w], dim=1), env_ids=env_ids)  # type: ignore
    held_asset.write_root_com_velocity_to_sim(wp.to_torch(held_asset.data.default_root_state)[env_ids, 7:], env_ids=env_ids)  # type: ignore


def grasp_held_asset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg,
    held_asset_diameter: float,
    flexible_angle: bool = True,
) -> None:
    robot: Articulation = env.scene[robot_cfg.name]
    joint_pos = wp.to_torch(robot.data.joint_pos)[:, robot_cfg.joint_ids][env_ids].clone()
    min_angle = held_asset_diameter / 2 * 1.15
    if flexible_angle:
        max_angle = wp.to_torch(robot.data.joint_pos_limits)[0, robot_cfg.joint_ids[0], 1]
        joint_pos[:] = (torch.rand((len(env_ids),), device=env.device) * (max_angle - min_angle) + min_angle).unsqueeze(1)
    else:
        joint_pos[:] = min_angle

    robot.write_joint_position_to_sim(joint_pos, robot_cfg.joint_ids, env_ids)  # type: ignore


class reset_end_effector_around_asset(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        fixed_asset_cfg: SceneEntityCfg = cfg.params.get("fixed_asset_cfg")  # type: ignore
        fixed_asset_offset: Offset = cfg.params.get("fixed_asset_offset")  # type: ignore
        pose_range_b: dict[str, tuple[float, float]] = cfg.params.get("pose_range_b")  # type: ignore
        robot_ik_cfg: SceneEntityCfg = cfg.params.get("robot_ik_cfg", SceneEntityCfg("robot"))

        range_list = [pose_range_b.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.wrist_idx = 6
        self.ranges = torch.tensor(range_list, device=env.device)
        self.fixed_asset: Articulation | RigidObject = env.scene[fixed_asset_cfg.name]
        self.fixed_asset_offset: Offset = fixed_asset_offset
        self.robot: Articulation = env.scene[robot_ik_cfg.name]
        self.joint_ids: list[int] | slice = robot_ik_cfg.joint_ids

        self.robot_ik_solver_cfg = DifferentialInverseKinematicsActionCfg(
            asset_name=robot_ik_cfg.name,
            joint_names=robot_ik_cfg.joint_names,  # type: ignore
            body_name=robot_ik_cfg.body_names,  # type: ignore
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            scale=1.0,
        )
        self.solver: DifferentialInverseKinematicsAction = None  # type: ignore
        self.grasp_angle_range  = (0.3, 0.7)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        fixed_asset_cfg: SceneEntityCfg,
        fixed_asset_offset: Offset,
        pose_range_b: dict[str, tuple[float, float]],
        robot_ik_cfg: SceneEntityCfg,
        ik_iterations: tuple[int, int] = (5, 10),
    ) -> None:
        if self.solver is None:
            self.solver = self.robot_ik_solver_cfg.class_type(self.robot_ik_solver_cfg, env)
        fixed_tip_pos_w, fixed_tip_quat_w = self.fixed_asset_offset.apply(self.fixed_asset)
        samples = math_utils.sample_uniform(self.ranges[:, 0], self.ranges[:, 1], (len(env_ids), 6), device=env.device)
        pos_b, quat_b = self.solver._compute_frame_pose()
        # for those non_reset_id, we will let ik solve for its current position
        pos_w = fixed_tip_pos_w[env_ids] + samples[:, 0:3]
        quat_w = math_utils.quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5])
        
        pos_b[env_ids], quat_b[env_ids] = math_utils.subtract_frame_transforms(
            wp.to_torch(self.robot.data.root_link_pos_w)[env_ids], wp.to_torch(self.robot.data.root_link_quat_w)[env_ids], pos_w, quat_w
        )
        self.solver.process_actions(torch.cat([pos_b, quat_b], dim=1))
        n_joints: int = self.robot.num_joints if isinstance(self.joint_ids, slice) else len(self.joint_ids)
        
        # Error Rate 75% ^ 10 = 0.05 (final error)
        lo, hi = ik_iterations
        k = int(torch.randint(low=lo, high=hi + 1, size=(1,)).item())
        for _ in range(k):
            self.solver.apply_actions()
            delta_joint_pos = 0.25 * (wp.to_torch(self.robot.data.joint_pos_target)[env_ids] - wp.to_torch(self.robot.data.joint_pos)[env_ids])
            self.robot.write_joint_position_to_sim(
                position=(delta_joint_pos + wp.to_torch(self.robot.data.joint_pos)[env_ids])[:, self.joint_ids],
                joint_ids=self.joint_ids,
                env_ids=env_ids,  # type: ignore
            )

        # wrist_low  = self.robot.data.joint_pos_limits[env_ids, self.wrist_idx, 0]
        # wrist_high = self.robot.data.joint_pos_limits[env_ids, self.wrist_idx, 1]
        # wrist_pos = (wrist_low + (wrist_high - wrist_low) * torch.rand_like(wrist_low)).view(len(env_ids), -1)
        # self.robot.write_joint_position_to_sim(position=wrist_pos, joint_ids=self.wrist_idx, env_ids=env_ids)
        self.robot.root_physx_view.get_jacobians()


def reset_root_state_uniform_on_offset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    offset: Offset,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = wp.to_torch(asset.data.default_root_state)[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples
    positions, orientations = offset.subtract(positions.view(-1, 3), orientations.view(-1, 4))

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)



class reset_accumulator(ManagerTermBase):
    """Accumulate validated reset states into a shared buffer and sample from it.

    During the pre-collection phase, reset states are generated and validated
    against acceptance conditions until the buffer is full. After that, every
    call samples from the buffer. Optionally keeps accumulating new valid states
    at runtime (``keep_accumulating=True``).

    All envs share a single ring buffer of validated states stored in
    env-origin-relative coordinates, so any env can be reset to any stored state.
    """

    _shared_buffer: StateBuffer | None = None

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.acceptance_conditions = cfg.params.get("acceptance_conditions")
        for key, val in self.acceptance_conditions.items():
            if hasattr(val, "class_type"):
                self.acceptance_conditions[key] = val.class_type(val, env)

        asset_keys = cfg.params.get("reset_assets")
        state_dim = factory_utils.get_reset_state(self._env, torch.tensor([0], device=env.device), asset_keys).shape[-1]
        buf_cfg: StateBufferCfg = cfg.params.get("state_buffer_cfg", StateBufferCfg())
        max_size = buf_cfg.size

        self.state_buffer = StateBuffer(max_size, state_dim, env.device)
        reset_accumulator._shared_buffer = self.state_buffer
        self.sampled_slots = torch.zeros(env.num_envs, device=env.device, dtype=torch.int)
        self.precollecting_phase = True
        self._tag_indices_bind: str | None = buf_cfg.tag_indices_bind
        self._tag_names_resolved = False
        self._sampling_cfg = cfg.params.get("sampling", UniformSamplingCfg())

        self.monitor_success_rate = torch.zeros(max_size, device=env.device)
        monitor_cfg: SuccessMonitorCfg = cfg.params.get("success_monitor_cfg", SuccessMonitorCfg(num_monitored_data=max_size, device=env.device))
        monitor_cfg.num_monitored_data = max_size
        monitor_cfg.device = env.device
        self.success_monitor = monitor_cfg.class_type(monitor_cfg, self.monitor_success_rate)

        self.success_rate = torch.zeros(max_size, device=env.device)
        self._success_rate_source = "monitor"
        if isinstance(self._sampling_cfg, BetaSamplingCfg) and "state_buffer" in self._sampling_cfg.success_rate_bind:
            self._success_rate_source = "success_estimator"

    # ------------------------------------------------------------------
    # Buffer accumulation
    # ------------------------------------------------------------------

    def _accumulate(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor, reset_term: EventTermCfg, reset_assets: list[str]):
        """Run a single reset attempt and store valid states in the buffer."""
        if not self._tag_names_resolved:
            buf_cfg: StateBufferCfg = self.cfg.params.get("state_buffer_cfg", StateBufferCfg())
            if buf_cfg.tag_names_bind is not None:
                self.state_buffer.set_tag_names(eval(buf_cfg.tag_names_bind))  # noqa: S307
            self._tag_names_resolved = True

        reset_term.func(env, env_ids, **reset_term.params)
        valid_mask = torch.ones(len(env_ids), dtype=torch.bool, device=env.device)
        for _, val in self.acceptance_conditions.items():
            valid_mask &= val(env, env_ids)

        valid_env_ids = env_ids[valid_mask]
        if valid_env_ids.numel() > 0:
            states = factory_utils.get_reset_state(self._env, valid_env_ids, reset_assets, is_relative=True)
            start, n = self.state_buffer.add(states)

            if self._tag_indices_bind is not None:
                all_tags = eval(self._tag_indices_bind)  # noqa: S307
                slot_indices = torch.arange(start, start + n, device=env.device)
                self.state_buffer.set_tags(slot_indices, all_tags[env_ids][valid_mask][:n])

        return env_ids[~valid_mask]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        reset_term: EventTermCfg,
        reset_assets: list[str] = [],
        acceptance_conditions: dict = {},
        state_buffer_cfg: StateBufferCfg = StateBufferCfg(),
        success_monitor_cfg: SuccessMonitorCfg | None = None,
        sampling: UniformSamplingCfg | BetaSamplingCfg = UniformSamplingCfg(),
        keep_accumulating: bool = False,
        report: bool = False,
        monitor_exclude_terms: list[str] = [],
    ):
        # 1. Pre-collect until buffer is full
        if self.precollecting_phase:
            all_env_ids = torch.arange(env.num_envs, device=env.device)
            pbar = tqdm(total=self.state_buffer.max_size, desc="reset_accumulator")
            while not self.state_buffer.is_full:
                prev = len(self.state_buffer)
                self._accumulate(env, all_env_ids, reset_term, reset_assets)
                pbar.update(len(self.state_buffer) - prev)
            pbar.close()
            self.precollecting_phase = False

        # 2. Update success monitor with episode outcomes, excluding specified terms
        progress = env.termination_manager.get_term_cfg("progress_context").func
        monitor_ids = env_ids
        if env_ids.numel() > 0 and monitor_exclude_terms:
            exclude_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            for term_name in monitor_exclude_terms:
                if term_name in env.termination_manager._term_names:
                    exclude_mask |= env.termination_manager._last_episode_dones[:, env.termination_manager._term_name_to_term_idx[term_name]]
            monitor_ids = env_ids[~exclude_mask[env_ids]]
        if monitor_ids.numel() > 0:
            self.success_monitor.success_update(self.sampled_slots[monitor_ids], progress.is_success[monitor_ids].float())

        # Sync the unified success_rate from the active source
        if self._success_rate_source == "success_estimator" and self.state_buffer.success_rates is not None:
            self.success_rate[:] = self.state_buffer.success_rates
        else:
            self.success_rate[:] = self.monitor_success_rate

        if report:
            log: dict[str, float] = {}
            if self.state_buffer.tag_names:
                tags = self.state_buffer.tags[:len(self.state_buffer)]
                names = self.state_buffer.tag_names
                monitor_means = tagged_report(self.monitor_success_rate, tags, names, reduction="mean")
                monitor_probs = beta_sampling_probs(torch.tensor(list(monitor_means.values()), device=env.device), target=0.5, kappa=1)
                log["Metrics/MonitorSuccessRate"] = self.monitor_success_rate.mean().item()
                for i, name in enumerate(names):
                    log[f"Metrics/MonitorSuccessRate/{name}"] = monitor_means[name]
                    log[f"Metrics/MonitorSampleProb/{name}"] = monitor_probs[i].item()
                if self.state_buffer.success_rates is not None:
                    estimator_means = tagged_report(self.state_buffer.success_rates, tags, names, reduction="mean")
                    estimator_probs = beta_sampling_probs(torch.tensor(list(estimator_means.values()), device=env.device), target=0.5, kappa=1)
                    log["Metrics/EstimatorSuccessRate"] = self.state_buffer.success_rates.mean().item()
                    for i, name in enumerate(names):
                        log[f"Metrics/EstimatorSuccessRate/{name}"] = estimator_means[name]
                        log[f"Metrics/EstimatorSampleProb/{name}"] = estimator_probs[i].item()

        # 3. Optionally accumulate more states
        if keep_accumulating:
            env_ids = self._accumulate(env, env_ids, reset_term, reset_assets)

        # 4. Sample a slot and apply the state
        if env_ids.numel() > 0:
            if isinstance(self._sampling_cfg, BetaSamplingCfg):
                probs = beta_sampling_probs(self.success_rate, self._sampling_cfg.target, self._sampling_cfg.kappa, self._sampling_cfg.temperature)
                slot_idx = torch.multinomial(probs, len(env_ids), replacement=True).to(torch.int32)
            else:
                slot_idx = torch.randint(0, self.state_buffer.max_size, (len(env_ids),), device=env.device)
            self.sampled_slots[env_ids] = slot_idx.to(self.sampled_slots.dtype)
            factory_utils.set_reset_state(self._env, self.state_buffer.sample(slot_idx), env_ids, reset_assets, is_relative=True)

        # 5. Log metrics
        if report:
            if "log" not in env.extras:
                env.extras["log"] = {}
            env.extras["log"].update(log)  # type: ignore


class TermChoice(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.term_partitions: dict[str, EventTermCfg] = cfg.params["terms"]  # type: ignore
        self.num_partitions = len(self.term_partitions)
        self.term_samples = torch.zeros((env.num_envs,), dtype=torch.int, device=env.device)
        self.term_success_rate = torch.zeros(self.num_partitions, device=env.device)
        self._sampling_cfg = cfg.params.get("sampling", UniformSamplingCfg())
        needs_monitor = cfg.params.get("report", False) or isinstance(self._sampling_cfg, BetaSamplingCfg)
        if needs_monitor:
            monitor_cfg: SuccessMonitorCfg = cfg.params.get(
                "success_monitor_cfg",
                SuccessMonitorCfg(num_monitored_data=self.num_partitions, device=env.device),
            )  # type: ignore
            monitor_cfg.num_monitored_data = self.num_partitions
            monitor_cfg.device = env.device
            self.success_monitor = monitor_cfg.class_type(monitor_cfg, self.term_success_rate)
        else:
            self.success_monitor = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        terms: dict[str, ManagerTermBase],
        sampling: UniformSamplingCfg | BetaSamplingCfg = UniformSamplingCfg(),
        success_monitor_cfg: SuccessMonitorCfg | None = None,
        report: bool = False,
    ) -> None:
        if self.num_partitions == 0:
            return
        if report:
            log = {f"Metrics/SuccessRate/{name}": self.term_success_rate[i].item() for i, name in enumerate(self.term_partitions.keys())}
            log["Metrics/SuccessRate"] = self.term_success_rate.mean().item()
        if self.success_monitor:
            success = env.termination_manager.get_term_cfg("progress_context").func.is_success
            self.success_monitor.success_update(self.term_samples[env_ids], success[env_ids].float())

        if isinstance(self._sampling_cfg, BetaSamplingCfg):
            rates = eval(self._sampling_cfg.success_rate_bind)  # noqa: S307
            probs = beta_sampling_probs(rates, self._sampling_cfg.target, self._sampling_cfg.kappa, self._sampling_cfg.temperature)
            self.term_samples[env_ids] = torch.multinomial(probs, len(env_ids), replacement=True).to(torch.int32)
            if report:
                log.update({f"Metrics/SampleProb/{name}": probs[i].item() for i, name in enumerate(self.term_partitions.keys())})
        else:
            self.term_samples[env_ids] = torch.randint(0, self.num_partitions, (env_ids.size(0),), device=env_ids.device, dtype=self.term_samples.dtype)

        i = 0
        for _, term_cfg in self.term_partitions.items():
            term_ids = env_ids[self.term_samples[env_ids] == i]
            if term_ids.numel() > 0:
                term_cfg.func(env, term_ids, **term_cfg.params)
            i += 1

        if report:
            if "log" not in env.extras:
                env.extras["log"] = {}
            env.extras["log"].update(log)  # type: ignore
        


class ChainedResetTerms(ManagerTermBase):

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.terms: dict[str, EventTermCfg] = cfg.params["terms"]  # type: ignore

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        terms: dict[str, callable],
        probability: float = 1.0,
    ) -> None:
        keep = torch.rand(env_ids.size(0), device=env_ids.device) < probability
        if not keep.any():
            return
        env_ids_to_reset = env_ids[keep]
        for _, term in terms.items():
            term.func(env, env_ids_to_reset, **term.params)  # type: ignore


# @torch.jit.script
def interpolate_grasp_quat(
    held_asset_grasp_point_quat_w: torch.Tensor,
    grasped_object_quat_in_ee_frame: torch.Tensor,
    secondary_z_axis: torch.Tensor | None = None,
    secondary_z_axis_weight: torch.Tensor | float = 0.3,
) -> torch.Tensor:
    if secondary_z_axis is not None:
        table_z_axis = secondary_z_axis
        leg_grasp_z_axis = math_utils.matrix_from_quat(held_asset_grasp_point_quat_w)[..., 2]
        leg_grasp_point_z_axis = math_utils.normalize((1.0 - secondary_z_axis_weight) * leg_grasp_z_axis + secondary_z_axis_weight * table_z_axis)

        # determine the closest y axis
        leg_grasp_point_y_axis_case1 = leg_grasp_point_z_axis.cross(table_z_axis)
        leg_grasp_point_y_axis_case2 = -leg_grasp_point_z_axis.cross(table_z_axis)
        robot_grasp_y = math_utils.matrix_from_quat(grasped_object_quat_in_ee_frame)[..., 1]
        dot_dist_1 = (robot_grasp_y * leg_grasp_point_y_axis_case1).sum(dim=1)
        dot_dist_2 = (robot_grasp_y * leg_grasp_point_y_axis_case2).sum(dim=1)
        leg_grasp_point_y_axis = torch.where(
            (dot_dist_1 > dot_dist_2).view(-1, 1), leg_grasp_point_y_axis_case1, leg_grasp_point_y_axis_case2
        )
        leg_grasp_point_x_axis = leg_grasp_point_y_axis.cross(leg_grasp_point_z_axis)
    else:
        leg_grasp_z_axis = math_utils.matrix_from_quat(held_asset_grasp_point_quat_w)[..., 2]
        leg_grasp_point_z_axis = math_utils.normalize(leg_grasp_z_axis)
        robot_grasp_y = math_utils.matrix_from_quat(grasped_object_quat_in_ee_frame)[..., 1]

        leg_grasp_x_axis = math_utils.matrix_from_quat(held_asset_grasp_point_quat_w)[..., 0]
        leg_grasp_y_axis = math_utils.matrix_from_quat(held_asset_grasp_point_quat_w)[..., 1]
        leg_grasp_neg_x_axis = -leg_grasp_x_axis.clone()
        leg_grasp_neg_y_axis = -leg_grasp_y_axis.clone()

        # Stack all candidate axes into a tensor of shape (num_envs, 4, 3)
        candidate_axes = torch.stack(
            [leg_grasp_x_axis, leg_grasp_neg_x_axis, leg_grasp_y_axis, leg_grasp_neg_y_axis], dim=1
        )  # shape: (N, 4, 3)

        # Compute dot products between each candidate axis and robot_grasp_y.
        # robot_grasp_y is (N, 3) and unsqueezed to (N, 1, 3) so that broadcasting gives (N, 4, 3).
        dot_products = (candidate_axes * robot_grasp_y.unsqueeze(1)).sum(dim=2)  # shape: (N, 4)
        # Get the index of the candidate with the maximum dot product for each environment.
        max_indices = dot_products.argmax(dim=1)  # shape: (N,)

        # Index the best candidate out.
        leg_grasp_point_y_axis = candidate_axes[torch.arange(candidate_axes.shape[0]), max_indices]  # shape: (N, 3)
        leg_grasp_point_x_axis = leg_grasp_point_y_axis.cross(leg_grasp_z_axis)

    # compose to grasp quat
    des_leg_grasp_quat_w = math_utils.quat_from_matrix(
        torch.stack((leg_grasp_point_x_axis, leg_grasp_point_y_axis, leg_grasp_point_z_axis), dim=1).transpose(1, 2)
    )
    return des_leg_grasp_quat_w