# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import math as math_utils

from ..assembly_keypoints import NIST_BOARD_CFG
from ..assembly_profile import AssemblyProfile, UniformPoseNoise
from ..assembly_profile_cfg import AssemblyProfileCfg, UniformPoseNoiseCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction

    from ..assembly_keypoints import Offset


def reset_fixed_asset_uniform(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_map: dict[str, str],
    pose_range: dict[str, tuple[float, float]],
):
    """Randomize the primary fixed asset uniformly about its nominal board pose.

    Samples ``pose_range`` about the pose the fixed asset would occupy on the board at its default
    placement, then writes it directly. Placing the fixed asset first (instead of deriving it from a
    yaw-randomized board) turns its distribution from a ring/donut into a filled uniform patch; the
    board is seated under it afterward by :func:`reset_board_under_fixed_asset`.

    Args:
        asset_map: Mapping from scene entity key to :class:`NistBoardKeyPointsCfg` attribute name.
            The ``"fixed_asset"`` entry selects the board keypoint used as the nominal center.
        pose_range: Per-axis ``(min, max)`` sample ranges, keys ``x``/``y``/``z`` [m] and
            ``roll``/``pitch``/``yaw`` [rad]. Missing axes stay at the nominal value.
    """
    nistboard: RigidObject = env.scene["nistboard"]
    fixed_asset: Articulation | RigidObject = env.scene["fixed_asset"]
    keypoint: Offset = getattr(NIST_BOARD_CFG, asset_map["fixed_asset"])

    board_default = wp.to_torch(nistboard.data.default_root_state)[env_ids]
    board_pos = board_default[:, 0:3] + env.scene.env_origins[env_ids]
    nominal_pos, nominal_quat = keypoint.combine(board_pos, board_default[:, 3:7])

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)
    new_pos = nominal_pos + samples[:, 0:3]
    new_quat = math_utils.quat_mul(
        nominal_quat, math_utils.quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5])
    )

    fixed_asset.write_root_pose_to_sim(torch.cat([new_pos, new_quat], dim=1), env_ids=env_ids)
    fixed_asset.write_root_velocity_to_sim(
        torch.zeros_like(wp.to_torch(fixed_asset.data.root_vel_w)[env_ids]), env_ids=env_ids
    )


def reset_board_under_fixed_asset(env: ManagerBasedRLEnv, env_ids: torch.Tensor, asset_map: dict[str, str]):
    """Seat the NIST board (and any extra board assets) under the already-placed fixed asset.

    Inverse of the board-first placement: solves the board root so ``board ∘ keypoint`` matches the
    fixed asset's current pose, then places every other asset in ``asset_map`` (e.g. non-held gears)
    at its own board keypoint. Run after :func:`reset_fixed_asset_uniform`.

    Args:
        asset_map: Mapping from scene entity key to :class:`NistBoardKeyPointsCfg` attribute name.
            ``"fixed_asset"`` selects the keypoint solved against; the rest ride along on the board.
    """
    nistboard: RigidObject = env.scene["nistboard"]
    fixed_asset: Articulation | RigidObject = env.scene["fixed_asset"]
    keypoint: Offset = getattr(NIST_BOARD_CFG, asset_map["fixed_asset"])

    fixed_pos = wp.to_torch(fixed_asset.data.root_pos_w)[env_ids]
    fixed_quat = wp.to_torch(fixed_asset.data.root_quat_w)[env_ids]
    board_pos, board_quat = keypoint.subtract(fixed_pos, fixed_quat)
    nistboard.write_root_pose_to_sim(torch.cat([board_pos, board_quat], dim=1), env_ids=env_ids)
    nistboard.write_root_velocity_to_sim(
        torch.zeros_like(wp.to_torch(nistboard.data.root_vel_w)[env_ids]), env_ids=env_ids
    )

    for scene_key, keypoint_attr in asset_map.items():
        if scene_key == "fixed_asset":
            continue
        extra: Articulation | RigidObject = env.scene[scene_key]
        extra_pos, extra_quat = getattr(NIST_BOARD_CFG, keypoint_attr).combine(board_pos, board_quat)
        extra.write_root_pose_to_sim(torch.cat([extra_pos, extra_quat], dim=1), env_ids=env_ids)
        extra.write_root_velocity_to_sim(torch.zeros_like(wp.to_torch(extra.data.root_vel_w)[env_ids]), env_ids=env_ids)


_PROFILE_CACHE: dict[int, AssemblyProfile] = {}
_POSE_NOISE_CACHE: dict[int, UniformPoseNoise] = {}


def _sweep_assembly_fraction(lo: float, hi: float, step: float = 0.001) -> Generator[tuple[float, float]]:
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
    pose_noise: UniformPoseNoiseCfg | None = None,
    debug_term: bool = False,
):
    """Seat the held asset somewhere along the assembly path of the fixed asset.

    Samples a fraction of the way along ``assembly_profile`` — ``0`` fully assembled, ``1`` at the
    point where the held asset enters the fixed one — and writes the held asset so its
    ``held_asset_align_offset`` lands on the sampled pose.

    Args:
        assembly_profile: Geometry of the assembly path, in the fixed asset frame.
        held_asset_align_offset: Point on the held asset placed on the path, e.g. the peg tip.
        assembly_fraction_range: ``(lo, hi)`` fraction range to sample. Values above ``1`` continue
            past the entry along the same direction.
        fixed_asset_cfg: The asset the path is measured against.
        held_asset_cfg: The asset being placed.
        pose_noise: Extra uniform pose noise on top of the sampled path pose, in the fixed asset
            frame — position [m] and euler angles [rad]. ``None`` places exactly on the path. The
            profile's own per-segment ``start_sampler`` still applies; this stacks on top of it.
        debug_term: Sweep the fraction range back and forth, rendering each step, instead of
            drawing one sample. Development aid only.
    """
    profile = _PROFILE_CACHE.get(id(assembly_profile))
    if profile is None:
        profile = assembly_profile.class_type(assembly_profile)
        _PROFILE_CACHE[id(assembly_profile)] = profile

    noise = None
    if pose_noise is not None:
        noise = _POSE_NOISE_CACHE.get(id(pose_noise))
        if noise is None:
            noise = pose_noise.class_type(pose_noise)
            _POSE_NOISE_CACHE[id(pose_noise)] = noise

    fixed_asset: RigidObject = env.scene[fixed_asset_cfg.name]
    held_asset: Articulation = env.scene[held_asset_cfg.name]

    fractions = _sweep_assembly_fraction(*assembly_fraction_range) if debug_term else iter([assembly_fraction_range])
    for frac_range in fractions:
        pos_offset, quat_offset = profile.sample(frac_range, len(env_ids), env.device)
        if noise is not None:
            # Same convention as the profile's own start samplers: the align point is displaced in
            # the fixed asset frame and the part is tilted about that displaced point.
            noise_pos, noise_quat = noise(len(env_ids), env.device)
            pos_offset = pos_offset + noise_pos
            quat_offset = math_utils.quat_mul(noise_quat, quat_offset)
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
    grasp_pos_w, grasp_quat_w = gripper_grasp_offset.combine(end_effector_pos_w, end_effector_quat_w)

    # Randomize the grasp target (at the grasp point) BEFORE solving for the asset root, so the
    # pose noise pivots about the grasp point and the graspable frame stays coincident with the
    # gripper. Applying the noise to the root pose afterward rotates about the (offset) root, which
    # swings the graspable point off the gripper — the asset stops being held at the grasp point.
    range_list = [held_asset_inhand_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)
    grasp_pos_w, grasp_quat_w = math_utils.combine_frame_transforms(
        grasp_pos_w,
        grasp_quat_w,
        samples[:, :3],
        math_utils.quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5]),
    )

    new_pos_w, new_quat_w = held_asset_graspable_offset.subtract(grasp_pos_w, grasp_quat_w)

    held_asset.write_root_link_pose_to_sim(torch.cat([new_pos_w, new_quat_w], dim=1), env_ids=env_ids)  # type: ignore
    held_asset.write_root_com_velocity_to_sim(
        wp.to_torch(held_asset.data.default_root_state)[env_ids, 7:], env_ids=env_ids
    )  # type: ignore


def grasp_held_asset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg,
    held_asset_diameter: float,
    flexible_angle: bool = True,
) -> None:
    robot: Articulation = env.scene[robot_cfg.name]
    joint_pos = wp.to_torch(robot.data.joint_pos)[:, robot_cfg.joint_ids][env_ids].clone()
    min_angle = held_asset_diameter / 2 * 1.05
    if flexible_angle:
        max_angle = wp.to_torch(robot.data.joint_pos_limits)[0, robot_cfg.joint_ids[0], 1]
        joint_pos[:] = (torch.rand((len(env_ids),), device=env.device) * (max_angle - min_angle) + min_angle).unsqueeze(
            1
        )
    else:
        joint_pos[:] = min_angle

    robot.write_joint_position_to_sim(joint_pos, robot_cfg.joint_ids, env_ids)  # type: ignore


class reset_end_effector_around_asset(ManagerTermBase):
    """Drive the gripper onto a sampled pose around an asset with a differential IK solve.

    Reports whether each solve actually landed on its target through :attr:`is_valid`, which
    :class:`~isaaclab_tasks.contrib.nist.utils.ChainedResetTerms` collects and the reset
    accumulator uses to reject the state instead of banking a missed grasp.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        fixed_asset_cfg: SceneEntityCfg = cfg.params.get("fixed_asset_cfg")  # type: ignore
        fixed_asset_offset: Offset = cfg.params.get("fixed_asset_offset")  # type: ignore
        pose_range_b: dict[str, tuple[float, float]] = cfg.params.get("pose_range_b")  # type: ignore
        robot_ik_cfg: SceneEntityCfg = cfg.params.get("robot_ik_cfg", SceneEntityCfg("robot"))
        robot_ik_body_offset: Offset = cfg.params.get("robot_ik_body_offset")  # type: ignore

        range_list = [pose_range_b.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.wrist_idx = 6
        self.ranges = torch.tensor(range_list, device=env.device)
        self.fixed_asset: Articulation | RigidObject = env.scene[fixed_asset_cfg.name]
        self.fixed_asset_offset: Offset = fixed_asset_offset
        self.robot: Articulation = env.scene[robot_ik_cfg.name]
        self.robot_ik_body_offset: Offset = robot_ik_body_offset
        self.joint_ids: list[int] | slice = robot_ik_cfg.joint_ids

        self.robot_ik_solver_cfg = DifferentialInverseKinematicsActionCfg(
            asset_name=robot_ik_cfg.name,
            joint_names=robot_ik_cfg.joint_names,  # type: ignore
            body_name=robot_ik_cfg.body_names,  # type: ignore
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            scale=1.0,
        )
        self.solver: DifferentialInverseKinematicsAction = None  # type: ignore
        self.grasp_angle_range = (0.3, 0.7)
        self.is_physx = "physx" in env.sim.physics_manager.__name__.lower()
        self.is_valid = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        """Whether the last solve for each env reached its target within ``pose_tolerance``."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        fixed_asset_cfg: SceneEntityCfg,
        fixed_asset_offset: Offset,
        pose_range_b: dict[str, tuple[float, float]],
        robot_ik_cfg: SceneEntityCfg,
        robot_ik_body_offset: Offset,
        upright_gripper: bool = False,
        ik_iterations: tuple[int, int] = (5, 10),
        pose_tolerance: tuple[float, float] | None = None,
    ) -> None:
        """Solve the gripper onto a pose sampled around ``fixed_asset_offset``.

        Args:
            fixed_asset_cfg: Asset the target pose is sampled around. Not necessarily the fixed
                asset — the strategies that grasp a part already on the board point this at the
                held asset.
            fixed_asset_offset: Keypoint on that asset the sample is centered on.
            pose_range_b: Per-axis ``(min, max)`` sample ranges about the keypoint, keys
                ``x``/``y``/``z`` [m] and ``roll``/``pitch``/``yaw`` [rad].
            robot_ik_cfg: The robot, its IK joints and the body being solved for.
            robot_ik_body_offset: Grasp frame of the gripper relative to that body.
            upright_gripper: Keep only the keypoint's yaw, so the sample is taken about a frame
                that is upright in the robot root frame rather than tilted with the asset.
            ik_iterations: ``(lo, hi)`` range the per-env iteration count is drawn from. Each
                iteration takes a quarter step, so the residual shrinks by ~0.75 per iteration.
            pose_tolerance: ``(position [m], orientation [rad])`` residual above which the solve is
                reported through :attr:`is_valid` as having missed. ``None`` reports every solve as
                valid, keeping whatever pose the iteration budget reached.
        """
        if self.solver is None:
            self.solver = self.robot_ik_solver_cfg.class_type(self.robot_ik_solver_cfg, env)
        fixed_keypoint_pos_w, fixed_keypoint_quat_w = self.fixed_asset_offset.apply(self.fixed_asset)
        samples = math_utils.sample_uniform(self.ranges[:, 0], self.ranges[:, 1], (len(env_ids), 6), device=env.device)
        pos_b, quat_b = self.solver._compute_frame_pose()
        robot_root_pos_w = wp.to_torch(self.robot.data.root_link_pos_w)[env_ids]
        robot_root_quat_w = wp.to_torch(self.robot.data.root_link_quat_w)[env_ids]
        grasp_reference_quat_w = fixed_keypoint_quat_w[env_ids]
        if upright_gripper:
            grasp_reference_quat_b = math_utils.quat_mul(math_utils.quat_inv(robot_root_quat_w), grasp_reference_quat_w)
            _, _, yaw = math_utils.euler_xyz_from_quat(grasp_reference_quat_b)
            zero = torch.zeros_like(yaw)
            grasp_reference_quat_w = math_utils.quat_mul(
                robot_root_quat_w, math_utils.quat_from_euler_xyz(zero, zero, yaw)
            )
        # for those non_reset_id, we will let ik solve for its current position
        grasp_pos_w, grasp_quat_w = math_utils.combine_frame_transforms(
            fixed_keypoint_pos_w[env_ids],
            grasp_reference_quat_w,
            samples[:, :3],
            math_utils.quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5]),
        )
        body_pos_w, body_quat_w = self.robot_ik_body_offset.subtract(grasp_pos_w, grasp_quat_w)

        pos_b[env_ids], quat_b[env_ids] = math_utils.subtract_frame_transforms(
            robot_root_pos_w,
            robot_root_quat_w,
            body_pos_w,
            body_quat_w,
        )
        self.solver.process_actions(torch.cat([pos_b, quat_b], dim=1))

        # Error Rate 75% ^ 10 = 0.05 (final error)
        lo, hi = ik_iterations
        k = int(torch.randint(low=lo, high=hi + 1, size=(1,)).item())
        # Clamp every write to the joint limits: unconstrained DLS steps walk joints past
        # their limits near workspace edges, and beyond-limit states written to sim produce
        # huge limit-constraint impulses (NaN on the mjwarp Newton solver at first step).
        limits = wp.to_torch(self.robot.data.joint_pos_limits)[env_ids][:, self.joint_ids]
        for _ in range(k):
            self.solver.apply_actions()
            delta_joint_pos = 0.25 * (
                wp.to_torch(self.robot.data.joint_pos_target)[env_ids] - wp.to_torch(self.robot.data.joint_pos)[env_ids]
            )
            new_joint_pos = (delta_joint_pos + wp.to_torch(self.robot.data.joint_pos)[env_ids])[:, self.joint_ids]
            self.robot.write_joint_position_to_sim(
                position=torch.clamp(new_joint_pos, limits[..., 0], limits[..., 1]),
                joint_ids=self.joint_ids,
                env_ids=env_ids,  # type: ignore
            )

        # A target near a workspace edge, a singularity, or a joint limit is one the DLS steps
        # never reach however long they run, so the iteration budget alone does not guarantee the
        # gripper ended up where it was sent. Measure the residual and let the caller drop the env.
        if pose_tolerance is None:
            self.is_valid[env_ids] = True
        else:
            reached_pos_b, reached_quat_b = self.solver._compute_frame_pose()
            pos_error = torch.norm(reached_pos_b[env_ids] - pos_b[env_ids], dim=-1)
            rot_error = math_utils.quat_error_magnitude(reached_quat_b[env_ids], quat_b[env_ids])
            self.is_valid[env_ids] = (pos_error < pose_tolerance[0]) & (rot_error < pose_tolerance[1])

        # wrist_low  = self.robot.data.joint_pos_limits[env_ids, self.wrist_idx, 0]
        # wrist_high = self.robot.data.joint_pos_limits[env_ids, self.wrist_idx, 1]
        # wrist_pos = (wrist_low + (wrist_high - wrist_low) * torch.rand_like(wrist_low)).view(len(env_ids), -1)
        # self.robot.write_joint_position_to_sim(position=wrist_pos, joint_ids=self.wrist_idx, env_ids=env_ids)
        if self.is_physx:
            self.robot.root_physx_view.get_jacobians()


# @torch.jit.script
