# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import TerminationTermCfg as DoneTermCfg
from . import states

# local cache for resolved gait body pairs to avoid repeated name lookups
_GAIT_PAIR_CACHE: dict[str, dict[tuple[str, ...], tuple[int, int]]] = {}

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import Articulation
    from isaaclab.sensors import ContactSensor

"""
MDP terminations.
"""


def success(
    env: ManagerBasedRLEnv,
    thresh: list[float, float, float, float],
    command: str = "goal_point",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[robot_cfg.name]
    cmd: torch.Tensor = env.command_manager.get_command(command)
    dist = cmd[:, :3].norm(2, -1)
    head = cmd[:, 3].abs()
    speed = asset.data.body_lin_vel_w[:, robot_cfg.body_ids].norm(2, dim=-1).amax(dim=1)
    joint_pos = asset.data.joint_pos[:, robot_cfg.joint_ids] - asset.data.default_joint_pos[:, robot_cfg.joint_ids]
    joint_pos_diff = torch.abs(joint_pos).amax(dim=1)
    return ((dist < thresh[0]) & (head < thresh[1])) & (speed < thresh[2]) & (joint_pos_diff < thresh[3])


class log(ManagerTermBase):

    def __init__(self, cfg: DoneTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        log_key = cfg.params.get("log_key")
        category = cfg.params.get("category", None)
        prefix = "Info"
        if isinstance(log_key, str):
            if log_key.startswith("eval"):
                log_key = eval(log_key[5:])
        if category is not None and isinstance(category, str):
            prefix = f"{prefix}/{category}"
        self.return_val = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        if isinstance(log_key, str):
            self.log = f"{prefix}/{log_key}"
        elif isinstance(log_key, list):
            self.log = [f"{prefix}/{key}" for key in log_key]
        else:
            raise KeyError("input key is neither str or list of str")
        self.func: callable = cfg.params.get("func")
        self.params: dict = {key: val for key, val in cfg.params.items() if key not in ["func", "log_key", "category"]}
        cfg.params = {}

    def __call__(self, env: ManagerBasedRLEnv):
        val = self.func(env, **self.params)
        env_log = env.extras["log"]
        if isinstance(self.log, str):
            env_log[self.log] = val
        elif isinstance(self.log, list):
            for i, key in enumerate(self.log):
                env_log[key] = float(val[i])
        return self.return_val


def mean_mech_energy_per_joint(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot: Articulation = env.scene[asset_cfg.name]
    applied_torque = robot.data.applied_torque[:, asset_cfg.joint_ids]
    joint_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
    work_per_joint = states.mechanical_work_per_joint(applied_torque, joint_vel, env.step_dt)
    return work_per_joint.mean(dim=0)


def total_average_mech_energy_per_joint(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot: Articulation = env.scene[asset_cfg.name]
    applied_torque = robot.data.applied_torque[:, asset_cfg.joint_ids]
    joint_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
    work_per_joint = states.mechanical_work_per_joint(applied_torque, joint_vel, env.step_dt)
    return work_per_joint.mean(dim=0).sum()


def mean_per_body_shock(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot: Articulation = env.scene[asset_cfg.name]
    per_body_incoming_wrench = torch.norm(robot.data.body_incoming_joint_wrench_b, dim=-1)
    return per_body_incoming_wrench.mean(dim=0)


def total_body_shock(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot: Articulation = env.scene[asset_cfg.name]
    per_body_incoming_wrench = torch.norm(robot.data.body_incoming_joint_wrench_b, dim=-1)
    return per_body_incoming_wrench.mean(dim=0).sum()


def forwardness(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    # Retrieve the robot and target data
    robot: Articulation = env.scene[asset_cfg.name]
    base_velocity = robot.data.root_lin_vel_b  # Robot's current base velocity vector
    speed = torch.linalg.vector_norm(base_velocity, ord=2, dim=-1)
    forward_comp = base_velocity[:, 0]
    forward_weight = forward_comp / (speed + 1e-6)
    return forward_weight.mean(dim=0)


def gait(
    env: ManagerBasedRLEnv,
    max_err: float,
    sync_pairs,
    async_pairs,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Scalar gait-quality metric in [0, 1].

    Users specify which foot pairs should be in sync (``sync_pairs``)
    and which should be anti-sync (``async_pairs``). The function
    computes a per-pair timing score and returns the average.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]  # type: ignore[name-defined]

    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time for gait metric computation.")

    air_time: torch.Tensor = contact_sensor.data.current_air_time
    contact_time: torch.Tensor = contact_sensor.data.current_contact_time

    def _resolve_pairs(pairs):
        if pairs is None:
            return []
        resolved = []
        for pair_names in pairs:
            if isinstance(pair_names, str):
                key = (pair_names,)
            else:
                key = tuple(pair_names)

            sensor_cache = _GAIT_PAIR_CACHE.setdefault(sensor_cfg.name, {})
            if key in sensor_cache:
                resolved.append(sensor_cache[key])
                continue

            body_ids, _ = contact_sensor.find_bodies(pair_names)
            if len(body_ids) < 2:
                raise ValueError(f"Pair {pair_names} did not resolve to at least two bodies.")

            pair_idx = (body_ids[0], body_ids[1])
            sensor_cache[key] = pair_idx
            resolved.append(pair_idx)
        return resolved

    sync_pairs_idx = _resolve_pairs(sync_pairs)
    async_pairs_idx = _resolve_pairs(async_pairs)

    if not sync_pairs_idx and not async_pairs_idx:
        raise ValueError("At least one sync or async pair must be provided for gait metric.")

    # map squared errors to scores in [0, 1]
    max_se = 2.0 * (max_err**2)
    if max_se <= 0.0:
        raise ValueError("Parameter 'max_err' must be positive for gait metric computation.")

    def _score_from_se(se: torch.Tensor) -> torch.Tensor:
        se_norm = torch.clamp(se / max_se, 0.0, 1.0)
        return 1.0 - se_norm

    scores = []
    for foot_0, foot_1 in sync_pairs_idx:
        se_sync = states.gait_sync_se(air_time, contact_time, foot_0, foot_1)
        scores.append(_score_from_se(se_sync))
    for foot_0, foot_1 in async_pairs_idx:
        se_async = states.gait_async_se(air_time, contact_time, foot_0, foot_1)
        scores.append(_score_from_se(se_async))

    gait_score = torch.stack(scores, dim=0).mean(dim=0)
    return gait_score
