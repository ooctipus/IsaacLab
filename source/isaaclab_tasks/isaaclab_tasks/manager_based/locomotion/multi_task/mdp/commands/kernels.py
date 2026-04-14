import enum
import torch
import warp as wp
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import axis_angle_from_quat, quat_inv, quat_mul

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


class ACTIVATION_KERNEL_ID(enum.IntEnum):
    TANH = 0
    LESS = 1
    GREATER = 2


class METRIC_KERNEL_ID(enum.IntEnum):
    GEOMETRIC = 0
    QUATERNION = 1


class STATE_KERNEL_ID(enum.IntEnum):
    JOINT_POS = 0
    JOINT_VEL = 1
    BODY_POS = 2
    BODY_QUAT = 3
    BODY_LIN_VEL = 4
    BODY_ANG_VEL = 5


class SAMPLER_KERNEL_ID(enum.IntEnum):
    UNIFORM = 0


# --- activation kernels (error -> score/predicate) ---
def tanh_kernel(error: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.tanh(error / std)


def less_kernel(error: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    return error < threshold


def greater_kernel(error: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    return error > threshold


ACTIVATION_KERNELS = (tanh_kernel, less_kernel, greater_kernel)


# --- metric kernels (x_cur, x_target -> scalar error) ---
def geometric_error(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(x, dim=-1)


def quaternion_error(quat: torch.Tensor) -> torch.Tensor:
    assert quat.shape[-1] == 4
    angle_axis = axis_angle_from_quat(quat)
    return torch.linalg.vector_norm(angle_axis, dim=-1)


METRIC_KERNELS = (geometric_error, quaternion_error)


# --- delta kernels (order: that - this)---
def geometric_subtract(x_cur: torch.Tensor, x_tgt: torch.Tensor) -> torch.Tensor:
    return x_tgt - x_cur


def quaternion_subtract(quat_cur: torch.Tensor, quat_tgt: torch.Tensor) -> torch.Tensor:
    assert quat_cur.shape[-1] == 4 and quat_tgt.shape[-1] == 4
    return quat_mul(quat_inv(quat_cur), quat_tgt)


DELTA_KERNELS = (geometric_subtract, quaternion_subtract)


# --- state kernels (env -> x_cur) ---
def joint_position(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    articulation: Articulation = env.scene[asset_cfg.name]
    return wp.to_torch(articulation.data.joint_pos)[env_ids, asset_cfg.joint_ids]


def joint_velocity(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    articulation: Articulation = env.scene[asset_cfg.name]
    return wp.to_torch(articulation.data.joint_vel)[env_ids, asset_cfg.joint_ids]


def body_position(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    articulation: Articulation = env.scene[asset_cfg.name]
    body_pos = wp.to_torch(articulation.data.body_pos_w)[env_ids, asset_cfg.body_ids]
    env_origins = env.scene.env_origins[env_ids]
    if env_origins.ndim == body_pos.ndim - 1:
        env_origins = env_origins.unsqueeze(-2)
    return body_pos - env_origins


def body_quaternion(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    articulation: Articulation = env.scene[asset_cfg.name]
    return wp.to_torch(articulation.data.body_quat_w)[env_ids, asset_cfg.body_ids]


def body_lin_velocity(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    articulation: Articulation = env.scene[asset_cfg.name]
    return wp.to_torch(articulation.data.body_lin_vel_w)[env_ids, asset_cfg.body_ids]


def body_ang_velocity(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    articulation: Articulation = env.scene[asset_cfg.name]
    return wp.to_torch(articulation.data.body_ang_vel_w)[env_ids, asset_cfg.body_ids]


STATE_KERNELS = (
    joint_position,
    joint_velocity,
    body_position,
    body_quaternion,
    body_lin_velocity,
    body_ang_velocity,
)


# --- sampler kernels (params -> target) ---
def uniform(params: torch.Tensor) -> torch.Tensor:
    # params: [..., 2*Dmax]
    mn = params[..., 0::2]  # [..., Dmax]
    rg = params[..., 1::2]  # [..., Dmax]
    return mn + torch.rand_like(mn) * rg


SAMPLER_KERNELS = (uniform,)
