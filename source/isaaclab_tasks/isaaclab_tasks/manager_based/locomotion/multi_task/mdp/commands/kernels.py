import enum
import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_error_magnitude

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
def geometric_error(x_cur: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(x_cur - x_target, dim=-1)


def quaternion_error(q_cur: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
    return quat_error_magnitude(q_cur, q_target)


METRIC_KERNELS = (geometric_error, quaternion_error)


# --- state kernels (env -> x_cur) ---
def joint_position(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    articulation: Articulation = env.scene[asset_cfg.name]
    return articulation.data.joint_pos[env_ids, asset_cfg.joint_ids]


def joint_velocity(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    articulation: Articulation = env.scene[asset_cfg.name]
    return articulation.data.joint_vel[env_ids, asset_cfg.joint_ids]


def body_position(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    articulation: Articulation = env.scene[asset_cfg.name]
    body_pos = articulation.data.body_pos_w[env_ids, asset_cfg.body_ids]
    env_origins = env.scene.env_origins[env_ids]
    if env_origins.ndim == body_pos.ndim - 1:
        env_origins = env_origins.unsqueeze(-2)
    return body_pos - env_origins


def body_quaternion(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    articulation: Articulation = env.scene[asset_cfg.name]
    return articulation.data.body_quat_w[env_ids, asset_cfg.body_ids]


def body_lin_velocity(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    articulation: Articulation = env.scene[asset_cfg.name]
    return articulation.data.body_lin_vel_w[env_ids, asset_cfg.body_ids]


def body_ang_velocity(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    articulation: Articulation = env.scene[asset_cfg.name]
    return articulation.data.body_ang_vel_w[env_ids, asset_cfg.body_ids]


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
    """Uniform sampler for padded interleaved [min, range] pairs.

    params: [..., 2*Dmax], padded with zeros at the end.
      pair i: [min_i, range_i]
    returns: [..., Dmax]
    """
    last = params.shape[-1]
    Dmax = last // 2
    pairs = params.view(*params.shape[:-1], Dmax, 2)
    mn = pairs[..., 0]
    rg = pairs[..., 1]
    return mn + torch.rand_like(mn) * rg


SAMPLER_KERNELS = (uniform,)
