from __future__ import annotations

import torch
from torch.nn import functional as F


# -----------------------------
# Direction / exploration state
# -----------------------------

def directional_alignment(position_b: torch.Tensor, velocity_b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between velocity and target position."""
    return F.cosine_similarity(velocity_b[..., :3], position_b[..., :3], dim=-1, eps=1e-6)


def forwardness(position_b: torch.Tensor, velocity_b: torch.Tensor) -> torch.Tensor:
    """Your exploration kernel with forward preference."""
    cos_align = F.cosine_similarity(velocity_b[:, :3], position_b, dim=-1, eps=1e-6)
    speed = torch.linalg.vector_norm(velocity_b, ord=2, dim=-1)
    forward_comp = velocity_b[:, 0].clamp_min(0)
    forward_weight = forward_comp / (speed + 1e-6)
    return cos_align * forward_weight


# -----------------------------
# Effort / energy-like state
# -----------------------------

def mechanical_work_per_joint(
    applied_torque: torch.Tensor,
    joint_vel: torch.Tensor,
    dt: float | torch.Tensor
) -> torch.Tensor:
    """Per-joint absolute mechanical work contribution."""
    return (applied_torque * joint_vel).abs() * dt


def average_mechanical_work_per_joint(
    applied_torque: torch.Tensor,
    joint_vel: torch.Tensor,
    dt: float | torch.Tensor
) -> torch.Tensor:
    """Per-joint absolute mechanical work contribution."""
    return (applied_torque * joint_vel).abs() * dt


def total_mechanical_work(applied_torque: torch.Tensor, joint_vel: torch.Tensor, dt: float | torch.Tensor
                          ) -> torch.Tensor:
    """Sum of work over joints."""
    return mechanical_work_per_joint(applied_torque, joint_vel, dt).sum(dim=1)


def per_body_incoming_wrench(body_incoming_joint_wrench_b: torch.Tensor) -> torch.Tensor:
    """Per-body L2 norm of incoming joint wrench. Input: (B, num_bodies, 6)."""
    return torch.norm(body_incoming_joint_wrench_b, dim=-1)  # (B, num_bodies)


def total_incoming_wrench(body_incoming_joint_wrench_b: torch.Tensor) -> torch.Tensor:
    """Sum of incoming wrench magnitudes across bodies."""
    return incoming_wrench_l2(body_incoming_joint_wrench_b).sum(dim=1)


# -----------------------------
# Stall / progress state
# -----------------------------

def base_speed(root_lin_vel_b: torch.Tensor) -> torch.Tensor:
    """Full 3D base speed."""
    return torch.linalg.vector_norm(root_lin_vel_b, ord=2, dim=-1)


# -----------------------------
# Contact / impact / slip state
# -----------------------------

def feet_lin_acc_l2(body_lin_acc_w: torch.Tensor, body_ids: torch.Tensor) -> torch.Tensor:
    """
    Sum of squared linear acceleration over selected bodies.

    body_lin_acc_w: (B, num_bodies, 3) or (B, ..., num_bodies, 3)
    """
    feet_acc = body_lin_acc_w[..., body_ids, :]
    return torch.sum(feet_acc * feet_acc, dim=(1, 2))


def feet_ang_acc_l2(body_ang_acc_w: torch.Tensor, body_ids: torch.Tensor) -> torch.Tensor:
    """Same as above but for angular acceleration."""
    feet_acc = body_ang_acc_w[..., body_ids, :]
    return torch.sum(feet_acc * feet_acc, dim=(1, 2))


def foot_planar_speed(body_com_lin_vel_w: torch.Tensor, body_ids: torch.Tensor) -> torch.Tensor:
    """
    Planar speed per selected body.

    body_com_lin_vel_w: (B, num_bodies, 3)
    returns: (B, K) speeds for each id in body_ids.
    """
    v_xy = body_com_lin_vel_w[:, body_ids, :2]
    return torch.linalg.vector_norm(v_xy, dim=-1)


# -----------------------------
# Posture / height / joints
# -----------------------------


def joint_position_error(joint_pos: torch.Tensor, default_joint_pos: torch.Tensor) -> torch.Tensor:
    """‖q - q_default‖ per environment."""
    return torch.linalg.vector_norm(joint_pos - default_joint_pos, dim=1)


# -----------------------------
# Gait-related state (contact timing)
# -----------------------------

def gait_sync_se(
    air_time: torch.Tensor,
    contact_time: torch.Tensor,
    foot_0: int,
    foot_1: int,
) -> torch.Tensor:
    """
    Squared error between air/contact times of two feet (sync term).
    air_time/contact_time: (B, num_feet)
    returns: (B,)
    """
    se_air = (air_time[:, foot_0] - air_time[:, foot_1]).pow(2)
    se_contact = (contact_time[:, foot_0] - contact_time[:, foot_1]).pow(2)
    return se_air + se_contact


def gait_async_se(
    air_time: torch.Tensor,
    contact_time: torch.Tensor,
    foot_0: int,
    foot_1: int,
) -> torch.Tensor:
    """
    Squared error for anti-synchrony between two feet.
    air_time/contact_time: (B, num_feet)
    returns: (B,)
    """
    se_0 = (air_time[:, foot_0] - contact_time[:, foot_1]).pow(2)
    se_1 = (contact_time[:, foot_0] - air_time[:, foot_1]).pow(2)
    return se_0 + se_1


def air_contact_times(
    current_air_time: torch.Tensor,
    current_contact_time: torch.Tensor,
    body_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract air and contact times for selected feet.

    returns:
        air:    (B, K)
        contact:(B, K)
    """
    air = current_air_time[:, body_ids]
    contact = current_contact_time[:, body_ids]
    return air, contact


def air_contact_variance(
    last_air_time: torch.Tensor,
    last_contact_time: torch.Tensor,
    body_ids: torch.Tensor,
    clip_max: float = 0.5,
) -> torch.Tensor:
    """
    Variance of clipped air/contact times across feet.

    returns: (B,)
    """
    air = last_air_time[:, body_ids].clamp_max(clip_max)
    contact = last_contact_time[:, body_ids].clamp_max(clip_max)
    return air.var(dim=1) + contact.var(dim=1)
