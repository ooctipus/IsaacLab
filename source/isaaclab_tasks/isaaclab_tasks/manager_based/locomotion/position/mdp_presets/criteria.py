# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot-agnostic retarget validation criteria.

These criteria work with any articulated robot without requiring
robot-specific constants. For robot-specific criteria (e.g. HAA
joint limits), see the per-robot preset modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import newton
import torch
import warp as wp

if TYPE_CHECKING:
    from ..mdp.kinematics import NewtonKinematics
    from ..mdp.retarget.buffer import RetargetBuffer


@dataclass
class FootPositionError:
    """Criterion: FK foot positions must match contact targets within tolerance [m].

    Args:
        kin: :class:`NewtonKinematics` instance.
        foot_ids: Newton body indices for the feet.
        max_err: Maximum position error per foot [m].
    """

    kin: NewtonKinematics
    foot_ids: list[int] = field(default_factory=list)
    max_err: float = 0.02

    def __call__(self, buffer: RetargetBuffer, N: int) -> torch.Tensor:
        nc = len(self.foot_ids)
        nb = self.kin.model.body_count
        tpl = newton.ModelBuilder()
        tpl.add_usd(self.kin.usd_path, collapse_fixed_joints=False)
        bldr = newton.ModelBuilder()
        for _ in range(N):
            bldr.add_world(tpl)
        fk_m = bldr.finalize(device=buffer.device)
        jq_t = buffer.joint_q_result_t[:N].contiguous().view(-1)
        fk_m.joint_q = wp.from_torch(jq_t)
        st = fk_m.state()
        newton.eval_fk(
            fk_m, fk_m.joint_q,
            wp.zeros(fk_m.joint_dof_count, dtype=float, device=buffer.device), st,
        )
        body_q = wp.to_torch(st.body_q).view(N, nb, 7)  # type: ignore[arg-type]
        ct = buffer.contact_targets_t[:N * nc].view(N, nc, 3)
        idx = torch.tensor(self.foot_ids, device=buffer.device, dtype=torch.long)
        err = (body_q[:, idx, :3] - ct).norm(dim=-1).max(dim=-1).values
        return err <= self.max_err


@dataclass
class JointMargin:
    """Criterion: revolute joints must stay within ``margin`` of their limits.

    Args:
        kin: :class:`NewtonKinematics` instance.
        margin: Fraction of joint range to keep as safety margin.
    """

    kin: NewtonKinematics
    margin: float = 0.1

    def __call__(self, buffer: RetargetBuffer, N: int) -> torch.Tensor:
        jl = wp.to_torch(self.kin.model.joint_limit_lower)
        ju = wp.to_torch(self.kin.model.joint_limit_upper)
        lo, hi = jl[6:], ju[6:]
        n_rev = lo.shape[0]
        jq = buffer.joint_q_result_t[:N, 7:7 + n_rev]
        safe_lo = lo + self.margin * (hi - lo)
        safe_hi = hi - self.margin * (hi - lo)
        violation = ((safe_lo - jq).clamp(min=0) + (jq - safe_hi).clamp(min=0)).max(dim=-1).values
        return violation <= 0


@dataclass
class HaaLimit:
    """Criterion: HAA (hip abduction) joint angles must not exceed ``max_angle`` [rad].

    Resolves DOF indices at construction time from a joint name regex
    via :meth:`NewtonKinematics.find_joint_dof_indices`.

    Args:
        kin: :class:`NewtonKinematics` instance for joint name resolution.
        joint_pattern: Regex matching HAA joint names (e.g. ``".*hip_joint"``).
        max_angle: Maximum absolute HAA angle [rad].
    """

    kin: NewtonKinematics
    joint_pattern: str = ".*hip.*"
    max_angle: float = 0.87

    def __post_init__(self):
        self._haa_indices = self.kin.find_joint_dof_indices(self.joint_pattern)

    def __call__(self, buffer: RetargetBuffer, N: int) -> torch.Tensor:
        if not self._haa_indices:
            return torch.ones(N, device=buffer.device, dtype=torch.bool)
        n_rev = buffer.joint_q_result_t.shape[1] - 7
        haa_idx = torch.tensor(
            [i for i in self._haa_indices if i < n_rev],
            device=buffer.device, dtype=torch.long,
        )
        if haa_idx.numel() == 0:
            return torch.ones(N, device=buffer.device, dtype=torch.bool)
        jq_rev = buffer.joint_q_result_t[:N, 7:]
        return jq_rev[:, haa_idx].abs().max(dim=-1).values <= self.max_angle


@dataclass
class BaseZError:
    """Criterion: base z must not deviate more than ``max_err`` from target [m].

    Args:
        max_err: Maximum absolute base z deviation [m].
    """

    max_err: float = 0.3

    def __call__(self, buffer: RetargetBuffer, N: int) -> torch.Tensor:
        base_z = buffer.joint_q_result_t[:N, 2]
        target_z = buffer.base_target_pos_t[:N, 2]
        return (base_z - target_z).abs() <= self.max_err
