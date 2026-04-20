# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Anymal-C robot preset. Activate with ``presets=anymal_c``."""

from __future__ import annotations

__all__: list[str] = []

import newton
import torch
import warp as wp
from isaaclab.assets import ArticulationCfg

import isaaclab_assets.robots.anymal as anymal

from .robot_presets import (
    AsyncFootPairsCfg,
    BaseBodyNameCfg,
    BaseContactBodyNamesCfg,
    ExperimentNameCfg,
    FootBodyNamesCfg,
    HeightScannerPrimPathCfg,
    NonFootBodyNamesCfg,
    RobotArticulationCfg,
    SyncFootPairsCfg,
)

_ANYMAL_C_CFG: ArticulationCfg = anymal.ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
_ANYMAL_C_CFG.spawn.usd_path = (  # type: ignore[attr-defined]
    "https://uwlab-assets.s3.us-west-004.backblazeb2.com/Robots/ANYbotics/ANYmal-C/anymal_c.usd"
)

RobotArticulationCfg.anymal_c = _ANYMAL_C_CFG
HeightScannerPrimPathCfg.anymal_c = "{ENV_REGEX_NS}/Robot/base"
BaseBodyNameCfg.anymal_c = "base"
BaseContactBodyNamesCfg.anymal_c = "base"
FootBodyNamesCfg.anymal_c = ".*FOOT.*"
NonFootBodyNamesCfg.anymal_c = "^(?!.*(?:(FOOT))).*$"
AsyncFootPairsCfg.anymal_c = (
    ("LF_FOOT", "RF_FOOT"),
    ("RH_FOOT", "LH_FOOT"),
    ("LF_FOOT", "LH_FOOT"),
    ("RF_FOOT", "RH_FOOT"),
)
SyncFootPairsCfg.anymal_c = (("LF_FOOT", "RH_FOOT"), ("RF_FOOT", "LH_FOOT"))
ExperimentNameCfg.anymal_c = "anymal_c_position_command"


# ---------------------------------------------------------------------------
# Retarget validation criteria for ANYmal-C
# ---------------------------------------------------------------------------

# HAA (hip abduction) DOF indices within the revolute joints (0-based after
# the 7 free-root coords, i.e. index into joint_q[7:])
ANYMAL_C_HAA_INDICES = [0, 3, 6, 9]


def foot_position_error(
    kin: object,
    foot_ids: list[int],
    max_err: float = 0.02,
):
    """Criterion: FK foot positions must match contact targets within tolerance [m].

    Args:
        kin: :class:`NewtonKinematics` instance.
        foot_ids: Newton body indices for the feet.
        max_err: Maximum position error per foot [m].
    """
    from isaaclab_tasks.manager_based.locomotion.position.mdp.retarget.buffer import RetargetBuffer

    def check(buffer: RetargetBuffer, N: int) -> torch.Tensor:
        nc = len(foot_ids)
        nb = kin.model.body_count  # type: ignore[attr-defined]
        tpl = newton.ModelBuilder()
        tpl.add_usd(kin.usd_path, collapse_fixed_joints=False)  # type: ignore[attr-defined]
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
        idx = torch.tensor(foot_ids, device=buffer.device, dtype=torch.long)
        err = (body_q[:, idx, :3] - ct).norm(dim=-1).max(dim=-1).values
        return err <= max_err

    return check


def joint_margin(
    kin: object,
    margin: float = 0.1,
):
    """Criterion: revolute joints must stay within ``margin`` of their limits.

    Args:
        kin: :class:`NewtonKinematics` instance.
        margin: Fraction of joint range to keep as safety margin.
    """
    from isaaclab_tasks.manager_based.locomotion.position.mdp.retarget.buffer import RetargetBuffer

    def check(buffer: RetargetBuffer, N: int) -> torch.Tensor:
        jl = wp.to_torch(kin.model.joint_limit_lower)  # type: ignore[union-attr]
        ju = wp.to_torch(kin.model.joint_limit_upper)  # type: ignore[union-attr]
        lo, hi = jl[6:], ju[6:]
        n_rev = lo.shape[0]
        jq = buffer.joint_q_result_t[:N, 7:7 + n_rev]
        safe_lo = lo + margin * (hi - lo)
        safe_hi = hi - margin * (hi - lo)
        violation = ((safe_lo - jq).clamp(min=0) + (jq - safe_hi).clamp(min=0)).max(dim=-1).values
        return violation <= 0

    return check


def haa_limit(max_angle: float = 0.87):
    """Criterion: HAA joint angles must not exceed ``max_angle`` [rad].

    Args:
        max_angle: Maximum absolute HAA angle [rad].
    """
    from isaaclab_tasks.manager_based.locomotion.position.mdp.retarget.buffer import RetargetBuffer

    def check(buffer: RetargetBuffer, N: int) -> torch.Tensor:
        n_rev = buffer.joint_q_result_t.shape[1] - 7
        haa_idx = torch.tensor(
            [i for i in ANYMAL_C_HAA_INDICES if i < n_rev],
            device=buffer.device, dtype=torch.long,
        )
        if haa_idx.numel() == 0:
            return torch.ones(N, device=buffer.device, dtype=torch.bool)
        jq_rev = buffer.joint_q_result_t[:N, 7:]
        return jq_rev[:, haa_idx].abs().max(dim=-1).values <= max_angle

    return check


def base_z_error(max_err: float = 0.3):
    """Criterion: base z must not deviate more than ``max_err`` from target [m].

    Args:
        max_err: Maximum absolute base z deviation [m].
    """
    from isaaclab_tasks.manager_based.locomotion.position.mdp.retarget.buffer import RetargetBuffer

    def check(buffer: RetargetBuffer, N: int) -> torch.Tensor:
        base_z = buffer.joint_q_result_t[:N, 2]
        target_z = buffer.base_target_pos_t[:N, 2]
        return (base_z - target_z).abs() <= max_err

    return check
