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

import numpy as np
import torch
import warp as wp

from .kinematic.ik_objectives.terrain_collision import _build_collision_probes

if TYPE_CHECKING:
    from ..mdp.retarget.buffer import RetargetBuffer
    from ..mdp.retarget.pipeline import RetargetPipeline
    from .kinematic import NewtonKinematics


@dataclass
class FootPositionError:
    """Criterion: FK foot positions must match contact targets within tolerance [m].

    Reads ``body_q`` directly from the buffer (populated by the
    pipeline after IK) -- no separate FK pass needed. The ``aggregate``
    knob selects how per-foot errors are combined: ``"max"`` bounds the
    worst foot, ``"sum"`` bounds the total drift across the polygon.

    Args:
        num_bodies: Number of bodies per robot in the Newton model.
        foot_ids: Newton body indices for the feet.
        max_err: Error threshold [m] applied to the aggregated value.
        aggregate: ``"max"`` or ``"sum"`` across feet.
    """

    num_bodies: int = 0
    foot_ids: list[int] = field(default_factory=list)
    max_err: float = 0.02
    aggregate: str = "max"

    def __call__(self, buffer: RetargetBuffer, N: int) -> torch.Tensor:
        nc = len(self.foot_ids)
        body_q = buffer.body_q_t[: N * self.num_bodies].view(N, self.num_bodies, 7)
        ct = buffer.contact_targets_t[: N * nc].view(N, nc, 3)
        idx = torch.tensor(self.foot_ids, device=buffer.device, dtype=torch.long)
        per_foot = (body_q[:, idx, :3] - ct).norm(dim=-1)  # [N, nc]
        if self.aggregate == "sum":
            err = per_foot.sum(dim=-1)
        elif self.aggregate == "max":
            err = per_foot.max(dim=-1).values
        else:
            raise ValueError(f"FootPositionError.aggregate must be 'max' or 'sum', got {self.aggregate!r}")
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
        jq = buffer.joint_q_result_t[:N, 7 : 7 + n_rev]
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
            device=buffer.device,
            dtype=torch.long,
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


@dataclass
class SupportPolygonStability:
    """Criterion: base (CoM proxy) must project inside the support polygon.

    Physically-grounded quasi-static stability check: under constant
    gravity, the CoM's vertical projection must lie within the convex
    hull of the ground-contact points (world-frame XY). Uses the
    base-link origin as a cheap CoM proxy -- for quadrupeds the base
    is near the CoM, so the approximation is tight.

    Implementation: sort contacts CCW around their centroid so edges
    are adjacent, then for each edge ``(v_i, v_{i+1})`` check the sign
    of the 2D cross product ``(v_{i+1} - v_i) × (p - v_i)``. All
    non-negative iff ``p`` is inside the convex hull.
    """

    def __call__(self, buffer: RetargetBuffer, N: int) -> torch.Tensor:
        nc = buffer.num_contacts
        ct_xy = buffer.contact_targets_t[: N * nc].view(N, nc, 3)[..., :2]
        base_xy = buffer.joint_q_result_t[:N, 0:2]

        centroid = ct_xy.mean(dim=1, keepdim=True)
        angles = torch.atan2(
            ct_xy[..., 1] - centroid[..., 1],
            ct_xy[..., 0] - centroid[..., 0],
        )
        order = angles.argsort(dim=1)
        poly = torch.gather(ct_xy, 1, order.unsqueeze(-1).expand(-1, -1, 2))

        v0 = poly
        v1 = torch.roll(poly, -1, dims=1)
        edge = v1 - v0
        to_p = base_xy.unsqueeze(1) - v0
        cross = edge[..., 0] * to_p[..., 1] - edge[..., 1] * to_p[..., 0]
        return (cross >= 0).all(dim=1)


@dataclass
class SolverCostOutlier:
    """Criterion: reject candidates whose final solver cost is an outlier.

    Residual IK divergence -- after all geometric constraints pass, the
    solver may still settle on a poor local minimum. Reject candidates
    whose cost exceeds ``threshold_multiplier * median(costs)``.

    Args:
        pipeline: The :class:`RetargetPipeline` whose per-run solver
            costs this criterion should read.
        threshold_multiplier: Multiplier on the batch median cost above
            which a candidate is rejected [unitless].
    """

    pipeline: RetargetPipeline
    threshold_multiplier: float = 3.0

    def __call__(self, buffer: RetargetBuffer, N: int) -> torch.Tensor:
        costs = self.pipeline._solver_costs[:N]
        median = costs.median()
        return costs < median * self.threshold_multiplier


@dataclass
class CollisionCheck:
    """Criterion: reject candidates where body probes penetrate terrain.

    Reads ``body_q`` from the buffer (populated by the pipeline after
    IK), transforms collision probe offsets into world space via a Warp
    kernel, and queries the terrain mesh for penetration.

    No separate FK pass -- uses the body transforms already stored in
    the buffer.

    Args:
        kin: :class:`NewtonKinematics` instance.
        wp_mesh: Warp mesh for terrain queries.
        exclude_bodies: Body indices to skip (e.g. feet).
        n_samples: Surface probe points per body.
        max_pen: Maximum allowed penetration depth [m].
    """

    kin: NewtonKinematics
    wp_mesh: object
    exclude_bodies: list[int] = field(default_factory=list)
    n_samples: int = 16
    max_pen: float = 0.02

    def __post_init__(self):
        bodies, offsets = _build_collision_probes(self.kin.builder, self.exclude_bodies, self.n_samples)
        self._n_probes = len(bodies)
        self._probe_body_np = np.array(bodies, dtype=np.int32)
        self._probe_offset_np = np.array(offsets, dtype=np.float32)
        self._wp_bufs: dict[str, wp.array] = {}

    def _get_wp_bufs(self, device: str) -> tuple[wp.array, wp.array]:
        """Lazily allocate device buffers for probe data."""
        if device not in self._wp_bufs:
            self._wp_bufs[device] = (
                wp.array(self._probe_body_np, dtype=wp.int32, device=device),
                wp.from_numpy(self._probe_offset_np, dtype=wp.vec3, device=device),
            )
        return self._wp_bufs[device]

    def __call__(self, buffer: RetargetBuffer, N: int) -> torch.Tensor:
        nb = self.kin.model.body_count
        body_q_wp = wp.from_torch(
            buffer.body_q_t[: N * nb].contiguous(),
            dtype=wp.transformf,
        )
        probe_body, probe_offset = self._get_wp_bufs(buffer.device)

        pen_out = torch.zeros(N * self._n_probes, device=buffer.device, dtype=torch.float32)
        wp_pen = wp.from_torch(pen_out)
        wp.launch(
            _collision_check_kernel,
            dim=[N, self._n_probes],
            inputs=[body_q_wp, nb, self.wp_mesh.id, probe_body, probe_offset, 2.0, self._n_probes],
            outputs=[wp_pen],
            device=buffer.device,
        )
        wp.synchronize()
        return pen_out.view(N, self._n_probes).max(dim=-1).values <= self.max_pen


@wp.kernel
def _collision_check_kernel(
    body_q: wp.array1d(dtype=wp.transformf),
    n_bodies: int,
    mesh_id: wp.uint64,
    probe_body: wp.array1d(dtype=wp.int32),
    probe_offset: wp.array1d(dtype=wp.vec3),
    max_dist: float,
    n_probes: int,
    penetration: wp.array1d(dtype=wp.float32),
):
    """Penetration depth for each (candidate, probe) pair.

    Combines two signals to handle both open and closed meshes:

    - ``-sign * dist``: correct for watertight meshes where
      ``mesh_query_point`` can determine inside/outside.
    - ``surface_z - probe_z``: correct for open heightfield terrains
      where ``sign`` is always +1.

    The max of both is used so penetration is detected either way.
    """
    row, probe_idx = wp.tid()
    tf = body_q[row * n_bodies + probe_body[probe_idx]]
    world_pos = wp.transform_point(tf, probe_offset[probe_idx])
    out_idx = row * n_probes + probe_idx
    query = wp.mesh_query_point(mesh_id, world_pos, max_dist)
    if query.result:
        surface_pt = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        dist = wp.length(world_pos - surface_pt)
        sign_pen = -query.sign * dist
        z_pen = surface_pt[2] - world_pos[2]
        penetration[out_idx] = wp.max(sign_pen, z_pen)
    else:
        penetration[out_idx] = 0.0
