# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-based kinematic model wrapper.

Wraps :class:`newton.Model` behind a single :class:`NewtonKinematics`
object that owns the model, ordered body names, and default stance.
The USD is parsed exactly once in ``__init__``.

No IsaacSim dependency -- only Newton + Warp.
"""

from __future__ import annotations

from pathlib import Path

import newton
import newton.ik as ik
import numpy as np
import warp as wp


class NewtonKinematics:
    """Newton kinematic model built from a USD file.

    Owns the :class:`newton.Model`, ordered body name list, and the
    default stance (computed via FK at construction time).

    Args:
        usd_path: Path to the robot USD file.
        device: Warp device string (e.g. ``"cuda:0"``).
        default_pos: Default root position ``(x, y, z)`` [m].
        default_quat: Default root orientation ``(x, y, z, w)`` quaternion.
        default_joint_pos: Default revolute joint positions ``[num_joints]``.
            If ``None``, all revolute joints default to zero.
        collapse_fixed_joints: Merge fixed joints for a simpler tree.
    """

    model: newton.Model
    """Finalized Newton model."""

    usd_path: str
    """Absolute path to the USD file used to build this model."""

    body_names: list[str]
    """Ordered body names (index ``i`` corresponds to Newton body ``i``)."""

    def __init__(
        self,
        usd_path: str | Path,
        device: str = "cuda:0",
        default_pos: tuple[float, float, float] = (0.0, 0.0, 0.6),
        default_quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        default_joint_pos: np.ndarray | None = None,
        *,
        collapse_fixed_joints: bool = False,
    ):
        self.usd_path = str(usd_path)
        builder = newton.ModelBuilder()
        result = builder.add_usd(self.usd_path, collapse_fixed_joints=collapse_fixed_joints)
        self.model = builder.finalize(device=device)
        path_body_map: dict[str, int] = result.get("path_body_map", {})
        names = [""] * self.model.body_count
        for path, idx in path_body_map.items():
            names[idx] = path.rsplit("/", 1)[-1]
        self.body_names = names

        jq = self.model.joint_q.numpy().copy()
        jq[0:3] = default_pos
        jq[3:7] = default_quat
        if default_joint_pos is not None:
            n = min(len(default_joint_pos), len(jq) - 7)
            jq[7:7 + n] = default_joint_pos[:n]
        state = self.eval_fk(wp.array(jq, dtype=float, device=device))
        self._default_joint_q = jq
        self._default_body_q = state.body_q.numpy()

    @property
    def device(self) -> str:
        return str(self.model.device)

    @property
    def default_joint_q(self) -> np.ndarray:
        """Default joint coordinates ``[joint_coord_count]`` (from FK at init)."""
        return self._default_joint_q

    @property
    def default_body_q(self) -> np.ndarray:
        """Default body transforms ``[body_count, 7]`` (from FK at init)."""
        return self._default_body_q

    def create_ik_solver(self, objectives: list, n_problems: int) -> ik.IKSolver:
        """Create an IK solver from user-provided objectives.

        Args:
            objectives: List of IK objectives (position, rotation,
                joint limit, etc.).
            n_problems: Number of parallel IK problems.

        Returns:
            Configured :class:`newton.ik.IKSolver`.
        """
        return ik.IKSolver(
            model=self.model,
            n_problems=n_problems,
            objectives=objectives,
            optimizer=ik.IKOptimizer.LM,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    def eval_fk(self, joint_q: wp.array, joint_qd: wp.array | None = None) -> newton.State:
        """Run forward kinematics.

        Args:
            joint_q: Joint coordinates [m or rad].
            joint_qd: Joint velocities (zeros if ``None``).

        Returns:
            Newton state with ``body_q`` populated.
        """
        state = self.model.state()
        if joint_qd is None:
            joint_qd = wp.zeros(self.model.joint_dof_count, dtype=float, device=self.device)
        newton.eval_fk(self.model, joint_q, joint_qd, state)
        return state
