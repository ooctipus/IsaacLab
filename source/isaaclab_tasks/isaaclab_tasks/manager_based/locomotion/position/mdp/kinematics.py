# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-based kinematic model wrapper.

Wraps :class:`newton.Model` behind a single :class:`NewtonKinematics`
object that owns the model, ordered body/joint names, and default stance.
The USD is parsed exactly once in ``__init__``.

No IsaacSim dependency -- only Newton + Warp.
"""

from __future__ import annotations

import re
from pathlib import Path

import newton
import newton.ik as ik
import numpy as np
import warp as wp


class NewtonKinematics:
    """Newton kinematic model built from a USD file.

    Owns the :class:`newton.Model`, ordered body/joint name lists, and
    the default stance (computed via FK at construction time).

    Args:
        usd_path: Path to the robot USD file.
        device: Warp device string (e.g. ``"cuda:0"``).
        default_pos: Default root position ``(x, y, z)`` [m].
        default_quat: Default root orientation ``(x, y, z, w)`` quaternion.
        default_joint_pos: Default revolute joint positions.  Accepts
            either a flat ``np.ndarray`` (indexed by DOF order) or a
            ``{regex_pattern: value}`` dict that is resolved against
            :attr:`joint_names` (same format as
            :attr:`ArticulationCfg.InitialStateCfg.joint_pos`).
            If ``None``, all revolute joints default to zero.
        collapse_fixed_joints: Merge fixed joints for a simpler tree.
    """

    model: newton.Model
    """Finalized Newton model."""

    usd_path: str
    """Absolute path to the USD file used to build this model."""

    body_names: list[str]
    """Ordered body names (index ``i`` corresponds to Newton body ``i``)."""

    joint_names: list[str]
    """Ordered joint names (index ``i`` corresponds to Newton joint ``i``)."""

    def __init__(
        self,
        usd_path: str | Path,
        device: str = "cuda:0",
        default_pos: tuple[float, float, float] = (0.0, 0.0, 0.6),
        default_quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        default_joint_pos: np.ndarray | dict[str, float] | None = None,
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

        path_joint_map: dict[str, int] = result.get("path_joint_map", {})
        jnames = [""] * self.model.joint_count
        for path, idx in path_joint_map.items():
            jnames[idx] = path.rsplit("/", 1)[-1]
        self.joint_names = jnames

        jq = self.model.joint_q.numpy().copy()
        jq[0:3] = default_pos
        jq[3:7] = default_quat
        if default_joint_pos is not None:
            if isinstance(default_joint_pos, dict):
                default_joint_pos = self._resolve_joint_pos_map(default_joint_pos)
            n = min(len(default_joint_pos), len(jq) - 7)
            jq[7:7 + n] = default_joint_pos[:n]
        state = self.eval_fk(wp.array(jq, dtype=float, device=device))
        self._default_joint_q = jq
        self._default_body_q = state.body_q.numpy()

    def _resolve_joint_pos_map(self, joint_pos_map: dict[str, float]) -> np.ndarray:
        """Resolve a ``{regex: value}`` dict to a flat joint position array.

        Uses ``joint_q_start`` to map each matched joint to its actual
        position in ``joint_q[7:]``, correctly skipping fixed and ball
        joints that contribute no DOFs.
        """
        n_coords = self.model.joint_coord_count - 7
        jpos = np.zeros(n_coords, dtype=np.float32)
        q_start = self.model.joint_q_start.numpy()
        joint_type = self.model.joint_type.numpy()
        for pattern, value in joint_pos_map.items():
            regex = re.compile(pattern)
            for jidx in range(1, len(self.joint_names)):
                if not regex.fullmatch(self.joint_names[jidx]):
                    continue
                # Only set revolute joints (type 1) -- they have exactly 1 coord
                if int(joint_type[jidx]) != 1:
                    continue
                qi = int(q_start[jidx]) - 7
                if 0 <= qi < n_coords:
                    jpos[qi] = value
        return jpos

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

    def find_joint_dof_indices(self, pattern: str) -> list[int]:
        """Find revolute-joint DOF indices matching a regex pattern.

        Returns indices into ``joint_q[7:]`` (i.e. excluding the 7
        free-root coordinates).  Uses ``joint_q_start`` for correct
        mapping even when the model contains non-revolute joints.

        Args:
            pattern: Regex matched against each joint name.

        Returns:
            Sorted list of matching DOF indices.
        """
        regex = re.compile(pattern)
        q_start = self.model.joint_q_start.numpy()
        joint_type = self.model.joint_type.numpy()
        indices = []
        for jidx in range(1, len(self.joint_names)):
            if int(joint_type[jidx]) != 1:  # revolute only
                continue
            if regex.fullmatch(self.joint_names[jidx]):
                indices.append(int(q_start[jidx]) - 7)
        return sorted(indices)

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
