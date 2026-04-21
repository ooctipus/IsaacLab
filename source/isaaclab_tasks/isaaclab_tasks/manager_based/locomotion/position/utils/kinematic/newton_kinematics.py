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
from dataclasses import MISSING
from pathlib import Path

import newton
import newton.ik as ik
import numpy as np
import warp as wp

from isaaclab.utils import configclass


@configclass
class NewtonKinematicsCfg:
    """Configuration for building a :class:`NewtonKinematics` model.

    Mirrors the constructor arguments so the model can be instantiated
    with ``NewtonKinematics(cfg)``.
    """

    usd_path: str = MISSING  # type: ignore[assignment]
    """Path to the robot USD file."""

    device: str = "cuda:0"
    """Warp device string."""

    default_pos: tuple[float, float, float] = (0.0, 0.0, 0.6)
    """Default root position ``(x, y, z)`` [m]."""

    default_quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """Default root orientation ``(x, y, z, w)`` quaternion."""

    default_joint_pos: dict[str, float] | None = None
    """Default revolute joint positions as ``{regex: value}`` dict, or ``None``."""

    collapse_fixed_joints: bool = False
    """Merge fixed joints for a simpler kinematic tree."""


class NewtonKinematics:
    """Newton kinematic model built from a USD file.

    Owns the :class:`newton.Model`, ordered body/joint name lists, and
    the default stance (computed via FK at construction time).

    Args:
        cfg: Kinematics configuration.
    """

    model: newton.Model
    """Finalized Newton model."""

    usd_path: str
    """Absolute path to the USD file used to build this model."""

    body_names: list[str]
    """Ordered body names (index ``i`` corresponds to Newton body ``i``)."""

    joint_names: list[str]
    """Ordered joint names (index ``i`` corresponds to Newton joint ``i``)."""

    def __init__(self, cfg: NewtonKinematicsCfg):
        self.cfg = cfg
        self.usd_path = str(cfg.usd_path)

        self.builder = newton.ModelBuilder()
        result = self.builder.add_usd(self.usd_path, collapse_fixed_joints=cfg.collapse_fixed_joints)
        self.model = self.builder.finalize(device=cfg.device)

        self._tpl = newton.ModelBuilder()
        self._tpl.add_usd(self.usd_path, collapse_fixed_joints=cfg.collapse_fixed_joints)

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
        jq[0:3] = cfg.default_pos
        jq[3:7] = cfg.default_quat
        if cfg.default_joint_pos is not None:
            resolved = self._resolve_joint_pos_map(cfg.default_joint_pos)
            n = min(len(resolved), len(jq) - 7)
            jq[7:7 + n] = resolved[:n]
        state = self.eval_fk(wp.array(jq, dtype=float, device=cfg.device))
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

    def find_body_indices(self, names: list[str]) -> list[int]:
        """Resolve body names to Newton body indices.

        Args:
            names: Body name strings (exact match).

        Returns:
            Corresponding Newton body indices.

        Raises:
            ValueError: If any name is not found.
        """
        indices = []
        for name in names:
            if name not in self.body_names:
                raise ValueError(f"Body '{name}' not found. Available: {self.body_names}")
            indices.append(self.body_names.index(name))
        return indices

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
            if int(joint_type[jidx]) != 1:
                continue
            if regex.fullmatch(self.joint_names[jidx]):
                indices.append(int(q_start[jidx]) - 7)
        return sorted(indices)

    def foot_geometry(self, foot_body_ids: list[int]) -> dict[str, np.ndarray | float]:
        """Derive foot geometry from the default stance.

        Args:
            foot_body_ids: Newton body indices for the feet.

        Returns:
            Dict with ``foot_offsets``, ``standing_height``,
            ``foot_ground_offset`` derived from default FK.
        """
        base_pos = self._default_body_q[0][:3]
        foot_pos = np.array([self._default_body_q[fid][:3] for fid in foot_body_ids])
        return {
            "foot_offsets": foot_pos - base_pos,
            "standing_height": float(base_pos[2] - foot_pos[:, 2].mean()),
            "foot_ground_offset": float(foot_pos[:, 2].min()),
        }

    def create_ik_solver(
        self,
        objectives: list,
        n_problems: int,
        jacobian_mode: ik.IKJacobianType = ik.IKJacobianType.ANALYTIC,
    ) -> ik.IKSolver:
        """Create an IK solver from user-provided objectives.

        Args:
            objectives: List of IK objectives (position, rotation,
                joint limit, etc.).
            n_problems: Number of parallel IK problems.
            jacobian_mode: Jacobian backend.  Use ``MIXED`` when
                combining analytic objectives with autodiff-only
                objectives.

        Returns:
            Configured :class:`newton.ik.IKSolver`.
        """
        return ik.IKSolver(
            model=self.model,
            n_problems=n_problems,
            objectives=objectives,
            optimizer=ik.IKOptimizer.LM,
            jacobian_mode=jacobian_mode,
        )

    def build_batched_model(self, n: int) -> newton.Model:
        """Build a batched Newton model with ``n`` robot instances.

        Uses the cached USD template so the file is not re-parsed.

        Args:
            n: Number of robot copies.

        Returns:
            Finalized Newton model with ``n`` articulations.
        """
        bldr = newton.ModelBuilder()
        for _ in range(n):
            bldr.add_world(self._tpl)
        return bldr.finalize(device=self.device)

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
