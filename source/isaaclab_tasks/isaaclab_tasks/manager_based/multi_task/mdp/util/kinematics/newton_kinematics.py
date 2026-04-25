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

import newton
import newton.ik as ik
import numpy as np
import warp as wp
from newton import GeoType
from newton._src.sim.ik.ik_common import eval_fk_batched as _newton_eval_fk_batched

from .newton_kinematics_cfg import NewtonKinematicsCfg  # re-exported for backcompat

__all__ = ["NewtonKinematics", "NewtonKinematicsCfg"]


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
            jq[7 : 7 + n] = resolved[:n]
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
        """Derive foot geometry from the default stance + collision shapes.

        ``foot_ground_offset`` is the z offset from the foot body's origin
        to the lowest point of its collision geometry — a pure-geometric
        quantity independent of the URDF default pose. Pipeline uses it
        to lift contact targets by this offset so IK places the foot's
        *sole* (not body origin) on the terrain surface.

        Args:
            foot_body_ids: Newton body indices for the feet.

        Returns:
            Dict with ``foot_offsets`` (body-to-base xyz at default),
            ``standing_height`` (default base-z minus default foot-mean-z),
            ``foot_ground_offset`` (negated min local-z of foot collision
            geometry, fallback to default foot-z if no shapes attached).
        """
        base_pos = self._default_body_q[0][:3]
        foot_pos = np.array([self._default_body_q[fid][:3] for fid in foot_body_ids])

        # Per-foot local-z-min from attached collision shapes. For each
        # shape type, compute the lowest-z offset the shape reaches in the
        # body frame (rotation assumed identity -- matches every foot
        # geometry we've seen in practice).
        builder = self.builder
        foot_ids_set = set(int(f) for f in foot_body_ids)
        z_min_local: float | None = None
        for si in range(len(builder.shape_body)):
            bid = int(builder.shape_body[si])
            if bid not in foot_ids_set:
                continue
            zmin = self._shape_local_z_min(
                int(builder.shape_type[si]),
                builder.shape_scale[si],
                builder.shape_transform[si],
                builder.shape_source[si],
            )
            if zmin is None:
                continue
            if z_min_local is None or zmin < z_min_local:
                z_min_local = zmin

        if z_min_local is not None:
            # foot_body_z + (-z_min_local) = terrain_z  →  sole on terrain.
            foot_ground_offset = float(-z_min_local)
        else:
            # Fallback: URDF default pose (assumes default places soles at z = 0).
            foot_ground_offset = float(foot_pos[:, 2].min())

        return {
            "foot_offsets": foot_pos - base_pos,
            "standing_height": float(base_pos[2] - foot_pos[:, 2].mean()),
            "foot_ground_offset": foot_ground_offset,
        }

    @staticmethod
    def _shape_local_z_min(
        shape_type: int,
        shape_scale,
        shape_transform,
        shape_source,
    ) -> float | None:
        """Lowest body-frame z coordinate reachable by a shape's surface.

        Handles the geometry primitives we encounter in practice (mesh,
        sphere, box, capsule, cylinder, plane). Returns ``None`` for
        unsupported types so the caller can fall back or skip.
        """
        pos_z = float(shape_transform[2])
        if shape_type == int(GeoType.MESH) or shape_type == int(GeoType.CONVEX_MESH):
            if shape_source is None or not hasattr(shape_source, "vertices"):
                return None
            verts = np.asarray(shape_source.vertices).reshape(-1, 3)
            if verts.size == 0:
                return None
            scale_z = float(shape_scale[2])
            return pos_z + float(verts[:, 2].min()) * scale_z
        if shape_type == int(GeoType.SPHERE):
            return pos_z - float(shape_scale[0])
        if shape_type == int(GeoType.BOX):
            return pos_z - 0.5 * float(shape_scale[2])
        if shape_type == int(GeoType.CAPSULE):
            return pos_z - float(shape_scale[1]) - float(shape_scale[0])
        if shape_type == int(GeoType.CYLINDER):
            return pos_z - float(shape_scale[1])
        return None

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

    def eval_fk(self, joint_q: wp.array, joint_qd: wp.array | None = None) -> newton.State:
        """Run forward kinematics for a single articulation.

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

    def eval_fk_batched(
        self,
        joint_q: wp.array,
        joint_qd: wp.array | None = None,
        body_q: wp.array | None = None,
        body_qd: wp.array | None = None,
    ) -> tuple[wp.array, wp.array]:
        """Run batched forward kinematics across ``N`` problems on the shared model.

        Wraps Newton's internal batched-FK kernel. All array arguments
        use a leading-``N`` batch dimension over the ``N`` parallel
        problems; all share the same kinematic model. Output arrays are
        allocated lazily when ``None``.

        Args:
            joint_q: Joint coordinates per problem, shape
                ``[N, joint_coord_count]`` [m or rad].
            joint_qd: Joint velocities per problem, shape
                ``[N, joint_dof_count]`` [m/s or rad/s]. Zero-filled if ``None``.
            body_q: Optional pre-allocated output for body transforms,
                shape ``[N, body_count]`` of :class:`warp.transformf`. Allocated if ``None``.
            body_qd: Optional pre-allocated output for body spatial velocities,
                shape ``[N, body_count]`` of :class:`warp.spatial_vectorf`. Allocated if ``None``.

        Returns:
            Tuple ``(body_q, body_qd)`` -- the (possibly freshly allocated) output arrays.
        """
        n = joint_q.shape[0]
        if joint_qd is None:
            joint_qd = wp.zeros((n, self.model.joint_dof_count), dtype=wp.float32, device=self.device)
        if body_q is None:
            body_q = wp.zeros((n, self.model.body_count), dtype=wp.transformf, device=self.device)
        if body_qd is None:
            body_qd = wp.zeros((n, self.model.body_count), dtype=wp.spatial_vectorf, device=self.device)
        _newton_eval_fk_batched(self.model, joint_q, joint_qd, body_q, body_qd)
        return body_q, body_qd
