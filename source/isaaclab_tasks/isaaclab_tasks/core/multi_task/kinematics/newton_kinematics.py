# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-based kinematic model wrapper.

Wraps :class:`newton.Model` behind a single :class:`NewtonKinematics`
object that owns the model, ordered body/joint names, and default stance.
The configured USD or MJCF is parsed exactly once in ``__init__``.

No IsaacSim dependency -- only Newton + Warp.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import newton
import newton.ik as ik
import numpy as np
import torch
import warp as wp
from newton import JointType
from newton._src.sim.ik.ik_common import eval_fk_batched as _newton_eval_fk_batched

from .newton_asset import resolve_newton_asset_path
from .newton_kinematics_cfg import NewtonKinematicsBuildCfg, NewtonKinematicsCfg

if TYPE_CHECKING:
    from isaaclab.assets import ArticulationCfg


class NewtonKinematics:
    """Newton kinematic model built from one USD or MJCF file.

    Owns the :class:`newton.Model`, ordered body/joint name lists, and
    the default stance (computed via FK at construction time).

    Args:
        cfg: Kinematics configuration.
    """

    @dataclass(frozen=True, slots=True)
    class Topology:
        """Canonical host-side topology and inertial facts parsed from one Newton model."""

        joint_type: np.ndarray
        joint_parent: np.ndarray
        joint_child: np.ndarray
        joint_transform_parent: np.ndarray
        joint_transform_child: np.ndarray
        joint_q_start: np.ndarray
        joint_qd_start: np.ndarray
        joint_dof_dim: np.ndarray
        joint_axis: np.ndarray
        joint_limit_lower: np.ndarray
        joint_limit_upper: np.ndarray
        body_parent: np.ndarray
        body_joint: np.ndarray
        dof_joint: np.ndarray
        body_dof_ancestry: np.ndarray
        joint_subtree_bodies: np.ndarray
        joint_subtree_offsets: np.ndarray
        joint_subtree_mass: np.ndarray
        joint_subtree_inverse_mass: np.ndarray
        body_mass: np.ndarray
        body_com: np.ndarray
        gravity: np.ndarray

        @property
        def body_count(self) -> int:
            """Number of rigid bodies."""
            return self.body_parent.shape[0]

        @property
        def joint_count(self) -> int:
            """Number of joints, including the root joint."""
            return self.joint_parent.shape[0]

        @property
        def coordinate_count(self) -> int:
            """Number of generalized-position coordinates."""
            return int(self.joint_q_start[-1])

        @property
        def dof_count(self) -> int:
            """Number of generalized velocities."""
            return int(self.joint_qd_start[-1])

    model: newton.Model
    """Finalized Newton model."""

    usd_path: str
    """Path to the USD file used to build this model, or an empty string."""

    mjcf_path: str
    """Path to the MJCF file used to build this model, or an empty string."""

    body_names: list[str]
    """Ordered body names (index ``i`` corresponds to Newton body ``i``)."""

    joint_names: list[str]
    """Ordered joint names (index ``i`` corresponds to Newton joint ``i``)."""

    joint_q_names: list[str]
    """Ordered labels for every Newton joint coordinate."""

    joint_qd_names: list[str]
    """Ordered labels for every Newton joint velocity."""
    topology: Topology
    """Canonical host-side topology and inertial facts parsed exactly once."""

    def __init__(self, cfg: NewtonKinematicsCfg):
        self.cfg = cfg

        usd_path = cfg.usd_path if isinstance(cfg.usd_path, str) and cfg.usd_path else None
        mjcf_path = cfg.mjcf_path if isinstance(cfg.mjcf_path, str) and cfg.mjcf_path else None
        if (usd_path is None) == (mjcf_path is None):
            raise ValueError("NewtonKinematicsCfg must define exactly one of usd_path or mjcf_path.")
        self.usd_path = resolve_newton_asset_path(usd_path) if usd_path is not None else ""
        self.mjcf_path = mjcf_path or ""

        self.builder = newton.ModelBuilder()
        if self.usd_path:
            result = self.builder.add_usd(self.usd_path, collapse_fixed_joints=cfg.collapse_fixed_joints)
        else:
            self.builder.add_mjcf(
                self.mjcf_path,
                collapse_fixed_joints=cfg.collapse_fixed_joints,
                enable_self_collisions=False,
                parse_meshes=False,
                parse_sites=False,
                parse_visuals=False,
            )
            result = None
        self.model = self.builder.finalize(device=cfg.device)

        if result is not None:
            # Keep the established USD-facing labels, including its unnamed
            # synthetic root joint. Newton's MJCF parser has no path maps.
            self.body_names = self._names_from_path_map(result.get("path_body_map", {}), self.model.body_count)
            self.joint_names = self._names_from_path_map(result.get("path_joint_map", {}), self.model.joint_count)
        else:
            self.body_names = self._names_from_model_labels(self.model.body_label, "body")
            self.joint_names = self._names_from_model_labels(self.model.joint_label, "joint")
        self.topology = self._build_topology(self.model)

        model_joint_names = [
            label.rsplit("/", 1)[-1] or f"joint_{index}" for index, label in enumerate(self.model.joint_label)
        ]
        self.joint_q_names = self._coordinate_names(
            model_joint_names,
            self.topology.joint_q_start,
            self.topology.joint_type,
            self.model.joint_coord_count,
            velocity=False,
        )
        self.joint_qd_names = self._coordinate_names(
            model_joint_names,
            self.topology.joint_qd_start,
            self.topology.joint_type,
            self.model.joint_dof_count,
            velocity=True,
        )

        # Root coordinate count: 7 for a free-floating base (3 position + 4
        # quaternion), 0 for a fixed base. Non-root joints occupy
        # ``joint_q[n_root_coords:]``. Reading it from the model (instead of
        # assuming a free root) lets the same wrapper drive fixed-base arms.
        self._n_root_coords = self._compute_root_coord_count()

        jq = self.model.joint_q.numpy().copy()
        if self._n_root_coords >= 7:
            # Free-floating base only: the first 7 coords are the root pose.
            jq[0:3] = cfg.default_pos
            jq[3:7] = cfg.default_quat
        if cfg.default_joint_pos is not None:
            resolved = self._resolve_joint_pos_map(cfg.default_joint_pos)
            n = min(len(resolved), len(jq) - self._n_root_coords)
            jq[self._n_root_coords : self._n_root_coords + n] = resolved[:n]
        state = self.eval_fk(wp.array(jq, dtype=float, device=cfg.device))
        self._default_joint_q = jq
        self._default_body_q = state.body_q.numpy()

    @classmethod
    def from_articulation(
        cls, cfg: NewtonKinematicsBuildCfg, articulation_cfg: ArticulationCfg, device: str | torch.device
    ) -> NewtonKinematics:
        """Build mechanics from one resolved IsaacLab articulation declaration.

        Args:
            cfg: Newton parse policy.
            articulation_cfg: Scene-owned articulation asset and initial state.
            device: Torch/Warp device used to finalize the kinematic model.

        Returns:
            Kinematics containing the articulation's parsed model and default state.
        """
        spawn = getattr(articulation_cfg, "spawn", None)
        usd_path = getattr(spawn, "usd_path", None)
        init_state = getattr(articulation_cfg, "init_state", None)
        if not isinstance(usd_path, str) or not usd_path or init_state is None:
            raise ValueError("Newton kinematics requires an articulation with a declared USD and initial state.")
        return cls(
            NewtonKinematicsCfg(
                usd_path=usd_path,
                device=str(device),
                default_pos=init_state.pos,
                default_quat=init_state.rot,
                default_joint_pos=init_state.joint_pos,
                collapse_fixed_joints=cfg.collapse_fixed_joints,
            )
        )

    @classmethod
    def _build_topology(cls, model: newton.Model) -> Topology:
        """Parse one immutable host-side topology and inertial record.

        Args:
            model: Finalized Newton model.

        Returns:
            Canonical topology whose NumPy arrays reject mutation.
        """

        def readonly(values, dtype) -> np.ndarray:
            array = np.asarray(values, dtype=dtype).copy()
            array.setflags(write=False)
            return array

        joint_count = model.joint_count
        body_count = model.body_count
        dof_count = model.joint_dof_count
        coordinate_count = model.joint_coord_count
        joint_type = np.asarray(model.joint_type.numpy(), dtype=np.int32)
        joint_parent = np.asarray(model.joint_parent.numpy(), dtype=np.int32)
        joint_child = np.asarray(model.joint_child.numpy(), dtype=np.int32)
        joint_transform_parent = np.asarray(model.joint_X_p.numpy(), dtype=np.float32)
        joint_transform_child = np.asarray(model.joint_X_c.numpy(), dtype=np.float32)
        joint_q_start = np.asarray(model.joint_q_start.numpy(), dtype=np.int32)
        joint_qd_start = np.asarray(model.joint_qd_start.numpy(), dtype=np.int32)
        joint_dof_dim = np.asarray(model.joint_dof_dim.numpy(), dtype=np.int32)
        joint_axis = np.asarray(model.joint_axis.numpy(), dtype=np.float32)
        joint_limit_lower = np.asarray(model.joint_limit_lower.numpy(), dtype=np.float32)
        joint_limit_upper = np.asarray(model.joint_limit_upper.numpy(), dtype=np.float32)
        body_mass = np.asarray(model.body_mass.numpy(), dtype=np.float32)
        body_com = np.asarray(model.body_com.numpy(), dtype=np.float32)
        gravity = np.asarray(model.gravity.numpy(), dtype=np.float32).reshape(-1, 3)[0]

        if (
            joint_type.shape != (joint_count,)
            or joint_parent.shape != (joint_count,)
            or joint_child.shape != (joint_count,)
            or joint_transform_parent.shape != (joint_count, 7)
            or joint_transform_child.shape != (joint_count, 7)
            or joint_q_start.shape != (joint_count + 1,)
            or joint_qd_start.shape != (joint_count + 1,)
            or joint_dof_dim.shape != (joint_count, 2)
            or joint_axis.shape != (dof_count, 3)
            or joint_limit_lower.shape != (dof_count,)
            or joint_limit_upper.shape != (dof_count,)
            or body_mass.shape != (body_count,)
            or body_com.shape != (body_count, 3)
            or int(joint_q_start[-1]) != coordinate_count
            or int(joint_qd_start[-1]) != dof_count
        ):
            raise ValueError("Newton model exposes inconsistent topology or inertial array shapes.")

        body_parent = np.full(body_count, -2, dtype=np.int32)
        body_joint = np.full(body_count, -1, dtype=np.int32)
        children: list[list[int]] = [[] for _ in range(body_count)]
        for joint_index, child in enumerate(joint_child):
            child_index = int(child)
            if child_index < 0:
                continue
            if child_index >= body_count or body_joint[child_index] >= 0:
                raise ValueError("Every Newton body must be owned by exactly one valid child joint.")
            parent_index = int(joint_parent[joint_index])
            if parent_index >= body_count:
                raise ValueError("Newton joint parent index exceeds the body count.")
            body_parent[child_index] = parent_index
            body_joint[child_index] = joint_index
            if parent_index >= 0:
                children[parent_index].append(child_index)
        if np.any(body_joint < 0) or np.count_nonzero(body_parent == -1) != 1:
            raise ValueError("Newton topology must contain one rooted articulation covering every body.")

        subtree_rows: list[list[int]] = []
        for child in joint_child:
            root = int(child)
            if root < 0:
                subtree_rows.append([])
                continue
            row: list[int] = []
            stack = [root]
            while stack:
                body = stack.pop()
                row.append(body)
                stack.extend(reversed(children[body]))
            subtree_rows.append(row)
        joint_subtree_offsets = np.zeros(joint_count + 1, dtype=np.int32)
        for joint_index, bodies in enumerate(subtree_rows):
            joint_subtree_offsets[joint_index + 1] = joint_subtree_offsets[joint_index] + len(bodies)
        joint_subtree_bodies = np.asarray(
            [body for bodies in subtree_rows for body in bodies],
            dtype=np.int32,
        )
        joint_subtree_mass = np.asarray(
            [body_mass[bodies].sum() if bodies else 0.0 for bodies in subtree_rows],
            dtype=np.float32,
        )
        joint_subtree_inverse_mass = np.zeros_like(joint_subtree_mass)
        positive_mass = joint_subtree_mass > 0.0
        joint_subtree_inverse_mass[positive_mass] = 1.0 / joint_subtree_mass[positive_mass]

        dof_joint = np.full(dof_count, -1, dtype=np.int32)
        body_dof_ancestry = np.zeros((body_count, dof_count), dtype=np.uint8)
        for joint_index, bodies in enumerate(subtree_rows):
            begin = int(joint_qd_start[joint_index])
            end = int(joint_qd_start[joint_index + 1])
            dof_joint[begin:end] = joint_index
            if bodies and end > begin:
                body_dof_ancestry[np.ix_(bodies, range(begin, end))] = 1
        if np.any(dof_joint < 0):
            raise ValueError("Newton joint velocity ranges must cover every degree of freedom exactly once.")

        return cls.Topology(
            joint_type=readonly(joint_type, np.int32),
            joint_parent=readonly(joint_parent, np.int32),
            joint_child=readonly(joint_child, np.int32),
            joint_transform_parent=readonly(joint_transform_parent, np.float32),
            joint_transform_child=readonly(joint_transform_child, np.float32),
            joint_q_start=readonly(joint_q_start, np.int32),
            joint_qd_start=readonly(joint_qd_start, np.int32),
            joint_dof_dim=readonly(joint_dof_dim, np.int32),
            joint_axis=readonly(joint_axis, np.float32),
            joint_limit_lower=readonly(joint_limit_lower, np.float32),
            joint_limit_upper=readonly(joint_limit_upper, np.float32),
            body_parent=readonly(body_parent, np.int32),
            body_joint=readonly(body_joint, np.int32),
            dof_joint=readonly(dof_joint, np.int32),
            body_dof_ancestry=readonly(body_dof_ancestry, np.uint8),
            joint_subtree_bodies=readonly(joint_subtree_bodies, np.int32),
            joint_subtree_offsets=readonly(joint_subtree_offsets, np.int32),
            joint_subtree_mass=readonly(joint_subtree_mass, np.float32),
            joint_subtree_inverse_mass=readonly(joint_subtree_inverse_mass, np.float32),
            body_mass=readonly(body_mass, np.float32),
            body_com=readonly(body_com, np.float32),
            gravity=readonly(gravity, np.float32),
        )

    @staticmethod
    def _names_from_path_map(path_map: dict[str, int], count: int) -> list[str]:
        """Return ordered leaf labels from a Newton USD parser path map."""
        names = [""] * count
        for path, index in path_map.items():
            names[index] = path.rsplit("/", 1)[-1]
        return names

    @staticmethod
    def _names_from_model_labels(labels: list[str], kind: str) -> list[str]:
        """Return unique ordered leaf labels from a finalized Newton model.

        Args:
            labels: Newton model labels in model index order.
            kind: Human-readable label kind for error messages.

        Returns:
            Model labels stripped to their final path components.

        Raises:
            ValueError: If a model label is empty or leaf labels are not unique.
        """
        names = [label.rsplit("/", 1)[-1] for label in labels]
        if any(not name for name in names):
            raise ValueError(f"Newton model contains an empty {kind} label.")
        if len(set(names)) != len(names):
            raise ValueError(f"Newton model {kind} leaf labels must be unique: {names}")
        return names

    @staticmethod
    def _coordinate_names(
        joint_names: list[str],
        starts: np.ndarray,
        joint_types: np.ndarray,
        count: int,
        *,
        velocity: bool,
    ) -> list[str]:
        """Derive ordered scalar coordinate labels from Newton joint ranges."""
        if len(starts) != len(joint_names) + 1 or len(joint_types) != len(joint_names):
            raise ValueError("Newton joint metadata lengths are inconsistent.")

        free_suffixes = (
            (
                "linear_velocity_x",
                "linear_velocity_y",
                "linear_velocity_z",
                "angular_velocity_x",
                "angular_velocity_y",
                "angular_velocity_z",
            )
            if velocity
            else (
                "position_x",
                "position_y",
                "position_z",
                "rotation_x",
                "rotation_y",
                "rotation_z",
                "rotation_w",
            )
        )
        ball_suffixes = (
            ("angular_velocity_x", "angular_velocity_y", "angular_velocity_z")
            if velocity
            else ("rotation_x", "rotation_y", "rotation_z", "rotation_w")
        )
        generic = "velocity" if velocity else "coordinate"
        names: list[str] = []
        for index, joint_name in enumerate(joint_names):
            start = int(starts[index])
            end = int(starts[index + 1])
            width = end - start
            if width == 0:
                continue
            if width == 1:
                names.append(joint_name)
                continue
            joint_type = int(joint_types[index])
            if joint_type == int(JointType.FREE):
                suffixes = free_suffixes
            elif joint_type == int(JointType.BALL):
                suffixes = ball_suffixes
            else:
                suffixes = tuple(f"{generic}_{offset}" for offset in range(width))
            if len(suffixes) != width:
                raise ValueError(f"Joint '{joint_name}' exposes {width} unexpected scalar coordinates.")
            names.extend(f"{joint_name}:{suffix}" for suffix in suffixes)
        if len(names) != count:
            raise ValueError(f"Newton joint ranges describe {len(names)} labels, expected {count}.")
        return names

    def _resolve_joint_pos_map(self, joint_pos_map: dict[str, float]) -> np.ndarray:
        """Resolve a ``{regex: value}`` dict to a flat joint position array.

        Uses ``joint_q_start`` to map each matched joint to its actual
        position in ``joint_q[n_root_coords:]`` (the non-root coordinates),
        accepting single-DoF position joints (revolute or prismatic) and
        skipping fixed/ball/free joints that carry no scalar default here.
        """
        n_root = self._n_root_coords
        n_coords = self.model.joint_coord_count - n_root
        jpos = np.zeros(n_coords, dtype=np.float32)
        q_start = self.topology.joint_q_start
        joint_type = self.topology.joint_type
        single_dof = (int(JointType.PRISMATIC), int(JointType.REVOLUTE))
        for pattern, value in joint_pos_map.items():
            regex = re.compile(pattern)
            for jidx in range(1, len(self.joint_names)):
                if not regex.fullmatch(self.joint_names[jidx]):
                    continue
                if int(joint_type[jidx]) not in single_dof:
                    continue
                qi = int(q_start[jidx]) - n_root
                if 0 <= qi < n_coords:
                    jpos[qi] = value
        return jpos

    def _compute_root_coord_count(self) -> int:
        """Number of ``joint_q`` coordinates consumed by the root joint.

        ``7`` for a free-floating base (joint 0 is a ``FREE`` joint: 3
        position + 4 quaternion), ``0`` for a fixed base (joint 0 is
        ``FIXED``). Derived from ``joint_q_start`` so it reflects the actual
        model layout regardless of the root joint type.
        """
        if self.model.joint_count <= 1:
            return int(self.model.joint_coord_count)
        return int(self.topology.joint_q_start[1] - self.topology.joint_q_start[0])

    @property
    def device(self) -> str:
        return str(self.model.device)

    @property
    def n_root_coords(self) -> int:
        """Coordinates the root joint consumes (7 free-floating, 0 fixed-base)."""
        return self._n_root_coords

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

    def find_joint_scalar_coordinates(self, pattern: str) -> tuple[list[int], list[int], list[str]]:
        """Find matching scalar joints as absolute position and velocity indices.

        Args:
            pattern: Regex matched against each joint name.

        Returns:
            Ordered absolute ``joint_q`` indices, absolute ``joint_qd`` indices,
            and matching joint names. Fixed, ball, and free joints are omitted
            because they do not have a one-to-one scalar q/qd representation.
        """
        regex = re.compile(pattern)
        q_start = self.topology.joint_q_start
        qd_start = self.topology.joint_qd_start
        coordinates: list[int] = []
        velocities: list[int] = []
        names: list[str] = []
        for joint_index, joint_name in enumerate(self.joint_names):
            q_begin, q_end = int(q_start[joint_index]), int(q_start[joint_index + 1])
            qd_begin, qd_end = int(qd_start[joint_index]), int(qd_start[joint_index + 1])
            if q_end - q_begin != 1 or qd_end - qd_begin != 1 or not regex.fullmatch(joint_name):
                continue
            coordinates.append(q_begin)
            velocities.append(qd_begin)
            names.append(joint_name)
        return coordinates, velocities, names

    def find_body_chain_joint_coordinates(self, body_name: str) -> tuple[list[int], list[int], list[str]]:
        """Return coordinate, velocity, and joint-name axes from the root to one body.

        Args:
            body_name: Exact body name at the end of the kinematic chain.

        Returns:
            Ordered joint-coordinate indices, joint-velocity indices, and one
            joint name per coordinate from root to body.
        """
        body = self.find_body_indices([body_name])[0]
        q_start = self.topology.joint_q_start
        qd_start = self.topology.joint_qd_start
        coordinates: list[int] = []
        velocities: list[int] = []
        names: list[str] = []
        while body >= 0:
            joint = int(self.topology.body_joint[body])
            joint_coordinates = list(range(int(q_start[joint]), int(q_start[joint + 1])))
            coordinates = joint_coordinates + coordinates
            velocities = list(range(int(qd_start[joint]), int(qd_start[joint + 1]))) + velocities
            names = [self.joint_names[joint]] * len(joint_coordinates) + names
            body = int(self.topology.body_parent[body])
        return coordinates, velocities, names

    def find_body_child_joint_coordinates(self, body_names: list[str]) -> tuple[list[int], list[int], list[str]]:
        """Return coordinates and velocities of joints whose children are named bodies.

        Args:
            body_names: Exact child-body names in the requested output order.

        Returns:
            Ordered joint-coordinate indices, joint-velocity indices, and one
            joint name per coordinate.
        """
        body_indices = self.find_body_indices(body_names)
        q_start = self.topology.joint_q_start
        qd_start = self.topology.joint_qd_start
        coordinates: list[int] = []
        velocities: list[int] = []
        names: list[str] = []
        for body in body_indices:
            joint = int(self.topology.body_joint[body])
            joint_coordinates = list(range(int(q_start[joint]), int(q_start[joint + 1])))
            coordinates.extend(joint_coordinates)
            velocities.extend(range(int(qd_start[joint]), int(qd_start[joint + 1])))
            names.extend([self.joint_names[joint]] * len(joint_coordinates))
        return coordinates, velocities, names

    def body_adjacency(self, max_joint_hops: int) -> np.ndarray:
        """Return body pairs separated by at most the declared joint hops.

        Args:
            max_joint_hops: Maximum number of parent-child edges between paired bodies.

        Returns:
            Symmetric uint8 adjacency matrix with shape [body_count, body_count].
        """
        if max_joint_hops < 0:
            raise ValueError("Maximum joint hops cannot be negative.")
        step = np.eye(self.topology.body_count, dtype=np.int32)
        for body, parent in enumerate(self.topology.body_parent):
            if parent >= 0:
                step[int(parent), body] = 1
                step[body, int(parent)] = 1
        reach = np.eye(self.topology.body_count, dtype=np.int32)
        for _ in range(max_joint_hops):
            reach = (reach @ step > 0).astype(np.int32)
        return (reach > 0).astype(np.uint8)

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

    def eval_fk_batched_torch(
        self,
        joint_q: torch.Tensor,
        joint_qd: torch.Tensor,
        body_q: torch.Tensor,
        body_qd: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run batched FK from Torch tensors into caller-owned Torch outputs.

        This path performs no tensor allocation and never repairs an input by
        copying it. Every tensor must be contiguous, ``float32``, and resident
        on the kinematic model's device.

        Args:
            joint_q: Joint coordinates [m or rad, depending on joint type],
                shape ``[N, joint_coord_count]``.
            joint_qd: Joint velocities [m/s or rad/s, depending on joint type],
                shape ``[N, joint_dof_count]``.
            body_q: Caller-owned body transforms, shape ``[N, body_count, 7]``.
                Components are position [m] followed by an ``xyzw`` quaternion.
            body_qd: Caller-owned body spatial velocities, shape
                ``[N, body_count, 6]``. Components are linear xyz [m/s]
                followed by angular xyz [rad/s].

        Returns:
            The unchanged ``(body_q, body_qd)`` output tensor objects.

        Raises:
            ValueError: If shape, dtype, stride, or device violates the contract.
        """
        batch_size = joint_q.shape[0] if joint_q.ndim > 0 else -1
        expected_shapes = (
            (joint_q, (batch_size, self.model.joint_coord_count), "joint_q"),
            (joint_qd, (batch_size, self.model.joint_dof_count), "joint_qd"),
            (body_q, (batch_size, self.model.body_count, 7), "body_q"),
            (body_qd, (batch_size, self.model.body_count, 6), "body_qd"),
        )
        for tensor, shape, name in expected_shapes:
            if tensor.dtype != torch.float32:
                raise ValueError(f"{name} must use float32; received {tensor.dtype}.")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous.")
            if str(wp.device_from_torch(tensor.device)) != self.device:
                raise ValueError(f"{name} must be on {self.device}; received {tensor.device}.")
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}; received {tuple(tensor.shape)}.")

        _newton_eval_fk_batched(
            self.model,
            wp.from_torch(joint_q, dtype=wp.float32),
            wp.from_torch(joint_qd, dtype=wp.float32),
            wp.from_torch(body_q, dtype=wp.transformf),
            wp.from_torch(body_qd, dtype=wp.spatial_vectorf),
        )
        return body_q, body_qd
