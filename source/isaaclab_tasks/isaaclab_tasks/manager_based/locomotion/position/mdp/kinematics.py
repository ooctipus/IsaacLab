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
        self.builder = newton.ModelBuilder()
        result = self.builder.add_usd(self.usd_path, collapse_fixed_joints=collapse_fixed_joints)
        self.model = self.builder.finalize(device=device)

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
                objectives (e.g. :class:`IKObjectiveGravityTorque`).

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


# ---------------------------------------------------------------------------
# IK objective: joint default regularization
# ---------------------------------------------------------------------------


@wp.kernel
def _default_residuals(
    joint_q: wp.array2d(dtype=wp.float32),
    target_q: wp.array1d(dtype=wp.float32),
    dof_to_coord: wp.array1d(dtype=wp.int32),
    weight: float,
    start_idx: int,
    residuals: wp.array2d(dtype=wp.float32),
):
    problem, dof_idx = wp.tid()
    coord_idx = dof_to_coord[dof_idx]
    if coord_idx < 0:
        return
    residuals[problem, start_idx + dof_idx] = weight * (joint_q[problem, coord_idx] - target_q[dof_idx])


@wp.kernel
def _default_jac_analytic(
    dof_to_coord: wp.array1d(dtype=wp.int32),
    weight: float,
    start_idx: int,
    jacobian: wp.array3d(dtype=wp.float32),
):
    problem, dof_idx = wp.tid()
    coord_idx = dof_to_coord[dof_idx]
    if coord_idx < 0:
        return
    jacobian[problem, start_idx + dof_idx, dof_idx] = weight


class IKObjectiveJointDefault(ik.IKObjective):
    """Penalize deviation from a target joint configuration.

    Pulls joints toward a reference pose (typically the robot's default
    standing stance), producing more natural-looking IK solutions.
    Inspired by biomechanical energy minimization: poses closer to the
    default require less static joint torque to maintain.

    Each DOF contributes one residual: ``weight * (q - q_target)``.
    The analytic Jacobian is a constant diagonal of ``weight``.
    Free-root DOFs (joint 0) are excluded by default.

    Args:
        target_joint_q: Target joint coordinates ``[joint_dof_count]`` [m or rad].
            Typically the robot's default revolute joint positions.
            Free-root entries are ignored when ``skip_root`` is ``True``.
        weight: Scalar multiplier for the regularization residual.
        skip_root: Exclude the free-root joint (joint 0) DOFs from
            regularization.  Defaults to ``True``.
    """

    def __init__(
        self,
        target_joint_q: wp.array,
        weight: float = 0.1,
        *,
        skip_root: bool = True,
    ) -> None:
        super().__init__()
        self.target_joint_q = target_joint_q
        self.weight = weight
        self.n_dofs = len(target_joint_q)
        self._skip_root = skip_root
        self.dof_to_coord: wp.array | None = None

    def init_buffers(self, model: newton.Model, jacobian_mode: ik.IKJacobianType) -> None:
        self._require_batch_layout()
        dof_to_coord_np = np.full(self.n_dofs, -1, dtype=np.int32)
        q_start_np = model.joint_q_start.numpy()
        qd_start_np = model.joint_qd_start.numpy()
        joint_dof_dim_np = model.joint_dof_dim.numpy()
        start_joint = 1 if self._skip_root else 0
        for j in range(start_joint, model.joint_count):
            dof0 = qd_start_np[j]
            coord0 = q_start_np[j]
            lin, ang = joint_dof_dim_np[j]
            for k in range(lin + ang):
                dof_to_coord_np[dof0 + k] = coord0 + k
        self.dof_to_coord = wp.array(dof_to_coord_np, dtype=wp.int32, device=self.device)

    def supports_analytic(self) -> bool:
        return True

    def residual_dim(self) -> int:
        return self.n_dofs

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx) -> None:
        count = joint_q.shape[0]
        wp.launch(
            _default_residuals,
            dim=[count, self.n_dofs],
            inputs=[joint_q, self.target_joint_q, self.dof_to_coord, self.weight, start_idx],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx) -> None:
        count = joint_q.shape[0]
        wp.launch(
            _default_jac_analytic,
            dim=[count, self.n_dofs],
            inputs=[self.dof_to_coord, self.weight, start_idx],
            outputs=[jacobian],
            device=self.device,
        )


# ---------------------------------------------------------------------------
# IK objective: gravity torque minimization
# ---------------------------------------------------------------------------


def _build_subtree_info(
    model: newton.Model,
) -> dict:
    """Extract revolute-joint subtree structure from a Newton Model.

    Returns a dict with:
        n_rev: number of revolute joints
        parent_bodies: ``[n_rev]`` parent body index per revolute joint
        axes_local: ``[n_rev, 3]`` joint axis in parent-body local frame
        downstream_bodies: flattened array of downstream body ids
        downstream_offsets: ``[n_rev + 1]`` CSR offsets into downstream_bodies
        subtree_mass: ``[n_rev]`` total mass of each subtree
    """
    jt = model.joint_type.numpy()
    jp = model.joint_parent.numpy()
    jc = model.joint_child.numpy()
    ja = model.joint_axis.numpy()
    bm = model.body_mass.numpy()
    n_joints = model.joint_count
    n_bodies = model.body_count

    # Build parent map: body -> list of child bodies
    children: dict[int, list[int]] = {i: [] for i in range(-1, n_bodies)}
    for j in range(n_joints):
        p, c = int(jp[j]), int(jc[j])
        children[p].append(c)

    def _get_subtree(root_body: int) -> list[int]:
        """BFS to get all bodies in the subtree rooted at root_body."""
        result = [root_body]
        queue = [root_body]
        while queue:
            b = queue.pop(0)
            for ch in children[b]:
                result.append(ch)
                queue.append(ch)
        return result

    rev_parent_bodies = []
    rev_axes = []
    all_downstream: list[list[int]] = []

    for j in range(n_joints):
        if int(jt[j]) != 1:  # revolute only
            continue
        rev_parent_bodies.append(int(jp[j]))
        rev_axes.append(ja[j].tolist() if j < len(ja) else [1, 0, 0])
        child_body = int(jc[j])
        all_downstream.append(_get_subtree(child_body))

    n_rev = len(rev_parent_bodies)
    flat_downstream = []
    offsets = [0]
    subtree_mass = []
    for ds in all_downstream:
        flat_downstream.extend(ds)
        offsets.append(len(flat_downstream))
        subtree_mass.append(float(sum(bm[b] for b in ds)))

    return {
        "n_rev": n_rev,
        "parent_bodies": np.array(rev_parent_bodies, dtype=np.int32),
        "axes_local": np.array(rev_axes, dtype=np.float32),
        "downstream_bodies": np.array(flat_downstream, dtype=np.int32),
        "downstream_offsets": np.array(offsets, dtype=np.int32),
        "subtree_mass": np.array(subtree_mass, dtype=np.float32),
    }


@wp.kernel
def _compute_subtree_com(
    body_q: wp.array2d(dtype=wp.transform),
    body_com: wp.array1d(dtype=wp.vec3),
    body_mass: wp.array1d(dtype=wp.float32),
    downstream_bodies: wp.array1d(dtype=wp.int32),
    downstream_offsets: wp.array1d(dtype=wp.int32),
    subtree_inv_mass: wp.array1d(dtype=wp.float32),
    subtree_com_out: wp.array2d(dtype=wp.vec3),
):
    """Compute weighted center of mass for each revolute joint's subtree."""
    row, jidx = wp.tid()
    start = downstream_offsets[jidx]
    end_val = downstream_offsets[jidx + 1]
    com = wp.vec3(0.0, 0.0, 0.0)
    for i in range(start, end_val):
        bid = downstream_bodies[i]
        com_world = wp.transform_point(body_q[row, bid], body_com[bid])
        com = com + body_mass[bid] * com_world
    subtree_com_out[row, jidx] = com * subtree_inv_mass[jidx]


@wp.kernel
def _gravity_torque_residuals(
    body_q: wp.array2d(dtype=wp.transform),
    subtree_com: wp.array2d(dtype=wp.vec3),
    subtree_mass: wp.array1d(dtype=wp.float32),
    joint_body: wp.array1d(dtype=wp.int32),
    joint_axis_local: wp.array1d(dtype=wp.vec3),
    gravity: wp.vec3,
    weight: float,
    start_idx: int,
    residuals: wp.array2d(dtype=wp.float32),
):
    """Compute gravity compensation torque residual per revolute joint."""
    row, jidx = wp.tid()
    parent_tf = body_q[row, joint_body[jidx]]
    axis_world = wp.transform_vector(parent_tf, joint_axis_local[jidx])
    joint_pos = wp.transform_get_translation(parent_tf)

    r = subtree_com[row, jidx] - joint_pos
    f = subtree_mass[jidx] * gravity
    torque = wp.dot(axis_world, wp.cross(r, f))

    residuals[row, start_idx + jidx] = weight * torque


@wp.kernel
def _grav_jac_fill_row(
    q_grad: wp.array2d(dtype=wp.float32),
    n_dofs: int,
    row_idx: int,
    jacobian: wp.array3d(dtype=wp.float32),
):
    """Copy autodiff gradient for one residual row into the Jacobian."""
    batch_idx = wp.tid()
    for d in range(n_dofs):
        jacobian[batch_idx, row_idx, d] = q_grad[batch_idx, d]


class IKObjectiveGravityTorque(ik.IKObjective):
    """Minimize static gravity compensation torques for natural poses.

    Computes the torque each revolute joint must exert to hold the
    current configuration against gravity, and penalizes it as a
    residual.  Lower torques correspond to more energy-efficient,
    natural-looking stances.

    Uses autodiff (``MIXED`` mode) -- FK is on the tape, so the solver
    gets gradients of torque w.r.t. joint angles automatically.

    Two Warp kernels are launched per residual evaluation:

    1. :func:`_compute_subtree_com` -- aggregates downstream body
       positions into a weighted center of mass per joint.
    2. :func:`_gravity_torque_residuals` -- computes
       ``weight * axis · (r × mg)`` per joint.

    Args:
        model: Newton model (used to extract kinematic tree).
        weight: Scalar multiplier for the torque residual.
    """

    def __init__(self, model: newton.Model, weight: float = 0.01) -> None:
        super().__init__()
        self.weight = weight

        info = _build_subtree_info(model)
        self.n_rev = info["n_rev"]
        self._parent_bodies_np = info["parent_bodies"]
        self._axes_local_np = info["axes_local"]
        self._downstream_bodies_np = info["downstream_bodies"]
        self._downstream_offsets_np = info["downstream_offsets"]
        self._subtree_mass_np = info["subtree_mass"]

        eps = 1e-10
        self._subtree_inv_mass_np = (1.0 / (self._subtree_mass_np + eps)).astype(np.float32)

        self._gravity_np = model.gravity.numpy()[0].astype(np.float32)
        self._body_com_np = model.body_com.numpy()
        self._body_mass_np = model.body_mass.numpy()

    def supports_analytic(self) -> bool:
        return False

    def residual_dim(self) -> int:
        return self.n_rev

    def init_buffers(self, model: newton.Model, jacobian_mode: ik.IKJacobianType) -> None:
        self._require_batch_layout()
        d = self.device
        self._joint_body = wp.array(self._parent_bodies_np, dtype=wp.int32, device=d)
        self._joint_axis_local = wp.from_numpy(self._axes_local_np, dtype=wp.vec3, device=d)
        self._downstream_bodies = wp.array(self._downstream_bodies_np, dtype=wp.int32, device=d)
        self._downstream_offsets = wp.array(self._downstream_offsets_np, dtype=wp.int32, device=d)
        self._subtree_mass = wp.array(self._subtree_mass_np, dtype=wp.float32, device=d)
        self._subtree_inv_mass = wp.array(self._subtree_inv_mass_np, dtype=wp.float32, device=d)
        self._body_com = wp.from_numpy(self._body_com_np, dtype=wp.vec3, device=d)
        self._body_mass_dev = wp.array(self._body_mass_np, dtype=wp.float32, device=d)
        self._gravity_vec = wp.vec3(*self._gravity_np.tolist())

        self._subtree_com_buf = wp.zeros((self.n_batch, self.n_rev), dtype=wp.vec3, device=d)

        # Autodiff seeds: one per residual row (backward separately per row)
        self._e_arrays = []
        for r in range(self.n_rev):
            e = np.zeros((self.n_batch, self.total_residuals), dtype=np.float32)
            for b in range(self.n_batch):
                e[b, self.residual_offset + r] = 1.0
            self._e_arrays.append(wp.array(e.flatten(), dtype=wp.float32, device=d))

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx) -> None:
        n_batch = body_q.shape[0]
        wp.launch(
            _compute_subtree_com,
            dim=[n_batch, self.n_rev],
            inputs=[
                body_q,
                self._body_com,
                self._body_mass_dev,
                self._downstream_bodies,
                self._downstream_offsets,
                self._subtree_inv_mass,
            ],
            outputs=[self._subtree_com_buf],
            device=self.device,
        )
        wp.launch(
            _gravity_torque_residuals,
            dim=[n_batch, self.n_rev],
            inputs=[
                body_q,
                self._subtree_com_buf,
                self._subtree_mass,
                self._joint_body,
                self._joint_axis_local,
                self._gravity_vec,
                self.weight,
                start_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof) -> None:
        self._require_batch_layout()
        n_dofs = dq_dof.shape[1]
        for r in range(self.n_rev):
            tape.backward(grads={tape.outputs[0]: self._e_arrays[r]})
            q_grad = tape.gradients[dq_dof]
            wp.launch(
                _grav_jac_fill_row,
                dim=self.n_batch,
                inputs=[q_grad, n_dofs, start_idx + r],
                outputs=[jacobian],
                device=self.device,
            )
            tape.zero()


# ---------------------------------------------------------------------------
# IK objective: terrain surface contact
# ---------------------------------------------------------------------------


@wp.kernel
def _terrain_contact_residuals(
    body_q: wp.array2d(dtype=wp.transform),
    mesh_id: wp.uint64,
    foot_body_indices: wp.array1d(dtype=wp.int32),
    weight: float,
    start_idx: int,
    residuals: wp.array2d(dtype=wp.float32),
):
    """Penalize distance from each foot to the nearest terrain surface."""
    row, foot_idx = wp.tid()
    body_idx = foot_body_indices[foot_idx]
    tf = body_q[row, body_idx]
    foot_pos = wp.transform_get_translation(tf)

    query = wp.mesh_query_point(mesh_id, foot_pos, 2.0)
    if query.result:
        closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        dist = wp.length(foot_pos - closest)
        residuals[row, start_idx + foot_idx] = weight * dist


class IKObjectiveTerrainContact(ik.IKObjective):
    """Penalize feet not touching the terrain surface.

    For each foot body, queries the terrain mesh for the closest surface
    point and produces a distance residual.  Uses ``wp.mesh_query_point``
    + ``wp.mesh_eval_position`` (both differentiable in Warp 1.12.1).

    The solver drives feet toward the terrain surface while other
    objectives (spread, gravity torques, base pose) determine the
    lateral position.

    Args:
        mesh_id: Warp mesh identifier (``wp.Mesh.id``).
        foot_body_indices: Newton body indices for the feet.
        weight: Residual weight.
    """

    def __init__(
        self,
        mesh_id: int,
        foot_body_indices: list[int],
        weight: float = 5.0,
    ) -> None:
        super().__init__()
        self.mesh_id = mesh_id
        self._foot_body_indices_np = np.array(foot_body_indices, dtype=np.int32)
        self.n_feet = len(foot_body_indices)
        self.weight = weight

    def supports_analytic(self) -> bool:
        return False

    def residual_dim(self) -> int:
        return self.n_feet

    def init_buffers(self, model: newton.Model, jacobian_mode: ik.IKJacobianType) -> None:
        self._require_batch_layout()
        d = self.device
        self._foot_body_indices = wp.array(self._foot_body_indices_np, dtype=wp.int32, device=d)

        self._e_arrays = []
        for r in range(self.n_feet):
            e = np.zeros((self.n_batch, self.total_residuals), dtype=np.float32)
            for b in range(self.n_batch):
                e[b, self.residual_offset + r] = 1.0
            self._e_arrays.append(wp.array(e.flatten(), dtype=wp.float32, device=d))

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx) -> None:
        n_batch = body_q.shape[0]
        wp.launch(
            _terrain_contact_residuals,
            dim=[n_batch, self.n_feet],
            inputs=[
                body_q,
                self.mesh_id,
                self._foot_body_indices,
                self.weight,
                start_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof) -> None:
        self._require_batch_layout()
        n_dofs = dq_dof.shape[1]
        for r in range(self.n_feet):
            tape.backward(grads={tape.outputs[0]: self._e_arrays[r]})
            q_grad = tape.gradients[dq_dof]
            wp.launch(
                _grav_jac_fill_row,
                dim=self.n_batch,
                inputs=[q_grad, n_dofs, start_idx + r],
                outputs=[jacobian],
                device=self.device,
            )
            tape.zero()


# ---------------------------------------------------------------------------
# IK objective: foot polygon area maximization
# ---------------------------------------------------------------------------


@wp.kernel
def _foot_spread_residuals(
    body_q: wp.array2d(dtype=wp.transform),
    foot_body_indices: wp.array1d(dtype=wp.int32),
    foot_ccw_order: wp.array1d(dtype=wp.int32),
    reference_area: wp.array1d(dtype=wp.float32),
    weight: float,
    start_idx: int,
    problem_idx: wp.array1d(dtype=wp.int32),
    residuals: wp.array2d(dtype=wp.float32),
):
    """Penalize foot polygon area shrinkage relative to reference."""
    row = wp.tid()
    base = problem_idx[row]

    # Shoelace formula for polygon area in CCW order
    area = float(0.0)
    for i in range(4):
        j = (i + 1) % 4
        fi = foot_body_indices[foot_ccw_order[i]]
        fj = foot_body_indices[foot_ccw_order[j]]
        pi = wp.transform_get_translation(body_q[row, fi])
        pj = wp.transform_get_translation(body_q[row, fj])
        area = area + (pi[0] * pj[1] - pj[0] * pi[1])
    area = wp.abs(area) * 0.5

    ratio = area / (reference_area[base] + 1.0e-6)
    residuals[row, start_idx] = weight * (1.0 - ratio)


class IKObjectiveFootSpread(ik.IKObjective):
    """Encourage feet to match the reference support polygon area.

    Computes the area of the foot polygon (shoelace formula in CCW
    order) and penalizes deviation from the reference area.
    The residual is ``weight * (1 - area/reference_area)``.

    Args:
        foot_body_indices: Newton body indices for the feet.
        foot_ccw_order: CCW winding order of feet (indices into foot_body_indices).
        reference_area: Per-problem reference polygon area ``[n_problems]``.
        weight: Residual weight.
    """

    def __init__(
        self,
        foot_body_indices: list[int],
        foot_ccw_order: list[int],
        reference_area: wp.array,
        weight: float = 0.5,
    ) -> None:
        super().__init__()
        self._foot_body_indices_np = np.array(foot_body_indices, dtype=np.int32)
        self._foot_ccw_order_np = np.array(foot_ccw_order, dtype=np.int32)
        self.reference_area = reference_area
        self.weight = weight

    def supports_analytic(self) -> bool:
        return False

    def residual_dim(self) -> int:
        return 1

    def init_buffers(self, model: newton.Model, jacobian_mode: ik.IKJacobianType) -> None:
        self._require_batch_layout()
        d = self.device
        self._foot_body_indices = wp.array(self._foot_body_indices_np, dtype=wp.int32, device=d)
        self._foot_ccw_order = wp.array(self._foot_ccw_order_np, dtype=wp.int32, device=d)

        e = np.zeros((self.n_batch, self.total_residuals), dtype=np.float32)
        for b in range(self.n_batch):
            e[b, self.residual_offset] = 1.0
        self._e_array = wp.array(e.flatten(), dtype=wp.float32, device=d)

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx) -> None:
        n_batch = body_q.shape[0]
        wp.launch(
            _foot_spread_residuals,
            dim=n_batch,
            inputs=[
                body_q,
                self._foot_body_indices,
                self._foot_ccw_order,
                self.reference_area,
                self.weight,
                start_idx,
                problem_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof) -> None:
        self._require_batch_layout()
        n_dofs = dq_dof.shape[1]
        tape.backward(grads={tape.outputs[0]: self._e_array})
        q_grad = tape.gradients[dq_dof]
        wp.launch(
            _grav_jac_fill_row,
            dim=self.n_batch,
            inputs=[q_grad, n_dofs, start_idx],
            outputs=[jacobian],
            device=self.device,
        )
        tape.zero()


# ---------------------------------------------------------------------------
# IK objective: stability margin (CoM over support centroid)
# ---------------------------------------------------------------------------


@wp.kernel
def _stability_margin_residuals(
    body_q: wp.array2d(dtype=wp.transform),
    body_mass: wp.array1d(dtype=wp.float32),
    n_bodies: int,
    active_mask: wp.array2d(dtype=wp.int32),
    foot_body_indices: wp.array1d(dtype=wp.int32),
    n_feet: int,
    total_mass_inv: float,
    weight: float,
    start_idx: int,
    residuals: wp.array2d(dtype=wp.float32),
):
    """Penalize CoM XY distance from active-feet centroid."""
    row = wp.tid()
    com_x = float(0.0)
    com_y = float(0.0)
    for b in range(n_bodies):
        pos = wp.transform_get_translation(body_q[row, b])
        com_x = com_x + body_mass[b] * pos[0]
        com_y = com_y + body_mass[b] * pos[1]
    com_x = com_x * total_mass_inv
    com_y = com_y * total_mass_inv

    cx = float(0.0)
    cy = float(0.0)
    n_active = float(0.0)
    for f in range(n_feet):
        if active_mask[row, f] > 0:
            fid = foot_body_indices[f]
            fp = wp.transform_get_translation(body_q[row, fid])
            cx = cx + fp[0]
            cy = cy + fp[1]
            n_active = n_active + 1.0
    if n_active > 0.0:
        cx = cx / n_active
        cy = cy / n_active

    dx = com_x - cx
    dy = com_y - cy
    dist = wp.sqrt(dx * dx + dy * dy + 1.0e-8)
    residuals[row, start_idx] = weight * dist


class IKObjectiveStabilityMargin(ik.IKObjective):
    """Center the CoM over the support polygon for static stability.

    Computes the XY distance from the robot's center of mass to the
    centroid of the active (ground-contact) feet and penalizes it.
    Smaller distance = more stable stance.

    Handles both quad (4 active feet) and tripod (3 active feet)
    configurations via a per-candidate active mask.

    Uses autodiff (``MIXED`` mode).

    Args:
        model: Newton model (for body masses).
        foot_body_indices: Newton body indices for the feet.
        active_mask: Per-candidate active foot mask ``[n_problems, nc]``,
            1 for feet on the ground, 0 for omitted feet.
        weight: Residual weight.
    """

    def __init__(
        self,
        model: newton.Model,
        foot_body_indices: list[int],
        active_mask: wp.array,
        weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.weight = weight
        self._foot_body_indices_np = np.array(foot_body_indices, dtype=np.int32)
        self.n_feet = len(foot_body_indices)
        self.n_bodies = model.body_count

        bm = model.body_mass.numpy()
        self._total_mass_inv = float(1.0 / (bm.sum() + 1e-10))
        self._body_mass_np = bm.astype(np.float32)
        self.active_mask = active_mask

    def supports_analytic(self) -> bool:
        return False

    def residual_dim(self) -> int:
        return 1

    def init_buffers(self, model: newton.Model, jacobian_mode: ik.IKJacobianType) -> None:
        self._require_batch_layout()
        d = self.device
        self._foot_body_indices = wp.array(self._foot_body_indices_np, dtype=wp.int32, device=d)
        self._body_mass_dev = wp.array(self._body_mass_np, dtype=wp.float32, device=d)

        e = np.zeros((self.n_batch, self.total_residuals), dtype=np.float32)
        for b in range(self.n_batch):
            e[b, self.residual_offset] = 1.0
        self._e_array = wp.array(e.flatten(), dtype=wp.float32, device=d)

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx) -> None:
        n_batch = body_q.shape[0]
        wp.launch(
            _stability_margin_residuals,
            dim=n_batch,
            inputs=[
                body_q,
                self._body_mass_dev,
                self.n_bodies,
                self.active_mask,
                self._foot_body_indices,
                self.n_feet,
                self._total_mass_inv,
                self.weight,
                start_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof) -> None:
        self._require_batch_layout()
        n_dofs = dq_dof.shape[1]
        tape.backward(grads={tape.outputs[0]: self._e_array})
        q_grad = tape.gradients[dq_dof]
        wp.launch(
            _grav_jac_fill_row,
            dim=self.n_batch,
            inputs=[q_grad, n_dofs, start_idx],
            outputs=[jacobian],
            device=self.device,
        )
        tape.zero()


# ---------------------------------------------------------------------------
# IK objective: terrain collision avoidance
# ---------------------------------------------------------------------------


@wp.kernel
def _terrain_collision_residuals(
    body_q: wp.array2d(dtype=wp.transform),
    mesh_id: wp.uint64,
    probe_body: wp.array1d(dtype=wp.int32),
    probe_offset: wp.array1d(dtype=wp.vec3),
    weight: float,
    margin: float,
    start_idx: int,
    residuals: wp.array2d(dtype=wp.float32),
):
    """Softplus penetration penalty per probe point."""
    row, probe_idx = wp.tid()
    body_idx = probe_body[probe_idx]
    tf = body_q[row, body_idx]
    probe_pos = wp.transform_point(tf, probe_offset[probe_idx])

    query = wp.mesh_query_point(mesh_id, probe_pos, 2.0)
    if query.result:
        closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        dist = wp.length(probe_pos - closest)
        penetration = -query.sign * dist
        pen = wp.log(1.0 + wp.exp(penetration / margin)) * margin
        residuals[row, start_idx + probe_idx] = weight * pen


def _build_collision_probes(
    builder: newton.ModelBuilder,
    exclude_bodies: list[int],
    n_samples: int = 16,
) -> tuple[list[int], list[tuple[float, float, float]]]:
    """Sample probe points on each body's actual mesh surface.

    Collects visual mesh vertices from the ``ModelBuilder``'s
    ``shape_source`` per body, then subsamples ``n_samples`` points
    using farthest-point sampling for uniform coverage.

    Args:
        builder: Newton model builder (before or after finalize) --
            provides ``shape_source`` with mesh vertex data.
        exclude_bodies: Body indices to skip (e.g. foot bodies).
        n_samples: Max number of surface sample points per body.

    Returns:
        ``(probe_bodies, probe_offsets)``
    """
    from collections import defaultdict

    # Collect all mesh vertices per body from visual shapes
    body_verts: dict[int, list[np.ndarray]] = defaultdict(list)
    for si in range(len(builder.shape_body)):
        bid = int(builder.shape_body[si])
        if bid in exclude_bodies:
            continue
        src = builder.shape_source[si]
        if src is not None and hasattr(src, "vertices") and len(src.vertices) > 0:
            # Apply shape transform (local offset within the body)
            verts = np.array(src.vertices, dtype=np.float32)
            if len(verts.shape) == 1:
                verts = verts.reshape(-1, 3)
            body_verts[bid].append(verts)

    probe_bodies: list[int] = []
    probe_offsets: list[tuple[float, float, float]] = []

    for bid in sorted(body_verts.keys()):
        all_v = np.concatenate(body_verts[bid], axis=0)
        if len(all_v) == 0:
            continue

        # Subsample using farthest-point sampling
        n = min(n_samples, len(all_v))
        if n <= 0:
            continue

        # Simple greedy FPS
        selected = [0]
        min_dists = np.full(len(all_v), np.inf)
        for _ in range(n - 1):
            last = all_v[selected[-1]]
            d = np.linalg.norm(all_v - last, axis=1)
            min_dists = np.minimum(min_dists, d)
            selected.append(int(np.argmax(min_dists)))

        for idx in selected:
            probe_bodies.append(bid)
            probe_offsets.append(tuple(float(x) for x in all_v[idx]))

    return probe_bodies, probe_offsets


class IKObjectiveTerrainCollision(ik.IKObjective):
    """Penalize robot body surface points penetrating the terrain mesh.

    Samples points on each body's AABB surface and penalizes any that
    are inside the terrain mesh.  No artificial collision radii -- the
    sample points represent the actual body surface, and signed
    distance from ``mesh_query_point`` determines penetration.

    Uses a softplus penalty for smooth gradients (same as Newton's
    collision avoidance example).

    Args:
        mesh_id: Warp mesh identifier (``wp.Mesh.id``).
        builder: Newton model builder (provides ``shape_source`` with
            mesh vertex data for surface sampling).
        exclude_bodies: Body indices to skip (e.g. foot bodies).
        weight: Residual weight.
        margin: Softplus temperature [m].
        n_samples: Number of surface sample points per body.
    """

    def __init__(
        self,
        mesh_id: int,
        builder: newton.ModelBuilder,
        exclude_bodies: list[int],
        weight: float = 3.0,
        margin: float = 0.05,
        n_samples: int = 16,
    ) -> None:
        super().__init__()
        self.mesh_id = mesh_id
        self.weight = weight
        self.margin = margin

        bodies, offsets = _build_collision_probes(builder, exclude_bodies, n_samples)
        self.n_probes = len(bodies)
        self._probe_body_np = np.array(bodies, dtype=np.int32)
        self._probe_offset_np = np.array(offsets, dtype=np.float32)

    def supports_analytic(self) -> bool:
        return False

    def residual_dim(self) -> int:
        return self.n_probes

    def init_buffers(self, model: newton.Model, jacobian_mode: ik.IKJacobianType) -> None:
        self._require_batch_layout()
        d = self.device
        self._probe_body = wp.array(self._probe_body_np, dtype=wp.int32, device=d)
        self._probe_offset = wp.from_numpy(self._probe_offset_np, dtype=wp.vec3, device=d)

        self._e_arrays = []
        for r in range(self.n_probes):
            e = np.zeros((self.n_batch, self.total_residuals), dtype=np.float32)
            for b in range(self.n_batch):
                e[b, self.residual_offset + r] = 1.0
            self._e_arrays.append(wp.array(e.flatten(), dtype=wp.float32, device=d))

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx) -> None:
        n_batch = body_q.shape[0]
        wp.launch(
            _terrain_collision_residuals,
            dim=[n_batch, self.n_probes],
            inputs=[
                body_q,
                self.mesh_id,
                self._probe_body,
                self._probe_offset,
                self.weight,
                self.margin,
                start_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof) -> None:
        self._require_batch_layout()
        n_dofs = dq_dof.shape[1]
        for r in range(self.n_probes):
            tape.backward(grads={tape.outputs[0]: self._e_arrays[r]})
            q_grad = tape.gradients[dq_dof]
            wp.launch(
                _grav_jac_fill_row,
                dim=self.n_batch,
                inputs=[q_grad, n_dofs, start_idx + r],
                outputs=[jacobian],
                device=self.device,
            )
            tape.zero()
