# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms for the dexsuite tasks."""

from __future__ import annotations

import numpy as np
import torch
import warp as wp
from typing import TYPE_CHECKING

from isaaclab.managers import EventTermCfg, ManagerTermBase, ManagerTermBaseCfg

from .utils import collect_body_collision_meshes, get_reset_state, sample_object_point_cloud, set_reset_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.assets import Articulation
    from .events_cfg import MeshClearanceCfg, SlabClearanceCfg


class conditional_reset(ManagerTermBase):
    """Run wrapped reset terms and guarantee the resulting states satisfy a criterion.

    Wraps a dict of ordinary reset event terms. The nested :class:`EventTermCfg` objects are
    resolved by the event manager's own at-play pass (nested term configs inside ``params``
    are processed recursively), so this term only *calls* them and never resolves functions,
    scene entities, or class terms itself.

    On the first reset, the wrapped terms are re-rolled and the states satisfying
    :paramref:`valid_criteria` are harvested into a buffer of :paramref:`buffer_size_per_group`
    samples per group (rejection sampling, amortized once). On subsequent resets the wrapped terms run
    exactly once; environments whose fresh state fails the criterion are overwritten with a
    random buffered sample instead of re-rolling, keeping resets constant-time.

    The captured state is the reset surface of the scene (see :func:`get_reset_state`):
    root pose/velocity plus joint positions/velocities of every articulation, and the root
    pose/velocity of every rigid object, buffered relative to the environment origins so a
    sample harvested in one environment can be replayed in another.

    With heterogeneous cloning (e.g. multi-asset spawned objects), environments are only
    interchangeable within the same unique asset combination: a state harvested in a cube
    environment is not a valid state for a capsule environment. The buffer is therefore
    partitioned by the scene's clone plan — an environment's column of the plan's clone mask
    is its asset-combination signature — as ``[num_groups * buffer_size_per_group]`` rows,
    and failing environments are only patched from their own group's partition.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._prefilled = False
        self._buffer: torch.Tensor | None = None
        self._reset_assets = list(env.scene.articulations) + list(env.scene.rigid_objects)
        self._group: torch.Tensor | None = None
        self._fill: torch.Tensor | None = None

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        terms: dict[str, EventTermCfg],
        valid_criteria: dict[str, ManagerTermBaseCfg],
        buffer_size_per_group: int = 20,
        max_prefill_iters: int = 50,
    ):
        """Apply the wrapped reset terms, patching criterion failures from the valid-state buffer.

        Args:
            env: The environment.
            env_ids: Environments being reset.
            terms: Reset event terms to wrap, applied in insertion order. Resolved by the
                event manager before the first call.
            valid_criteria: Criteria as term configs (e.g. :class:`SlabClearanceCfg`,
                :class:`MeshClearanceCfg`), each evaluated as
                ``func(env, env_ids, **params) -> BoolTensor`` over the freshly reset
                environments and combined with logical AND. Resolved by the event manager
                like any nested term config.
            buffer_size_per_group: Number of valid states to bank per unique asset
                combination during the prefill phase.
            max_prefill_iters: Re-roll budget for the prefill phase.
        """

        def roll_once() -> torch.Tensor:
            for term in terms.values():
                term.func(env, env_ids, **term.params)
            # no explicit refresh needed: state writes invalidate the FK timestamps and the
            # criteria's kinematic reads recompute on demand
            ok = torch.ones(len(env_ids), dtype=torch.bool, device=env_ids.device)
            for criterion in valid_criteria.values():
                ok &= criterion.func(env, env_ids, **criterion.params)
            return ok

        if not self._prefilled:
            plan = getattr(env.scene, "clone_plan", None)
            if plan is None:
                self._group = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
            else:
                # envs sharing a clone-mask column are clones of the same unique asset combination
                mask = plan.clone_mask.to(device=env.device, dtype=torch.uint8)
                self._group = torch.unique(mask.T, dim=0, return_inverse=True)[1]
            num_groups = int(self._group.max().item()) + 1
            self._fill = torch.zeros(num_groups, dtype=torch.long, device=env.device)
            for _ in range(max_prefill_iters):
                valid_ids = env_ids[roll_once()]
                for group in torch.unique(self._group[valid_ids]).tolist():
                    filled = int(self._fill[group])
                    take = valid_ids[self._group[valid_ids] == group][: buffer_size_per_group - filled]
                    if len(take) == 0:
                        continue
                    state = get_reset_state(env, take, self._reset_assets, is_relative=True)
                    if self._buffer is None:
                        capacity = num_groups * buffer_size_per_group
                        self._buffer = torch.empty(capacity, state.shape[-1], device=state.device, dtype=state.dtype)
                    row = group * buffer_size_per_group + filled
                    self._buffer[row : row + len(take)] = state
                    self._fill[group] += len(take)
                if bool((self._fill >= buffer_size_per_group).all()):
                    break
            if bool((self._fill == 0).any()):
                empty = torch.nonzero(self._fill == 0).view(-1).tolist()
                raise RuntimeError(
                    f"conditional_reset: no valid state found during prefill for asset-combination"
                    f" group(s) {empty}; the criterion may be unsatisfiable under the wrapped"
                    " reset terms for those environments."
                )
            self._prefilled = True

        ok = roll_once()
        # Branchless patching: draw a donor row for every resetting env from its own
        # asset-combination partition and blend by the validity mask. Checking the mask on
        # the host instead (``bool(ok.all())``, per-group ``.tolist()``/``.sum()`` loop)
        # stalls the CPU on the whole GPU pipeline once per step — measured at half of the
        # env.step host time at 4096 envs, where some env resets nearly every step.
        # ``rand * fill`` floors to a uniform draw in ``[0, fill)`` per env, so partially
        # filled partitions stay in-range. Valid envs are rewritten with their own freshly
        # read state — an identity round-trip through the same write path.
        groups = self._group[env_ids]
        donor = (torch.rand(len(env_ids), device=env_ids.device) * self._fill[groups]).long()
        rows = groups * buffer_size_per_group + donor
        fresh = get_reset_state(env, env_ids, self._reset_assets, is_relative=True)
        blended = torch.where(ok.unsqueeze(-1), fresh, self._buffer[rows])
        set_reset_state(env, blended, env_ids, self._reset_assets, is_relative=True)


@wp.func
def _slab_signed_dist(
    point_env: wp.vec3,
    slab_top: wp.array(dtype=wp.float32),
    slab_x: wp.array2d(dtype=wp.float32),
    slab_y: wp.array2d(dtype=wp.float32),
    slab_has_x: wp.array(dtype=wp.int32),
    slab_has_y: wp.array(dtype=wp.int32),
    margin: float,
    current: float,
) -> float:
    result = current
    for s in range(slab_top.shape[0]):
        inside = True
        if slab_has_x[s] != 0 and (point_env[0] < slab_x[s, 0] - margin or point_env[0] > slab_x[s, 1] + margin):
            inside = False
        if slab_has_y[s] != 0 and (point_env[1] < slab_y[s, 0] - margin or point_env[1] > slab_y[s, 1] + margin):
            inside = False
        if inside:
            result = wp.min(result, point_env[2] - slab_top[s])
    return result


@wp.kernel
def _object_points_slab_min(
    env_ids: wp.array(dtype=wp.int32),
    obj_points: wp.array2d(dtype=wp.vec3),
    obj_pose: wp.array(dtype=wp.transformf),
    env_origins: wp.array(dtype=wp.vec3),
    slab_top: wp.array(dtype=wp.float32),
    slab_x: wp.array2d(dtype=wp.float32),
    slab_y: wp.array2d(dtype=wp.float32),
    slab_has_x: wp.array(dtype=wp.int32),
    slab_has_y: wp.array(dtype=wp.int32),
    margin: float,
    out_min: wp.array(dtype=wp.float32),
):
    i, k = wp.tid()
    env = env_ids[i]
    point_env = wp.transform_point(obj_pose[env], obj_points[env, k]) - env_origins[env]
    dist = _slab_signed_dist(point_env, slab_top, slab_x, slab_y, slab_has_x, slab_has_y, margin, out_min[i])
    wp.atomic_min(out_min, i, dist)


@wp.kernel
def _robot_vertices_slab_min(
    env_ids: wp.array(dtype=wp.int32),
    vertices: wp.array(dtype=wp.vec3),
    vertex_body: wp.array(dtype=wp.int32),
    body_pose: wp.array2d(dtype=wp.transformf),
    env_origins: wp.array(dtype=wp.vec3),
    slab_top: wp.array(dtype=wp.float32),
    slab_x: wp.array2d(dtype=wp.float32),
    slab_y: wp.array2d(dtype=wp.float32),
    slab_has_x: wp.array(dtype=wp.int32),
    slab_has_y: wp.array(dtype=wp.int32),
    margin: float,
    out_min: wp.array(dtype=wp.float32),
):
    i, k = wp.tid()
    env = env_ids[i]
    point_env = wp.transform_point(body_pose[env, vertex_body[k]], vertices[k]) - env_origins[env]
    dist = _slab_signed_dist(point_env, slab_top, slab_x, slab_y, slab_has_x, slab_has_y, margin, out_min[i])
    wp.atomic_min(out_min, i, dist)


@wp.kernel
def _object_points_mesh_min(
    env_ids: wp.array(dtype=wp.int32),
    obj_points: wp.array2d(dtype=wp.vec3),
    obj_pose: wp.array(dtype=wp.transformf),
    body_pose: wp.array2d(dtype=wp.transformf),
    mesh_ids: wp.array(dtype=wp.uint64),
    mesh_body: wp.array(dtype=wp.int32),
    max_dist: float,
    out_min: wp.array(dtype=wp.float32),
):
    i, k = wp.tid()
    env = env_ids[i]
    point_world = wp.transform_point(obj_pose[env], obj_points[env, k])
    dist = out_min[i]
    for m in range(mesh_ids.shape[0]):
        point_local = wp.transform_point(wp.transform_inverse(body_pose[env, mesh_body[m]]), point_world)
        query = wp.mesh_query_point_sign_winding_number(mesh_ids[m], point_local, max_dist)
        if query.result:
            mesh_dist = wp.length(point_local - wp.mesh_eval_position(mesh_ids[m], query.face, query.u, query.v))
            if query.sign < 0.0:
                mesh_dist = -mesh_dist
            dist = wp.min(dist, mesh_dist)
    wp.atomic_min(out_min, i, dist)


class mesh_clearance(ManagerTermBase):
    """Valid when the object's surface clears the robot's collision meshes.

    Reset draws can place the object overlapping the arm; the solver resolves the overlap
    ballistically at episode birth. Checks the object's surface point cloud against the
    robot's collision meshes with Warp signed-distance queries — the winding-number sign
    catches full containment.

    The object point cloud comes from the same sampler as the point-cloud observation (per
    clone-plan prototype, geometry-keyed cache), so with the default count the cloud is
    shared, not recomputed. Robot collision meshes are extracted once from the USD collision
    prims and baked into each body's frame.

    Configured with :class:`MeshClearanceCfg`; called as ``(env, env_ids) -> BoolTensor``,
    ``True`` where the state is valid.
    """

    cfg: MeshClearanceCfg

    def __init__(self, cfg: MeshClearanceCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        device = env.device
        self._robot: Articulation = env.scene[cfg.asset_name]
        self._object = env.scene[cfg.object_name]
        points = sample_object_point_cloud(
            env.num_envs, cfg.num_object_points, self._object.cfg.prim_path, device=device
        )
        self._obj_points = wp.from_torch(points.contiguous(), dtype=wp.vec3)

        body_meshes, _ = collect_body_collision_meshes(self._robot, cfg.body_names)
        self._meshes = []
        mesh_body = []
        for body_id, mesh in body_meshes.items():
            self._meshes.append(
                wp.Mesh(
                    points=wp.array(mesh.vertices, dtype=wp.vec3, device=device),
                    indices=wp.array(mesh.faces.reshape(-1), dtype=wp.int32, device=device),
                    support_winding_number=True,
                )
            )
            mesh_body.append(body_id)
        self._mesh_ids = wp.array([mesh.id for mesh in self._meshes], dtype=wp.uint64, device=device)
        self._mesh_body = wp.array(mesh_body, dtype=wp.int32, device=device)
        # query horizon: must exceed both the clearance and plausible penetration depths so
        # contained points still resolve a (negative) signed distance
        self._max_dist = max(4.0 * cfg.min_clearance, 0.15)

    def __call__(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> torch.Tensor:
        num = len(env_ids)
        out_min = wp.full(num, 1.0e6, dtype=wp.float32, device=env.device)
        wp.launch(
            _object_points_mesh_min,
            dim=(num, self.cfg.num_object_points),
            inputs=[
                wp.from_torch(env_ids.to(torch.int32).contiguous()),
                self._obj_points,
                self._object.data.root_link_pose_w.warp,
                self._robot.data.body_link_pose_w.warp,
                self._mesh_ids,
                self._mesh_body,
                self._max_dist,
                out_min,
            ],
            device=env.device,
        )
        return wp.to_torch(out_min) >= self.cfg.min_clearance


class slab_clearance(ManagerTermBase):
    """Valid when the object's surface and the robot's collision geometry clear the slabs.

    Reset draws can pose the arm into the table (depenetration slams joints to several times
    their velocity limit within steps) and spawn long shapes with random orientation
    intersecting the tabletop. Checks the object's surface point cloud and the robot's
    collision-mesh vertices against horizontal obstacle slabs in the environment frame.

    Configured with :class:`SlabClearanceCfg`; called as ``(env, env_ids) -> BoolTensor``,
    ``True`` where the state is valid.
    """

    cfg: SlabClearanceCfg

    def __init__(self, cfg: SlabClearanceCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        device = env.device
        self._robot: Articulation = env.scene[cfg.asset_name]
        self._object = env.scene[cfg.object_name]
        points = sample_object_point_cloud(
            env.num_envs, cfg.num_object_points, self._object.cfg.prim_path, device=device
        )
        self._obj_points = wp.from_torch(points.contiguous(), dtype=wp.vec3)

        body_meshes, _ = collect_body_collision_meshes(self._robot, cfg.body_names)
        vertices, vertex_body = [], []
        for body_id, mesh in body_meshes.items():
            vertices.append(mesh.vertices)
            vertex_body.extend([body_id] * len(mesh.vertices))
        self._vertices = wp.array(np.concatenate(vertices), dtype=wp.vec3, device=device)
        self._vertex_body = wp.array(vertex_body, dtype=wp.int32, device=device)

        slabs = cfg.obstacle_slabs
        self._slab_args = [
            wp.array(np.array([top_z for _, _, top_z in slabs], dtype=np.float32), device=device),
            wp.array(np.array([x or (0.0, 0.0) for x, _, _ in slabs], dtype=np.float32).reshape(-1, 2), device=device),
            wp.array(np.array([y or (0.0, 0.0) for _, y, _ in slabs], dtype=np.float32).reshape(-1, 2), device=device),
            wp.array(np.array([x is not None for x, _, _ in slabs], dtype=np.int32), device=device),
            wp.array(np.array([y is not None for _, y, _ in slabs], dtype=np.int32), device=device),
        ]
        self._env_origins = wp.from_torch(env.scene.env_origins.contiguous(), dtype=wp.vec3)

    def __call__(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> torch.Tensor:
        num = len(env_ids)
        out_min = wp.full(num, 1.0e6, dtype=wp.float32, device=env.device)
        ids = wp.from_torch(env_ids.to(torch.int32).contiguous())
        wp.launch(
            _object_points_slab_min,
            dim=(num, self.cfg.num_object_points),
            inputs=[
                ids,
                self._obj_points,
                self._object.data.root_link_pose_w.warp,
                self._env_origins,
                *self._slab_args,
                self.cfg.min_clearance,
                out_min,
            ],
            device=env.device,
        )
        wp.launch(
            _robot_vertices_slab_min,
            dim=(num, len(self._vertices)),
            inputs=[
                ids,
                self._vertices,
                self._vertex_body,
                self._robot.data.body_link_pose_w.warp,
                self._env_origins,
                *self._slab_args,
                self.cfg.min_clearance,
                out_min,
            ],
            device=env.device,
        )
        return wp.to_torch(out_min) >= self.cfg.min_clearance
