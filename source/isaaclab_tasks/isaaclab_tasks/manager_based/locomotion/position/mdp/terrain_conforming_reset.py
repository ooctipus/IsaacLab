# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain-conforming reset event powered by the retarget pipeline.

Owns a :class:`RetargetPipeline` and a pre-computed state buffer.
The pipeline is run once (or periodically) to generate a pool of valid
reset states; each ``__call__`` draws from that pool.

Other manager terms (commands, curriculum) can access the reset data
through :attr:`state_buffer` and :meth:`get_reset_state`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .kinematics import NewtonKinematics
from .retarget import RetargetBuffer, RetargetPipeline, RetargetPipelineCfg

from ..mdp_presets.sampling import SupportPolygonSampler, SupportPolygonSamplerCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import EventTermCfg


class TerrainConformingReset(ManagerTermBase):
    """Reset robots into terrain-conforming stances from a pre-computed pool.

    On initialization:
        1. Builds a :class:`NewtonKinematics` model from the robot USD.
        2. Computes default stance geometry (foot offsets, standing height).
        3. Creates a :class:`RetargetPipeline`.

    On first call (or when the pool is exhausted):
        Runs the pipeline against the terrain mesh to generate a pool of
        valid (joint_q, base_pose) pairs.

    On each reset call:
        Draws states from the pool and writes them to PhysX.

    Required params:
        - ``asset_cfg``: :class:`SceneEntityCfg` for the robot articulation.
        - ``foot_cfg``: :class:`SceneEntityCfg` with ``body_names`` for the
          foot bodies (e.g. ``SceneEntityCfg("robot", body_names=".*FOOT.*")``).

    Example::

        EventTermCfg(
            func=TerrainConformingReset,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "foot_cfg": SceneEntityCfg("robot", body_names=".*FOOT.*"),
            },
        )
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset = env.scene[asset_cfg.name]

        usd_path = self.asset.cfg.spawn.usd_path
        if usd_path.startswith("http"):
            raise RuntimeError(
                f"TerrainConformingReset requires a local USD path, got: {usd_path}"
            )

        # Foot body names resolved by the framework from SceneEntityCfg
        foot_cfg: SceneEntityCfg = cfg.params["foot_cfg"]
        foot_body_names: list[str] = foot_cfg.body_names  # type: ignore[assignment]

        default_jpos = wp.to_torch(self.asset.data.default_joint_pos)[0].cpu().numpy()

        kin = NewtonKinematics(
            usd_path, device=self.device,
            default_pos=self.asset.cfg.init_state.pos,
            default_quat=self.asset.cfg.init_state.rot,
            default_joint_pos=default_jpos,
        )
        newton_foot_ids = [kin.body_names.index(n) for n in foot_body_names]

        base_pos = kin.default_body_q[0][:3]
        foot_positions = np.array([kin.default_body_q[fid][:3] for fid in newton_foot_ids])
        foot_offsets = foot_positions - base_pos
        standing_height = float(base_pos[2] - foot_positions[:, 2].mean())
        foot_ground_offset = float(foot_positions[:, 2].min())

        # Pipeline config
        pipeline_cfg = cfg.params.get("pipeline_cfg", RetargetPipelineCfg(device=self.device))
        if isinstance(pipeline_cfg, dict):
            pipeline_cfg = RetargetPipelineCfg(**pipeline_cfg)

        sampler = SupportPolygonSampler(
            SupportPolygonSamplerCfg(),
            foot_offsets=foot_offsets,
            foot_ground_offset=foot_ground_offset,
            standing_height=standing_height,
            default_joint_q=kin.default_joint_q,
        )

        import newton.ik as _ik

        def _objectives_factory(n_problems):
            device = kin.device
            contact_objs = [
                _ik.IKObjectivePosition(
                    link_index=fid, link_offset=wp.vec3(0, 0, 0),
                    target_positions=wp.zeros(n_problems, dtype=wp.vec3, device=device), weight=1.0,
                )
                for fid in newton_foot_ids
            ]
            base_pos_obj = _ik.IKObjectivePosition(
                link_index=0, link_offset=wp.vec3(0, 0, 0),
                target_positions=wp.zeros(n_problems, dtype=wp.vec3, device=device), weight=0.05,
            )
            base_rot_obj = _ik.IKObjectiveRotation(
                link_index=0, link_offset_rotation=wp.quat_identity(),
                target_rotations=wp.zeros(n_problems, dtype=wp.vec4, device=device), weight=0.5,
            )
            jl_obj = _ik.IKObjectiveJointLimit(
                joint_limit_lower=kin.model.joint_limit_lower,
                joint_limit_upper=kin.model.joint_limit_upper, weight=10.0,
            )
            return [*contact_objs, base_pos_obj, base_rot_obj, jl_obj], contact_objs, base_pos_obj, base_rot_obj

        self.pipeline = RetargetPipeline(
            kin=kin,
            sampler=sampler,
            objectives_factory=_objectives_factory,
            cfg=pipeline_cfg,
            contact_body_ids=newton_foot_ids,
        )

        # State pool
        self._pool_joint_q: torch.Tensor | None = None
        self._pool_base_pose: torch.Tensor | None = None
        self._pool_size: int = 0
        self._pool_cursor: int = 0

        self._pool_desired = int(cfg.params.get("pool_size", 200))
        self._velocity_range = cfg.params.get("velocity_range", None)

        # Strip consumed params
        cfg.params = {}

    @property
    def state_buffer(self) -> RetargetBuffer:
        """Direct access to the pipeline's GPU buffer (zero-copy)."""
        return self.pipeline.buffer

    def get_reset_state(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw ``n`` (joint_q, base_pose) pairs from the pool.

        If the pool is exhausted, wraps around.

        Returns:
            ``(joint_q [n, num_joints], base_pose [n, 7])`` on ``self.device``.
        """
        if self._pool_size == 0:
            raise RuntimeError("No valid reset states available. Run the pipeline first.")

        indices = torch.arange(n, device=self.device) + self._pool_cursor
        indices = indices % self._pool_size
        self._pool_cursor = (self._pool_cursor + n) % self._pool_size

        return self._pool_joint_q[indices], self._pool_base_pose[indices]

    def _fill_pool(self, env: ManagerBasedEnv) -> None:
        """Run the retarget pipeline against the terrain mesh."""
        terrain = env.scene.terrain
        wp_mesh = terrain.warp_meshes[0] if hasattr(terrain, "warp_meshes") else None
        if wp_mesh is None:
            return

        origin = np.zeros(3)
        buf = self.pipeline.run(wp_mesh, origin, self._pool_desired)

        if buf.num_selected == 0:
            return

        n = buf.num_selected
        sel_t = buf._selected[:n].long()
        selected_jq = buf.joint_q_result_t[sel_t]
        self._pool_base_pose = selected_jq[:, :7].contiguous()
        self._pool_joint_q = selected_jq[:, 7:].contiguous()
        self._pool_size = n
        self._pool_cursor = 0

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg | None = None,
    ):
        N = env_ids.shape[0]
        if N == 0:
            return

        if self._pool_size == 0:
            self._fill_pool(env)

        if self._pool_size == 0:
            return

        joint_q, base_pose = self.get_reset_state(N)

        self.asset.write_root_pose_to_sim_index(root_pose=base_pose, env_ids=env_ids)

        zero_vel = torch.zeros(N, 6, device=self.device)
        if self._velocity_range:
            keys = ["x", "y", "z", "roll", "pitch", "yaw"]
            for i, k in enumerate(keys):
                if k in self._velocity_range:
                    lo, hi = self._velocity_range[k]
                    zero_vel[:, i].uniform_(lo, hi)
        self.asset.write_root_velocity_to_sim_index(root_velocity=zero_vel, env_ids=env_ids)
        self.asset.write_joint_position_to_sim_index(position=joint_q, env_ids=env_ids)
        self.asset.write_joint_velocity_to_sim_index(
            velocity=torch.zeros_like(joint_q), env_ids=env_ids
        )
