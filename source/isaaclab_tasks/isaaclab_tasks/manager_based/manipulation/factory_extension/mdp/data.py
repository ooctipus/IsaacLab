from __future__ import annotations

import torch
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence
from isaaclab.utils import math as math_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from . import key_point_maths as advanced_math
from isaaclab.managers import DataTerm

if TYPE_CHECKING:
    from isaaclab.assets import RigidObjectCollection
    from ..tasks import SuccessCondition
    from .data_cfg import AlignmentMetricCfg, DiameterLookUpCfg, KeyPointTrackerCfg


# from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers
# frame_marker_cfg = FRAME_MARKER_CFG.copy()  # type: ignore
# frame_marker_cfg.markers["frame"].scale = (0.025, 0.025, 0.025)
# pose_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/alignment_debug"))


class AlignmentMetric(DataTerm):
    @dataclass
    class AlignmentMetricSpec:
        pos_threshold: torch.Tensor
        rot_threshold: torch.Tensor
        pos_std: torch.Tensor
        rot_std: torch.Tensor
        metric_mask: torch.Tensor

    @dataclass
    class AlignmentData:
        pose_align: torch.Tensor
        pose_align_against: torch.Tensor
        align_asset_index: torch.Tensor
        align_offset_index: torch.Tensor
        align_against_asset_index: torch.Tensor
        align_against_offset_index: torch.Tensor
        pos_threshold: torch.Tensor
        rot_threshold: torch.Tensor
        pos_delta: torch.Tensor
        rot_delta: torch.Tensor
        pos_error: torch.Tensor
        rot_error: torch.Tensor
        pos_std: torch.Tensor
        rot_std: torch.Tensor
        pos_aligned: torch.Tensor
        rot_aligned: torch.Tensor
        pos_mask: torch.Tensor
        rot_mask: torch.Tensor

        def pos_error_w(self):
            return self.pos_error[..., :3]

    def __init__(self, cfg: AlignmentMetricCfg, env: DataManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.ALL_ENV_INDICES = torch.arange(env.num_envs)
        self.env: DataManagerBasedRLEnv = env
        self.spec = self.AlignmentMetricSpec(*self._filling_alignment_metric_table(cfg.spec, env))
        self.align_kp_cfg, self.align_against_kp_cfg = cfg.align_kp_cfg, cfg.align_against_kp_cfg
        self.data_manager_hook = cfg.data_manager_hook
        self._data = self.AlignmentData(
            pose_align=torch.zeros((env.num_envs, 7), device=env.device),
            pose_align_against=torch.zeros((env.num_envs, 7), device=env.device),
            align_asset_index=torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
            align_offset_index=torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
            align_against_asset_index=torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
            align_against_offset_index=torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
            pos_threshold=torch.zeros((env.num_envs, 3), device=env.device),
            rot_threshold=torch.zeros((env.num_envs, 3), device=env.device),
            pos_delta=torch.zeros((env.num_envs, 3), device=env.device),
            rot_delta=torch.zeros((env.num_envs, 3), device=env.device),
            pos_error=torch.zeros((env.num_envs, 3), device=env.device),
            rot_error=torch.zeros((env.num_envs, 3), device=env.device),
            pos_std=torch.zeros((env.num_envs, 3), device=env.device),
            rot_std=torch.zeros((env.num_envs, 3), device=env.device),
            pos_aligned=torch.zeros((env.num_envs), dtype=torch.bool, device=env.device),
            rot_aligned=torch.zeros((env.num_envs), dtype=torch.bool, device=env.device),
            pos_mask=torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.bool),
            rot_mask=torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.bool),
        )
        self.metric_mask = torch.zeros((env.num_envs, 6), device=env.device, dtype=torch.bool)

        self.context_id_callback = cfg.context_id_callback

        self.context_id_param = cfg.context_id_param

    @property
    def data(self) -> AlignmentData:
        return self._data

    def command_reset(self, env_ids):
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = self.ALL_ENV_INDICES
        task_ids = self.context_id_callback(self.env, env_ids, **self.context_id_param)
        self.data.pos_threshold[env_ids] = self.spec.pos_threshold[task_ids]
        self.data.rot_threshold[env_ids] = self.spec.rot_threshold[task_ids]
        self.data.pos_std[env_ids] = self.spec.pos_std[task_ids]
        self.data.rot_std[env_ids] = self.spec.rot_std[task_ids]
        self.data.pos_mask[env_ids] = self.spec.metric_mask[task_ids, :3]
        self.data.rot_mask[env_ids] = self.spec.metric_mask[task_ids, 3:]
        self.metric_mask[env_ids] = self.spec.metric_mask[task_ids]

    def _update_data(self, env_ids: torch.Tensor | slice | None = None) -> None:
        data_manager = eval(f'self.{self.data_manager_hook}')
        align_against_kp, align_against_kp_mask = self.align_against_kp_cfg.get(data_manager)  # (n_envs, 1, n_assets, n_offsets, 7)
        align_kp, align_kp_mask = self.align_kp_cfg.get(data_manager)  # (n_envs, 1, n_assets, n_offsets, 7)
        n_assets_align_against, n_offsets_align_against = align_against_kp.shape[2:4]
        n_assets_align, n_offsets_align = align_kp.shape[2:4]
        num_envs = len(self.ALL_ENV_INDICES)

        align_pose, align_idx, align_against_pose, align_against_idx = advanced_math.minimum_error_pair_selection(
            align_kp.view(num_envs, -1, 7),
            align_kp_mask.view(num_envs, -1).bool(),
            align_against_kp.view(num_envs, -1, 7),
            align_against_kp_mask.view(num_envs, -1).bool(),
            self.metric_mask
        )

        self.data.pose_align[:] = align_pose
        self.data.pose_align_against[:] = align_against_pose

        self.data.align_asset_index[:] = align_idx // n_offsets_align
        self.data.align_offset_index[:] = align_idx % n_offsets_align
        self.data.align_against_asset_index[:] = align_against_idx // n_offsets_align_against
        self.data.align_against_offset_index[:] = align_against_idx % n_offsets_align_against

        pos_delta, rot_delta = advanced_math.compute_pose_error(
            self.data.pose_align_against[:, :3], self.data.pose_align_against[:, 3:],
            self.data.pose_align[:, :3], self.data.pose_align[:, 3:],
        )

        self.data.pos_delta = pos_delta * self.metric_mask[:, :3]
        self.data.rot_delta = math_utils.wrap_to_pi(rot_delta) * self.metric_mask[:, 3:]

        self.data.pos_error[:] = self.data.pos_delta.abs()
        self.data.rot_error[:] = self.data.rot_delta.abs()
        self.data.pos_aligned[:] = torch.all(self.data.pos_error < self.data.pos_threshold, dim=1)
        self.data.rot_aligned[:] = torch.all(self.data.rot_error < self.data.rot_threshold, dim=1)

    def _filling_alignment_metric_table(self, spec: list[SuccessCondition], env: DataManagerBasedRLEnv):
        # filling tables # don't include nist board
        device = env.device
        pos_threshold = torch.as_tensor([c.pos_threshold for c in spec], device=device)
        rot_threshold = torch.as_tensor([c.rot_threshold for c in spec], device=device)
        pos_std = torch.as_tensor([c.pos_std for c in spec], device=device)
        rot_std = torch.as_tensor([c.rot_std for c in spec], device=device)
        pose_mask = torch.as_tensor([c.pose_components_mask() for c in spec], device=device, dtype=torch.bool)

        return pos_threshold, rot_threshold, pos_std, rot_std, pose_mask

    def _update_metrics(self):
        pass

    def _update_infos(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


class KeyPointsTracker(DataTerm):
    """Key Points Tracker, tracks key points organized under the same semantics across all context.
    Recap: Semantic-KeyPoints are organized around meaningful KeyPoints-Contexts,
    e.g.
        Context: 'alignment' -> Semantic-KeyPoints:  'aligning', 'aligning_against'
        Context: 'manipulation' -> Semantic-KeyPoints:  'asset_grasp', 'robot_object_held'
        Context: 'resets' -> Semantic-KeyPoints:  'on_table', 'on_board'

    As a DataTerm used in ManagerBasedRLEnv, the KeyPointsTracker is responsible for tracking the key points grouped in
    contexts. and serve as the database for querying key points for applications. The input for the class needs to be
    a list of contexts. Each context contains different key points but they have the same semantics.
    e.g.
        Context: 'alignment -> Semantic-KeyPoints:  'aligning: kp1', 'aligning_against: kp2'
        Context: 'alignment -> Semantic-KeyPoints:  'aligning: kp3', 'aligning_against: kp4'
        Context: 'alignment -> Semantic-KeyPoints:  'aligning: kp5', 'aligning_against: kp6'
        Context: 'alignment -> Semantic-KeyPoints:  'aligning: kp5', 'aligning_against: kp2'

    For different contexts, please use different KeyPointsTracker instances.

    """
    @dataclass
    class KeyPointsSpec:

        key_points_asset_id: torch.Tensor
        """asset ids the key points origins from. Shape: (num_contexts, num_key_points, max_num_assets)"""

        key_points_asset_id_mask: torch.Tensor
        """mask for the asset ids. Shape: (num_contexts, num_key_points, max_num_assets)"""

        asset_id_key_points: torch.Tensor
        """Shape: (num_assets, num_key_points, max_num_contexts)"""

        asset_id_key_points_mask: torch.Tensor
        """Shape: (num_assets, num_key_points, max_num_contexts)"""

        key_points_name: list[str]
        """list of key points' tasks semantics. Shape: (num_key_points,)"""

        key_points_offset: torch.Tensor
        """list of key points' offsets. Shape: (num_contexts, num_key_points, max_num_assets, max_num_offsets, 7)"""

        key_points_mask: torch.Tensor
        """list of key points' masks. Shape: (num_contexts, num_key_points, max_num_assets, max_num_offsets)"""

    @dataclass
    class KeyPointsData:

        key_points_src: torch.Tensor
        """key points root pose (num_envs, num_key_points, num_assets, num_offsets, 7)"""

        key_points: torch.Tensor
        """key points currently tracking (num_envs, num_key_points, num_assets, num_offsets, 7)"""

        key_points_name: list[str]
        """list of key points' tasks semantics. Shape: (num_key_points,)"""

        key_points_mask: torch.Tensor
        """key points' masks indicating validity of the offset. Shape: (num_envs, num_key_points, num_assets, num_offsets)"""

        key_points_offset: torch.Tensor
        """key points' offsets, some may be invalid, used with mask. Shape: (num_envs, num_key_points, num_assets, num_offsets, 7)"""

        key_points_asset_id: torch.Tensor
        """assets where the key points are attached. Shape: (num_envs, num_key_points, max_num_assets)"""

        key_points_asset_id_mask: torch.Tensor
        """mask for the asset ids. Shape: (num_envs, num_key_points, max_num_assets)"""

    def __init__(self, cfg: KeyPointTrackerCfg, env: DataManagerBasedRLEnv):
        self._env = env
        self.spec = self.KeyPointsSpec(*self._filling_key_point_offset_table(cfg.spec, env))
        self.update_only_on_reset = cfg.update_only_on_reset
        super().__init__(cfg, env)

        num_key_points = self.spec.key_points_offset.shape[1]
        self.max_num_assets = self.spec.key_points_offset.shape[2]
        self.max_num_offsets = self.spec.key_points_offset.shape[3]

        self.assets: RigidObjectCollection = env.scene["assets"]
        self._ALL_KPS_INDICES = torch.arange(num_key_points, device=env.device)
        self._ALL_ENV_INDICES = torch.arange(env.num_envs, device=env.device)

        key_points_group_size = (env.num_envs, num_key_points, self.max_num_assets, self.max_num_offsets)

        self.context_id_callback = cfg.context_id_callback

        self.context_id_param = cfg.context_id_param

        self._data = self.KeyPointsData(
            key_points_src=torch.zeros((*key_points_group_size, 7), device=env.device),
            key_points=torch.zeros((*key_points_group_size, 7), device=env.device),
            key_points_name=self.spec.key_points_name,
            key_points_mask=torch.zeros((*key_points_group_size,), device=env.device, dtype=torch.bool),
            key_points_offset=torch.zeros((*key_points_group_size, 7), device=env.device),
            key_points_asset_id=torch.zeros((env.num_envs, num_key_points, self.max_num_assets), device=env.device).to(torch.long),
            key_points_asset_id_mask=torch.zeros((env.num_envs, num_key_points, self.max_num_assets), device=env.device, dtype=torch.bool),
        )

    @property
    def data(self) -> KeyPointsData:
        return self._data

    def command_reset(self, env_ids: torch.Tensor):
        # update dynamic data - changes at every step
        context_ids = self.context_id_callback(self._env, env_ids, **self.context_id_param)
        self._data.key_points_asset_id[env_ids] = self.spec.key_points_asset_id[context_ids]
        self._data.key_points_asset_id_mask[env_ids] = self.spec.key_points_asset_id_mask[context_ids]
        self._update_data(env_ids, self._ALL_KPS_INDICES, reset=True)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = self._ALL_ENV_INDICES
        self.command_reset(env_ids)
        return super().reset(env_ids)

    def _update_data(
        self,
        env_ids: torch.Tensor | None = None,
        key_point_ids: torch.Tensor | list[int] | None = None,
        reset: bool = False,
    ):
        if self.update_only_on_reset and not reset:
            return
        if key_point_ids is None:
            key_point_ids = self._ALL_KPS_INDICES
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES
        context_ids = self.context_id_callback(self._env, env_ids, **self.context_id_param)
        E, K, A, OF = len(env_ids), len(key_point_ids), self.max_num_assets, self.max_num_offsets
        env_ids = env_ids[:, None]
        asset_ids = self._data.key_points_asset_id[env_ids, key_point_ids].view(E, -1)  # (E, K * A)
        assets_pose = self.assets.data.object_state_w[env_ids, asset_ids, :7]  # (E, K * A, 7)
        # (E, K, A, O, 7) = (E, K, A, 1, 7)
        self._data.key_points_src[env_ids, key_point_ids, :] = assets_pose.view(E, K, A, 7).unsqueeze(-2)

        self._data.key_points_offset[env_ids, key_point_ids, :] = self.spec.key_points_offset[context_ids[:, None], key_point_ids]
        self._data.key_points_mask[env_ids, key_point_ids, :] = self.spec.key_points_mask[context_ids[:, None], key_point_ids]
        task_key_points_target_pos, task_key_points_target_quat = math_utils.combine_frame_transforms(
            self._data.key_points_src[env_ids, key_point_ids, :, :, :3].view(-1, 3),
            self._data.key_points_src[env_ids, key_point_ids, :, :, 3:].view(-1, 4),
            self._data.key_points_offset[env_ids, key_point_ids, :, :, :3].view(-1, 3),
            self._data.key_points_offset[env_ids, key_point_ids, :, :, 3:].view(-1, 4),
        )
        self._data.key_points[env_ids, key_point_ids, :, :, :3] = task_key_points_target_pos.view(E, K, A, OF, 3)
        self._data.key_points[env_ids, key_point_ids, :, :, 3:] = task_key_points_target_quat.view(E, K, A, OF, 4)

    def _filling_key_point_offset_table(self, spec, env):
        C = len(spec)
        P = len(next(iter(spec.values())).key_points.ordered_points())

        # 1) Flatten out (i, j, asset_id, offsets) for every sub-cfg & every id
        entries = []
        for i, ctx in enumerate(spec.values()):
            for j, kp_cfg in enumerate(ctx.key_points.ordered_points()):
                # normalize to a list of BaseTaskKeyPointCfg
                subs = (
                    kp_cfg.values() if isinstance(kp_cfg, dict) else
                    kp_cfg if isinstance(kp_cfg, list) else
                    [kp_cfg]
                )
                for sub in subs:
                    sub.root.resolve(env.scene)
                    ids = sub.root.object_collection_ids # (A_i,)
                    offsets = sub.offsets_group().to(env.device)  # (O_i,7)
                    for a in ids.tolist():
                        entries.append((i, j, a, offsets))

        # 2) Discover the max dims
        counts = Counter((i, j) for i, j, _, _ in entries)
        A_max = max(counts.values(), default=0)
        O_max = max((offs.shape[0] for *_, offs in entries), default=0)

        # 3) Allocate with “empty” defaults
        device = env.device
        kp_asset_id = torch.full((C, P, A_max), -1, device=device, dtype=torch.long)
        kp_asset_id_mask = torch.zeros((C, P, A_max), device=device, dtype=torch.bool)

        kp_offset = torch.zeros((C, P, A_max, O_max, 7), device=device)
        kp_offset_mask = torch.zeros((C, P, A_max, O_max), device=device, dtype=torch.bool)
        identity = torch.tensor([0, 0, 0, 1, 0, 0, 0], device=device, dtype=torch.float)
        kp_offset[:] = identity

        # 4) One pass to fill
        #    keep per-(i,j) counters so we write into the right “column”
        counters = [[0] * P for _ in range(C)]
        for i, j, a, offs in entries:
            idx = counters[i][j]
            kp_asset_id[i, j, idx] = a
            kp_asset_id_mask[i, j, idx] = True
            n = offs.shape[0]
            kp_offset[i, j, idx, :n] = offs
            kp_offset_mask[i, j, idx, :n] = True
            counters[i][j] += 1

        # 5) build the reverse lookup
        asset_id_kp, asset_id_kp_mask = self.build_reverse_lookup(kp_asset_id, env.scene["assets"].num_objects)
        kp_name = next(iter(spec.values())).key_points.ordered_names()
        return kp_asset_id, kp_asset_id_mask, asset_id_kp, asset_id_kp_mask, kp_name, kp_offset, kp_offset_mask

    def build_reverse_lookup(self, kp_asset_id: torch.Tensor, num_assets: int):
        num_contexts, num_key_points, _ = kp_asset_id.shape

        # 1) Count how many contexts each (asset,point) pair appears in
        counts = torch.zeros((num_assets, num_key_points), dtype=torch.long, device=self._env.device)
        for context_idx in range(num_contexts):
            for key_point_idx in range(num_key_points):
                for asset_id in kp_asset_id[context_idx, key_point_idx].tolist():
                    if asset_id >= 0:
                        counts[asset_id, key_point_idx] += 1
        max_ctx = int(counts.max().item())

        # 2) Allocate outputs, initialized to “empty”
        asset_id_key_points = torch.full((num_assets, num_key_points, max_ctx), -1, device=self._env.device)
        asset_id_key_points_mask = torch.zeros((num_assets, num_key_points, max_ctx), dtype=torch.bool, device=self._env.device)

        # 3) Fill them
        # keep a write‐pointer per (asset, point)
        next_slot = [[0] * num_key_points for _ in range(num_assets)]
        for context_idx in range(num_contexts):
            for key_point_idx in range(num_key_points):
                for asset_id in kp_asset_id[context_idx, key_point_idx].tolist():
                    if asset_id < 0:
                        continue
                    slot = next_slot[asset_id][key_point_idx]
                    asset_id_key_points[asset_id, key_point_idx, slot] = context_idx
                    asset_id_key_points_mask[asset_id, key_point_idx, slot] = True
                    next_slot[asset_id][key_point_idx] += 1

        return asset_id_key_points, asset_id_key_points_mask

    def _update_metrics(self):
        pass

    def _update_infos(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        self.visualizers: dict[str, VisualizationMarkers] = {}
        if debug_vis:
            for i, name in enumerate(self.spec.key_points_name):
                visualizer_cfg: VisualizationMarkersCfg = self.cfg.visualizer_cfg.copy().replace(prim_path=f"/Visuals/Data/{name}")
                self.visualizers[name] = VisualizationMarkers(visualizer_cfg)
                self.visualizers[name].set_visibility(True)
        else:
            for i, name in enumerate(self.spec.key_points_name):
                if name in self.visualizers:
                    self.visualizers[name].set_visibility(False)

    def _debug_vis_callback(self, event):
        for i, name in enumerate(self.spec.key_points_name):
            key_points = self.data.key_points[:, i]
            kp_mask = self.data.key_points_mask[:, i]
            key_points[kp_mask][:, :3]
            self.visualizers[name].visualize(
                translations=key_points[kp_mask][:, :3],
                orientations=key_points[kp_mask][:, 3:]
            )


class DiameterLookUp(DataTerm):

    def __init__(self, cfg: DiameterLookUpCfg, env: DataManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.env = env
        self.diameter_table = torch.tensor(cfg.manipulation_diameters, device=env.device)
        self.held_asset_diameter = torch.zeros((env.num_envs,), device=env.device)
        self.context_id_callback = cfg.context_id_callback
        self.context_id_param = cfg.context_id_param

    def __str__(self) -> str:
        msg = "DiameterData:\n"
        return msg

    @property
    def data(self) -> torch.Tensor:
        return self.held_asset_diameter

    def command_reset(self, env_ids: Sequence[int] | slice | None = None):
        context_ids = self.context_id_callback(self.env, env_ids, **self.context_id_param)
        self.held_asset_diameter[env_ids] = self.diameter_table[context_ids]

    def _update_data(self, env_ids: torch.Tensor | slice | None = None):
        pass

    def _update_metrics(self):
        pass

    def _update_infos(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass
