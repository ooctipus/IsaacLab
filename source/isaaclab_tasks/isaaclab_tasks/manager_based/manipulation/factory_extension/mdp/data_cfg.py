from __future__ import annotations
import torch
from dataclasses import MISSING
from collections.abc import Mapping
from typing import Literal

import isaaclab.sim as sim_utils

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.managers import DataTermCfg, DataTerm, DataManager
from isaaclab.envs import ManagerBasedRLEnv

from ..tasks import BaseTaskKeyPointCfg, SuccessCondition
from .data import DiameterLookUp, KeyPointsTracker, AlignmentMetric
from . import key_point_maths


@configclass
class DataCfg:
    term: str = ""

    resolved: bool = False

    def resolve(self, data_term: DataTerm):
        self._data = data_term.data
        self.resolved = True

    def get(self, data_manager: DataManager):
        if not self.resolved:
            self.resolve(data_manager.get_term(self.term))
        return self._data


@configclass
class AlignmentDataCfg(DataCfg):

    alignment_attr: None | Literal["pos_error", "rot_error", "pos_threshold", "rot_threshold", "pos_aligned", "rot_aligned"] = None

    def resolve(self, data_term: DataTerm):
        assert isinstance(data_term, AlignmentMetric)
        alignment_data = data_term.data
        if self.alignment_attr is None:
            self._data = alignment_data
        else:
            self._data = getattr(alignment_data, self.alignment_attr)

        self.resolved = True

    def get(self, data_manager):
        return super().get(data_manager)


@configclass
class KeyPointDataCfg(DataCfg):

    is_spec: bool = False

    kp_attr: Literal["key_points", "key_points_src", "key_points_asset_id", "asset_id_key_points", "key_points_offset", "key_points_name"] = "key_points"

    kp_names: None | str | list[str] = None
    """asset_align, asset_align_against, robot_root, robot_object_held, asset_grasp"""

    kp_ids: None | slice | list[int] = None

    def resolve(self, data_term: DataTerm):
        assert isinstance(data_term, KeyPointsTracker)
        if not self.is_spec:
            key_points_data = data_term.data
            self._data = getattr(key_points_data, self.kp_attr)
            if "offset" in self.kp_attr or "src" in self.kp_attr:
                self._mask = key_points_data.key_points_mask
            else:
                self._mask = getattr(key_points_data, self.kp_attr + "_mask")
        else:
            key_points_data = data_term.spec
            if self.kp_attr:
                mask_entry = "key_points_mask" if "offset" in self.kp_attr else self.kp_attr + "_mask"
                self._data = getattr(key_points_data, self.kp_attr)
                self._mask = getattr(key_points_data, mask_entry)
            else:
                self._data = key_points_data

        # determine kp_ids
        if self.kp_names is None:
            self.kp_ids = slice(None)
        elif isinstance(self.kp_names, str):
            self.kp_ids = [key_points_data.key_points_name.index(self.kp_names)]
        elif isinstance(self.kp_names, list):
            self.kp_ids = [key_points_data.key_points_name.index(kp_name) for kp_name in self.kp_names]

        self.resolved = True

    def get(self, data_manager: DataManager):
        if not self.resolved:
            self.resolve(data_manager.get_term(self.term))
        if self.kp_attr:
            data: torch.Tensor = self._data[:, self.kp_ids].clone()
            mask: torch.Tensor = self._mask[:, self.kp_ids].clone()
            return data, mask
        else:
            return self._data


@configclass
class AlignmentMetricCfg(DataTermCfg):

    class_type: type[AlignmentMetric] = AlignmentMetric

    spec: list[SuccessCondition] = []

    context_id_callback: callable = None

    context_id_param: dict = {}

    data_manager_hook: str = 'env.data_manager'

    align_kp_cfg: KeyPointDataCfg = MISSING  # type:ignore

    align_against_kp_cfg: KeyPointDataCfg = MISSING  # type:ignore


@configclass
class DiameterLookUpCfg(DataTermCfg):
    """Configuration for diameter data term."""

    class_type: type[DiameterLookUp] = DiameterLookUp

    manipulation_diameters: list[float] = []

    context_id_callback: callable = None

    context_id_param: dict = {}


@configclass
class KeyPointTrackerCfg(DataTermCfg):
    """Configuration for key point tracker."""

    class_type: type[KeyPointsTracker] = KeyPointsTracker

    spec: Mapping[str, BaseTaskKeyPointCfg] = {}

    context_id_callback: callable = None

    context_id_param: dict = {}

    update_only_on_reset: bool = False

    visualizer_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Data/",
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.02, 0.02, 0.02),
            )
        },
    )


def task_id_callback(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> torch.Tensor:
    return env.command_manager.get_command("task_command")[env_ids]


def asset_id_callback_deterministic(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,    # shape (E,)
    kp_asset_id_cfg: KeyPointDataCfg,  # shape (E, A)
    asset_to_kps_cfg: KeyPointDataCfg,  # shape (num_assets, 1, n_contexts)
) -> torch.Tensor:
    #    align_asset_id:      align_mask:       (n_envs, 1, n_assets)
    align_asset_id, align_mask = kp_asset_id_cfg.get(env.data_manager)  # (n_envs, 1, n_assets)
    asset_to_kps, asset_to_kps_mask = asset_to_kps_cfg.get(env.data_manager)  # (num_assets, 1, n_contexts)
    align_asset_id, align_mask = align_asset_id.squeeze(1)[env_ids], align_mask.squeeze(1)[env_ids]  # (n_envs, n_assets)
    asset_to_kps, asset_to_kps_mask = asset_to_kps.squeeze(1), asset_to_kps_mask.squeeze(1)  # (num_assets, n_contexts)
    chosen_assets, _ = key_point_maths.select_valid(align_asset_id, align_mask, dim=1, strategy="first")
    per_env_contexts, per_env_mask = asset_to_kps[chosen_assets], asset_to_kps_mask[chosen_assets]  # (E, n_contexts)
    chosen_ctxs, _ = key_point_maths.select_valid(per_env_contexts, per_env_mask, dim=1, strategy="first")

    return chosen_ctxs


def random_context_callback(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,  # shape (E,)
    reset_kp_asset_id_cfg: KeyPointDataCfg,  # shape (E, 1, n_assets)
) -> torch.Tensor:
    #  spec: (num_contexts, 1, max_num_assets), mask: (num_contexts, 1, max_num_assets)
    kp_asset_id_spec, _ = reset_kp_asset_id_cfg.get(env.data_manager)
    num_contexts = kp_asset_id_spec.shape[0]
    context_idx = torch.randint(low=0, high=num_contexts, size=(env_ids.shape[0],), device=env_ids.device)
    return context_idx
