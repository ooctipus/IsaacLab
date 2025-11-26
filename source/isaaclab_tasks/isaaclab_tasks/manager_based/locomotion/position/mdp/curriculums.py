# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import isaaclab.sim as sim_utils

from isaaclab.managers import ManagerTermBase
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from .success_monitor_cfg import SuccessMonitorCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from .commands import RelativeStateCommand


class terrain_spawn_goal_pair_success_rate_levels(ManagerTermBase):
    """Success-rate curriculum over spawn→target patch pairs per (level, type).

    Tracks success for every possible (spawn_patch, target_patch) connection at each
    terrain (level, type). The total monitored count is:
        num_levels * num_types * num_patches_spawn * num_patches_targets

    For each sampled assignment, it sets the env origin to the sampled spawn location
    (instead of the terrain tile origin) and sets the goal command to the sampled
    target location.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        debug_vis = cfg.params.get("debug_vis", True)

        self.env = env
        self.goal_term: RelativeStateCommand = env.command_manager.get_term("goal_point")

        # Disable term's own resampling: curriculum owns cmd_indices AND cmd_buf writes
        self.goal_term.resample_indices = lambda env_ids: None

        # Discrete command table size
        self.num_discrete_cmd = int(self.goal_term.spec.num_descretized_cmd)

        # Success monitor over discrete command indices [0..num_discrete_cmd)
        success_monitor_cfg = SuccessMonitorCfg(
            monitored_history_len=cfg.params.get("history_len", 100),
            num_monitored_data=self.num_discrete_cmd,
            device=env.device,
        )
        self.success_monitor = success_monitor_cfg.class_type(success_monitor_cfg)

        # simple result dict
        self._result: dict[str, torch.Tensor] = {
            "all_success": torch.zeros((), dtype=torch.float, device=env.device),
        }

        # Visualization: we can still build spawn→target lines from descretized_cmd[0:terrain_count]
        if debug_vis:
            self._init_path_visuals_from_discrete()

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        debug_vis: bool = False,
        kappa: float = 2.0,
    ):
        if env_ids.numel() == 0:
            return self._result

        # 1) SUCCESS UPDATE — use cmd_indices *before* overwriting them
        prev_idx = self.goal_term.cmd_indices[env_ids]
        time_out_mask = env.termination_manager.get_term("time_out")[env_ids]  # [len]

        # group_err: [N_env, 4]  (pos, rot, lin_vel, ang_vel)
        success = self.goal_term.get_state_error()[env_ids] < self.goal_term._reward_scales
        success_mask = time_out_mask & success.all(dim=1)
        self.success_monitor.success_update(prev_idx, success_mask)

        # 2) SAMPLE NEXT DISCRETE COMMANDS
        choices = self.success_monitor.sample_by_target_rate(env_ids, target=0.33, kappa=kappa)
        self.goal_term.cmd_indices[env_ids] = choices.to(self.goal_term.cmd_indices.dtype)

        # 3) APPLY DISCRETE COMMAND ROWS TO cmd_buf[0] AND cmd_mask
        rows = self.goal_term.spec.descretized_cmd[choices]        # [len(env_ids), 15]
        env.scene.terrain.env_origins.index_copy_(0, env_ids, rows[:, 0:3])

        # 4) LOGGING / VISUALIZATION
        success_rates = self.success_monitor.get_success_rate()  # [num_discrete_cmd]
        self._result["all_success"].copy_(success_rates.mean())

        if debug_vis and hasattr(self, "frame_visualizer"):
            self._recolor_lines(success_rates)

        return self._result

    def _get_connecting_lines(self, start_pos: torch.Tensor, end_pos: torch.Tensor):
        v = end_pos - start_pos                    # [N,3]
        l = v.norm(2, dim=-1).clamp_min(1e-12)     # [N]
        p = (start_pos + end_pos) * 0.5            # [N,3]
        z = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(v)
        b = v / l.unsqueeze(-1)                    # normalized direction
        c = torch.cross(z, b, dim=-1)
        w = 1.0 + (z * b).sum(-1, keepdim=True)
        q = torch.cat([w, c], dim=-1)              # [N,4] (w, x, y, z)
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return p, q, l

    def _init_path_visuals_from_discrete(self) -> None:
        rows = self.goal_term.spec.descretized_cmd           # [N,15]
        mask_pos = self.goal_term.spec.descretized_mask[:, 0:3].any(dim=-1)  # [N] bool

        # Only visualize position-like discrete commands
        line_indices = torch.arange(self.num_discrete_cmd, device=self.device)[mask_pos]  # [N_pos]
        rows_pos = rows[mask_pos]                                                     # [N_pos,15]

        start = rows_pos[:, 0:3].clone()  # spawn
        end = rows_pos[:, 3:6].clone()  # target

        Lp, Lq, Ll = self._get_connecting_lines(start, end)
        self._n_lines = Lp.size(0)

        # Remember which discrete command index corresponds to each line
        self._line_indices = line_indices  # [N_pos]

        MARKER_CFG = VisualizationMarkersCfg(
            markers={
                "line_0": sim_utils.CylinderCfg(radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))),
                "line_1": sim_utils.CylinderCfg(radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.0))),
                "line_2": sim_utils.CylinderCfg(radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.0))),
                "line_3": sim_utils.CylinderCfg(radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.3, 0.0))),
                "line_4": sim_utils.CylinderCfg(radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.0))),
                "line_5": sim_utils.CylinderCfg(radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.0))),
                "line_6": sim_utils.CylinderCfg(radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.6, 0.0))),
                "line_7": sim_utils.CylinderCfg(radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.7, 0.0))),
                "line_8": sim_utils.CylinderCfg(radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.0))),
                "line_9": sim_utils.CylinderCfg(radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))),
            }
        )
        self.frame_visualizer = VisualizationMarkers(MARKER_CFG.replace(prim_path="/World/Visuals/CurriculumPaths"))

        Sc = torch.ones(self._n_lines, 3, device=self.device)
        Sc[:, 2] = Ll

        self._marker_idx = torch.full((self._n_lines,), 5, dtype=torch.int32, device=self.device)
        self.frame_visualizer.visualize(translations=Lp, orientations=Lq, scales=Sc, marker_indices=self._marker_idx)

    def _recolor_lines(self, success: torch.Tensor) -> None:
        """Recolor lines according to success per discrete command index."""
        if not hasattr(self, "frame_visualizer"):
            return
        # success is [num_discrete_cmd], but we only have lines for a subset:
        line_success = success[self._line_indices]  # [N_pos]

        bins = torch.clamp((line_success * 9.0).round().to(torch.int32), 0, 9)
        self._marker_idx[:] = bins
        self.frame_visualizer.visualize(marker_indices=self._marker_idx)

def skip_reward_term(env: ManagerBasedRLEnv, env_ids: Sequence[int], reward_term: str):
    term_cfg = env.reward_manager.get_term_cfg(reward_term)
    if term_cfg.weight == 0.0:
        return
    success_monitor = getattr(env.curriculum_manager.cfg, "terrain_levels").func.success_monitor
    success_rate = success_monitor.get_success_rate().mean()
    if (success_rate > 0.2 and env.common_step_counter > 100) or env.common_step_counter > 20000:
        # Set weight to zero so manager skips computing it
        term_cfg.weight = 0.0
        # Additionally, replace the callable with a zero-return stub
        if hasattr(term_cfg.func, "reset"):
            # keep simple lambda style, but make signatures flexible to avoid TypeErrors
            term_cfg.func.reset = lambda *args, **kwargs: None
            term_cfg.func.__call__ = lambda *args, **kwargs: torch.zeros(env.num_envs, device=env.device)
        else:
            term_cfg.func = lambda env, **kwargs: torch.zeros(env.num_envs, device=env.device)


def stricten_success_term(env: ManagerBasedRLEnv, env_ids: Sequence[int], term: str):
    term_cfg = env.termination_manager.get_term_cfg(term)
    success_monitor = getattr(env.curriculum_manager.cfg, "terrain_levels").func.success_monitor
    success_rate = success_monitor.get_success_rate().mean()
    if (success_rate > 0.1 and env.common_step_counter > 100):
        term_cfg.params["thresh"][2] = 0.5
        term_cfg.params["thresh"][3] = 0.5


def activate_reward_term(env: ManagerBasedRLEnv, env_ids: Sequence[int], reward_term: str):
    term_cfg = env.reward_manager.get_term_cfg(reward_term)
    if env.common_step_counter < 5000 and term_cfg.weight != 0.0:
        term_cfg.weight = 0.0
    if env.common_step_counter > 5000 and term_cfg.weight == 0.0:
        term_cfg.weight = 250.0
