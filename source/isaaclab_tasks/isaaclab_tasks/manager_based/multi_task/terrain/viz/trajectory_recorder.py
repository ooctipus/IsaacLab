# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Top-down 2D trajectory video recorder for terrain locomotion debugging.

Captures per-step (robot_xy, target_xy, instant_success) for a small
random subset of envs over a fixed window, then renders a matplotlib
animation as mp4 with one panel per env. The robot is drawn as a
filled circle that turns **green** while ``instant_success`` is True
(all error groups below threshold this step) and **red** otherwise,
with a faded trail of recent positions so a stationary "succeeding
and waiting for hold time" robot is visually distinct from a
stationary "doing nothing" robot.

The recorder is implemented as a :class:`gymnasium.Wrapper` so it
slots into ``train.py`` next to the existing ``RecordVideo`` wrapper
and is independent of the env's manager system. Output mp4 files are
written to a folder under ``log_dir``; rsl_rl's logger picks them up
automatically via its ``*.mp4`` glob and uploads to W&B.

Why this exists: the standard 3D rgb video shows a single env's
rendered scene, which can mislead when the per-task success rate is
high but the rendered robot looks lost (the rendered env may be one
of the few that didn't learn while the population mean is high). The
2D trajectory video shows many envs at once with explicit
success-state colouring, so "stationary because it's at goal" and
"stationary because it gave up" are unambiguous.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
import torch
import warp as wp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _to_torch(arr) -> torch.Tensor:
    """Accept either a ``warp`` array or a ``torch`` tensor and return torch."""
    if isinstance(arr, torch.Tensor):
        return arr
    return wp.to_torch(arr)


class TrajectoryVideoWrapper(gym.Wrapper):
    """Periodically record a 2D top-down trajectory video.

    Args:
        env: The (possibly already-wrapped) gym env.
        video_folder: Output directory for ``trajectory_step_<N>.mp4`` files.
        step_trigger: Callable ``(step) -> bool`` that decides when a new
            recording starts; mirrors :class:`gym.wrappers.RecordVideo`.
        video_length: Number of policy steps per recording.
        command_name: Command-term name to read targets and the success
            criterion from.
        max_envs: Cap on the number of envs rendered. Picked uniformly at
            random at the start of each recording for diversity.
        fps: Frames per second of the output mp4.
        trail_steps: Length of the per-env position trail in the animation.
    """

    def __init__(
        self,
        env: gym.Env,
        video_folder: str | Path,
        step_trigger: Callable[[int], bool],
        video_length: int,
        command_name: str = "goal_point",
        max_envs: int = 16,
        fps: int = 30,
        trail_steps: int = 30,
    ) -> None:
        super().__init__(env)
        self.video_folder = Path(video_folder)
        self.video_folder.mkdir(parents=True, exist_ok=True)
        self.step_trigger = step_trigger
        self.video_length = int(video_length)
        self.command_name = command_name
        self.max_envs = int(max_envs)
        self.fps = int(fps)
        self.trail_steps = int(trail_steps)

        self._step_count = 0
        self._recording = False
        self._frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._env_subset: np.ndarray | None = None

    # ------------------------------------------------------------------ gym

    def step(self, action):
        result = self.env.step(action)

        if not self._recording and self.step_trigger(self._step_count):
            self._start_recording()
        if self._recording:
            self._capture_frame()
            if len(self._frames) >= self.video_length:
                self._finish_recording()

        self._step_count += 1
        return result

    # ----------------------------------------------------------------- impl

    def _start_recording(self) -> None:
        unwrapped: ManagerBasedRLEnv = self.env.unwrapped  # type: ignore[assignment]
        n_envs = int(unwrapped.num_envs)
        n = min(self.max_envs, n_envs)
        # Deterministic-per-recording subset: seed by step counter so the
        # same indices reappear if the recording is re-run; helps visual
        # comparison across windows.
        rng = np.random.default_rng(self._step_count)
        self._env_subset = rng.choice(n_envs, size=n, replace=False).astype(np.int64)
        self._frames = []
        self._recording = True

    def _capture_frame(self) -> None:
        unwrapped: ManagerBasedRLEnv = self.env.unwrapped  # type: ignore[assignment]
        cmd = unwrapped.command_manager.get_term(self.command_name)

        robot = unwrapped.scene["robot"]
        robot_pos = _to_torch(robot.data.root_pos_w)[:, :2]

        env_origins = _to_torch(unwrapped.scene.env_origins)[:, :2]

        # Targets are stored in the command buffer at row 0, columns 0:3.
        # ``cmd.cmd_buf`` is torch (state_command.py constructs it with
        # torch.zeros); accessing as a tensor is safe.
        target_pos = cmd.cmd_buf[:, 0, :2]

        # ``instant_success`` -- all active error groups below threshold THIS
        # step. Computed inline so we don't depend on the metrics dict being
        # populated yet at capture time.
        err = cmd._err
        thr = cmd._reward_scales[cmd.cmd_ids.long()]
        success = (err < thr).all(dim=1)

        idx = torch.as_tensor(self._env_subset, device=robot_pos.device, dtype=torch.long)
        robot_xy = (robot_pos - env_origins)[idx].detach().cpu().numpy()
        target_xy = (target_pos - env_origins)[idx].detach().cpu().numpy()
        success_np = success[idx].detach().cpu().numpy()

        self._frames.append((robot_xy, target_xy, success_np))

    def _finish_recording(self) -> None:
        out_path = self.video_folder / f"trajectory_step_{self._step_count:08d}.mp4"
        try:
            self._render_video(out_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[TrajectoryVideoWrapper] Failed to render {out_path}: {exc}")
        self._frames = []
        self._recording = False
        self._env_subset = None

    # --------------------------------------------------------------- render

    def _render_video(self, out_path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FFMpegWriter, FuncAnimation

        # Plot bounds: union of all robot + target positions across the
        # window, with a small margin for legibility.
        all_xy = np.concatenate(
            [np.concatenate([f[0], f[1]], axis=0) for f in self._frames],
            axis=0,
        )
        x_min, y_min = all_xy.min(axis=0) - 0.5
        x_max, y_max = all_xy.max(axis=0) + 0.5

        n = int(self._env_subset.shape[0])  # type: ignore[union-attr]
        ncols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(2.3 * ncols, 2.3 * nrows), squeeze=False)
        axes_flat = axes.flatten()
        for ax in axes_flat[n:]:
            ax.set_visible(False)

        robot_artists = []
        target_artists = []
        trail_artists = []
        title_artists = []
        env_subset = self._env_subset  # type: ignore[assignment]
        for i in range(n):
            ax = axes_flat[i]
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=6)
            ax.grid(True, lw=0.3, alpha=0.3)
            (trail,) = ax.plot([], [], lw=0.6, c="gray", alpha=0.6, zorder=1)
            target = ax.scatter([], [], marker="*", s=80, c="black", zorder=3)
            robot = ax.scatter([], [], s=40, c="red", zorder=2)
            title = ax.set_title(f"env {int(env_subset[i])}", fontsize=8)
            trail_artists.append(trail)
            target_artists.append(target)
            robot_artists.append(robot)
            title_artists.append(title)

        trail_x: list[list[float]] = [[] for _ in range(n)]
        trail_y: list[list[float]] = [[] for _ in range(n)]

        suptitle = fig.suptitle("", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96))

        def update(frame_idx: int):
            robot_xy, target_xy, success = self._frames[frame_idx]
            for i in range(n):
                rx, ry = float(robot_xy[i, 0]), float(robot_xy[i, 1])
                tx, ty = float(target_xy[i, 0]), float(target_xy[i, 1])
                trail_x[i].append(rx)
                trail_y[i].append(ry)
                if len(trail_x[i]) > self.trail_steps:
                    trail_x[i].pop(0)
                    trail_y[i].pop(0)

                trail_artists[i].set_data(trail_x[i], trail_y[i])
                target_artists[i].set_offsets(np.asarray([[tx, ty]]))
                robot_artists[i].set_offsets(np.asarray([[rx, ry]]))
                robot_artists[i].set_color("limegreen" if bool(success[i]) else "crimson")
            n_success = int(np.sum(success))
            suptitle.set_text(
                f"trajectory @ step {self._step_count - self.video_length + frame_idx + 1}   (success: {n_success}/{n})"
            )
            return list(robot_artists) + list(target_artists) + list(trail_artists) + [suptitle]

        anim = FuncAnimation(
            fig,
            update,
            frames=len(self._frames),
            blit=False,
            interval=int(1000 / self.fps),
        )
        writer = FFMpegWriter(fps=self.fps, bitrate=1800)
        anim.save(str(out_path), writer=writer)
        plt.close(fig)
