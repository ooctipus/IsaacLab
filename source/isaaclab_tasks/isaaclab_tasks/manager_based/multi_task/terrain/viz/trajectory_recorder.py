# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Top-down 2D trajectory video recorder for terrain locomotion debugging.

A small helper class -- no gym/manager glue, just capture + render -- that
the :func:`isaaclab_tasks...mdp.events.record_trajectory_video` event term
drives. Captures per-step (robot_xy, target_xy, instant_success) for a
random subset of envs over a fixed window, then renders a matplotlib
animation as mp4 with one panel per env. The robot is drawn as a filled
circle that turns **green** while ``instant_success`` is True and **red**
otherwise, with a faded trail of recent positions so a stationary
"succeeding and waiting for hold time" robot is visually distinct from
a stationary "doing nothing" robot.

Why this exists: the standard 3D rgb video shows a single env's rendered
scene, which can mislead when the per-task success rate is high but the
rendered robot looks lost (the rendered env may be one of the few that
didn't learn while the population mean is high). The 2D trajectory video
shows many envs at once with explicit success-state colouring, so
"stationary because it's at goal" and "stationary because it gave up"
are unambiguous.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class TrajectoryRecorder:
    """Stateful capture + render helper for the trajectory video.

    Owns a running window of captured frames and a recording flag. The
    caller is responsible for invoking :meth:`maybe_record` every step
    with the latest per-env state; the recorder decides internally
    when to start a window, when to close it, and when to render.

    The math is layout-agnostic: the caller passes already-local
    ``robot_xy`` and ``target_xy`` (env-origin subtracted), plus the
    ``instant_success`` boolean.
    """

    def __init__(
        self,
        video_folder: str | Path,
        video_interval: int,
        video_length: int,
        max_envs: int = 16,
        fps: int = 30,
        trail_steps: int = 30,
        wandb_tag: str = "Curriculum/trajectory_video",
    ) -> None:
        self.video_folder = Path(video_folder)
        self.video_folder.mkdir(parents=True, exist_ok=True)
        self.video_interval = int(video_interval)
        self.video_length = int(video_length)
        self.max_envs = int(max_envs)
        self.fps = int(fps)
        self.trail_steps = int(trail_steps)
        # Tag used for the W&B upload. We log under a *unique* key (rather
        # than relying on rsl_rl's ``*.mp4`` glob, which uploads every mp4
        # under the hardcoded ``"video"`` key and lets the standard 3D
        # RecordVideo overwrite us). Mirrors how ``_log_spawn_scatter``
        # writes to ``env.extras["log_images"]["Curriculum/spawn_scatter"]``
        # so the panel coexists with the other Curriculum/* dashboards.
        self.wandb_tag = wandb_tag

        self._step_count = 0
        self._recording = False
        self._frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._env_subset: np.ndarray | None = None

    def maybe_record(
        self,
        robot_xy: np.ndarray,
        target_xy: np.ndarray,
        success: np.ndarray,
    ) -> None:
        """Advance one step.

        Args:
            robot_xy: ``[num_envs, 2]`` env-local xy of every robot.
            target_xy: ``[num_envs, 2]`` env-local xy of every target.
            success: ``[num_envs]`` bool flag (e.g. ``instant_success``).
        """
        if not self._recording and self._step_count % self.video_interval == 0:
            self._start_recording(num_envs=int(robot_xy.shape[0]))

        if self._recording:
            assert self._env_subset is not None
            self._frames.append(
                (
                    robot_xy[self._env_subset].copy(),
                    target_xy[self._env_subset].copy(),
                    success[self._env_subset].copy(),
                )
            )
            if len(self._frames) >= self.video_length:
                self._finish_recording()

        self._step_count += 1

    # ----------------------------------------------------------------- impl

    def _start_recording(self, num_envs: int) -> None:
        n = min(self.max_envs, num_envs)
        # Deterministic-per-recording subset: seed by step counter so the
        # same indices reappear if the recording is re-run; helps visual
        # comparison across windows.
        rng = np.random.default_rng(self._step_count)
        self._env_subset = rng.choice(num_envs, size=n, replace=False).astype(np.int64)
        self._frames = []
        self._recording = True

    def _finish_recording(self) -> None:
        out_path = self.video_folder / f"trajectory_step_{self._step_count:08d}.mp4"
        try:
            self._render_video(out_path)
            self._upload_to_wandb(out_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[TrajectoryRecorder] Failed to render {out_path}: {exc}")
        self._frames = []
        self._recording = False
        self._env_subset = None

    def _upload_to_wandb(self, mp4_path: Path) -> None:
        """Log the rendered mp4 to W&B under :attr:`wandb_tag` if wandb is active.

        Direct ``wandb.log`` (rather than relying on rsl_rl's ``*.mp4`` glob,
        which uses a hardcoded ``"video"`` key and would overwrite alongside
        the standard 3D ``RecordVideo`` mp4s). No-op when wandb isn't
        installed or when no ``wandb.init`` has been called yet.

        After the upload, the file is renamed from ``.mp4`` to
        ``.mp4.archived`` so rsl_rl's ``rglob("*.mp4")`` (logger.py:290)
        won't re-upload it under the hardcoded ``"video"`` key on every
        subsequent iteration -- otherwise the standard 3D-video panel
        gets polluted with our trajectory videos and the upload bandwidth
        compounds with iteration count.
        """
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return
        wandb.log({self.wandb_tag: wandb.Video(str(mp4_path), format="mp4")}, step=self._step_count)
        # Rename so rsl_rl's mp4 glob can't find it again.
        try:
            mp4_path.rename(mp4_path.with_suffix(".mp4.archived"))
        except OSError as exc:
            print(f"[TrajectoryRecorder] Failed to archive {mp4_path}: {exc}")

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
            ax.set_title(f"env {int(env_subset[i])}", fontsize=8)
            trail_artists.append(trail)
            target_artists.append(target)
            robot_artists.append(robot)

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
