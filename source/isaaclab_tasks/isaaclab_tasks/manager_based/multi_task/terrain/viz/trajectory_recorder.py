# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Top-down 2D trajectory video recorder for terrain locomotion debugging.

A small helper class -- no gym/manager glue, just capture + render -- that
the :func:`isaaclab_tasks...mdp.events.record_trajectory_video` event term
drives. Captures per-step ``(robot_xy, target_xy, task_done)`` for a
random 10% subsample of envs over a fixed window, then renders one shared
world-frame scatter as a gif (or mp4 when ffmpeg is on PATH) via
vectorized numpy splatting. Robots paint **green** while
``task_done`` holds and **red** otherwise; targets are black stars.

Why this exists: the standard 3D rgb video shows a single env's rendered
scene, which can mislead when the per-task success rate is high but the
rendered robot looks lost (the rendered env may be one of the few that
didn't learn while the population mean is high). The 2D trajectory video
shows the *population* at once with explicit success-state colouring, so
"stationary because it's at goal" and "stationary because it gave up"
are unambiguous in aggregate.

Render path: pure numpy + Pillow (no matplotlib draw cycle, no FFMpeg
shell-out unless explicitly available). At n=1638 dots × 200 frames ×
480×480 resolution the render takes ~1.4 s on a single CPU thread when
a terrain background is supplied (master-palette gif, ~10 MiB) and
~0.5 s without (~2 MiB) -- small enough to run inline in the env-step
loop given the typical 5000-step ``video_interval``.
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
    task lifecycle completion boolean.
    """

    # Render 10% of the env population per window. Hard-coded rather
    # than configurable -- at typical 16k-env locomotion runs that's
    # ~1.6k dots, which renders in ~1.4 s with terrain bg and reads
    # cleanly. Smaller setups still work (always >= 1).
    SUBSAMPLE_FRACTION = 0.1

    # Maximum canvas side (px) along the longer terrain axis. The shorter
    # side is computed from the bg-extent aspect ratio so a rectangular
    # terrain doesn't get squashed into a square. 480 px on the long
    # side reads ~1.6k dots cleanly at 2-3 px each while keeping the
    # 200-frame gif compact.
    CANVAS_MAX_PX = 480

    # Master gif palette: precomputed once from the baked terrain canvas
    # so every frame quantizes against the *same* color table. Without
    # this, PIL builds a per-frame adaptive palette which (a) makes
    # successive frames look mildly flickery and (b) bloats the file
    # by storing a fresh palette per frame. 64 colors is plenty for a
    # heightmap + four overlay colours (success / fail / target / bg).
    GIF_PALETTE_COLORS = 64

    def __init__(
        self,
        video_folder: str | Path,
        video_interval: int,
        video_length: int,
        fps: int = 30,
        wandb_tag: str = "Sampler/trajectory_video",
        background_image: np.ndarray | None = None,
        background_extent: tuple[float, float, float, float] | None = None,
    ) -> None:
        """
        Args:
            background_image: Optional ``[H, W]`` float heightmap (NaN for
                misses) returned by :func:`render_terrain_background`. When
                provided, becomes the per-frame canvas and its
                :paramref:`background_extent` overrides the
                observed-positions auto-bounds, so the dots are drawn in
                world frame and align with the terrain underneath.
            background_extent: ``(xmin, xmax, ymin, ymax)`` [m] in world
                frame matching :paramref:`background_image`. Required when
                :paramref:`background_image` is set.
        """
        self.video_folder = Path(video_folder)
        self.video_folder.mkdir(parents=True, exist_ok=True)
        self.video_interval = max(1, int(video_interval))
        self.video_length = max(1, int(video_length))
        self.fps = int(fps)

        # Pre-bake the background canvas at CANVAS_PX × CANVAS_PX once.
        # Per-frame cost is then a single np.copy of a small uint8 buffer
        # plus the dot splat; no PIL resize / colormap lookup hot-path.
        # Plus a shared "master palette" image so every frame quantizes
        # against the same 64 colors -- avoids per-frame palette tables
        # and the resulting flicker / bloat.
        self._bg_canvas: np.ndarray | None = None
        self._bg_extent: tuple[float, float, float, float] | None = None
        self._gif_palette_image: object | None = None
        # Square fallback when no bg is supplied -- we only learn the
        # observed-positions aspect at render time, but a square canvas
        # at the same max budget reads fine.
        self._canvas_w: int = self.CANVAS_MAX_PX
        self._canvas_h: int = self.CANVAS_MAX_PX
        if background_image is not None:
            if background_extent is None:
                raise ValueError("background_extent required when background_image is set")
            from PIL import Image

            from .terrain_background import heightmap_to_rgb

            xmin, xmax, ymin, ymax = background_extent
            world_w = max(xmax - xmin, 1e-6)
            world_h = max(ymax - ymin, 1e-6)
            if world_w >= world_h:
                self._canvas_w = self.CANVAS_MAX_PX
                self._canvas_h = max(1, int(round(self.CANVAS_MAX_PX * world_h / world_w)))
            else:
                self._canvas_h = self.CANVAS_MAX_PX
                self._canvas_w = max(1, int(round(self.CANVAS_MAX_PX * world_w / world_h)))

            rgb = heightmap_to_rgb(background_image)
            # Flip rows so row 0 corresponds to world ymax (cartesian
            # "north up"). The heightmap is produced with row 0 = ymin
            # because matplotlib's imshow(origin="lower") -- used by the
            # spawn-scatter dashboard -- expects that convention. Our
            # PIL/numpy render path has no imshow to invert the axis,
            # and we map dots with ``py = (ymax - xy_y) * scale_y`` so
            # robots at high y go to the top. Flipping the bg here is
            # what aligns the two conventions.
            rgb = np.ascontiguousarray(rgb[::-1])
            img = Image.fromarray(rgb, mode="RGB").resize((self._canvas_w, self._canvas_h), resample=Image.BILINEAR)
            self._bg_canvas = np.asarray(img, dtype=np.uint8)
            self._bg_extent = tuple(background_extent)  # type: ignore[assignment]
            # Master palette: bg-adaptive colors (most of the budget) plus
            # explicit reserved slots for the four overlay colours. PIL's
            # ``quantize(colors=N)`` runs median-cut on pixel histograms,
            # so the bg gradient (~230k pixels) drowns out any small dot
            # swatch we paint in. We reserve 4 trailing entries by hand
            # to guarantee the dots survive the quantization snap.
            #
            # The reference image MUST have one pixel per palette index --
            # ``frame.quantize(palette=ref)`` only matches against palette
            # entries that appear in ``ref``'s pixel data, not the full
            # 256-slot table. A flat ``range(N_PALETTE)`` strip ensures
            # every entry is "live".
            n_bg_slots = self.GIF_PALETTE_COLORS - 4
            bg_palette_img = img.quantize(colors=n_bg_slots, method=Image.MEDIANCUT)
            pal = list(bg_palette_img.getpalette()[: n_bg_slots * 3])
            pal += [255, 255, 255]  # bg fill (no-bg fallback)
            pal += [0, 0, 0]  # target / outline
            pal += [50, 205, 50]  # success (limegreen)
            pal += [220, 20, 60]  # fail (crimson)
            # PIL palettes are flat 768-byte buffers (256 RGB triplets);
            # pad the unused slots with zero so the buffer is well-formed.
            pal += [0] * (768 - len(pal))
            palette_img = Image.new("P", (self.GIF_PALETTE_COLORS, 1))
            palette_img.putpalette(pal)
            palette_img.putdata(list(range(self.GIF_PALETTE_COLORS)))
            self._gif_palette_image = palette_img

        # Tag used for the W&B upload. We log under a *unique* key (rather
        # than relying on rsl_rl's ``*.mp4`` glob, which uploads every mp4
        # under the hardcoded ``"video"`` key and lets the standard 3D
        # RecordVideo overwrite us). This mirrors the sampler image logger's
        # direct W&B upload path so Sampler/* panels coexist.
        self.wandb_tag = wandb_tag

        self._step_count = 0
        self._recording = False
        self._frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._env_subset: np.ndarray | None = None

    def capture_env_indices(self, num_envs: int) -> np.ndarray | None:
        """Advance idle steps cheaply and return env indices during recording."""
        if not self._recording:
            if self._step_count % self.video_interval != 0:
                self._step_count += 1
                return None
            self._start_recording(num_envs=num_envs)
        return self._env_subset

    def append_frame(
        self,
        robot_xy: np.ndarray,
        target_xy: np.ndarray,
        success: np.ndarray,
    ) -> None:
        self._frames.append((robot_xy.copy(), target_xy.copy(), success.copy()))
        if len(self._frames) >= self.video_length:
            self._finish_recording()
        self._step_count += 1

    # ----------------------------------------------------------------- impl

    def _start_recording(self, num_envs: int) -> None:
        n = max(1, int(num_envs * self.SUBSAMPLE_FRACTION))
        # Deterministic-per-recording subset: seed by step counter so the
        # same indices reappear if the recording is re-run; helps visual
        # comparison across windows.
        rng = np.random.default_rng(self._step_count)
        self._env_subset = rng.choice(num_envs, size=n, replace=False).astype(np.int64)
        self._frames = []
        self._recording = True

    def _finish_recording(self) -> None:
        """Render + upload the just-captured window inline.

        Render of one shared world-frame scatter at ~1.6k dots × 200
        frames takes ~1.4 s with terrain bg / ~0.5 s without -- short
        enough vs. the typical 5000-step ``video_interval`` (~156 PPO
        iterations) that blocking the env loop here is < 0.5% wall-time.
        """
        import shutil

        assert self._env_subset is not None
        # ffmpeg on PATH → mp4 (smaller, more portable in W&B); else fall
        # back to gif via PIL (no apt deps). The cluster docker image
        # doesn't ship ffmpeg today, so the gif path is currently the
        # production one.
        ext = "mp4" if shutil.which("ffmpeg") is not None else "gif"
        out_path = self.video_folder / f"trajectory_step_{self._step_count:08d}.{ext}"
        try:
            self._render_video(out_path, self._frames, self._env_subset, self._step_count)
            self._upload_to_wandb(out_path, self._step_count)
        except Exception as exc:  # noqa: BLE001
            print(f"[TrajectoryRecorder] Failed to render {out_path}: {exc}")
        self._frames = []
        self._recording = False
        self._env_subset = None

    def _upload_to_wandb(self, video_path: Path, step_count: int) -> None:
        """Log the rendered video to W&B under :attr:`wandb_tag` if wandb is active.

        Direct ``wandb.log`` (rather than relying on rsl_rl's ``*.mp4``
        glob, which uploads every mp4 under a hardcoded ``"video"`` key
        and would overwrite the standard 3D ``RecordVideo`` panel).
        No-op when wandb isn't installed or when no ``wandb.init`` has
        been called yet.

        Step axis: rsl_rl logs *every* metric, image, and video at
        ``step=iteration`` (see ``rsl_rl/utils/wandb_utils.py``). Our
        per-env-step counter advances ``num_steps_per_env`` times faster,
        so logging at our counter would push wandb's monotonic step far
        past rsl_rl's iteration; subsequent ``wandb.log(..., step=it)``
        calls with ``it < our step`` then get a warning and glued onto
        the wrong axis. Using ``wandb.run.step`` (whatever step rsl_rl
        last advanced to) keeps our video aligned with the rest of the
        dashboard and avoids the non-monotonic warning entirely.

        For mp4 outputs the file is renamed to ``.mp4.archived`` after
        upload so rsl_rl's ``rglob("*.mp4")`` (logger.py:290) can't
        re-upload it under the hardcoded ``"video"`` key on every
        subsequent iteration. Gifs aren't picked up by that glob, so the
        rename is mp4-only.
        """
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            print(f"[TrajectoryRecorder] {video_path.name} rendered but wandb.run is None — skipping upload.")
            return
        wandb_step = int(getattr(wandb.run, "step", 0))
        fmt = video_path.suffix.lstrip(".")
        wandb.log({self.wandb_tag: wandb.Video(str(video_path), format=fmt)}, step=wandb_step)
        print(
            f"[TrajectoryRecorder] Uploaded {video_path.name} (window step={step_count}) to W&B at step={wandb_step}."
        )
        if video_path.suffix == ".mp4":
            try:
                video_path.rename(video_path.with_suffix(".mp4.archived"))
            except OSError as exc:
                print(f"[TrajectoryRecorder] Failed to archive {video_path}: {exc}")

    def _render_video(
        self,
        out_path: Path,
        frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        env_subset: np.ndarray,
        end_step: int,
    ) -> None:
        """Render ``frames`` to ``out_path`` via vectorized numpy splatting.

        At ~10% of 16384 envs (~1.6k dots × 200 frames) matplotlib's
        draw cycle dominates render time -- a single FuncAnimation pass
        takes ~17 s even at n=16 because each frame triggers a full
        layout + artist update + canvas blit. We bypass that here:
        paint each frame as a flat ``[H, W, 3] uint8`` numpy array via
        vectorized index assignment, then encode via Pillow (gif) or
        imageio-ffmpeg (mp4). Trails are dropped (too many segments at
        scale; keyframe-rate dot motion already conveys "stationary vs
        walking").
        """
        from PIL import Image

        n = int(env_subset.shape[0])
        if n == 0 or len(frames) == 0:
            return

        # Coordinate frame: prefer the terrain-bg extent (world frame,
        # fixed across windows so successive videos overlay consistently);
        # else fall back to the observed-positions union with a margin.
        if self._bg_extent is not None:
            x_min, x_max, y_min, y_max = self._bg_extent
        else:
            all_xy = np.concatenate([np.concatenate([f[0], f[1]], axis=0) for f in frames], axis=0)
            margin = 0.5
            x_min, y_min = all_xy.min(axis=0) - margin
            x_max, y_max = all_xy.max(axis=0) + margin

        H, W = self._canvas_h, self._canvas_w
        dot_radius = max(1, int(round(2 + 2 / np.sqrt(max(n, 1)))))
        scale_x = (W - 1) / max(x_max - x_min, 1e-6)
        scale_y = (H - 1) / max(y_max - y_min, 1e-6)

        def world_to_pixel(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Vectorized world → pixel projection. Y flipped (image origin top-left)."""
            px = np.clip(((xy[:, 0] - x_min) * scale_x).astype(np.int32), 0, W - 1)
            py = np.clip(((y_max - xy[:, 1]) * scale_y).astype(np.int32), 0, H - 1)
            return px, py

        # Pre-compute the (dy, dx) splat kernel once -- a filled square
        # of side 2*dot_radius+1. Cheaper to reuse than to re-mesh per
        # frame.
        rng = np.arange(-dot_radius, dot_radius + 1)
        ky, kx = np.meshgrid(rng, rng, indexing="ij")
        ky, kx = ky.ravel(), kx.ravel()
        # Slightly larger kernel for targets so they're distinguishable.
        rng_t = np.arange(-dot_radius - 1, dot_radius + 2)
        kty, ktx = np.meshgrid(rng_t, rng_t, indexing="ij")
        kty, ktx = kty.ravel(), ktx.ravel()

        # Color palette.
        bg_color = np.array([255, 255, 255], dtype=np.uint8)
        target_color = np.array([0, 0, 0], dtype=np.uint8)
        success_color = np.array([50, 205, 50], dtype=np.uint8)  # limegreen
        fail_color = np.array([220, 20, 60], dtype=np.uint8)  # crimson

        rendered_frames: list[Image.Image] = []
        for frame_idx, (robot_xy, target_xy, success) in enumerate(frames):
            if self._bg_canvas is not None:
                buf = self._bg_canvas.copy()
            else:
                buf = np.broadcast_to(bg_color, (H, W, 3)).copy()

            # Targets first so robots paint over them on overlap.
            tx, ty = world_to_pixel(target_xy)
            tys = (ty[:, None] + kty[None, :]).ravel()
            txs = (tx[:, None] + ktx[None, :]).ravel()
            mask = (tys >= 0) & (tys < H) & (txs >= 0) & (txs < W)
            buf[tys[mask], txs[mask]] = target_color

            # Robots, vectorized splat across all envs in one assignment.
            rx, ry = world_to_pixel(robot_xy)
            rys = (ry[:, None] + ky[None, :]).ravel()  # [n*K]
            rxs = (rx[:, None] + kx[None, :]).ravel()
            colors = np.where(success[:, None], success_color, fail_color)  # [n, 3]
            colors = np.broadcast_to(colors[:, None, :], (n, ky.size, 3)).reshape(-1, 3)
            mask = (rys >= 0) & (rys < H) & (rxs >= 0) & (rxs < W)
            buf[rys[mask], rxs[mask]] = colors[mask]

            # Suppress unused-frame_idx lint -- frame_idx kept in case
            # we later overlay a step counter or progress bar.
            _ = frame_idx
            _ = end_step

            rendered_frames.append(Image.fromarray(buf, mode="RGB"))

        if out_path.suffix == ".gif":
            duration_ms = int(round(1000.0 / max(self.fps, 1)))
            # Quantize every frame against the shared master palette
            # baked at recorder init. Each frame is then a 'P'-mode image
            # with the same palette table, which the gif encoder stores
            # *once* in the file header rather than once per frame --
            # this is the difference between a 27 MiB gif and ~10 MiB
            # for 200 frames over a complex terrain bg.
            if self._gif_palette_image is not None:
                quantized = [f.quantize(palette=self._gif_palette_image, dither=Image.NONE) for f in rendered_frames]
            else:
                quantized = rendered_frames
            quantized[0].save(
                str(out_path),
                save_all=True,
                append_images=quantized[1:],
                duration=duration_ms,
                loop=0,
                optimize=False,  # the master-palette path already handles savings
            )
        else:
            # mp4 path -- imageio with the ffmpeg plugin. Only reachable
            # when ffmpeg is on PATH (the launcher checks shutil.which).
            import imageio.v2 as imageio

            with imageio.get_writer(str(out_path), fps=self.fps, codec="libx264", quality=7) as wr:
                for img in rendered_frames:
                    wr.append_data(np.asarray(img))
