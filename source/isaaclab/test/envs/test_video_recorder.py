# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for VideoRecorder and VideoRecorderCfg.

All tests are pure-Python mocks — no simulation context or Kit app required.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from isaaclab.envs.utils.video_recorder import VideoRecorder, _parse_source
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

_FRAME = np.ones((8, 12, 3), dtype=np.uint8) * 128


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_moviepy():
    """Stub out ImageSequenceClip so tests run without moviepy installed.

    Tests that specifically validate the ImportError path re-patch to None
    inside their own context managers, which takes precedence over this stub.
    """
    with patch("isaaclab.envs.utils.video_recorder.ImageSequenceClip", MagicMock()):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(**overrides) -> VideoRecorderCfg:
    defaults = dict(source="visualizer", output_dir="/tmp/test_videos", fps=30, video_length=4, video_interval=0)
    cfg = VideoRecorderCfg()
    for k, v in {**defaults, **overrides}.items():
        setattr(cfg, k, v)
    return cfg


class _FakeViz:
    def __init__(self, viz_type: str, frame: np.ndarray | None = None):
        self.cfg = SimpleNamespace(visualizer_type=viz_type)
        self._frame = frame if frame is not None else _FRAME.copy()
        self.render_calls = 0

    def render_rgb_array(self) -> np.ndarray:
        self.render_calls += 1
        return self._frame


def _make_env(visualizers=(), sensors: dict | None = None):
    env = MagicMock()
    env.sim.visualizers = list(visualizers)
    env.scene.sensors = sensors or {}
    return env


# ---------------------------------------------------------------------------
# _parse_source
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,expected",
    [
        ("visualizer", ("visualizer", "", "")),
        ("visualizer:kit", ("visualizer", "kit", "")),
        ("visualizer:newton_gl:streaming_view", ("visualizer", "newton_gl", "streaming_view")),
        ("sensor:tiled_camera", ("sensor", "tiled_camera", "")),
        ("  visualizer:kit  ", ("visualizer", "kit", "")),
    ],
)
def test_parse_source(source, expected):
    assert _parse_source(source) == expected


# ---------------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------------


def test_init_raises_value_error_for_unknown_source_kind():
    with pytest.raises(ValueError, match="Unrecognized source kind"):
        VideoRecorder(_cfg(source="badkind:foo"), _make_env())


def test_init_raises_import_error_when_moviepy_missing():
    with patch("isaaclab.envs.utils.video_recorder.ImageSequenceClip", None):
        with pytest.raises(ImportError, match="moviepy"):
            VideoRecorder(_cfg(), _make_env())


def test_init_continues_clip_index_after_existing_files(tmp_path):
    output_dir = tmp_path / "videos"
    output_dir.mkdir()
    (output_dir / "clip_0000.mp4").touch()
    (output_dir / "clip_0007.mp4").touch()
    (output_dir / "clip_final.mp4").touch()
    (output_dir / "other_0008.mp4").touch()

    recorder = VideoRecorder(_cfg(output_dir=str(output_dir), output_filename_prefix="clip"), _make_env())

    assert recorder._clip_index == 8


def test_init_starts_clip_index_at_zero_for_empty_output_dir(tmp_path):
    output_dir = tmp_path / "videos"
    output_dir.mkdir()

    recorder = VideoRecorder(_cfg(output_dir=str(output_dir)), _make_env())

    assert recorder._clip_index == 0


# ---------------------------------------------------------------------------
# Trigger logic
# ---------------------------------------------------------------------------


def test_trigger_one_shot_fires_once():
    """video_interval=0 → single clip starts at step 1 and never re-triggers."""
    viz = _FakeViz("kit")
    recorder = VideoRecorder(
        _cfg(source="visualizer:kit", video_length=2, video_interval=0), _make_env(visualizers=[viz])
    )
    closed_count = [0]

    def counting_close():
        recorder._frames = []
        recorder._recording = False
        closed_count[0] += 1

    with patch.object(recorder, "_close_clip", side_effect=counting_close):
        for _ in range(8):
            recorder.step()
    assert closed_count[0] == 1


def test_trigger_recurring_fires_periodically():
    """video_interval=3 + video_length=1 → exactly 3 clips over 9 steps, first at step 1."""
    viz = _FakeViz("kit")
    recorder = VideoRecorder(
        _cfg(source="visualizer:kit", video_length=1, video_interval=3), _make_env(visualizers=[viz])
    )
    close_count = [0]
    trigger_steps = []

    def counting_close():
        close_count[0] += 1
        trigger_steps.append(recorder._step_count)
        recorder._frames = []
        recorder._recording = False

    with patch.object(recorder, "_close_clip", side_effect=counting_close):
        for _ in range(9):
            recorder.step()
    assert close_count[0] == 3
    # First clip should trigger at step 1 (not step 3).
    assert trigger_steps[0] == 1, f"Expected first clip at step 1, got step {trigger_steps[0]}"


# ---------------------------------------------------------------------------
# Frame collection, step_offset, frame_stride
# ---------------------------------------------------------------------------


def test_step_accumulates_frames_and_closes_clip():
    """Recorder collects exactly video_length frames then calls _close_clip."""
    viz = _FakeViz("kit")
    recorder = VideoRecorder(_cfg(source="visualizer:kit", video_length=3), _make_env(visualizers=[viz]))
    with patch.object(recorder, "_close_clip") as mock_close:
        for _ in range(3):
            recorder.step()
        assert mock_close.call_count == 1
    assert viz.render_calls == 3


def test_step_offset_delays_first_trigger():
    """step_offset=5 means the first clip starts at step 6, not step 1."""
    viz = _FakeViz("kit")
    recorder = VideoRecorder(
        _cfg(source="visualizer:kit", step_offset=5, video_length=100, video_interval=0),
        _make_env(visualizers=[viz]),
    )
    for _ in range(5):
        recorder.step()
    assert not recorder._recording
    recorder.step()  # step 6 — triggers
    assert recorder._recording


def test_frame_stride_subsamples_frames():
    """frame_stride=2 captures one frame every 2 steps."""
    viz = _FakeViz("kit")
    recorder = VideoRecorder(
        _cfg(source="visualizer:kit", video_length=4, frame_stride=2), _make_env(visualizers=[viz])
    )
    with patch.object(recorder, "_close_clip"):
        for _ in range(4):
            recorder.step()
    assert viz.render_calls == 2


# ---------------------------------------------------------------------------
# Visualizer frame routing
# ---------------------------------------------------------------------------


def test_visualizer_source_auto_picks_first_with_render_rgb_array():
    viz = _FakeViz("kit")
    recorder = VideoRecorder(_cfg(source="visualizer"), _make_env(visualizers=[viz]))
    recorder.step()
    assert viz.render_calls == 1


def test_visualizer_source_auto_no_visualizer_logs_and_returns_none(caplog):
    """source='visualizer' with no visualizers logs an error once and returns None instead of raising."""
    import logging

    recorder = VideoRecorder(_cfg(source="visualizer"), _make_env(visualizers=[]))
    with caplog.at_level(logging.ERROR, logger="isaaclab.envs.utils.video_recorder"):
        frame = recorder._get_frame()
    assert frame is None
    assert any("no recording-capable visualizer" in r.message for r in caplog.records)


def test_visualizer_newton_gl_selects_newton_gl():
    """source='visualizer:newton_gl' selects a Newton GL visualizer."""
    viz = _FakeViz("newton_gl")
    recorder = VideoRecorder(_cfg(source="visualizer:newton_gl"), _make_env(visualizers=[viz]))
    frame = recorder._get_frame()
    assert viz.render_calls == 1
    assert frame is not None


# ---------------------------------------------------------------------------
# Sensor frame routing
# ---------------------------------------------------------------------------


def test_sensor_source_reads_rgb():
    import torch

    rgb = torch.ones((1, 8, 12, 3), dtype=torch.uint8) * 200
    sensor = MagicMock()
    sensor.data.output = {"rgb": rgb}
    recorder = VideoRecorder(_cfg(source="sensor:tiled_camera"), _make_env(sensors={"tiled_camera": sensor}))
    frame = recorder._get_frame()
    assert frame is not None
    assert frame.shape == (8, 12, 3)


def test_sensor_source_missing_logs_and_returns_none(caplog):
    """Missing sensor logs an error (listing available sensors) and returns None instead of raising."""
    import logging

    sensors = {"tiled_camera": MagicMock()}
    recorder = VideoRecorder(_cfg(source="sensor:missing"), _make_env(sensors=sensors))
    with caplog.at_level(logging.ERROR, logger="isaaclab.envs.utils.video_recorder"):
        frame = recorder._get_frame()
    assert frame is None
    assert any("tiled_camera" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Clip writing
# ---------------------------------------------------------------------------


def test_close_clip_writes_mp4_via_moviepy():
    frames = [_FRAME.copy(), _FRAME.copy()]
    recorder = VideoRecorder(_cfg(output_dir="/tmp/test_clips", fps=10), _make_env())
    recorder._frames = frames
    recorder._recording = True

    mock_clip = MagicMock()
    with patch("isaaclab.envs.utils.video_recorder.ImageSequenceClip", return_value=mock_clip) as mock_cls:
        with patch("isaaclab.envs.utils.video_recorder.os.makedirs"):
            recorder._close_clip()

    mock_cls.assert_called_once_with(frames, fps=10)
    mock_clip.write_videofile.assert_called_once()
    assert not recorder._recording
    assert recorder._frames == []


def test_close_with_empty_frame_buffer_does_not_write():
    recorder = VideoRecorder(_cfg(), _make_env())
    recorder._recording = True
    recorder._frames = []
    mock_cls = MagicMock()
    with patch("isaaclab.envs.utils.video_recorder.ImageSequenceClip", mock_cls):
        recorder.close()
    mock_cls.assert_not_called()


# ---------------------------------------------------------------------------
# Minor 13: keep_last_n_clips pruning
# ---------------------------------------------------------------------------


def test_keep_last_n_clips_prunes_old_clips():
    """keep_last_n_clips=2 removes the oldest clip once a third is written."""

    recorder = VideoRecorder(
        _cfg(output_dir="/tmp/test_prune", keep_last_n_clips=2),
        _make_env(),
    )
    recorder._clip_index = 3
    removed = []

    def fake_remove(path):
        removed.append(path)

    with patch("isaaclab.envs.utils.video_recorder.os.path.isdir", return_value=True):
        with patch(
            "isaaclab.envs.utils.video_recorder.os.listdir",
            return_value=["clip_0000.mp4", "clip_0001.mp4", "clip_0002.mp4"],
        ):
            with patch("isaaclab.envs.utils.video_recorder.os.remove", side_effect=fake_remove):
                recorder._maybe_delete_old_clips()

    # After 3 clips with keep_last_n_clips=2, clip index 0 should be removed.
    assert any("_0000.mp4" in p for p in removed), f"Expected clip 0 to be removed, got: {removed}"


def test_keep_last_n_clips_prunes_only_existing_sparse_clips():
    """Sparse clip indices must not trigger one deletion attempt per missing index."""

    recorder = VideoRecorder(
        _cfg(output_dir="/tmp/test_sparse_prune", keep_last_n_clips=2),
        _make_env(),
    )
    recorder._clip_index = 10_000
    removed = []

    def fake_remove(path):
        removed.append(path)

    with patch("isaaclab.envs.utils.video_recorder.os.path.isdir", return_value=True):
        with patch(
            "isaaclab.envs.utils.video_recorder.os.listdir",
            return_value=["clip_0001.mp4", "clip_9997.mp4", "clip_9998.mp4", "other_0000.mp4"],
        ):
            with patch("isaaclab.envs.utils.video_recorder.os.remove", side_effect=fake_remove):
                recorder._maybe_delete_old_clips()

    assert removed == ["/tmp/test_sparse_prune/clip_0001.mp4", "/tmp/test_sparse_prune/clip_9997.mp4"]


# ---------------------------------------------------------------------------
# Minor 14: partial-clip close() flush
# ---------------------------------------------------------------------------


def test_close_flushes_partial_clip():
    """close() with non-empty _frames and _recording=True flushes the clip."""
    recorder = VideoRecorder(_cfg(output_dir="/tmp/test_partial", fps=10), _make_env())
    recorder._frames = [_FRAME.copy()]
    recorder._recording = True

    mock_clip = MagicMock()
    with patch("isaaclab.envs.utils.video_recorder.ImageSequenceClip", return_value=mock_clip):
        with patch("isaaclab.envs.utils.video_recorder.os.makedirs"):
            recorder.close()

    mock_clip.write_videofile.assert_called_once()
    assert not recorder._recording
    assert recorder._frames == []


# ---------------------------------------------------------------------------
# Minor 15: one-shot trigger uses mock _close_clip (no disk I/O)
# ---------------------------------------------------------------------------


def test_trigger_one_shot_fires_once_no_disk_io():
    """video_interval=0 → single clip starts at step 1; _close_clip called exactly once (mocked)."""
    viz = _FakeViz("kit")
    recorder = VideoRecorder(
        _cfg(source="visualizer:kit", video_length=2, video_interval=0), _make_env(visualizers=[viz])
    )
    close_count = [0]

    def counting_close():
        recorder._frames = []
        recorder._recording = False
        close_count[0] += 1

    with patch.object(recorder, "_close_clip", side_effect=counting_close):
        for _ in range(8):
            recorder.step()

    assert close_count[0] == 1
