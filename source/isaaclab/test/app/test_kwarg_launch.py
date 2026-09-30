# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import logging

import pytest
from isaaclab_newton.physics import VBDSolverCfg

import isaaclab.app as app_module
import isaaclab.app.app_launcher as app_launcher_module
import isaaclab.app.sim_launcher as sim_launcher
import isaaclab.utils as utils_module
from isaaclab.app import AppLauncher
from isaaclab.app.sim_launcher import Scan, _get_kit_runtime_sources

pytestmark = pytest.mark.integration


def test_make_physics_cfg_builds_core_vbd():
    physics_cfg = sim_launcher.make_physics_cfg("newton_vbd")

    assert isinstance(physics_cfg, VBDSolverCfg)


@pytest.mark.usefixtures("mocker")
def test_livestream_launch_with_kwargs(mocker):
    """Test launching with keyword arguments."""
    # everything defaults to None
    app_launcher = AppLauncher(headless=True, livestream=1)
    app = app_launcher.app
    assert app_launcher._livestream == 1
    assert app_launcher._headless is True

    # close the app on exit
    app.close()


def test_explicit_experience_requires_isaac_sim_runtime():
    """An explicit Kit experience must override a kitless physics configuration."""
    scan = Scan(
        resolved_physics_cfg=None,
        effective_cfg=object(),
        visualizer_intent={"has_any_visualizers": False, "has_kit_visualizer": False},
        has_ovrtx=False,
        has_kit_camera=False,
        has_kit_physics=False,
        has_kitless_physics=True,
        has_ovphysx_physics=False,
        needs_kit=False,
    )
    args = argparse.Namespace(experience="isaaclab.python.kit")

    assert _get_kit_runtime_sources(scan, args)


@pytest.mark.parametrize(
    "visualizer_intent, xr, headless_env, livestream, expected_headless, expected_has_window",
    [
        ({"has_any_visualizers": True, "has_kit_visualizer": True}, False, 0, 0, False, True),
        ({"has_any_visualizers": True, "has_kit_visualizer": True}, True, 0, 0, False, True),
        ({"has_any_visualizers": True, "has_kit_visualizer": True}, True, 1, 0, True, False),
        ({"has_any_visualizers": True, "has_kit_visualizer": False}, True, 0, 0, True, False),
        ({"has_any_visualizers": False, "has_kit_visualizer": False}, False, 0, 0, True, False),
        ({"has_any_visualizers": False, "has_kit_visualizer": False}, True, 0, 1, True, True),
    ],
)
def test_resolved_config_visualizer_controls_headless(
    visualizer_intent: dict,
    xr: bool,
    headless_env: int,
    livestream: int,
    expected_headless: bool,
    expected_has_window: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    """Only a Kit visualizer in the resolved config opens a local viewport."""
    monkeypatch.setenv("HEADLESS", str(headless_env))
    monkeypatch.delenv("XR", raising=False)
    launcher = AppLauncher.__new__(AppLauncher)
    launcher._livestream = livestream
    args = {"visualizer_intent": visualizer_intent, "xr": xr}

    launcher._resolve_visualizer_intent(args)
    launcher._resolve_xr_settings(args)
    launcher._resolve_headless_settings(args, livestream_arg=-1, livestream_env=0)

    assert launcher._headless is expected_headless
    assert launcher.has_window is expected_has_window


def test_launch_simulation_preserves_failure_exit_code(monkeypatch: pytest.MonkeyPatch):
    close_args = {}

    class _FakeApp:
        def close(self, *, exit_code: int = 0) -> None:
            close_args["exit_code"] = exit_code

    class _FakeAppLauncher:
        def __init__(self, _launcher_args):
            self.app = _FakeApp()

    scan = sim_launcher.Scan(
        resolved_physics_cfg=None,
        effective_cfg=object(),
        visualizer_intent={"has_any_visualizers": False, "has_kit_visualizer": False},
        has_ovrtx=False,
        has_kit_camera=False,
        has_kit_physics=True,
        has_kitless_physics=False,
        has_ovphysx_physics=False,
        needs_kit=True,
    )
    monkeypatch.setattr(sim_launcher, "scan", lambda cfg, physics: scan)
    monkeypatch.setattr(sim_launcher, "_ensure_isaac_sim_available", lambda: None)
    monkeypatch.setattr(app_module, "AppLauncher", _FakeAppLauncher)
    monkeypatch.setattr(utils_module, "has_kit", lambda: False)

    with pytest.raises(RuntimeError, match="sentinel"):
        with sim_launcher.launch_simulation(object(), argparse.Namespace()):
            raise RuntimeError("sentinel")

    assert close_args == {"exit_code": 1}


def test_launch_simulation_auto_enables_kit_camera_without_launcher_args(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LIVESTREAM", raising=False)
    received_args = {}

    class _FakeApp:
        def close(self) -> None:
            pass

    class _FakeAppLauncher:
        def __init__(self, launcher_args):
            received_args.update(launcher_args)
            self.app = _FakeApp()

    scan = sim_launcher.Scan(
        resolved_physics_cfg=None,
        effective_cfg=object(),
        visualizer_intent={"has_any_visualizers": False, "has_kit_visualizer": False},
        has_ovrtx=False,
        has_kit_camera=True,
        has_kit_physics=False,
        has_kitless_physics=False,
        has_ovphysx_physics=False,
        needs_kit=True,
    )

    def _scan(_cfg, launcher_args):
        assert launcher_args == {}
        return scan

    monkeypatch.setattr(sim_launcher, "scan", _scan)
    monkeypatch.setattr(sim_launcher, "_ensure_isaac_sim_available", lambda: None)
    monkeypatch.setattr(app_module, "AppLauncher", _FakeAppLauncher)
    monkeypatch.setattr(utils_module, "has_kit", lambda: False)

    with sim_launcher.launch_simulation(object()):
        pass

    assert received_args["enable_cameras"] is True


def test_deferred_cuda_device_synchronizes_torch_and_warp(monkeypatch: pytest.MonkeyPatch):
    """The post-Kit device hook must synchronize both CUDA runtimes."""
    devices = []
    monkeypatch.setattr(app_launcher_module, "set_cuda_device", devices.append)
    launcher = AppLauncher.__new__(AppLauncher)
    launcher._deferred_cuda_device_id = 2

    launcher._set_deferred_cuda_device()

    assert devices == [2]


def test_limit_cpu_threads_forwarded_to_simulation_app(monkeypatch: pytest.MonkeyPatch):
    """A SimulationApp thread limit must survive AppLauncher config resolution."""
    monkeypatch.setenv("HEADLESS", "0")
    monkeypatch.setenv("LIVESTREAM", "0")
    monkeypatch.setenv("XR", "0")

    launcher = AppLauncher.__new__(AppLauncher)
    monkeypatch.setattr(launcher, "_resolve_experience_file", lambda _launcher_args: None)

    launcher._config_resolution({"headless": True, "device": "cpu", "limit_cpu_threads": 1})

    assert launcher._sim_app_config["limit_cpu_threads"] == 1


@pytest.mark.parametrize(
    ("headless", "livestream", "xr", "visualizer_intent", "disabled"),
    [
        (True, 0, False, None, True),
        (False, 0, False, {"has_any_visualizers": True, "has_kit_visualizer": True}, False),
        (True, 1, False, None, False),
        (True, 0, True, None, False),
    ],
)
def test_stage_and_viewport_startup_forwarded_to_simulation_app(
    monkeypatch: pytest.MonkeyPatch,
    headless: bool,
    livestream: int,
    xr: bool,
    visualizer_intent: dict | None,
    disabled: bool,
):
    """Skip the throwaway stage and only update a viewport that will be shown."""
    monkeypatch.setenv("HEADLESS", "0")
    monkeypatch.setenv("LIVESTREAM", "0")
    monkeypatch.setenv("XR", "0")
    launcher = AppLauncher.__new__(AppLauncher)
    monkeypatch.setattr(launcher, "_resolve_experience_file", lambda _launcher_args: None)

    launcher._config_resolution(
        {
            "headless": headless,
            "livestream": livestream,
            "xr": xr,
            "device": "cpu",
            "visualizer_intent": visualizer_intent,
        }
    )

    assert launcher._sim_app_config["create_new_stage"] is False
    assert launcher._sim_app_config["disable_viewport_updates"] is disabled


class _DummySettings:
    def __init__(self):
        self.values = {}

    def set_string(self, path: str, value: str) -> None:
        self.values[path] = value

    def set_int(self, path: str, value: int) -> None:
        self.values[path] = value

    def set_bool(self, path: str, value: bool) -> None:
        self.values[path] = value


@pytest.mark.parametrize("deterministic", [True, False])
def test_load_extensions_publishes_deterministic_setting(monkeypatch: pytest.MonkeyPatch, deterministic: bool):
    """Publish ``/isaaclab/render/deterministic`` from ``_load_extensions``."""
    launcher = AppLauncher.__new__(AppLauncher)
    launcher._deterministic_rendering = deterministic
    launcher._python_logging_level = logging.ERROR
    launcher._headless = True
    launcher._livestream = 0
    launcher._enable_cameras = False
    launcher._offscreen_render = False
    launcher._render_viewport = False
    launcher._xr = False
    launcher._video_enabled = False

    settings = _DummySettings()
    monkeypatch.setattr(app_launcher_module, "initialize_carb_settings", lambda: None)
    monkeypatch.setattr(app_launcher_module, "get_settings_manager", lambda: settings)
    monkeypatch.setattr(app_launcher_module, "apply_python_logging_level", lambda _level: None)

    launcher._load_extensions()

    assert settings.values["/isaaclab/render/deterministic"] is deterministic


@pytest.mark.parametrize(
    ("headless", "livestream", "xr", "expected_has_gui", "expected_xr_auto_start"),
    [
        pytest.param(False, 0, False, True, False, id="local-window"),
        pytest.param(True, 0, False, False, False, id="headless"),
        pytest.param(True, 1, False, True, False, id="livestream"),
        pytest.param(True, 0, True, True, True, id="xr"),
        # XR with a window: the operator starts the session, so it must not auto-start.
        pytest.param(False, 0, True, True, False, id="xr-windowed"),
        # ...but a windowless XR run must, however that windowless state was reached.
        pytest.param(True, 1, True, True, True, id="xr-livestream"),
    ],
)
def test_load_extensions_publishes_has_gui_setting(
    monkeypatch: pytest.MonkeyPatch,
    headless: bool,
    livestream: int,
    xr: bool,
    expected_has_gui: bool,
    expected_xr_auto_start: bool,
):
    """Publish the GUI and XR auto-start state consumed by SimulationContext and the teleop stack.

    Asserted at the publication site rather than by recomputing the expression, so that changing
    what is published fails here instead of silently agreeing with a restated formula.
    """
    launcher = AppLauncher.__new__(AppLauncher)
    launcher._deterministic_rendering = False
    launcher._python_logging_level = logging.ERROR
    launcher._headless = headless
    launcher._livestream = livestream
    launcher._enable_cameras = False
    launcher._offscreen_render = False
    launcher._render_viewport = False
    launcher._xr = xr
    launcher._video_enabled = False

    settings = _DummySettings()
    monkeypatch.setattr(app_launcher_module, "initialize_carb_settings", lambda: None)
    monkeypatch.setattr(app_launcher_module, "get_settings_manager", lambda: settings)
    monkeypatch.setattr(app_launcher_module, "apply_python_logging_level", lambda _level: None)

    launcher._load_extensions()

    assert settings.values["/isaaclab/has_gui"] is expected_has_gui
    assert settings.values["/isaaclab/xr/auto_start"] is expected_xr_auto_start


def _resolve_headless_for_case(monkeypatch: pytest.MonkeyPatch, visualizer_intent: dict | None) -> bool:
    monkeypatch.setenv("HEADLESS", "0")
    launcher = AppLauncher.__new__(AppLauncher)
    launcher._livestream = 0
    launcher_args = {"visualizer_intent": visualizer_intent}
    launcher._resolve_visualizer_intent(launcher_args)
    launcher._resolve_headless_settings(launcher_args, livestream_arg=-1, livestream_env=0)
    return launcher._headless


@pytest.mark.parametrize(
    ("intent", "expected_headless"),
    [
        ({"has_any_visualizers": True, "has_kit_visualizer": True}, False),
        ({"has_any_visualizers": True, "has_kit_visualizer": False}, True),
        ({"has_any_visualizers": False, "has_kit_visualizer": False}, True),
        (None, True),
    ],
)
def test_config_visualizer_intent_controls_window(intent, expected_headless, monkeypatch: pytest.MonkeyPatch):
    assert _resolve_headless_for_case(monkeypatch, intent) is expected_headless


def test_invalid_visualizer_intent_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HEADLESS", "0")
    launcher = AppLauncher.__new__(AppLauncher)
    with pytest.raises(ValueError, match="visualizer_intent"):
        launcher._resolve_visualizer_intent({"visualizer_intent": {"has_any_visualizers": "yes"}})


def _new_launcher_for_experience_check():
    launcher = AppLauncher.__new__(AppLauncher)
    launcher._enable_cameras = False
    launcher._headless = False
    launcher._xr = False
    launcher._deterministic_rendering = False
    launcher.is_isaac_sim_version_5 = lambda: False
    return launcher


def test_rejects_isaacsim_full_streaming_experience_with_livestream(tmp_path, monkeypatch: pytest.MonkeyPatch):
    experience = tmp_path / "isaacsim.exp.full.streaming.kit"
    experience.write_text('[dependencies]\n"isaacsim.exp.full" = {}\n', encoding="utf-8")
    monkeypatch.setenv("EXP_PATH", str(tmp_path))
    launcher = _new_launcher_for_experience_check()
    launcher._livestream = 2

    with pytest.raises(ValueError, match="depends on 'isaacsim.exp.full'"):
        launcher._resolve_experience_file({"experience": str(experience)})


def test_rejects_custom_experience_with_isaacsim_full_dependency_and_livestream(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    experience = tmp_path / "merged.kit"
    experience.write_text('[dependencies]\n"isaaclab.python" = {}\n"isaacsim.exp.full" = {}\n', encoding="utf-8")
    monkeypatch.setenv("EXP_PATH", str(tmp_path))
    launcher = _new_launcher_for_experience_check()
    launcher._livestream = 2

    with pytest.raises(ValueError, match="depends on 'isaacsim.exp.full'"):
        launcher._resolve_experience_file({"experience": str(experience)})


def test_allows_isaacsim_full_streaming_experience_when_livestream_disabled(tmp_path, monkeypatch: pytest.MonkeyPatch):
    experience = tmp_path / "isaacsim.exp.full.streaming.kit"
    experience.write_text('[dependencies]\n"isaacsim.exp.full" = {}\n', encoding="utf-8")
    monkeypatch.setenv("EXP_PATH", str(tmp_path))
    launcher = _new_launcher_for_experience_check()
    launcher._livestream = 0

    launcher._resolve_experience_file({"experience": str(experience)})

    assert launcher._sim_experience_file == str(experience)


def test_constructor_reports_missing_isaac_sim(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(app_launcher_module, "SimulationApp", None)

    with pytest.raises(ImportError, match="requires the full Isaac Sim runtime"):
        AppLauncher()


def test_is_available_reflects_simulation_app_presence(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(app_launcher_module, "SimulationApp", None)
    assert AppLauncher.is_available() is False

    monkeypatch.setattr(app_launcher_module, "SimulationApp", object())
    assert AppLauncher.is_available() is True


def test_has_gui_reads_published_setting():
    from isaaclab.app.settings_manager import get_settings_manager

    settings = get_settings_manager()
    original = settings.get("/isaaclab/has_gui")
    try:
        settings.set_bool("/isaaclab/has_gui", True)
        assert AppLauncher.has_gui() is True

        settings.set_bool("/isaaclab/has_gui", False)
        assert AppLauncher.has_gui() is False
    finally:
        settings.set_bool("/isaaclab/has_gui", bool(original))
