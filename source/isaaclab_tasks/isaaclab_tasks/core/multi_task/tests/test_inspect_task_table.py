# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the shared simulator-free task-table inspector."""

from __future__ import annotations

import ast
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "inspect_task_table.py"
_MULTI_TASK_ROOT = Path(__file__).parents[1]
_OBSOLETE_TABLE_TOOLS = (
    "scripts/visualize_states.py",
    "terrain/scripts/validate_spawn_points.py",
    "terrain/scripts/preview_spawn_scatter.py",
    "terrain/scripts/trace_ik.py",
    "terrain/scripts/sampler_metrics.py",
    "terrain/scripts/profile_pipeline.py",
    "terrain/scripts/bench_fused_sampler.py",
    "factory/scripts/inspect_factory_reset_state.py",
)


def _load_inspector():
    spec = importlib.util.spec_from_file_location("test_inspect_task_table_module", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inspector_source_has_no_simulator_or_domain_imports() -> None:
    """The shared inspector cannot import simulation runtime or domain builders."""
    tree = ast.parse(_SCRIPT_PATH.read_text())
    imports = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names] + [
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ]
    forbidden_modules = (
        "isaaclab.app",
        "isaaclab.envs",
        "isaaclab.sim",
        "isaacsim",
        "omni",
        "carb",
        "isaaclab_tasks.core.multi_task.factory",
        "isaaclab_tasks.core.multi_task.motion",
        "isaaclab_tasks.core.multi_task.terrain",
    )
    assert not [name for name in imports if name.startswith(forbidden_modules)]
    assert not {"AppLauncher", "SimulationContext", "launch_simulation"}.intersection(
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    )


def test_obsolete_task_table_tools_are_absent() -> None:
    """Only the shared production-table inspector may own these visualization workflows."""
    assert [path for path in _OBSOLETE_TABLE_TOOLS if (_MULTI_TASK_ROOT / path).exists()] == []


def test_inspector_import_loads_no_simulator_runtime() -> None:
    """Importing the inspector alone is stdlib-only and does not require Viser."""
    before = set(sys.modules)
    _load_inspector()
    loaded = set(sys.modules) - before
    assert not [name for name in loaded if name.startswith(("isaacsim", "omni", "carb", "isaaclab.app"))]


def test_viser_dependency_fails_with_exact_install_guidance(monkeypatch) -> None:
    """A missing optional viewer exits before configuration or table construction."""
    inspector = _load_inspector()
    real_import = __import__

    def import_without_viser(name, *args, **kwargs):
        if name == "viser":
            raise ModuleNotFoundError("No module named 'viser'", name="viser")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", import_without_viser)
    with pytest.raises(SystemExit) as error:
        inspector.main(["--task", "Isaac-Position-v0", "--command", "goal_point"])
    assert str(error.value) == "Install Viser with: ./isaaclab.sh -i 'visualizer[viser]'"


def test_cli_declares_only_task_and_command() -> None:
    """Hydra owns every runtime override beyond task and command selection."""
    tree = ast.parse(_SCRIPT_PATH.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    flags = {
        call.args[0].value
        for call in ast.walk(main)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "add_argument"
        and call.args
        and isinstance(call.args[0], ast.Constant)
    }

    assert flags == {"--task", "--command"}


@pytest.mark.parametrize(("timed", "expected"), ((False, "static"), (True, "timed")))
def test_main_uses_hydra_device_and_table_timing(monkeypatch, capsys, timed: bool, expected: str) -> None:
    """Main builds once on the resolved device and dispatches from table metadata."""
    inspector = _load_inspector()
    calls = []
    logger_enabled = []
    view = SimpleNamespace(
        sequences=SimpleNamespace(sequence_count=20, frame_count=40, is_timed=timed),
        state_bank=SimpleNamespace(row_count=12),
    )
    table = SimpleNamespace(view=view)

    class FakeTableCfg:
        def build(self, command_cfg, scene_cfg, device):
            logger_enabled.append(logging.getLogger(inspector._TASK_FAMILY_LOGGER).level == logging.INFO)
            calls.append(("build", command_cfg, scene_cfg, device))
            return table

    command_cfg = SimpleNamespace(task_table=FakeTableCfg())
    scene_cfg = object()
    env_cfg = SimpleNamespace(
        commands=SimpleNamespace(motion=command_cfg),
        scene=scene_cfg,
        sim=SimpleNamespace(device="cuda:3"),
    )

    import isaaclab_tasks.utils.hydra as hydra

    def resolve(task, agent):
        calls.append(("resolve", task, agent, tuple(sys.argv[1:])))
        return env_cfg, None

    monkeypatch.setattr(hydra, "resolve_task_config", resolve)
    monkeypatch.setitem(sys.modules, "viser", ModuleType("viser"))
    viewer_module = ModuleType("newton.viewer")
    viewer_module.ViewerViser = object
    monkeypatch.setitem(sys.modules, "newton.viewer", viewer_module)
    monkeypatch.setattr(
        inspector,
        "_inspect_static",
        lambda selected_view, viewer, count: calls.append(("static", selected_view, viewer, count)),
    )
    monkeypatch.setattr(
        inspector,
        "_inspect_timed",
        lambda selected_view, viewer, count: calls.append(("timed", selected_view, viewer, count)),
    )

    inspector.main(
        [
            "--task",
            "Isaac-Motion-Imitation-v0",
            "--command",
            "motion",
            "sim.device=cuda:3",
        ]
    )

    assert calls[0] == ("resolve", "Isaac-Motion-Imitation-v0", "", ("sim.device=cuda:3",))
    assert calls[1] == ("build", command_cfg, scene_cfg, "cuda:3")
    assert calls[2] == (expected, view, object, 16)
    assert logger_enabled == [True]
    output = capsys.readouterr().out
    assert "Task table built: seconds=" in output
    assert "states=12 sequences=20 frames=40" in output


def test_static_view_uses_declared_spacing_and_two_frames_per_sequence(monkeypatch) -> None:
    """Static inspection repeats exact endpoint states and applies table-owned layout."""
    inspector = _load_inspector()
    captured = {}

    class Sequences:
        offsets = torch.tensor((0, 2, 4), dtype=torch.int64)

        @staticmethod
        def state_rows(sequence_indices, frame_indices):
            captured["sequence"] = sequence_indices
            return 10 * sequence_indices + frame_indices

    class Kinematics:
        world_spacing = (0.0, 0.0, 0.0)

        @staticmethod
        def joint_q_into(_state_bank, state_rows, _out):
            captured["rows"] = state_rows

    class Viewer:
        def __init__(self):
            self.world_offsets = SimpleNamespace(numpy=lambda: np.zeros((4, 3), dtype=np.float32))
            self.spacing = None
            self.closed = False

        def set_model(self, _model):
            pass

        def set_world_offsets(self, spacing):
            self.spacing = spacing

        def begin_frame(self, _time):
            pass

        def log_state(self, _state):
            pass

        def end_frame(self):
            pass

        def is_running(self):
            return False

        def close(self):
            self.closed = True

    viewer = Viewer()
    view = SimpleNamespace(
        sequences=Sequences(),
        state_bank=object(),
        kinematic_view=Kinematics(),
        points=(),
        lines=(),
        quality=None,
    )

    def fake_repeat(_view, count):
        return count, None, torch.empty(4, 1), None, None

    import newton

    monkeypatch.setattr(inspector, "_repeat_kinematic_model", fake_repeat)
    monkeypatch.setattr(newton, "eval_fk", lambda *_args: None)

    inspector._inspect_static(view, lambda: viewer, 2)

    assert viewer.spacing == (0.0, 0.0, 0.0)
    assert viewer.closed
    assert captured["sequence"].tolist() == [0, 0, 1, 1]
    assert captured["rows"].tolist() == [0, 1, 10, 11]


def test_exact_timeline_holds_other_sequences_between_integer_frames() -> None:
    """Mixed sample periods advance only at stored-frame timestamps and never interpolate."""
    inspector = _load_inspector()

    assert list(inspector._timeline_events((3, 4), (0.5, 0.25))) == [
        (0.0, (0, 0)),
        (0.25, (0, 1)),
        (0.5, (1, 2)),
        (0.75, (1, 3)),
        (1.0, (2, 3)),
    ]


def test_timed_loop_contains_no_tensor_construction_or_sequence_lookup() -> None:
    """Timed playback precomputes its event indices before entering the live loop."""
    tree = ast.parse(_SCRIPT_PATH.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_inspect_timed")
    loop = next(node for node in ast.walk(function) if isinstance(node, ast.While))
    calls = [node for node in ast.walk(loop) if isinstance(node, ast.Call)]

    assert not [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr in {"tensor", "as_tensor", "state_rows", "cpu", "tolist"}
    ]


def test_kinematic_model_adds_shared_geometry_once_and_state_geometry_per_world() -> None:
    """Global geometry stays shared while state geometry repeats for each displayed world."""
    import newton

    from isaaclab_tasks.core.multi_task.mdp.commands.state_command import TaskTableKinematicView

    inspector = _load_inspector()
    shared_builder = newton.ModelBuilder()
    shared_builder.add_shape_box(body=-1, hx=4.0, hy=4.0, hz=0.1)
    state_builder = newton.ModelBuilder()
    body = state_builder.add_body(label="robot")
    state_builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1)
    state_model = state_builder.finalize(device="cpu")
    kinematics = TaskTableKinematicView(
        model_builder_state=state_builder,
        model_builder_shared=shared_builder,
        joint_q_default=torch.from_numpy(state_model.joint_q.numpy().copy()),
        root_entity_names=(),
        root_state_indices=torch.empty(0, dtype=torch.int64),
        root_q_indices=torch.empty(0, 7, dtype=torch.int64),
        joint_coordinate_names=(),
        joint_state_indices=torch.empty(0, dtype=torch.int64),
        joint_q_indices=torch.empty(0, dtype=torch.int64),
    )

    model, _state, joint_q, _joint_q_warp, _joint_qd_zero = inspector._repeat_kinematic_model(
        SimpleNamespace(kinematic_view=kinematics), 3
    )

    assert model.world_count == 3
    assert model.body_count == 3
    assert model.shape_count == 4
    assert model.shape_world.numpy().tolist() == [-1, 0, 1, 2]
    assert joint_q.shape == (3, state_model.joint_coord_count)
