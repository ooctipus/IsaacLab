# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the shared simulator-free task-table inspector."""

from __future__ import annotations

import ast
import importlib.util
import itertools
import logging
import sys
import tracemalloc
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


def test_base_task_table_inspection_does_not_slice_position_or_factory_views() -> None:
    """The shared display limit is advisory unless a domain owns early construction selection."""
    from isaaclab_tasks.core.multi_task.mdp.commands.state_command import StateCommandCfg

    view = object()
    calls = []

    def build(command_cfg, scene_cfg, device):
        calls.append((command_cfg, scene_cfg, device))
        return SimpleNamespace(view=view)

    table_cfg = StateCommandCfg.TaskTableCfg(class_type=build)
    command_cfg = SimpleNamespace(task_table=table_cfg)
    scene_cfg = object()

    assert table_cfg.build_inspection_view(command_cfg, scene_cfg, "cuda:2", sequence_limit=1) is view
    assert calls == [(command_cfg, scene_cfg, "cuda:2")]


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
        def build_inspection_view(self, command_cfg, scene_cfg, device, *, sequence_limit):
            logger_enabled.append(logging.getLogger(inspector._TASK_FAMILY_LOGGER).level == logging.INFO)
            calls.append(("build_inspection_view", command_cfg, scene_cfg, device, sequence_limit))
            return table.view

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
    assert calls[1] == ("build_inspection_view", command_cfg, scene_cfg, "cuda:3", 16)
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
        (0.0, ((0, 0), (1, 0))),
        (0.25, ((1, 1),)),
        (0.5, ((0, 1), (1, 2))),
        (0.75, ((1, 3),)),
        (1.0, ((0, 2),)),
    ]


def test_timed_loop_contains_no_tensor_construction_or_sequence_lookup() -> None:
    """Timed playback updates preallocated row indices without materializing event tensors."""
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
    assert "event_state_rows" not in {node.id for node in ast.walk(function) if isinstance(node, ast.Name)}
    assert not [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "tuple"
        and any(
            isinstance(descendant, ast.Call)
            and isinstance(descendant.func, ast.Name)
            and descendant.func.id == "_timeline_events"
            for descendant in ast.walk(node)
        )
    ]


def test_timed_view_streams_mixed_clocks_into_preallocated_state_rows(monkeypatch) -> None:
    """Mixed-rate playback resolves one O(sequence-count) state vector per sparse event."""
    import newton

    inspector = _load_inspector()
    captured_rows = []
    frame_times = []

    class Sequences:
        offsets = torch.tensor((0, 3, 7), dtype=torch.int64)
        state_indices = torch.tensor((10, 11, 12, 20, 21, 22, 23), dtype=torch.int64)
        frame_dt = torch.tensor((0.5, 0.25), dtype=torch.float32)

    class Kinematics:
        world_spacing = (0.0, 0.0, 0.0)

        @staticmethod
        def joint_q_into(_state_bank, state_rows, _out):
            captured_rows.append(state_rows.clone())

    class Viewer:
        def __init__(self):
            self.world_offsets = SimpleNamespace(numpy=lambda: np.zeros((2, 3), dtype=np.float32))
            self.closed = False

        def set_model(self, _model):
            pass

        def set_world_offsets(self, _spacing):
            pass

        def begin_frame(self, frame_time):
            frame_times.append(frame_time)

        def log_state(self, _state):
            pass

        def end_frame(self):
            pass

        def is_running(self):
            return len(captured_rows) < 5

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

    monkeypatch.setattr(
        inspector,
        "_repeat_kinematic_model",
        lambda _view, count: (count, None, torch.empty((count, 1)), None, None),
    )
    timeline_events = inspector._timeline_events

    def require_previous_event_rendered(*args):
        for event_index, event in enumerate(timeline_events(*args)):
            assert len(captured_rows) == event_index
            yield event

    monkeypatch.setattr(inspector, "_timeline_events", require_previous_event_rendered)
    monkeypatch.setattr(inspector.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(newton, "eval_fk", lambda *_args: None)

    inspector._inspect_timed(view, lambda: viewer, 2)

    assert viewer.closed
    assert frame_times == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert [rows.tolist() for rows in captured_rows] == [
        [10, 20],
        [10, 21],
        [11, 22],
        [11, 23],
        [12, 23],
    ]


def test_timeline_merge_uses_bounded_memory_for_mixed_rate_long_clips() -> None:
    """Huge clip lengths affect event count, not the scheduler's resident memory."""
    inspector = _load_inspector()
    frame_counts = tuple(10_000_000 + index for index in range(16))
    frame_dt = tuple(1.0 / (24.0 + index) for index in range(16))

    tracemalloc.start()
    events = inspector._timeline_events(frame_counts, frame_dt)
    consumed = sum(1 for _ in itertools.islice(events, 4096))
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert consumed == 4096
    assert peak_bytes < 100_000


def test_evidence_renderer_logs_source_target_contact_and_rejection_geometry() -> None:
    """Table-owned trajectory evidence reaches Viser without domain reconstruction."""
    import warp as wp

    wp.init()
    inspector = _load_inspector()
    calls = []

    class Viewer:
        def log_points(self, name, points, **_kwargs):
            calls.append(("points", name, tuple(points.shape)))

        def log_lines(self, name, starts, ends, **_kwargs):
            calls.append(("lines", name, tuple(starts.shape), tuple(ends.shape)))

        def log_scalar(self, name, value):
            calls.append(("scalar", name, float(value)))

    source = torch.zeros((2, 2, 3), dtype=torch.float32)
    target = torch.ones((2, 2, 3), dtype=torch.float32)
    rejected = torch.tensor((False, True), dtype=torch.bool).reshape(2, 1)
    view = SimpleNamespace(
        points=(
            SimpleNamespace(
                name="source_landmarks", points=source, scope="state", valid=None, radius=0.01, color=(0.3, 1.0, 0.4)
            ),
            SimpleNamespace(
                name="target_landmarks", points=target, scope="state", valid=None, radius=0.01, color=(0.2, 0.7, 1.0)
            ),
            SimpleNamespace(
                name="rejected_frames",
                points=target[:, :1],
                scope="state",
                valid=rejected,
                radius=0.03,
                color=(1.0, 0.1, 0.1),
            ),
        ),
        lines=(
            SimpleNamespace(
                name="contact_intervals",
                endpoints=torch.stack((source[:, :1], target[:, :1]), dim=2),
                scope="state",
                valid=None,
                color=(1.0, 0.75, 0.1),
                width=0.01,
            ),
        ),
        quality=SimpleNamespace(names=("accepted",), values=torch.tensor(((1.0,), (0.0,))), scope="sequence"),
    )
    inspector._log_evidence(Viewer(), view, torch.tensor((0, 1)), torch.zeros((2, 3)), view.quality.values)

    assert [call[1] for call in calls] == [
        "evidence/source_landmarks",
        "evidence/target_landmarks",
        "evidence/rejected_frames",
        "evidence/contact_intervals",
        "quality/accepted/world_0",
        "quality/accepted/world_1",
    ]
    assert calls[0][2] == (4, 3)
    assert calls[2][2] == (1, 3)
    assert calls[3][2:4] == ((2, 3), (2, 3))


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
