# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for strict Newton physics incident capture and replay buffering."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics import NewtonCfg, NewtonDebugCaptureCfg, NewtonDebugReplayCfg
from isaaclab_newton.physics._debug_archive import load_archive
from isaaclab_newton.physics._debug_capture import DebugCaptureError
from isaaclab_newton.physics._incident_recorder import PhysicsIncidentRecorder

wp.init()


class _FakeState:
    """Small Newton-like state backed by CPU Warp arrays."""

    def __init__(self, body_count: int, joint_count: int) -> None:
        self.body_q = wp.zeros(body_count, dtype=wp.transformf, device="cpu")
        self.body_qd = wp.zeros(body_count, dtype=wp.spatial_vectorf, device="cpu")
        self.joint_q = wp.zeros(joint_count, dtype=wp.float32, device="cpu")
        self.joint_qd = wp.zeros(joint_count, dtype=wp.float32, device="cpu")
        self.global_metric = np.float64(0.0)


class _FakeModel:
    """Newton-like model exposing discoverable world-frequency metadata."""

    def __init__(self, *, world_count: int = 1, body_count: int = 4, joint_count: int = 4) -> None:
        self.world_count = world_count
        self.attribute_frequency = {
            "body_q": "body",
            "body_qd": "body",
            "joint_q": "joint_coord",
            "joint_qd": "joint_dof",
        }
        self._body_count = body_count
        self._joint_count = joint_count
        if world_count >= 1:
            if body_count % world_count or joint_count % world_count:
                raise ValueError("Fake model entity counts must divide evenly across worlds.")
            self.body_world_start = _warp_int_array(
                [index * body_count // world_count for index in range(world_count + 1)]
            )
            self.joint_coord_world_start = _warp_int_array(
                [index * joint_count // world_count for index in range(world_count + 1)]
            )
            self.joint_dof_world_start = _warp_int_array(
                [index * joint_count // world_count for index in range(world_count + 1)]
            )

    def state(self) -> _FakeState:
        """Create an allocated state matching this model."""
        return _FakeState(self._body_count, self._joint_count)


class _FakeSolver:
    """Solver with a field unknown to the recorder implementation."""

    def __init__(self) -> None:
        self.adaptive_vector = np.asarray([1.25, 2.5], dtype=np.float32)
        self.mode_inactive = np.asarray([0.0, 0.0], dtype=np.float32)
        self.iterations = np.int32(7)


class _EmptyState:
    """State whose public surface contains no capturable values."""

    def __init__(self) -> None:
        self.callback = lambda: None


class _EmptySolver:
    """Solver whose public surface contains no capturable values."""

    def __init__(self) -> None:
        self.callback = lambda: None


@dataclasses.dataclass(frozen=True, slots=True)
class _TriggerResult:
    reason: str
    world_ids: tuple[int, ...] = ()
    global_scope: bool = False


class _FakeContacts:
    """Allocated contact provider with semantic partition anchors."""

    def __init__(self) -> None:
        self.rigid_contact_count = np.asarray([2], dtype=np.int32)
        self.rigid_contact_shape0 = np.asarray([0, 2, 0], dtype=np.int32)
        self.rigid_contact_shape1 = np.asarray([1, 3, 0], dtype=np.int32)
        self.rigid_contact_force = np.asarray([11.0, 22.0, 0.0], dtype=np.float32)
        self.unrelated_capacity = np.asarray([71.0, 72.0, 73.0], dtype=np.float32)


class _FakeNarrowPhase:
    """Nested collision buffers unknown to the recorder implementation."""

    def __init__(self) -> None:
        self.shape_aabb_lower = np.arange(12, dtype=np.float32).reshape(4, 3)
        self.shape_aabb_upper = self.shape_aabb_lower + 0.5


class _FakeCollisionPipeline:
    """External pipeline with a top-level model alias and nested AABBs."""

    def __init__(self, model: _FakeModel) -> None:
        self.model = model
        self.broad_phase_pair_count = np.asarray([2], dtype=np.int32)
        self.broad_phase_shape_pairs = np.asarray([[0, 1], [2, 3], [0, 0]], dtype=np.int32)
        self.shape_pairs_filtered = np.asarray([[9, 8], [7, 6], [5, 4]], dtype=np.int32)
        self.narrow_phase = _FakeNarrowPhase()
        self.arbitrary_metric = np.asarray([3.5], dtype=np.float32)


class _CollisionPipelineWithForbiddenReconstruction(_FakeCollisionPipeline):
    """Pipeline that fails if incident capture re-runs collision detection."""

    def __init__(self, model: _FakeModel) -> None:
        super().__init__(model)
        self.contacts_calls = 0
        self.collide_calls = 0

    def contacts(self) -> _FakeContacts:
        """Reject hidden contact allocations by the recorder."""
        self.contacts_calls += 1
        raise AssertionError("incident capture must use the live post-step contacts")

    def collide(self, state: _FakeState, contacts: _FakeContacts) -> None:
        """Reject mutation of live collision-pipeline state by the recorder."""
        self.collide_calls += 1
        raise AssertionError("incident capture must not re-run collision detection")


class _FakeMjWarpSolver:
    """Small solver wrapper around real MuJoCo-Warp Data and Contact types."""

    def __init__(self, data: object) -> None:
        self.mjw_data = data


class _FakeOperationProvider:
    """Generic transition-operation provider with a mutable strict schema."""

    def __init__(self) -> None:
        self.bind_count = 0
        self.close_count = 0
        self.fail_bind = False
        self.fail_close = False
        self.hook_installed = False
        self.bound_solver = None
        self.payload = {"residual": np.asarray([0.25, 0.5], dtype=np.float32)}

    def bind(self, solver: object) -> None:
        """Bind this provider to one stable solver."""
        self.bind_count += 1
        self.bound_solver = solver
        self.hook_installed = True
        if self.fail_bind:
            raise RuntimeError("partial bind failure")

    def snapshot(self) -> dict[str, np.ndarray]:
        """Return the current transient operation values."""
        return self.payload

    def close(self) -> None:
        """Release this provider exactly once."""
        self.close_count += 1
        if self.fail_close:
            raise RuntimeError("close failure")
        self.hook_installed = False


def _warp_int_array(values: list[int]) -> wp.array:
    """Create a CPU Warp integer array."""
    return wp.array(np.asarray(values, dtype=np.int32), dtype=wp.int32, device="cpu")


def _collision_model() -> _FakeModel:
    """Create two worlds with two shapes per world."""
    model = _FakeModel(world_count=2, body_count=4, joint_count=4)
    model.shape_world = _warp_int_array([0, 0, 1, 1])
    model.shape_world_start = _warp_int_array([0, 2, 4])
    model.attribute_frequency["shape_world"] = "shape"
    return model


def _fill(array: wp.array, value: float) -> None:
    """Fill a one-dimensional Warp float array through an exact host copy."""
    wp.copy(
        array,
        wp.array(np.full(array.shape, value, dtype=np.float32), dtype=wp.float32, device="cpu"),
    )


def _state(model: _FakeModel, *, joint_value: float = 0.0, global_incident: bool = False) -> _FakeState:
    """Create a state with a recognizable joint value and optional global failure."""
    state = model.state()
    _fill(state.joint_q, joint_value)
    state.global_metric = np.float64(np.inf if global_incident else 0.0)
    return state


def _inject_body_inf(state: _FakeState, body_index: int) -> None:
    """Inject positive infinity into one body transform component."""
    values = state.body_q.numpy()
    values[body_index, 0] = np.inf
    wp.copy(state.body_q, wp.array(values, dtype=wp.transformf, device="cpu"))


def _inject_joint_velocity_inf(state: _FakeState, joint_index: int) -> None:
    """Inject positive infinity into one joint velocity."""
    values = state.joint_qd.numpy()
    values[joint_index] = np.inf
    wp.copy(state.joint_qd, wp.array(values, dtype=wp.float32, device="cpu"))


def _minimal_mjwarp_solver(
    *,
    distances: list[float] | None = None,
) -> _FakeMjWarpSolver:
    """Create a sparse real MuJoCo-Warp Data object with contact and Hessian fields."""
    from mujoco_warp import Contact, Data

    contact = object.__new__(Contact)
    contact.dist = wp.array(
        np.asarray([1.0, 2.0, 3.0, np.nan] if distances is None else distances, dtype=np.float32),
        dtype=wp.float32,
        device="cpu",
    )
    contact.worldid = wp.array(
        np.asarray([0, 1, 1, 99], dtype=np.int32),
        dtype=wp.int32,
        device="cpu",
    )

    data = object.__new__(Data)
    data.contact = contact
    data.nacon = wp.array(np.asarray([3], dtype=np.int32), dtype=wp.int32, device="cpu")
    data.naconmax = 4
    data.nworld = 2
    data.qLD = wp.array(
        np.arange(8, dtype=np.float32).reshape(2, 2, 2),
        dtype=wp.float32,
        device="cpu",
    )
    return _FakeMjWarpSolver(data)


def _capture_cfg(output_dir: Path, **overrides) -> NewtonDebugCaptureCfg:
    """Build a complete, lightweight recorder configuration for unit tests."""
    values = {
        "history_length": 3,
        "output_dir": str(output_dir),
        "failed_worlds_only": True,
        "max_incidents": 5,
        "halt_on_incident": False,
        "fail_on_capture_error": True,
        "max_gpu_bytes": 1024**2,
        "record_model": False,
        "record_control": False,
        "record_contacts": False,
        "record_solver": False,
        "record_operations": False,
    }
    values.update(overrides)
    return NewtonDebugCaptureCfg(**values)


def _archives(output_dir: Path) -> list[Path]:
    """Return incident artifacts in stable filename order."""
    return sorted(output_dir.glob("physics_incident_*.npz"))


def _recorder(
    model: _FakeModel,
    cfg: NewtonDebugCaptureCfg,
    *,
    state: _FakeState | None = None,
    control=None,
    solver=None,
    contacts=None,
    collision_pipeline=None,
    scene_exporter=None,
    context_provider=None,
    operation_provider=None,
    triggers=None,
) -> PhysicsIncidentRecorder:
    """Create a recorder with explicit stable providers."""
    return PhysicsIncidentRecorder(
        model,
        cfg,
        state=model.state() if state is None else state,
        control=control,
        solver=solver,
        contacts=contacts,
        collision_pipeline=collision_pipeline,
        scene_exporter=scene_exporter,
        context_provider=context_provider,
        operation_provider=operation_provider,
        triggers=triggers,
    )


def _dispatch(
    recorder: PhysicsIncidentRecorder,
    model: _FakeModel,
    post_state: _FakeState,
    *,
    sim_time: float,
    trigger_results: dict[str, object] | None = None,
) -> None:
    """Run one recorder dispatch with an explicit finite pre-state."""
    recorder.capture_pre(model.state())
    recorder.step(post_state, sim_time=sim_time, trigger_results=trigger_results)


def test_cold_incident_records_live_applied_control_without_replay(tmp_path: Path):
    """A non-replay artifact preserves the control applied after pre-dispatch capture."""
    model = _FakeModel()
    control = SimpleNamespace(joint_f=np.asarray([0.0], dtype=np.float32))
    recorder = _recorder(
        model,
        _capture_cfg(tmp_path, record_control=True),
        control=control,
    )
    recorder.capture_pre(model.state())
    control.joint_f[:] = 7.0

    recorder.step(_state(model, global_incident=True), sim_time=0.1)

    [archive] = _archives(tmp_path)
    arrays, _ = load_archive(archive)
    np.testing.assert_array_equal(arrays["incident__control__joint_f"], [7.0])
    assert not any(key.startswith("replay__") for key in arrays)


def test_nonfinite_applied_control_triggers_a_cold_incident(tmp_path: Path):
    """Control participates in strict non-finite detection when explicitly selected."""
    model = _FakeModel()
    control = SimpleNamespace(joint_f=np.asarray([0.0], dtype=np.float32))
    recorder = _recorder(
        model,
        _capture_cfg(
            tmp_path,
            record_control=True,
            detect_nonfinite_in=("control",),
        ),
        control=control,
    )
    recorder.capture_pre(model.state())
    control.joint_f[:] = np.nan

    recorder.step(model.state(), sim_time=0.1)

    [archive] = _archives(tmp_path)
    arrays, manifest = load_archive(archive)
    assert arrays["incident__nonfinite_paths"].tolist() == ["control.joint_f"]
    assert np.isnan(arrays["incident__control__joint_f"]).all()
    assert manifest["metadata"]["incident"]["global"] is True


def test_inf_is_detected_and_exported_with_its_exact_path(tmp_path: Path):
    """Positive infinity is an incident, not only NaN."""
    model = _FakeModel()
    recorder = _recorder(model, _capture_cfg(tmp_path))
    state = model.state()
    _inject_joint_velocity_inf(state, 2)

    _dispatch(recorder, model, state, sim_time=0.125)

    [archive] = _archives(tmp_path)
    arrays, manifest = load_archive(archive)
    assert np.isposinf(arrays["history__state__joint_qd"]).any()
    assert arrays["incident__nonfinite_paths"].tolist() == ["state.joint_qd"]
    assert manifest["status"] == "complete"
    assert manifest["metadata"]["incident"]["global"] is True
    assert manifest["metadata"]["incident"]["world_id"] is None


@pytest.mark.parametrize(
    ("history_length", "values", "expected_values", "expected_times"),
    [
        (5, [10.0, 20.0], [10.0, 20.0], [0.0, 0.1]),
        (3, [1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0], [0.1, 0.2, 0.3]),
    ],
)
def test_history_exports_only_valid_frames_in_chronological_order(
    tmp_path: Path,
    history_length: int,
    values: list[float],
    expected_values: list[float],
    expected_times: list[float],
):
    """Both partially filled and wrapped histories exclude stale slots."""
    model = _FakeModel()
    recorder = _recorder(model, _capture_cfg(tmp_path, history_length=history_length))
    assert recorder.history_length == history_length
    for index, value in enumerate(values):
        _dispatch(
            recorder,
            model,
            _state(model, joint_value=value, global_incident=index == len(values) - 1),
            sim_time=index * 0.1,
        )

    [archive] = _archives(tmp_path)
    arrays, _ = load_archive(archive)
    assert arrays["history_length"].item() == history_length
    assert arrays["history_valid_count"].item() == len(expected_values)
    np.testing.assert_allclose(arrays["history_sim_time"], expected_times)
    np.testing.assert_allclose(arrays["history__state__joint_q"][:, 0], expected_values)


def test_first_incident_contains_the_real_pre_step_state(tmp_path: Path):
    """The first history entry never substitutes an uninitialized pre-state."""
    model = _FakeModel()
    recorder = _recorder(model, _capture_cfg(tmp_path))
    recorder.capture_pre(_state(model, joint_value=42.0))

    recorder.step(_state(model, joint_value=99.0, global_incident=True), sim_time=0.1)

    [archive] = _archives(tmp_path)
    arrays, _ = load_archive(archive)
    np.testing.assert_allclose(arrays["pre__state__joint_q"], 42.0)
    np.testing.assert_allclose(arrays["history__state__joint_q"][-1], 99.0)


def test_substep_observation_requires_explicit_configuration(tmp_path: Path):
    """A disabled per-substep path does not allocate or accept substep capture."""
    model = _FakeModel()
    recorder = _recorder(model, _capture_cfg(tmp_path, capture_per_substep=False))
    recorder.capture_pre(model.state())

    with pytest.raises(DebugCaptureError, match="capture_per_substep=True"):
        recorder.observe_substep(model.state(), sim_time=0.0, substep_idx=0)


def test_first_substep_requires_a_real_pre_dispatch_state(tmp_path: Path):
    """Per-substep capture fails loudly when capture_pre was not called."""
    model = _FakeModel()
    recorder = _recorder(model, _capture_cfg(tmp_path, capture_per_substep=True))

    with pytest.raises(DebugCaptureError, match=r"capture_pre\(\).*current physics dispatch"):
        recorder.observe_substep(model.state(), sim_time=0.0, substep_idx=0)


def test_step_requires_and_consumes_a_fresh_pre_state(tmp_path: Path):
    """Each dispatch rejects a missing or stale pre-state snapshot."""
    model = _FakeModel()
    recorder = _recorder(model, _capture_cfg(tmp_path))

    with pytest.raises(DebugCaptureError, match=r"step\(\).*capture_pre"):
        recorder.step(model.state(), sim_time=0.0)

    recorder.capture_pre(model.state())
    recorder.step(model.state(), sim_time=0.0)

    with pytest.raises(DebugCaptureError, match=r"step\(\).*capture_pre"):
        recorder.step(model.state(), sim_time=0.1)


def test_capture_pre_rejects_an_unconsumed_dispatch(tmp_path: Path):
    """A second pre-state cannot silently replace the first one."""
    model = _FakeModel()
    recorder = _recorder(model, _capture_cfg(tmp_path))
    recorder.capture_pre(model.state())

    with pytest.raises(DebugCaptureError, match="previous dispatch"):
        recorder.capture_pre(model.state())


def test_one_event_writes_one_artifact_per_failed_world_and_global_scope(tmp_path: Path):
    """Simultaneous failures remain independently inspectable without duplicate events."""
    model = _FakeModel(world_count=2)
    recorder = _recorder(model, _capture_cfg(tmp_path, max_incidents=1))
    state = model.state()
    _inject_body_inf(state, 0)
    _inject_body_inf(state, 2)
    state.global_metric = np.float64(np.inf)

    _dispatch(recorder, model, state, sim_time=0.5)

    archives = _archives(tmp_path)
    assert len(archives) == 3
    incidents = []
    for archive in archives:
        arrays, manifest = load_archive(archive)
        incident = manifest["metadata"]["incident"]
        incidents.append((incident["world_id"], incident["global"]))
        if incident["world_id"] is not None:
            np.testing.assert_array_equal(arrays["failed_world_ids"], [incident["world_id"]])
            assert arrays["incident__nonfinite_paths"].tolist() == ["state.body_q"]
        else:
            assert arrays["incident__nonfinite_paths"].tolist() == ["state.global_metric"]
    assert sorted(incidents, key=lambda item: (-1 if item[0] is None else item[0])) == [
        (None, True),
        (0, False),
        (1, False),
    ]


def test_halt_policy_is_independent_of_incident_retention(tmp_path: Path):
    """Halting is immediate when enabled and never implied by max_incidents."""
    halt_dir = tmp_path / "halt"
    model = _FakeModel()
    halting = _recorder(model, _capture_cfg(halt_dir, max_incidents=5, halt_on_incident=True))
    _dispatch(halting, model, _state(model, global_incident=True), sim_time=0.0)
    assert halting.halted is True
    assert len(_archives(halt_dir)) == 1

    continue_dir = tmp_path / "continue"
    continuing = _recorder(model, _capture_cfg(continue_dir, max_incidents=1, halt_on_incident=False))
    _dispatch(continuing, model, _state(model, global_incident=True), sim_time=0.0)
    _dispatch(continuing, model, _state(model), sim_time=0.1)
    _dispatch(continuing, model, _state(model, global_incident=True), sim_time=0.2)
    assert continuing.halted is False
    assert len(_archives(continue_dir)) == 1


def test_repeated_rearmed_incidents_are_recorded_when_halt_is_disabled(tmp_path: Path):
    """A clean observation re-arms a scope for a later independent incident."""
    model = _FakeModel()
    recorder = _recorder(model, _capture_cfg(tmp_path, max_incidents=2, halt_on_incident=False))
    _dispatch(recorder, model, _state(model, global_incident=True), sim_time=0.0)
    _dispatch(recorder, model, _state(model), sim_time=0.1)
    _dispatch(recorder, model, _state(model, global_incident=True), sim_time=0.2)

    archives = _archives(tmp_path)
    assert len(archives) == 2
    incident_times = sorted(load_archive(path)[1]["metadata"]["incident"]["sim_time_seconds"] for path in archives)
    assert incident_times == [0.0, 0.2]
    assert recorder.halted is False


@pytest.mark.parametrize(
    ("cfg_overrides", "missing_provider"),
    [
        ({"record_solver": True}, "solver"),
        ({"record_control": True}, "control"),
        ({"record_contacts": True}, "contacts"),
        ({"record_operations": True}, "operation_provider"),
        (
            {
                "replay": NewtonDebugReplayCfg(
                    enabled=True,
                    record_state=True,
                    record_control=True,
                    record_solver=False,
                    record_contacts=False,
                )
            },
            "control",
        ),
        (
            {
                "replay": NewtonDebugReplayCfg(
                    enabled=True,
                    record_state=False,
                    record_control=False,
                    record_solver=False,
                    record_contacts=False,
                    record_operations=True,
                )
            },
            "operation_provider",
        ),
    ],
)
def test_missing_required_provider_fails_during_recorder_initialization(
    tmp_path: Path,
    cfg_overrides: dict,
    missing_provider: str,
):
    """A required provider fails at load time instead of waiting for an incident."""
    model = _FakeModel()

    with pytest.raises(DebugCaptureError, match=missing_provider):
        _recorder(model, _capture_cfg(tmp_path, **cfg_overrides))

    assert _archives(tmp_path) == []


def test_missing_required_state_fails_during_recorder_initialization(tmp_path: Path):
    """The state schema is a required load-time dependency."""
    model = _FakeModel()

    with pytest.raises(DebugCaptureError, match="state"):
        PhysicsIncidentRecorder(
            model,
            _capture_cfg(tmp_path),
            state=None,
            control=None,
            solver=None,
        )


@pytest.mark.parametrize(
    "world_count",
    [True, 0, -1, 1.5, "2"],
    ids=["boolean", "zero", "negative", "float", "string"],
)
def test_world_count_requires_a_positive_non_boolean_integer(
    tmp_path: Path,
    world_count,
):
    """World count cannot be coerced, clamped, or silently accepted."""
    model = _FakeModel()
    model.world_count = world_count

    with pytest.raises(DebugCaptureError, match=r"world_count.*non-boolean integer.*1"):
        _recorder(model, _capture_cfg(tmp_path))


@pytest.mark.parametrize(
    "frequency_map",
    [
        {1: "body", "1": "joint_coord"},
        {"": "body"},
    ],
    ids=["numeric-collision", "empty"],
)
def test_attribute_frequency_keys_are_not_coerced(
    tmp_path: Path,
    frequency_map: dict,
):
    """Invalid keys cannot normalize into valid or colliding field names."""
    model = _FakeModel()
    model.attribute_frequency = frequency_map

    with pytest.raises(DebugCaptureError, match=r"attribute_frequency keys.*non-empty strings"):
        _recorder(model, _capture_cfg(tmp_path))


def test_duplicate_world_start_frequency_fails_during_initialization(tmp_path: Path):
    """Case-normalized duplicate ownership anchors cannot overwrite each other."""
    model = _FakeModel(world_count=2)
    model.BODY_world_start = _warp_int_array([0, 2, 4])

    with pytest.raises(DebugCaptureError, match=r"frequency 'body'.*duplicate start anchors"):
        _recorder(model, _capture_cfg(tmp_path))


def test_explicit_frequency_requires_a_matching_world_start_anchor(tmp_path: Path):
    """Declared entity ownership cannot downgrade to an ambiguous global field."""
    model = _FakeModel(world_count=2)
    del model.body_world_start

    with pytest.raises(DebugCaptureError, match=r"attribute_frequency.*body_q.*without.*world_start"):
        _recorder(model, _capture_cfg(tmp_path))


def test_explicit_frequency_extent_mismatch_fails_during_initialization(tmp_path: Path):
    """Declared entity ownership must exactly match its world-start extent."""
    model = _FakeModel(world_count=2)

    with pytest.raises(DebugCaptureError, match=r"state.body_q.*extent 3.*expected 4"):
        _recorder(
            model,
            _capture_cfg(tmp_path),
            state=_FakeState(body_count=3, joint_count=4),
        )


def test_symbolic_nworld_extent_mismatch_fails_during_initialization(tmp_path: Path):
    """A symbolic nworld dimension must exactly equal the finalized world count."""
    from mujoco_warp import Data

    model = _FakeModel(world_count=2)
    data = object.__new__(Data)
    data.solver_niter = wp.zeros(3, dtype=wp.int32, device="cpu")

    with pytest.raises(DebugCaptureError, match=r"nworld.*solver_niter.*extent 3.*expected 2"):
        _recorder(
            model,
            _capture_cfg(tmp_path, record_solver=True),
            solver=_FakeMjWarpSolver(data),
        )


@pytest.mark.parametrize("invalid_world", [-2, 2])
def test_shape_world_owner_range_is_validated_before_partitioning(
    tmp_path: Path,
    invalid_world: int,
):
    """Pair rows cannot disappear through an invalid shape-to-world owner."""
    model = _collision_model()
    owners = np.asarray([0, 0, 1, 1], dtype=np.int32)
    owners[2] = invalid_world
    model.shape_world = _warp_int_array(owners.tolist())

    with pytest.raises(DebugCaptureError, match=r"shape_world.*outside.*\[-1, 2\)"):
        _recorder(
            model,
            _capture_cfg(tmp_path, record_contacts=True),
            contacts=_FakeContacts(),
        )


def test_archive_key_collision_fails_during_initialization(tmp_path: Path):
    """Sanitized field names can never silently overwrite an archive array."""
    model = _FakeModel()

    def context_provider() -> dict[str, np.ndarray]:
        return {
            "a-b": np.zeros(1, dtype=np.float32),
            "a_b": np.ones(1, dtype=np.float32),
        }

    with pytest.raises(DebugCaptureError, match=r"Archive key collision.*a-b.*a_b"):
        _recorder(model, _capture_cfg(tmp_path), context_provider=context_provider)


def test_empty_required_state_schema_fails_during_initialization(tmp_path: Path):
    """A non-None state must still contribute a complete capturable schema."""
    model = _FakeModel()

    with pytest.raises(DebugCaptureError, match=r"state.*no capturable fields"):
        PhysicsIncidentRecorder(
            model,
            _capture_cfg(tmp_path),
            state=_EmptyState(),
            control=None,
            solver=None,
        )


def test_empty_required_provider_schema_fails_during_initialization(tmp_path: Path):
    """A present provider with no data is not treated as successfully bound."""
    model = _FakeModel()

    with pytest.raises(DebugCaptureError, match=r"solver.*no capturable fields|no capturable.*solver"):
        _recorder(
            model,
            _capture_cfg(tmp_path, record_solver=True),
            solver=_EmptySolver(),
        )


def test_unknown_solver_fields_are_discovered_without_a_hardcoded_field_table(tmp_path: Path):
    """A newly introduced solver array is captured through schema discovery."""
    model = _FakeModel()
    solver = _FakeSolver()
    recorder = _recorder(
        model,
        _capture_cfg(tmp_path, record_solver=True),
        solver=solver,
    )

    _dispatch(recorder, model, _state(model, global_incident=True), sim_time=0.0)

    [archive] = _archives(tmp_path)
    arrays, manifest = load_archive(archive)
    np.testing.assert_array_equal(arrays["incident__solver__adaptive_vector"], solver.adaptive_vector)
    assert "incident.solver.adaptive_vector" in manifest["metadata"]["providers"]["selected_fields"]


def test_coupled_solver_mapping_is_discovered_for_incident_and_replay(tmp_path: Path):
    """Every coupled solver slot contributes strict incident and replay fields."""

    class _CoupledSlot:
        def __init__(self, values: tuple[float, float]):
            self.adaptive_vector = np.asarray(values, dtype=np.float32)
            self.iterations = np.int32(3)

    model = _FakeModel()
    rigid = _CoupledSlot((10.0, 11.0))
    soft = _CoupledSlot((20.0, 21.0))
    solvers = {"rigid": rigid, "soft": soft}
    replay = NewtonDebugReplayCfg(
        enabled=True,
        record_state=False,
        record_control=False,
        record_solver=True,
        record_contacts=False,
    )
    recorder = _recorder(
        model,
        _capture_cfg(tmp_path, record_solver=True, replay=replay),
        solver=solvers,
    )

    recorder.record_step_replay_pre(model.state(), sim_time=0.0)
    soft.adaptive_vector[:] = (30.0, 31.0)
    recorder.record_step_replay_post(model.state())
    _dispatch(recorder, model, _state(model, global_incident=True), sim_time=0.1)

    [archive] = _archives(tmp_path)
    arrays, manifest = load_archive(archive)
    np.testing.assert_array_equal(
        arrays["incident__solver__key_rigid__adaptive_vector"],
        [10.0, 11.0],
    )
    np.testing.assert_array_equal(
        arrays["incident__solver__key_soft__adaptive_vector"],
        [30.0, 31.0],
    )
    np.testing.assert_array_equal(
        arrays["replay__pre__solver__key_rigid__adaptive_vector"],
        [[10.0, 11.0]],
    )
    np.testing.assert_array_equal(
        arrays["replay__post__solver__key_rigid__adaptive_vector"],
        [[10.0, 11.0]],
    )
    np.testing.assert_array_equal(
        arrays["replay__pre__solver__key_soft__adaptive_vector"],
        [[20.0, 21.0]],
    )
    np.testing.assert_array_equal(
        arrays["replay__post__solver__key_soft__adaptive_vector"],
        [[30.0, 31.0]],
    )
    selected = manifest["metadata"]["providers"]["selected_fields"]
    assert "incident.solver['rigid'].adaptive_vector" in selected
    assert "incident.solver['soft'].adaptive_vector" in selected


def test_coupled_solver_mapping_schema_drift_fails_loudly(tmp_path: Path):
    """A new field in one coupled solver slot invalidates the frozen schema."""

    class _DriftSlot:
        def __init__(self):
            self.adaptive_vector = np.asarray([1.0, 2.0], dtype=np.float32)

    model = _FakeModel()
    solvers = {"rigid": _DriftSlot(), "soft": _DriftSlot()}
    recorder = _recorder(
        model,
        _capture_cfg(tmp_path, record_solver=True),
        solver=solvers,
    )
    solvers["soft"].new_workspace = np.asarray([9.0], dtype=np.float32)

    with pytest.raises(DebugCaptureError, match=r"solver\['soft'\]\.new_workspace"):
        _dispatch(recorder, model, model.state(), sim_time=0.0)


def test_initialization_warning_reports_complete_unrecorded_inventory(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    """Every known ignored and unallocated resource is disclosed without truncation."""
    model = _FakeModel()
    solver = SimpleNamespace(
        residual=np.asarray([0.0], dtype=np.float32),
        unavailable=None,
        callback=lambda: None,
    )

    with caplog.at_level(logging.WARNING, logger="isaaclab_newton.physics._incident_recorder"):
        recorder = _recorder(
            model,
            _capture_cfg(tmp_path, record_solver=True),
            solver=solver,
        )

    inventory = []
    for plan_name, plan in (
        ("model", recorder._model_plan),
        ("state", recorder._state_plan),
        ("incident", recorder._incident_plan),
    ):
        for category, entries in (("unallocated", plan.unallocated), ("ignored", plan.ignored)):
            inventory.extend(f"plan={plan_name}; category={category}; path={entry.display_path};" for entry in entries)
    assert inventory
    assert "Complete capture inventory follows:" in caplog.text
    for expected in inventory:
        assert expected in caplog.text


def test_bound_provider_schema_drift_fails_instead_of_dropping_the_field(tmp_path: Path):
    """Replacing a bound solver array cannot silently change an artifact schema."""
    model = _FakeModel()
    solver = _FakeSolver()
    recorder = _recorder(
        model,
        _capture_cfg(tmp_path, record_solver=True),
        solver=solver,
    )
    solver.adaptive_vector = np.zeros(3, dtype=np.float32)

    with pytest.raises(DebugCaptureError, match=r"adaptive_vector.*shape"):
        _dispatch(recorder, model, _state(model, global_incident=True), sim_time=0.0)


def test_replay_operation_recording_requires_enabled_replay_without_binding(tmp_path: Path):
    """Invalid disabled replay fails before binding its operation provider."""
    model = _FakeModel()
    operations = _FakeOperationProvider()
    cfg = _capture_cfg(
        tmp_path,
        replay=NewtonDebugReplayCfg(enabled=False, record_operations=True),
    )

    with pytest.raises(ValueError, match=r"record_operations=True.*replay.enabled=True"):
        _recorder(
            model,
            cfg,
            solver=_FakeSolver(),
            operation_provider=operations,
        )
    assert operations.bind_count == 0


def test_partial_operation_bind_failure_closes_exactly_once(tmp_path: Path):
    """A provider that installs a hook before raising is closed immediately."""
    model = _FakeModel()
    operations = _FakeOperationProvider()
    operations.fail_bind = True

    with pytest.raises(DebugCaptureError, match=r"bind\(solver\).*partial bind failure"):
        _recorder(
            model,
            _capture_cfg(tmp_path, record_operations=True),
            solver=_FakeSolver(),
            operation_provider=operations,
        )

    assert operations.bind_count == 1
    assert operations.close_count == 1
    assert operations.hook_installed is False


def test_partial_operation_bind_and_cleanup_failures_remain_retryable(tmp_path: Path):
    """Both failures are retained and failed cleanup ownership can be retried."""
    model = _FakeModel()
    operations = _FakeOperationProvider()
    operations.fail_bind = True
    operations.fail_close = True
    recorder = PhysicsIncidentRecorder.__new__(PhysicsIncidentRecorder)

    with pytest.raises(DebugCaptureError) as exc_info:
        recorder.__init__(
            model,
            _capture_cfg(tmp_path, record_operations=True),
            state=model.state(),
            control=None,
            solver=_FakeSolver(),
            operation_provider=operations,
        )

    error = exc_info.value
    assert "partial bind failure" in str(error.operation_error)
    assert "close failure" in str(error.cleanup_error)
    assert "Provider ownership is retained" in str(error)
    assert operations.bind_count == 1
    assert operations.close_count == 1
    assert operations.hook_installed is True
    assert recorder._operation_bound is True

    operations.fail_close = False
    recorder._close_operation_provider()
    assert operations.close_count == 2
    assert operations.hook_installed is False
    assert recorder._operation_bound is False
    recorder._close_operation_provider()
    assert operations.close_count == 2


def test_replay_operation_provider_binds_once_and_archives_generic_transition_data(tmp_path: Path):
    """Replay-only operation snapshots omit the cold incident provider."""
    model = _FakeModel()
    solver = _FakeSolver()
    operations = _FakeOperationProvider()
    cfg = _capture_cfg(
        tmp_path,
        replay=NewtonDebugReplayCfg(
            enabled=True,
            record_state=False,
            record_control=False,
            record_solver=False,
            record_contacts=False,
            record_operations=True,
        ),
    )
    recorder = _recorder(
        model,
        cfg,
        solver=solver,
        operation_provider=operations,
    )

    assert operations.bind_count == 1
    assert operations.bound_solver is solver
    recorder.record_step_replay_pre(model.state(), sim_time=0.0)
    recorder.record_step_replay_post(model.state())
    _dispatch(recorder, model, _state(model, global_incident=True), sim_time=0.1)

    [archive] = _archives(tmp_path)
    arrays, _ = load_archive(archive)
    np.testing.assert_allclose(
        arrays["replay__pre__operations__key_residual"],
        [[0.25, 0.5]],
    )
    assert not any(key.startswith("incident__operations__") for key in arrays)
    recorder.clear()
    recorder.clear()
    assert operations.close_count == 1


def test_incident_operation_capture_archives_without_replay(tmp_path: Path):
    """Cold incidents retain latest operations without allocating replay history."""
    model = _FakeModel()
    solver = _FakeSolver()
    operations = _FakeOperationProvider()
    recorder = _recorder(
        model,
        _capture_cfg(tmp_path, record_operations=True),
        solver=solver,
        operation_provider=operations,
    )
    operations.payload["residual"][:] = [3.0, 4.0]

    _dispatch(recorder, model, _state(model, global_incident=True), sim_time=0.1)

    [archive] = _archives(tmp_path)
    arrays, _ = load_archive(archive)
    np.testing.assert_array_equal(arrays["incident__operations__key_residual"], [3.0, 4.0])
    assert not any(key.startswith("replay__") for key in arrays)
    assert operations.bind_count == 1
    assert operations.bound_solver is solver


@pytest.mark.parametrize("drift", ["shape", "key"])
def test_operation_snapshot_schema_drift_fails_loudly(tmp_path: Path, drift: str):
    """Operation providers cannot change fields after initialization."""
    model = _FakeModel()
    operations = _FakeOperationProvider()
    cfg = _capture_cfg(
        tmp_path,
        replay=NewtonDebugReplayCfg(
            enabled=True,
            record_state=False,
            record_control=False,
            record_solver=False,
            record_contacts=False,
            record_operations=True,
        ),
    )
    recorder = _recorder(
        model,
        cfg,
        solver=_FakeSolver(),
        operation_provider=operations,
    )
    if drift == "shape":
        operations.payload["residual"] = np.zeros(3, dtype=np.float32)
    else:
        operations.payload["new_operation"] = np.zeros(1, dtype=np.float32)

    with pytest.raises(DebugCaptureError, match=r"schema|residual|new_operation"):
        recorder.record_step_replay_pre(model.state())


def test_ambiguous_workflow_context_is_archived_without_world_slicing(tmp_path: Path):
    """Context extents matching world count remain global without ownership metadata."""
    model = _FakeModel(world_count=2)
    per_world = np.asarray([10, 20], dtype=np.int64)

    def context_provider() -> dict[str, object]:
        return {"per_world": per_world, "workflow_step": np.int64(7)}

    cfg = _capture_cfg(
        tmp_path,
        replay=NewtonDebugReplayCfg(
            enabled=True,
            record_state=True,
            record_control=False,
            record_solver=False,
            record_contacts=False,
        ),
    )
    recorder = _recorder(model, cfg, context_provider=context_provider)
    recorder.record_step_replay_pre(model.state(), sim_time=0.0)
    recorder.record_step_replay_post(model.state())

    state = model.state()
    _inject_body_inf(state, 0)
    _dispatch(recorder, model, state, sim_time=0.1)

    [archive] = _archives(tmp_path)
    arrays, _ = load_archive(archive)
    np.testing.assert_array_equal(arrays["incident__context__key_per_world"], [10, 20])
    np.testing.assert_array_equal(arrays["incident__context__key_workflow_step"], 7)
    np.testing.assert_array_equal(arrays["replay__pre__context__key_per_world"], [[10, 20]])
    np.testing.assert_array_equal(arrays["replay__post__context__key_per_world"], [[10, 20]])


def test_empty_workflow_context_freezes_an_empty_schema(tmp_path: Path):
    """A universal context provider may explicitly freeze no workflow fields."""
    model = _FakeModel()
    recorder = _recorder(model, _capture_cfg(tmp_path), context_provider=lambda: {})

    _dispatch(recorder, model, model.state(), sim_time=0.0)

    assert _archives(tmp_path) == []


def test_empty_workflow_context_rejects_late_keys(tmp_path: Path):
    """Keys cannot appear after an empty context schema was frozen."""
    model = _FakeModel()
    context: dict[str, object] = {}
    recorder = _recorder(model, _capture_cfg(tmp_path), context_provider=lambda: dict(context))
    context["late_key"] = np.int64(1)

    with pytest.raises(DebugCaptureError, match=r"schema|late_key"):
        _dispatch(recorder, model, model.state(), sim_time=0.0)


def test_workflow_context_provider_failure_is_not_swallowed(tmp_path: Path):
    """A workflow provider exception stops capture with its original detail."""
    model = _FakeModel()
    calls = 0

    def context_provider() -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("workflow context unavailable")
        return {"workflow_step": np.int64(0)}

    recorder = _recorder(model, _capture_cfg(tmp_path), context_provider=context_provider)

    with pytest.raises(DebugCaptureError, match="context_provider.*workflow context unavailable"):
        _dispatch(recorder, model, model.state(), sim_time=0.0)


@pytest.mark.parametrize("fail_on_capture_error", [False, True])
def test_runtime_provider_failure_writes_partial_artifact_and_optionally_raises(
    tmp_path: Path,
    fail_on_capture_error: bool,
):
    """A cold-path failure is visible in both status and error policy."""
    model = _FakeModel()

    def broken_scene_exporter(path: str, world_ids: list[int]) -> None:
        del path, world_ids
        raise RuntimeError("scene export failed")

    recorder = _recorder(
        model,
        _capture_cfg(tmp_path, fail_on_capture_error=fail_on_capture_error),
        scene_exporter=broken_scene_exporter,
    )
    if fail_on_capture_error:
        with pytest.raises(DebugCaptureError, match="saved as partial.*scene"):
            _dispatch(recorder, model, _state(model, global_incident=True), sim_time=0.0)
    else:
        _dispatch(recorder, model, _state(model, global_incident=True), sim_time=0.0)

    [archive] = _archives(tmp_path)
    arrays, manifest = load_archive(archive, allowed_statuses={"partial"})
    assert arrays["history_valid_count"].item() == 1
    assert manifest["status"] == "partial"
    assert manifest["metadata"]["incomplete"] is True
    assert manifest["metadata"]["errors"] == [
        {
            "provider": "scene",
            "type": "builtins.RuntimeError",
            "message": "scene export failed",
        }
    ]


def test_state_history_memory_budget_fails_before_allocation(tmp_path: Path):
    """An undersized total budget reports every state field instead of truncating."""
    model = _FakeModel()
    cfg = _capture_cfg(tmp_path, history_length=100, max_gpu_bytes=1)

    with pytest.raises(DebugCaptureError) as exc_info:
        _recorder(model, cfg)

    message = str(exc_info.value)
    assert "state history" in message.lower()
    assert "max_gpu_bytes=1" in message
    assert "state.body_q" in message
    assert "state.joint_qd" in message


def test_replay_memory_budget_uses_only_the_remaining_total_budget_at_initialization(tmp_path: Path):
    """Replay binding preflights pre/post slots before allocating them."""
    model = _FakeModel()
    cfg = _capture_cfg(
        tmp_path,
        history_length=3,
        max_gpu_bytes=1500,
        replay=NewtonDebugReplayCfg(
            enabled=True,
            record_state=True,
            record_control=False,
            record_solver=False,
            record_contacts=False,
        ),
    )
    with pytest.raises(DebugCaptureError) as exc_info:
        _recorder(model, cfg)

    message = str(exc_info.value)
    assert "replay" in message.lower()
    assert "remaining" in message.lower()
    assert "max_gpu_bytes=1500" in message
    assert "replay.state.body_q" in message


@pytest.mark.parametrize(
    "debug_capture",
    [
        NewtonDebugCaptureCfg(capture_per_substep=True),
        NewtonDebugCaptureCfg(replay=NewtonDebugReplayCfg(enabled=True)),
        NewtonDebugCaptureCfg(record_operations=True),
    ],
)
def test_cuda_graph_rejects_host_managed_debug_capture(debug_capture: NewtonDebugCaptureCfg):
    """CUDA graph capture cannot hide incompatible host-managed debug paths."""
    with pytest.raises(ValueError, match="CUDA.graph.*debug_capture"):
        NewtonCfg(use_cuda_graph=True, debug_capture=debug_capture)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("history_length", 0),
        ("max_incidents", 0),
        ("max_gpu_bytes", 0),
    ],
)
def test_debug_capture_rejects_nonpositive_limits(field: str, value: int):
    """Invalid limits fail instead of being silently clamped."""
    cfg = NewtonDebugCaptureCfg(**{field: value})

    with pytest.raises(ValueError, match=field):
        NewtonCfg(debug_capture=cfg)


def test_debug_capture_defaults_fail_loudly_and_halt_safely():
    """The safe default never continues after an incomplete or non-finite capture."""
    cfg = NewtonDebugCaptureCfg()

    assert cfg.halt_on_incident is True
    assert cfg.fail_on_capture_error is True
    assert cfg.include_private_fields is True


def test_debug_capture_rejects_non_boolean_private_field_selection():
    """Private-field discovery cannot be accidentally enabled by a truthy value."""
    cfg = NewtonDebugCaptureCfg(include_private_fields=1)

    with pytest.raises(TypeError, match="include_private_fields.*bool"):
        NewtonCfg(debug_capture=cfg)


def test_custom_trigger_exports_finite_world_and_global_scopes(tmp_path: Path):
    """Registered trigger results create incidents without requiring non-finite state."""
    model = _FakeModel(world_count=2)
    recorder = _recorder(
        model,
        _capture_cfg(tmp_path),
        triggers={"energy_spike": lambda _context: None},
    )

    _dispatch(
        recorder,
        model,
        model.state(),
        sim_time=0.25,
        trigger_results={
            "energy_spike": _TriggerResult(
                reason="energy exceeded the workflow threshold",
                world_ids=(1,),
                global_scope=True,
            )
        },
    )

    archives = _archives(tmp_path)
    assert len(archives) == 2
    scopes = set()
    for archive in archives:
        arrays, manifest = load_archive(archive)
        incident = manifest["metadata"]["incident"]
        scopes.add(incident["scope"])
        assert arrays["incident__nonfinite_paths"].tolist() == []
        assert arrays["incident__trigger_names"].tolist() == ["energy_spike"]
        assert incident["trigger_reasons"] == {"energy_spike": "energy exceeded the workflow threshold"}
        expected_joint_rows = 4 if incident["global"] else 2
        assert arrays["history__state__joint_q"].shape[1] == expected_joint_rows
    assert scopes == {"global", "world000001"}


def test_custom_trigger_suppresses_persistent_scope_until_rearmed(tmp_path: Path):
    """One active trigger scope emits once and an omitted result re-arms it."""
    model = _FakeModel(world_count=2)
    recorder = _recorder(
        model,
        _capture_cfg(tmp_path, max_incidents=3),
        triggers={"energy_spike": lambda _context: None},
    )
    result = {"energy_spike": _TriggerResult(reason="persistent", world_ids=(0,))}

    _dispatch(recorder, model, model.state(), sim_time=0.0, trigger_results=result)
    _dispatch(recorder, model, model.state(), sim_time=0.1, trigger_results=result)
    _dispatch(recorder, model, model.state(), sim_time=0.2, trigger_results={})
    _dispatch(recorder, model, model.state(), sim_time=0.3, trigger_results=result)

    assert len(_archives(tmp_path)) == 2
    times = sorted(load_archive(path)[1]["metadata"]["incident"]["sim_time_seconds"] for path in _archives(tmp_path))
    assert times == [0.0, 0.3]


@pytest.mark.parametrize("name", ["BadName", "bad-name", "9bad", ""])
def test_trigger_registry_rejects_non_snake_case_names(tmp_path: Path, name: str):
    """Trigger names are stable archive identifiers and fail before capture."""
    model = _FakeModel()

    with pytest.raises(ValueError, match="lower snake case"):
        _recorder(
            model,
            _capture_cfg(tmp_path),
            triggers={name: lambda _context: None},
        )


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (_TriggerResult(reason="", world_ids=(0,)), "reason"),
        (SimpleNamespace(reason="bad", world_ids=[0], global_scope=False), "tuple"),
        (_TriggerResult(reason="bad", world_ids=(0, 0)), "duplicates"),
        (_TriggerResult(reason="bad", world_ids=(2,)), "outside"),
        (SimpleNamespace(reason="bad", world_ids=(0,), global_scope=1), "bool"),
        (_TriggerResult(reason="bad"), "at least one"),
    ],
)
def test_trigger_result_validation_fails_loudly(
    tmp_path: Path,
    result: object,
    expected: str,
):
    """Malformed callback results cannot silently create the wrong scope."""
    model = _FakeModel(world_count=2)
    recorder = _recorder(
        model,
        _capture_cfg(tmp_path),
        triggers={"energy_spike": lambda _context: None},
    )

    with pytest.raises(DebugCaptureError, match=expected):
        _dispatch(
            recorder,
            model,
            model.state(),
            sim_time=0.0,
            trigger_results={"energy_spike": result},
        )


def test_trigger_and_nonfinite_detection_merge_into_one_world_artifact(tmp_path: Path):
    """Simultaneous trigger and non-finite evidence share one event artifact."""
    model = _FakeModel(world_count=2)
    recorder = _recorder(
        model,
        _capture_cfg(tmp_path),
        triggers={"energy_spike": lambda _context: None},
    )
    state = model.state()
    _inject_body_inf(state, 2)

    _dispatch(
        recorder,
        model,
        state,
        sim_time=0.5,
        trigger_results={"energy_spike": _TriggerResult(reason="energy spike", world_ids=(1,))},
    )

    [archive] = _archives(tmp_path)
    arrays, manifest = load_archive(archive)
    assert arrays["incident__nonfinite_paths"].tolist() == ["state.body_q"]
    assert arrays["incident__trigger_names"].tolist() == ["energy_spike"]
    assert manifest["metadata"]["incident"]["world_id"] == 1


def test_per_substep_trigger_uses_the_exact_substep_pre_state(tmp_path: Path):
    """Solver-post trigger capture records substep metadata without a NaN."""
    model = _FakeModel(world_count=2)
    recorder = _recorder(
        model,
        _capture_cfg(tmp_path, capture_per_substep=True),
        triggers={"constraint_jump": lambda _context: None},
    )
    recorder.capture_pre(_state(model, joint_value=42.0))
    recorder.observe_substep(
        _state(model, joint_value=99.0),
        sim_time=0.1,
        substep_idx=0,
        trigger_results={"constraint_jump": _TriggerResult(reason="constraint jump", world_ids=(0,))},
    )

    [archive] = _archives(tmp_path)
    arrays, _ = load_archive(archive)
    assert arrays["incident__failed_substep_idx"].item() == 0
    assert arrays["incident__last_finite_substep_idx"].item() == -1
    np.testing.assert_allclose(arrays["pre__state__joint_q"], 42.0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"record_collision_pipeline": True},
        {
            "replay": NewtonDebugReplayCfg(
                enabled=True,
                record_state=False,
                record_control=False,
                record_solver=False,
                record_contacts=False,
                record_collision_pipeline=True,
            )
        },
    ],
)
def test_collision_pipeline_is_a_strict_required_provider(
    tmp_path: Path,
    overrides: dict,
):
    """Incident and replay pipeline capture both fail at recorder load time."""
    model = _collision_model()

    with pytest.raises(DebugCaptureError, match="collision_pipeline"):
        _recorder(model, _capture_cfg(tmp_path, **overrides))


def test_collision_capture_preserves_live_post_pipeline_without_recollision(
    tmp_path: Path,
):
    """Incident serialization never calls collide or mutates live post-step evidence."""
    model = _collision_model()
    contacts = _FakeContacts()
    pipeline = _CollisionPipelineWithForbiddenReconstruction(model)
    post_pipeline = np.arange(12, dtype=np.float32).reshape(4, 3) + 100.0
    pipeline.narrow_phase.shape_aabb_lower[:] = post_pipeline
    recorder = _recorder(
        model,
        _capture_cfg(
            tmp_path,
            record_model=True,
            record_contacts=True,
            record_collision_pipeline=True,
        ),
        contacts=contacts,
        collision_pipeline=pipeline,
        triggers={"contact_jump": lambda _context: None},
    )
    recorder.capture_pre(_state(model, joint_value=42.0))
    recorder.step(
        model.state(),
        sim_time=0.1,
        trigger_results={"contact_jump": _TriggerResult(reason="contact changed", world_ids=(1,))},
    )

    [archive] = _archives(tmp_path)
    arrays, manifest = load_archive(archive)
    np.testing.assert_array_equal(
        arrays["incident__collision_pipeline__narrow_phase__shape_aabb_lower"],
        post_pipeline,
    )
    np.testing.assert_array_equal(arrays["incident__contacts__rigid_contact_force"], [22.0])
    np.testing.assert_array_equal(
        arrays["incident__contacts__unrelated_capacity"],
        [71.0, 72.0, 73.0],
    )
    np.testing.assert_array_equal(
        arrays["incident__collision_pipeline__shape_pairs_filtered"],
        pipeline.shape_pairs_filtered,
    )
    np.testing.assert_array_equal(pipeline.narrow_phase.shape_aabb_lower, post_pipeline)
    np.testing.assert_array_equal(contacts.rigid_contact_force, [11.0, 22.0, 0.0])
    assert pipeline.contacts_calls == 0
    assert pipeline.collide_calls == 0
    assert "incident__model__shape_world" in arrays
    assert not any(key.startswith("incident__collision_pipeline__model__") for key in arrays)
    contracts = {entry["contract"] for entry in manifest["metadata"]["providers"]["partition_contracts"]}
    assert contracts == {"newton_rigid_contacts", "newton_broad_phase_pairs"}


def test_collision_pipeline_replay_wraps_slices_and_rejects_schema_drift(tmp_path: Path):
    """Collision replay remains chronological, world-focused, and schema-frozen."""
    model = _collision_model()
    pipeline = _FakeCollisionPipeline(model)
    replay = NewtonDebugReplayCfg(
        enabled=True,
        record_state=False,
        record_control=False,
        record_solver=False,
        record_contacts=False,
        record_collision_pipeline=True,
    )
    recorder = _recorder(
        model,
        _capture_cfg(tmp_path, history_length=2, replay=replay),
        collision_pipeline=pipeline,
        triggers={"broad_phase_jump": lambda _context: None},
    )

    for index in range(3):
        pre = np.arange(12, dtype=np.float32).reshape(4, 3) + index * 100.0
        pipeline.narrow_phase.shape_aabb_lower[:] = pre
        recorder.record_step_replay_pre(model.state(), sim_time=float(index))
        pipeline.narrow_phase.shape_aabb_lower[:] = pre + 50.0
        recorder.record_step_replay_post(model.state())

    _dispatch(
        recorder,
        model,
        model.state(),
        sim_time=3.0,
        trigger_results={"broad_phase_jump": _TriggerResult(reason="pair churn", world_ids=(1,))},
    )

    [archive] = _archives(tmp_path)
    arrays, _ = load_archive(archive)
    np.testing.assert_array_equal(arrays["replay__sim_time"], [1.0, 2.0])
    np.testing.assert_array_equal(
        arrays["replay__pre__collision_pipeline__narrow_phase__shape_aabb_lower"][:, 0, 0],
        [100.0, 200.0],
    )
    np.testing.assert_array_equal(
        arrays["replay__post__collision_pipeline__narrow_phase__shape_aabb_lower"][:, 0, 0],
        [150.0, 250.0],
    )
    np.testing.assert_array_equal(
        arrays["replay__pre__collision_pipeline__broad_phase_pair_count"],
        [[1], [1]],
    )
    np.testing.assert_array_equal(
        arrays["replay__pre__collision_pipeline__broad_phase_shape_pairs"],
        [[2, 3], [2, 3]],
    )
    np.testing.assert_array_equal(
        arrays["replay__pre__collision_pipeline__broad_phase_shape_pairs__slot_offsets"],
        [0, 1, 2],
    )
    np.testing.assert_array_equal(
        arrays["replay__pre__collision_pipeline__shape_pairs_filtered"],
        np.stack([pipeline.shape_pairs_filtered, pipeline.shape_pairs_filtered]),
    )

    drift_dir = tmp_path / "drift"
    drift_pipeline = _FakeCollisionPipeline(model)
    drift_recorder = _recorder(
        model,
        _capture_cfg(drift_dir, replay=replay),
        collision_pipeline=drift_pipeline,
    )
    drift_pipeline.narrow_phase.shape_aabb_lower = np.zeros((5, 3), dtype=np.float32)
    with pytest.raises(DebugCaptureError, match=r"shape_aabb_lower.*shape"):
        drift_recorder.record_step_replay_pre(model.state())


def test_real_mjwarp_contact_rows_are_sliced_but_ambiguous_hessian_is_full(
    tmp_path: Path,
):
    """Semantic contacts localize while unannotated qLD remains complete."""
    model = _FakeModel(world_count=2)
    solver = _minimal_mjwarp_solver()
    replay = NewtonDebugReplayCfg(
        enabled=True,
        record_state=False,
        record_control=False,
        record_solver=True,
        record_contacts=False,
    )
    recorder = _recorder(
        model,
        _capture_cfg(tmp_path, record_solver=True, replay=replay),
        solver=solver,
        triggers={"solver_jump": lambda _context: None},
    )
    recorder.record_step_replay_pre(model.state(), sim_time=0.0)
    solver.mjw_data.contact.dist = wp.array(
        np.asarray([10.0, 20.0, 30.0, np.nan], dtype=np.float32),
        dtype=wp.float32,
        device="cpu",
    )
    recorder.record_step_replay_post(model.state())

    _dispatch(
        recorder,
        model,
        model.state(),
        sim_time=0.1,
        trigger_results={"solver_jump": _TriggerResult(reason="solver diagnostic", world_ids=(1,))},
    )

    [archive] = _archives(tmp_path)
    arrays, manifest = load_archive(archive)
    np.testing.assert_array_equal(
        arrays["incident__solver__mjw_data__contact__dist"],
        [20.0, 30.0],
    )
    np.testing.assert_array_equal(
        arrays["incident__solver__mjw_data__contact__worldid"],
        [1, 1],
    )
    np.testing.assert_array_equal(arrays["incident__solver__mjw_data__nacon"], [2])
    qld = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    np.testing.assert_array_equal(arrays["incident__solver__mjw_data__qLD"], qld)
    np.testing.assert_array_equal(arrays["replay__pre__solver__mjw_data__qLD"], [qld])
    np.testing.assert_array_equal(
        arrays["replay__pre__solver__mjw_data__contact__dist"],
        [2.0, 3.0],
    )
    np.testing.assert_array_equal(
        arrays["replay__post__solver__mjw_data__contact__dist"],
        [20.0, 30.0],
    )
    np.testing.assert_array_equal(
        arrays["replay__pre__solver__mjw_data__contact__dist__slot_offsets"],
        [0, 2],
    )
    np.testing.assert_array_equal(arrays["replay__pre__solver__mjw_data__nacon"], [[2]])
    contracts = manifest["metadata"]["providers"]["partition_contracts"]
    assert any(
        entry["contract"] == "symbolic_world_rows:naconmax" and entry["count_path"] == "incident.solver.mjw_data.nacon"
        for entry in contracts
    )


def test_real_mjwarp_nonfinite_detection_ignores_inactive_capacity_rows(
    tmp_path: Path,
):
    """Stale NaN capacity is ignored while an active contact non-finite is scoped."""
    model = _FakeModel(world_count=2)
    solver = _minimal_mjwarp_solver()
    recorder = _recorder(
        model,
        _capture_cfg(
            tmp_path,
            record_solver=True,
            detect_nonfinite_in=("solver",),
        ),
        solver=solver,
    )

    _dispatch(recorder, model, model.state(), sim_time=0.0)
    assert _archives(tmp_path) == []

    solver.mjw_data.contact.dist = wp.array(
        np.asarray([1.0, np.inf, 3.0, np.nan], dtype=np.float32),
        dtype=wp.float32,
        device="cpu",
    )
    _dispatch(recorder, model, model.state(), sim_time=0.1)

    [archive] = _archives(tmp_path)
    arrays, manifest = load_archive(archive)
    assert manifest["metadata"]["incident"]["world_id"] == 1
    assert arrays["incident__nonfinite_paths"].tolist() == ["solver.mjw_data.contact.dist"]


def test_incident_archive_is_frozen_before_live_reset_mutation(tmp_path: Path):
    """State, solver, contacts, and context survive immediate post-capture reset."""
    model = _collision_model()
    solver = _FakeSolver()
    contacts = _FakeContacts()
    per_world_context = np.asarray([7.0, 8.0], dtype=np.float32)
    recorder = _recorder(
        model,
        _capture_cfg(tmp_path, record_contacts=True, record_solver=True),
        solver=solver,
        contacts=contacts,
        context_provider=lambda: {"per_world": per_world_context},
    )
    recorder.capture_pre(_state(model, joint_value=42.0))
    post_state = model.state()
    _inject_body_inf(post_state, 0)
    recorder.step(post_state, sim_time=0.1)

    post_state.body_q.zero_()
    solver.adaptive_vector[:] = 0.0
    contacts.rigid_contact_force[:] = 0.0
    per_world_context[:] = 0.0

    [archive] = _archives(tmp_path)
    arrays, _ = load_archive(archive)
    np.testing.assert_array_equal(arrays["pre__state__joint_q"], [42.0, 42.0])
    assert np.isposinf(arrays["history__state__body_q"]).any()
    np.testing.assert_array_equal(arrays["incident__solver__adaptive_vector"], [1.25, 2.5])
    np.testing.assert_array_equal(arrays["incident__contacts__rigid_contact_force"], [11.0])
    np.testing.assert_array_equal(arrays["incident__context__key_per_world"], [7.0, 8.0])


@pytest.mark.parametrize(
    ("model", "failed_worlds_only"),
    [
        (_FakeModel(), True),
        (_FakeModel(world_count=2), False),
    ],
)
def test_full_or_single_world_capture_does_not_require_partition_anchors(
    tmp_path: Path,
    model: _FakeModel,
    failed_worlds_only: bool,
):
    """Semantic row anchors are required only for focused multi-world output."""
    contacts = SimpleNamespace(values=np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    recorder = _recorder(
        model,
        _capture_cfg(
            tmp_path,
            failed_worlds_only=failed_worlds_only,
            record_contacts=True,
        ),
        contacts=contacts,
        triggers={"global_probe": lambda _context: None},
    )

    _dispatch(
        recorder,
        model,
        model.state(),
        sim_time=0.0,
        trigger_results={"global_probe": _TriggerResult(reason="full capture", global_scope=True)},
    )

    [archive] = _archives(tmp_path)
    arrays, _ = load_archive(archive)
    np.testing.assert_array_equal(arrays["incident__contacts__values"], contacts.values)


def test_ambiguous_world_count_extent_detects_global_and_preserves_full_array(tmp_path: Path):
    """A runtime world-count-leading shape never implies world ownership."""
    model = _FakeModel(world_count=2)
    model.new_diagnostic = np.asarray([[1.0, 2.0], [3.0, np.inf]], dtype=np.float32)
    recorder = _recorder(
        model,
        _capture_cfg(
            tmp_path,
            record_model=True,
            detect_nonfinite_in=("model",),
        ),
    )

    _dispatch(recorder, model, model.state(), sim_time=0.0)

    [archive] = _archives(tmp_path)
    arrays, manifest = load_archive(archive)
    assert manifest["metadata"]["incident"]["global"] is True
    assert manifest["metadata"]["incident"]["world_id"] is None
    assert arrays["incident__nonfinite_paths"].tolist() == ["model.new_diagnostic"]
    np.testing.assert_array_equal(arrays["incident__model__new_diagnostic"], model.new_diagnostic)


def test_solver_name_collision_does_not_inherit_model_attribute_frequency(tmp_path: Path):
    """A solver field named like Newton state remains ambiguous and global."""
    model = _FakeModel(world_count=2)
    solver = SimpleNamespace(
        body_q=np.asarray([1.0, 2.0, np.inf, 4.0], dtype=np.float32),
    )
    recorder = _recorder(
        model,
        _capture_cfg(
            tmp_path,
            record_solver=True,
            detect_nonfinite_in=("solver",),
        ),
        solver=solver,
    )

    _dispatch(recorder, model, model.state(), sim_time=0.0)

    [archive] = _archives(tmp_path)
    arrays, manifest = load_archive(archive)
    assert manifest["metadata"]["incident"]["global"] is True
    assert manifest["metadata"]["incident"]["world_id"] is None
    assert arrays["incident__nonfinite_paths"].tolist() == ["solver.body_q"]
    np.testing.assert_array_equal(arrays["incident__solver__body_q"], solver.body_q)


def test_contact_nonfinite_detection_uses_semantic_active_rows(tmp_path: Path):
    """Contact non-finites are localized through shape ownership anchors."""
    model = _collision_model()
    contacts = _FakeContacts()
    contacts.rigid_contact_force[1] = np.inf
    recorder = _recorder(
        model,
        _capture_cfg(
            tmp_path,
            record_contacts=True,
            detect_nonfinite_in=("contacts",),
        ),
        contacts=contacts,
    )

    _dispatch(recorder, model, model.state(), sim_time=0.0)

    [archive] = _archives(tmp_path)
    arrays, manifest = load_archive(archive)
    assert manifest["metadata"]["incident"]["world_id"] == 1
    assert arrays["incident__nonfinite_paths"].tolist() == ["contacts.rigid_contact_force"]
    np.testing.assert_array_equal(arrays["incident__contacts__rigid_contact_force"], [np.inf])


def test_ambiguous_collision_pipeline_nonfinite_is_global_and_full(
    tmp_path: Path,
):
    """Nested array extents do not inherit unrelated model row ownership."""
    model = _collision_model()
    pipeline = _FakeCollisionPipeline(model)
    pipeline.narrow_phase.shape_aabb_lower[2, 0] = np.inf
    recorder = _recorder(
        model,
        _capture_cfg(
            tmp_path,
            record_collision_pipeline=True,
            detect_nonfinite_in=("collision_pipeline",),
        ),
        collision_pipeline=pipeline,
    )

    _dispatch(recorder, model, model.state(), sim_time=0.0)

    [archive] = _archives(tmp_path)
    arrays, manifest = load_archive(archive)
    assert manifest["metadata"]["incident"]["global"] is True
    assert manifest["metadata"]["incident"]["world_id"] is None
    assert arrays["incident__nonfinite_paths"].tolist() == ["collision_pipeline.narrow_phase.shape_aabb_lower"]
    np.testing.assert_array_equal(
        arrays["incident__collision_pipeline__narrow_phase__shape_aabb_lower"],
        pipeline.narrow_phase.shape_aabb_lower,
    )


def test_ambiguous_operation_nonfinite_is_global_full_and_replay_independent(
    tmp_path: Path,
):
    """Incident-only operations detect globally without extent-based slicing."""
    model = _FakeModel(world_count=2)
    operations = _FakeOperationProvider()
    recorder = _recorder(
        model,
        _capture_cfg(
            tmp_path,
            record_operations=True,
            detect_nonfinite_in=("operations",),
        ),
        solver=_FakeSolver(),
        operation_provider=operations,
    )
    operations.payload["residual"] = np.asarray([0.25, np.inf], dtype=np.float32)

    _dispatch(recorder, model, model.state(), sim_time=0.0)

    [archive] = _archives(tmp_path)
    arrays, manifest = load_archive(archive)
    assert manifest["metadata"]["incident"]["global"] is True
    assert manifest["metadata"]["incident"]["world_id"] is None
    assert arrays["incident__nonfinite_paths"].tolist() == ["operations['residual']"]
    np.testing.assert_array_equal(arrays["incident__operations__key_residual"], [0.25, np.inf])
    assert not any(key.startswith("replay__") for key in arrays)


def test_symbolic_row_selection_requires_world_and_count_anchors(tmp_path: Path):
    """Filtering out an active-row anchor fails during recorder initialization."""
    model = _FakeModel(world_count=2)
    solver = _minimal_mjwarp_solver()
    replay = NewtonDebugReplayCfg(
        enabled=True,
        record_state=False,
        record_control=False,
        record_solver=True,
        record_contacts=False,
        include_fields=(
            "solver.mjw_data.contact.dist",
            "solver.mjw_data.nacon",
        ),
    )

    with pytest.raises(DebugCaptureError, match="world-id anchor"):
        _recorder(
            model,
            _capture_cfg(tmp_path, replay=replay),
            solver=solver,
        )


def test_pinned_newton_collision_pipeline_is_discovered_without_alias_duplication(
    tmp_path: Path,
):
    """The pinned real CollisionPipeline binds nested buffers through discovery."""
    import newton

    builder = newton.ModelBuilder()
    body = builder.add_body(mass=1.0)
    builder.add_shape_sphere(body, radius=0.5)
    model = builder.finalize(device="cpu")
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=16)
    recorder = PhysicsIncidentRecorder(
        model,
        _capture_cfg(
            tmp_path,
            history_length=1,
            max_gpu_bytes=16 * 1024**2,
            record_model=True,
            record_collision_pipeline=True,
        ),
        state=model.state(),
        control=None,
        solver=None,
        collision_pipeline=pipeline,
    )

    selected = {field.display_path for field in recorder._incident_binding.fields}
    assert "incident.collision_pipeline.broad_phase_pair_count" in selected
    assert "incident.collision_pipeline.broad_phase_shape_pairs" in selected
    assert "incident.collision_pipeline.narrow_phase.shape_aabb_lower" in selected
    assert "incident.collision_pipeline.narrow_phase.shape_aabb_upper" in selected
    assert any(path.startswith("incident.model.") for path in selected)
    assert not any(path.startswith("incident.collision_pipeline.model.") for path in selected)


def test_replay_packs_ragged_collision_rows_with_chronological_offsets(
    tmp_path: Path,
):
    """Wrapped replay stores variable active-row counts without pickle or padding."""
    model = _collision_model()
    pipeline = _FakeCollisionPipeline(model)
    pipeline.broad_phase_shape_pairs[:] = np.asarray(
        [[2, 3], [2, 3], [2, 3]],
        dtype=np.int32,
    )
    replay = NewtonDebugReplayCfg(
        enabled=True,
        record_state=False,
        record_control=False,
        record_solver=False,
        record_contacts=False,
        record_collision_pipeline=True,
    )
    recorder = _recorder(
        model,
        _capture_cfg(tmp_path, history_length=3, replay=replay),
        collision_pipeline=pipeline,
        triggers={"pair_churn": lambda _context: None},
    )

    for index, count in enumerate((2, 0, 1, 3)):
        pipeline.broad_phase_pair_count[0] = count
        recorder.record_step_replay_pre(model.state(), sim_time=float(index))
        recorder.record_step_replay_post(model.state())

    _dispatch(
        recorder,
        model,
        model.state(),
        sim_time=4.0,
        trigger_results={"pair_churn": _TriggerResult(reason="ragged pair rows", world_ids=(1,))},
    )

    [archive] = _archives(tmp_path)
    arrays, manifest = load_archive(archive)
    np.testing.assert_array_equal(arrays["replay__sim_time"], [1.0, 2.0, 3.0])
    [contract] = manifest["metadata"]["replay_partition_contracts"]
    assert contract["contract"] == "newton_broad_phase_pairs"
    assert contract["row_encoding"] == "concatenated_rows_with_slot_offsets"
    np.testing.assert_array_equal(
        arrays["replay__pre__collision_pipeline__broad_phase_pair_count"],
        [[0], [1], [3]],
    )
    np.testing.assert_array_equal(
        arrays["replay__pre__collision_pipeline__broad_phase_shape_pairs__slot_offsets"],
        [0, 0, 1, 4],
    )
    np.testing.assert_array_equal(
        arrays["replay__pre__collision_pipeline__broad_phase_shape_pairs"],
        [[2, 3], [2, 3], [2, 3], [2, 3]],
    )


def test_ambiguous_context_nonfinite_uses_latest_snapshot_as_global(tmp_path: Path):
    """Workflow arrays without ownership metadata trigger a complete global incident."""
    model = _FakeModel(world_count=2)
    per_world = np.asarray([1.0, 2.0], dtype=np.float32)
    recorder = _recorder(
        model,
        _capture_cfg(tmp_path, detect_nonfinite_in=("context",)),
        context_provider=lambda: {"per_world": per_world},
    )
    per_world[1] = np.inf

    _dispatch(recorder, model, model.state(), sim_time=0.0)

    [archive] = _archives(tmp_path)
    arrays, manifest = load_archive(archive)
    assert manifest["metadata"]["incident"]["global"] is True
    assert manifest["metadata"]["incident"]["world_id"] is None
    assert arrays["incident__nonfinite_paths"].tolist() == ["context['per_world']"]
    np.testing.assert_array_equal(arrays["incident__context__key_per_world"], [1.0, np.inf])


@pytest.mark.parametrize("context_provider", [None, lambda: {}])
def test_context_detection_requires_a_capturable_context_at_bind(
    tmp_path: Path,
    context_provider,
):
    """Missing and empty workflow schemas fail before simulation starts."""
    model = _FakeModel()

    with pytest.raises(DebugCaptureError, match="context.*no capturable fields"):
        _recorder(
            model,
            _capture_cfg(tmp_path, detect_nonfinite_in=("context",)),
            context_provider=context_provider,
        )


def test_reset_rearm_allows_immediate_world_recurrence_without_clean_step(
    tmp_path: Path,
):
    """Reset worlds rearm independently while non-reset failures remain active."""
    model = _FakeModel(world_count=2)
    recorder = _recorder(
        model,
        _capture_cfg(tmp_path, max_incidents=2, halt_on_incident=False),
    )
    both_bad = model.state()
    _inject_body_inf(both_bad, 0)
    _inject_body_inf(both_bad, 2)
    _dispatch(recorder, model, both_bad, sim_time=0.0)
    assert len(_archives(tmp_path)) == 2

    recorder.rearm_reset_worlds((0,))
    both_bad_again = model.state()
    _inject_body_inf(both_bad_again, 0)
    _inject_body_inf(both_bad_again, 2)
    _dispatch(recorder, model, both_bad_again, sim_time=0.1)

    archives = _archives(tmp_path)
    assert len(archives) == 3
    incidents = [load_archive(path)[1]["metadata"]["incident"] for path in archives]
    repeated = [incident for incident in incidents if incident["sim_time_seconds"] == 0.1]
    assert [(incident["world_id"], incident["global"]) for incident in repeated] == [(0, False)]


def test_reset_rearm_preserves_global_scope_without_explicit_all_world_reset(
    tmp_path: Path,
):
    """Global triggers rearm only when the caller proves every world was reset."""
    model = _FakeModel(world_count=2)
    recorder = _recorder(
        model,
        _capture_cfg(tmp_path, max_incidents=2),
        triggers={"global_probe": lambda _context: None},
    )
    result = {"global_probe": _TriggerResult(reason="global diagnostic", global_scope=True)}
    _dispatch(recorder, model, model.state(), sim_time=0.0, trigger_results=result)

    recorder.rearm_reset_worlds((0, 1), all_worlds=False)
    _dispatch(recorder, model, model.state(), sim_time=0.1, trigger_results=result)
    assert len(_archives(tmp_path)) == 1

    recorder.rearm_reset_worlds((0, 1), all_worlds=True)
    _dispatch(recorder, model, model.state(), sim_time=0.2, trigger_results=result)
    assert len(_archives(tmp_path)) == 2


@pytest.mark.parametrize(
    ("world_ids", "all_worlds", "expected"),
    [
        ([0], False, "tuple"),
        ((True,), False, "integers"),
        ((0, 0), False, "duplicates"),
        ((2,), False, "outside"),
        ((0,), 1, "bool"),
        ((0,), True, "exactly every allocated world"),
    ],
)
def test_reset_rearm_validates_scope_strictly(
    tmp_path: Path,
    world_ids,
    all_worlds,
    expected: str,
):
    """Invalid reset masks cannot accidentally rearm unrelated incidents."""
    model = _FakeModel(world_count=2)
    recorder = _recorder(model, _capture_cfg(tmp_path))

    with pytest.raises((TypeError, ValueError, DebugCaptureError), match=expected):
        recorder.rearm_reset_worlds(world_ids, all_worlds=all_worlds)


def test_reset_rearm_rejects_an_open_dispatch(tmp_path: Path):
    """Rearming after capture_pre cannot change the active dispatch semantics."""
    model = _FakeModel(world_count=2)
    recorder = _recorder(model, _capture_cfg(tmp_path))
    recorder.capture_pre(model.state())

    with pytest.raises(DebugCaptureError, match="before capture_pre"):
        recorder.rearm_reset_worlds((0,))


@pytest.mark.parametrize("fail_close", [False, True])
def test_operation_provider_closes_on_partial_recorder_initialization(
    tmp_path: Path,
    fail_close: bool,
):
    """A bound provider is closed when later schema binding fails."""
    model = _FakeModel()
    operations = _FakeOperationProvider()
    operations.fail_close = fail_close
    replay = NewtonDebugReplayCfg(
        enabled=True,
        record_state=False,
        record_control=False,
        record_solver=False,
        record_contacts=False,
        record_operations=True,
    )
    expected = "operation_provider.close" if fail_close else "matched no fields"

    with pytest.raises(DebugCaptureError, match=expected):
        _recorder(
            model,
            _capture_cfg(
                tmp_path,
                include_fields=("missing_provider_field",),
                replay=replay,
            ),
            solver=_FakeSolver(),
            operation_provider=operations,
        )

    assert operations.bind_count == 1
    assert operations.close_count == 1


def test_operation_provider_close_error_retains_retryable_ownership(
    tmp_path: Path,
):
    """A failed close releases rings but keeps provider ownership for retry."""
    model = _FakeModel()
    operations = _FakeOperationProvider()
    replay = NewtonDebugReplayCfg(
        enabled=True,
        record_state=False,
        record_control=False,
        record_solver=False,
        record_contacts=False,
        record_operations=True,
    )
    recorder = _recorder(
        model,
        _capture_cfg(tmp_path, replay=replay),
        solver=_FakeSolver(),
        operation_provider=operations,
    )
    operations.fail_close = True

    with pytest.raises(DebugCaptureError, match="operation_provider.close.*close failure"):
        recorder.clear()

    assert operations.close_count == 1
    assert recorder.history_length == 0
    operations.fail_close = False
    recorder.clear()
    assert operations.close_count == 2
    recorder.clear()
    assert operations.close_count == 2


def test_nonfinite_scan_filter_excludes_inactive_nan_but_archives_full_value(
    tmp_path: Path,
):
    """Mode-inactive non-finites can be ignored without dropping debug evidence."""
    model = _FakeModel(world_count=2)
    solver = _FakeSolver()
    solver.mode_inactive[:] = np.nan
    recorder = _recorder(
        model,
        _capture_cfg(
            tmp_path,
            failed_worlds_only=False,
            record_solver=True,
            detect_nonfinite_in=("solver",),
            detect_nonfinite_exclude_fields=("solver.mode_inactive",),
        ),
        solver=solver,
        triggers={"manual_probe": lambda _context: None},
    )

    _dispatch(
        recorder,
        model,
        model.state(),
        sim_time=0.0,
        trigger_results={"manual_probe": _TriggerResult(reason="inspect inactive mode", global_scope=True)},
    )

    [archive] = _archives(tmp_path)
    arrays, manifest = load_archive(archive)
    assert arrays["incident__nonfinite_paths"].tolist() == []
    np.testing.assert_array_equal(
        arrays["incident__solver__mode_inactive"],
        [np.nan, np.nan],
    )
    scan = manifest["metadata"]["providers"]["nonfinite_detection"]
    assert "incident.solver.mode_inactive" not in scan["fields"]
    assert "incident.solver.mode_inactive" in manifest["metadata"]["providers"]["selected_fields"]


def test_nonfinite_metadata_lists_only_scannable_recorded_fields(tmp_path: Path):
    """Integer metadata is archived but never advertised as a scan target."""
    model = _FakeModel()
    solver = _FakeSolver()
    recorder = _recorder(
        model,
        _capture_cfg(
            tmp_path,
            record_solver=True,
            detect_nonfinite_in=("solver",),
        ),
        solver=solver,
        triggers={"manual_probe": lambda _context: None},
    )

    _dispatch(
        recorder,
        model,
        model.state(),
        sim_time=0.0,
        trigger_results={"manual_probe": _TriggerResult(reason="inspect scan schema", global_scope=True)},
    )

    [archive] = _archives(tmp_path)
    arrays, manifest = load_archive(archive)
    scan = manifest["metadata"]["providers"]["nonfinite_detection"]
    assert scan["fields"] == [
        "incident.solver.adaptive_vector",
        "incident.solver.mode_inactive",
    ]
    assert "incident.solver.iterations" in manifest["metadata"]["providers"]["selected_fields"]
    assert arrays["incident__solver__iterations"].item() == 7


@pytest.mark.parametrize(
    "value",
    [np.int32(7), np.bool_(True), "inactive"],
    ids=["integer", "boolean", "string"],
)
def test_nonfinite_scan_rejects_exact_non_scannable_include(
    tmp_path: Path,
    value,
):
    """Exact scan patterns cannot bind integer, boolean, or string fields."""
    model = _FakeModel()
    solver = SimpleNamespace(
        residual=np.asarray([0.0], dtype=np.float32),
        target=value,
    )

    with pytest.raises(DebugCaptureError, match=r"non-scannable.*solver.target"):
        _recorder(
            model,
            _capture_cfg(
                tmp_path,
                record_solver=True,
                detect_nonfinite_in=("solver",),
                detect_nonfinite_include_fields=("solver.target",),
            ),
            solver=solver,
        )


def test_each_nonfinite_provider_requires_a_scannable_field(tmp_path: Path):
    """A provider with only integer fields cannot hide behind another provider."""
    model = _FakeModel()
    solver = SimpleNamespace(iterations=np.int32(7))

    with pytest.raises(DebugCaptureError, match=r"eliminated every.*provider 'solver'"):
        _recorder(
            model,
            _capture_cfg(
                tmp_path,
                record_solver=True,
                detect_nonfinite_in=("state", "solver"),
            ),
            solver=solver,
        )


def test_nonfinite_scan_filter_defaults_preserve_all_recorded_detection(tmp_path: Path):
    """Default scan patterns continue to detect every recorded provider field."""
    model = _FakeModel(world_count=2)
    solver = _FakeSolver()
    solver.mode_inactive[:] = np.nan
    recorder = _recorder(
        model,
        _capture_cfg(
            tmp_path,
            record_solver=True,
            detect_nonfinite_in=("solver",),
        ),
        solver=solver,
    )

    _dispatch(recorder, model, model.state(), sim_time=0.0)

    [archive] = _archives(tmp_path)
    arrays, manifest = load_archive(archive)
    assert manifest["metadata"]["incident"]["global"] is True
    assert arrays["incident__nonfinite_paths"].tolist() == ["solver.mode_inactive"]
    np.testing.assert_array_equal(arrays["incident__solver__mode_inactive"], [np.nan, np.nan])


@pytest.mark.parametrize(
    ("field_name", "patterns", "expected"),
    [
        ("detect_nonfinite_include_fields", ("solver.missing_field",), "include pattern"),
        ("detect_nonfinite_exclude_fields", ("solver.missing_field",), "exclude pattern"),
    ],
)
def test_nonfinite_scan_filter_rejects_unmatched_patterns_at_bind(
    tmp_path: Path,
    field_name: str,
    patterns: tuple[str, ...],
    expected: str,
):
    """Upstream field renames cannot silently weaken non-finite detection."""
    model = _FakeModel(world_count=2)

    with pytest.raises(DebugCaptureError, match=expected):
        _recorder(
            model,
            _capture_cfg(
                tmp_path,
                record_solver=True,
                detect_nonfinite_in=("solver",),
                **{field_name: patterns},
            ),
            solver=_FakeSolver(),
        )


def test_nonfinite_scan_filter_rejects_eliminated_provider(tmp_path: Path):
    """Each configured provider must retain at least one recorded scan field."""
    model = _FakeModel(world_count=2)

    with pytest.raises(DebugCaptureError, match=r"eliminated every.*provider 'solver'"):
        _recorder(
            model,
            _capture_cfg(
                tmp_path,
                record_solver=True,
                detect_nonfinite_in=("state", "solver"),
                detect_nonfinite_exclude_fields=("solver.*",),
            ),
            solver=_FakeSolver(),
        )


def test_operation_scan_filter_ignores_mode_inactive_nan_and_archives_it(
    tmp_path: Path,
):
    """Inactive transient buffers stay diagnostic without causing false incidents."""
    model = _FakeModel(world_count=2)
    operations = _FakeOperationProvider()
    operations.payload["mode_inactive"] = np.asarray([0.0, 0.0], dtype=np.float32)
    recorder = _recorder(
        model,
        _capture_cfg(
            tmp_path,
            failed_worlds_only=False,
            record_operations=True,
            detect_nonfinite_in=("operations",),
            detect_nonfinite_exclude_fields=("operations*mode_inactive*",),
        ),
        solver=_FakeSolver(),
        operation_provider=operations,
        triggers={"manual_probe": lambda _context: None},
    )
    operations.payload["mode_inactive"][:] = np.nan

    _dispatch(
        recorder,
        model,
        model.state(),
        sim_time=0.0,
        trigger_results={"manual_probe": _TriggerResult(reason="inspect inactive operation", global_scope=True)},
    )

    [archive] = _archives(tmp_path)
    arrays, _ = load_archive(archive)
    assert arrays["incident__nonfinite_paths"].tolist() == []
    np.testing.assert_array_equal(
        arrays["incident__operations__key_mode_inactive"],
        [np.nan, np.nan],
    )
