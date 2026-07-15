# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Route-independent target-coordinate certification tests."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from isaaclab_newton.sim import NewtonMjcfFileCfg

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from isaaclab_tasks.core.multi_task.kinematics import (
    KinematicTree,
    NewtonKinematics,
    NewtonKinematicsBuildCfg,
    time_backward_difference_segmented,
    time_forward_difference_segmented,
)
from isaaclab_tasks.core.multi_task.motion.data import MotionClipIndex, MotionGeneralizedCoordinates
from isaaclab_tasks.core.multi_task.motion.mdp.commands.commands_cfg import (
    MotionAnalyticFamilyCfg,
    MotionExactFamilyCfg,
    MotionGroundPenetrationCriterionCfg,
    MotionTargetCoordinateCriterionCfg,
    MotionTargetCoordinateLimitsCriterionCfg,
    MotionTrajectoryFamilyCfg,
)
from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table import (
    _QUALITY_NAMES,
    _TARGET_COORDINATE_QUALITY_NAMES,
    _TARGET_COORDINATE_QUALITY_START,
    _TARGET_COORDINATE_QUALITY_STOP,
    _TRAJECTORY_METRIC_NAMES,
)
from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table_builder import (
    _certify_target_coordinates,
    _stored_corpus_quality,
    motion_criterion_ground_penetration,
    motion_criterion_target_coordinate_limits,
    motion_criterion_target_coordinates,
)
from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_trajectory import _trajectory_corpus_quality
from isaaclab_tasks.core.multi_task.motion.robots.g1.frames import _time_forward_difference_segmented
from isaaclab_tasks.core.multi_task.motion.robots.g1.reference import G1FrameBuilder, g1_frame_builder
from isaaclab_tasks.core.multi_task.motion.robots.smpl.reference import (
    SmplFrameBuilder,
    _time_select_euler_xyz_branches_segmented,
    smpl_frame_builder,
)

_SHA256 = "0" * 64
_MJCF = """
<mujoco model="coordinate_certificate">
  <compiler angle="radian"/>
  <worldbody>
    <body name="root">
      <freejoint name="root_joint"/>
      <geom name="root_geom" type="sphere" size="0.05" mass="1.0"/>
      <body name="link" pos="0 0 0.2">
        <joint name="joint" type="hinge" axis="0 1 0" range="-1 1"/>
        <geom name="link_geom" type="capsule" size="0.02 0.1" mass="0.5"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


class _Target:
    """Minimal target exposing the production coordinate contract."""

    version = "coordinate_certificate_target_v1"
    construction_identity_sha256 = _SHA256
    joint_names = ("joint",)
    reference_frame_names = ("root", "link")
    materialization_minimum_frames = 2

    def __init__(self, kinematics: NewtonKinematics) -> None:
        self.kinematics = kinematics
        self.kinematic_tree = KinematicTree.from_newton(kinematics)
        self.collision_probe_body_indices = torch.tensor((0,), dtype=torch.int64, device=kinematics.device)
        self.collision_probe_offsets_m = torch.zeros((1, 3), dtype=torch.float32, device=kinematics.device)
        self.collision_probe_contact_slots = torch.tensor((-1,), dtype=torch.int64, device=kinematics.device)
        self.collision_probe_normal_channel_slots = torch.tensor((-1,), dtype=torch.int64, device=kinematics.device)
        self.collision_geometry_identity_sha256 = "1" * 64

    def allocate_coordinates(self, frame_count: int, *, device: str | torch.device) -> MotionGeneralizedCoordinates:
        return MotionGeneralizedCoordinates(torch.empty((frame_count, 8), device=device), None)

    def coordinates_from_newton(
        self, joint_q: torch.Tensor, clip_index: MotionClipIndex
    ) -> MotionGeneralizedCoordinates:
        if joint_q.shape != (clip_index.total_frames, 8):
            raise ValueError("Test target coordinates differ from the declared clips.")
        return MotionGeneralizedCoordinates(joint_q, None)

    def write_joint_position_newton(self, coordinates: MotionGeneralizedCoordinates, output: torch.Tensor) -> None:
        output.copy_(coordinates.joint_q)

    def write_nonroot_velocity_canonical(
        self,
        joint_q: torch.Tensor,
        clip_offsets: torch.Tensor,
        step_seconds: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        velocity = time_backward_difference_segmented(joint_q[:, 7:], clip_offsets.to(torch.int64), step_seconds)
        output[:, 6:].copy_(velocity)

    def materialize_coordinates(self, coordinates: MotionGeneralizedCoordinates, clip_index: MotionClipIndex):
        raise AssertionError("Coordinate certification must not materialize target frames.")


@pytest.fixture(scope="module")
def target(tmp_path_factory: pytest.TempPathFactory) -> _Target:
    """Build one scene-derived free-root target with hard position and velocity limits."""
    path = tmp_path_factory.mktemp("coordinate_certificate") / "robot.xml"
    path.write_text(_MJCF, encoding="utf-8")
    articulation = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=NewtonMjcfFileCfg(asset_path=str(path)),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
        actuators={
            "joint": ImplicitActuatorCfg(
                joint_names_expr=["joint"],
                velocity_limit_sim=0.5,
                effort_limit_sim=10.0,
                stiffness=0.0,
                damping=0.0,
            )
        },
    )
    kinematics = NewtonKinematics.from_articulation(
        NewtonKinematicsBuildCfg(collapse_fixed_joints=False), articulation, "cpu"
    )
    return _Target(kinematics)


def _clips() -> MotionClipIndex:
    """Return two three-frame clips whose flat boundary contains a large jump."""
    clip = MotionClipIndex.Clip
    return MotionClipIndex(
        source_content_sha256=_SHA256,
        skeleton_identity_sha256s=(_SHA256,),
        clips=(
            clip("first", 3, 2.0, _SHA256, 0),
            clip("second", 3, 2.0, _SHA256, 0),
        ),
    )


def _coordinates() -> MotionGeneralizedCoordinates:
    """Return finite, normalized, limit-valid coordinates for :func:`_clips`."""
    joint_q = torch.zeros((6, 8), dtype=torch.float32)
    joint_q[:, 6] = 1.0
    joint_q[:, 7] = torch.tensor((0.0, 0.1, 0.2, -0.9, -0.8, -0.7))
    return MotionGeneralizedCoordinates(joint_q, None)


def _three_frame_corpus(
    fps_values: tuple[float, ...], rates: tuple[float, ...]
) -> tuple[MotionClipIndex, MotionGeneralizedCoordinates]:
    """Return varied valid three-frame clips sampled from declared continuous rates [rad/s]."""
    if len(fps_values) != len(rates):
        raise ValueError("Test corpus rates and sampling frequencies must have equal lengths.")
    clip = MotionClipIndex.Clip
    index = MotionClipIndex(
        source_content_sha256=_SHA256,
        skeleton_identity_sha256s=(_SHA256,),
        clips=tuple(clip(f"clip_{row}", 3, fps, _SHA256, 0) for row, fps in enumerate(fps_values)),
    )
    joint_q = torch.zeros((index.total_frames, 8), dtype=torch.float32)
    joint_q[:, 6] = 1.0
    for row, (fps, rate) in enumerate(zip(fps_values, rates, strict=True)):
        start, stop = index.offsets[row : row + 2]
        joint_q[start:stop, 7] = -0.6 + 0.1 * (row % 4) + torch.arange(3) * (rate / fps)
    return index, MotionGeneralizedCoordinates(joint_q, None)


def _column(name: str) -> int:
    return _TARGET_COORDINATE_QUALITY_NAMES.index(name)


@pytest.mark.parametrize("device_type", ("cpu", "cuda"))
def test_g1_forward_difference_restarts_and_repeats_each_penultimate_edge(device_type: str) -> None:
    """The released G1 tail law remains local to every retained clip."""
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(device_type)
    values = torch.tensor((0.0, 1.0, 3.0, 100.0, 104.0, 110.0), dtype=torch.float32, device=device)[:, None]
    offsets = torch.tensor((0, 3, 6), dtype=torch.int64, device=device)
    steps = torch.tensor((0.5, 2.0), dtype=torch.float32, device=device)

    actual = _time_forward_difference_segmented(values, offsets, steps)
    expected_parts = []
    for segment, step in zip(values.split(3), steps):
        difference = (segment[1:] - segment[:-1]) / step
        expected_parts.append(torch.cat((difference, difference[-2:-1])))

    torch.testing.assert_close(actual, torch.cat(expected_parts))


@pytest.mark.parametrize("device_type", ("cpu", "cuda"))
def test_public_g1_forward_difference_wrapper_warns_and_delegates(device_type: str) -> None:
    """The committed shared API remains one explicit deprecation boundary."""
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(device_type)
    values = torch.tensor((0.0, 1.0, 3.0), dtype=torch.float32, device=device)[:, None]
    offsets = torch.tensor((0, 3), dtype=torch.int64, device=device)
    steps = torch.tensor((0.5,), dtype=torch.float32, device=device)
    expected = _time_forward_difference_segmented(values, offsets, steps)

    with pytest.warns(DeprecationWarning, match="owned by G1 frame construction"):
        actual = time_forward_difference_segmented(values, offsets, steps)

    torch.testing.assert_close(actual, expected)


def _humenv_euler_branch_oracle(values: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Return the literal released clip-local coupled XYZ branch recurrence."""
    output = values.clone()
    for start, stop in zip(offsets[:-1].tolist(), offsets[1:].tolist(), strict=True):
        for frame in range(start + 1, stop - 1):
            difference = output[frame] - output[frame - 1]
            passes = 0
            while difference.abs().max() >= 3.0:
                changed = difference.abs().sum(dim=-1) >= 3.0
                branch = output[frame, changed].clone()
                branch[:, 0] = torch.pi + branch[:, 0]
                branch[:, 1] = torch.pi - branch[:, 1]
                branch[:, 2] = torch.pi + branch[:, 2]
                branch[branch > torch.pi] -= 2.0 * torch.pi
                branch[branch < -torch.pi] += 2.0 * torch.pi
                output[frame, changed] = branch
                difference = output[frame] - output[frame - 1]
                passes += 1
                if passes > 1:
                    break
    return output


@pytest.mark.parametrize("device_type", ("cpu", "cuda"))
def test_smpl_humenv_branch_selection_matches_coupled_released_recurrence(device_type: str) -> None:
    """SMPL owns global joint coupling, clip restarts, two passes, and untouched tails."""
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(device_type)
    values = torch.zeros((8, 2, 3), dtype=torch.float32, device=device)
    values[1, 0] = values.new_tensor((-3.04, 1.74, -2.94))
    values[1, 1] = 1.1
    values[2, 0] = values.new_tensor((3.1, 0.0, 0.0))
    values[3] = values.new_tensor(((3.12, -3.12, 3.12), (-3.12, 3.12, -3.12)))
    values[4] = values.new_tensor(((2.9, -2.8, 2.7), (1.0, -1.0, 1.0)))
    values[5, 0] = values.new_tensor((-0.24, -0.34, -0.44))
    values[5, 1] = values.new_tensor((1.1, 1.1, 1.1))
    values[7] = values.new_tensor(((-3.12, 3.12, -3.12), (3.12, -3.12, 3.12)))
    offsets = torch.tensor((0, 4, 8), dtype=torch.int64, device=device)

    expected = _humenv_euler_branch_oracle(values, offsets)
    actual = _time_select_euler_xyz_branches_segmented(values, offsets)

    torch.testing.assert_close(actual, expected, atol=1.0e-6, rtol=0.0)
    assert not torch.equal(actual[1, 1], values[1, 1])
    torch.testing.assert_close(actual[3], values[3])
    torch.testing.assert_close(actual[4], values[4])
    torch.testing.assert_close(actual[7], values[7])


def test_final_quality_keeps_one_target_coordinate_block_for_every_route() -> None:
    """Exact, analytic, and trajectory routes publish the same certified target-coordinate evidence."""
    assert "constraint_feasible" not in _QUALITY_NAMES
    clip_index = SimpleNamespace(clips=(object(), object()))
    coordinates = MotionGeneralizedCoordinates(torch.zeros((2, 1), dtype=torch.float32), None)
    evidence = torch.arange(2 * len(_TARGET_COORDINATE_QUALITY_NAMES), dtype=torch.float32).reshape(2, -1)
    accepted = torch.tensor((True, False))
    stored = _stored_corpus_quality(
        SimpleNamespace(
            clip_index=clip_index,
            coordinates=coordinates,
            target_coordinate_evidence=evidence,
        ),
        accepted,
    )
    trajectory_metrics = torch.arange(2 * len(_TRAJECTORY_METRIC_NAMES), dtype=torch.float32).reshape(2, -1)
    trajectory = _trajectory_corpus_quality(
        SimpleNamespace(
            clip_index=clip_index,
            coordinates=coordinates,
            trajectory_quality=trajectory_metrics,
            target_coordinate_evidence=evidence,
            constraint_geometry_feasible=torch.tensor((True, False)),
            inner_solve_converged=torch.tensor((False, True)),
            nonlinear_refinement_required=torch.tensor((True, False)),
            nonlinear_phases_converged=torch.tensor((False, False)),
        ),
        accepted,
    )

    assert stored.shape == trajectory.shape == (2, len(_QUALITY_NAMES))
    torch.testing.assert_close(stored[:, _TARGET_COORDINATE_QUALITY_START:_TARGET_COORDINATE_QUALITY_STOP], evidence)
    torch.testing.assert_close(
        trajectory[:, _TARGET_COORDINATE_QUALITY_START:_TARGET_COORDINATE_QUALITY_STOP], evidence
    )
    zero_metrics = {"contact_applicable", "contact_stable_frame_channel_count"}
    for metric_index, name in enumerate(_TRAJECTORY_METRIC_NAMES):
        stored_metric = stored[:, _QUALITY_NAMES.index(name)]
        if name in zero_metrics:
            torch.testing.assert_close(stored_metric, torch.zeros(2))
        else:
            assert torch.isnan(stored_metric).all()
        torch.testing.assert_close(trajectory[:, _QUALITY_NAMES.index(name)], trajectory_metrics[:, metric_index])
    torch.testing.assert_close(
        trajectory[:, _QUALITY_NAMES.index("constraint_geometry_feasible")], torch.tensor((1.0, 0.0))
    )
    torch.testing.assert_close(trajectory[:, _QUALITY_NAMES.index("inner_solve_converged")], torch.tensor((0.0, 1.0)))
    torch.testing.assert_close(
        trajectory[:, _QUALITY_NAMES.index("nonlinear_refinement_required")], torch.tensor((1.0, 0.0))
    )
    torch.testing.assert_close(
        trajectory[:, _QUALITY_NAMES.index("nonlinear_phases_converged")], torch.tensor((0.0, 0.0))
    )


def test_certificate_is_read_only_and_clip_bounded(target: _Target) -> None:
    """Certification preserves coordinates and never differentiates across clip boundaries."""
    source = _coordinates()
    coordinates = MotionGeneralizedCoordinates(source.joint_q, torch.zeros((6, 7), dtype=torch.float32))
    joint_q_before = coordinates.joint_q.clone()
    joint_qd_before = coordinates.joint_qd.clone()

    evidence = _certify_target_coordinates(target, coordinates, _clips())

    torch.testing.assert_close(coordinates.joint_q, joint_q_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(coordinates.joint_qd, joint_qd_before, rtol=0.0, atol=0.0)
    assert evidence.shape == (2, len(_TARGET_COORDINATE_QUALITY_NAMES))
    torch.testing.assert_close(
        evidence[:, _column("canonical_joint_velocity_limit_ratio")], torch.full((2,), 0.4), atol=1.0e-6, rtol=0.0
    )
    assert motion_criterion_target_coordinates(
        None, SimpleNamespace(target_coordinate_evidence=evidence), torch.arange(2)
    ).tolist() == [True, True]


@pytest.mark.parametrize(("penetration_m", "expected"), ((0.002, True), (0.0021, False)))
def test_ground_certificate_enforces_two_millimeter_target_boundary(
    target: _Target, penetration_m: float, expected: bool
) -> None:
    """Every coordinate route shares the same finite 2 mm target-collider ground certificate."""
    coordinates = _coordinates()
    coordinates.joint_q[:, 2].fill_(-penetration_m)
    evidence = _certify_target_coordinates(target, coordinates, _clips())
    torch.testing.assert_close(
        evidence[:, _column("ground_penetration_max_m")],
        torch.full((2,), penetration_m),
        atol=1.0e-7,
        rtol=0.0,
    )
    accepted = motion_criterion_ground_penetration(
        MotionGroundPenetrationCriterionCfg(), SimpleNamespace(target_coordinate_evidence=evidence), torch.arange(2)
    )
    assert accepted.tolist() == [expected, expected]


def test_ground_certificate_maps_nonfinite_probe_to_infinity(target: _Target) -> None:
    """A non-finite world probe fails closed as infinite ground penetration."""
    coordinates = _coordinates()
    coordinates.joint_q[1, 0] = float("nan")

    evidence = _certify_target_coordinates(target, coordinates, _clips())

    penetration = evidence[:, _column("ground_penetration_max_m")]
    assert torch.isposinf(penetration[0])
    torch.testing.assert_close(penetration[1], torch.tensor(0.0))
    accepted = motion_criterion_ground_penetration(
        MotionGroundPenetrationCriterionCfg(), SimpleNamespace(target_coordinate_evidence=evidence), torch.arange(2)
    )
    assert accepted.tolist() == [False, True]


def test_certificate_ignores_stored_velocity_when_final_positions_and_clock_match(target: _Target) -> None:
    """Route-local stored velocities cannot change the canonical certificate."""
    source = _coordinates()
    stored_slow = torch.zeros((source.frame_count, 7), dtype=torch.float32)
    stored_fast = stored_slow.clone()
    stored_fast[:, 6] = 4.0

    slow_evidence = _certify_target_coordinates(
        target, MotionGeneralizedCoordinates(source.joint_q.clone(), stored_slow), _clips()
    )
    fast_evidence = _certify_target_coordinates(
        target, MotionGeneralizedCoordinates(source.joint_q.clone(), stored_fast), _clips()
    )

    torch.testing.assert_close(slow_evidence, fast_evidence, rtol=0.0, atol=0.0)


def test_reference_coordinate_families_preserve_source_while_trajectory_requires_robot_feasibility() -> None:
    """Exact and analytic references report physical diagnostics without rejecting source-owned coordinates."""
    from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionCommandsCfg

    evidence = torch.ones((1, len(_TARGET_COORDINATE_QUALITY_NAMES)), dtype=torch.float32)
    evidence[:, _column("root_quaternion_norm_error")] = 0.0
    evidence[:, _column("joint_position_limit_violation_max_rad")] = 0.4
    evidence[:, _column("joint_position_limits_satisfied")] = 0.0
    evidence[:, _column("canonical_joint_velocity_limit_ratio")] = 3.0
    evidence[:, _column("canonical_joint_velocity_limits_satisfied")] = 0.0
    candidate = SimpleNamespace(target_coordinate_evidence=evidence)
    rows = torch.tensor((0,))

    assert motion_criterion_target_coordinates(None, candidate, rows).tolist() == [True]
    assert motion_criterion_target_coordinate_limits(None, candidate, rows).tolist() == [False]
    root_families = MotionCommandsCfg().motion.task_table.families
    direct_families = (MotionExactFamilyCfg(), MotionAnalyticFamilyCfg(), *root_families[:2])
    for family in direct_families:
        assert tuple(type(criterion) for criterion in family.criteria) == (MotionTargetCoordinateCriterionCfg,)
    trajectory_criteria = MotionTrajectoryFamilyCfg().criteria
    assert any(isinstance(criterion, MotionTargetCoordinateCriterionCfg) for criterion in trajectory_criteria)
    assert any(isinstance(criterion, MotionTargetCoordinateLimitsCriterionCfg) for criterion in trajectory_criteria)
    assert any(isinstance(criterion, MotionGroundPenetrationCriterionCfg) for criterion in trajectory_criteria)


def test_certificate_uses_physical_time_for_canonical_velocity(target: _Target) -> None:
    """The same continuous target motion has identical velocity evidence at 30 and 60 Hz."""
    clip = MotionClipIndex.Clip
    indices = tuple(
        MotionClipIndex(
            source_content_sha256=_SHA256,
            skeleton_identity_sha256s=(_SHA256,),
            clips=(clip(f"motion_{int(fps)}", 3, fps, _SHA256, 0),),
        )
        for fps in (30.0, 60.0)
    )
    coordinates = []
    for fps in (30.0, 60.0):
        joint_q = torch.zeros((3, 8), dtype=torch.float32)
        joint_q[:, 6] = 1.0
        joint_q[:, 7] = torch.arange(3, dtype=torch.float32) * (0.25 / fps)
        coordinates.append(MotionGeneralizedCoordinates(joint_q, None))

    evidence_30 = _certify_target_coordinates(target, coordinates[0], indices[0])
    evidence_60 = _certify_target_coordinates(target, coordinates[1], indices[1])

    torch.testing.assert_close(
        evidence_30[:, _column("canonical_joint_velocity_limit_ratio")],
        evidence_60[:, _column("canonical_joint_velocity_limit_ratio")],
        atol=1.0e-6,
        rtol=0.0,
    )


def test_certificate_materializes_one_corpus_and_one_cpu_fk_batch(
    target: _Target, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Thousands of clips do not cause per-clip conversion, velocity, or FK launches."""
    index, coordinates = _three_frame_corpus((30.0,) * 1919, (0.1,) * 1919)
    write_position = target.write_joint_position_newton
    write_velocity = target.write_nonroot_velocity_canonical
    evaluate_fk = target.kinematics.eval_fk_batched_torch
    calls = {"position": 0, "velocity": 0, "fk": 0}

    def counted_position(coordinates, output):
        calls["position"] += 1
        return write_position(coordinates, output)

    def counted_velocity(joint_q, clip_offsets, step_seconds, output):
        calls["velocity"] += 1
        return write_velocity(joint_q, clip_offsets, step_seconds, output)

    def counted_fk(joint_q, joint_qd, body_q, body_qd):
        calls["fk"] += 1
        return evaluate_fk(joint_q, joint_qd, body_q, body_qd)

    monkeypatch.setattr(target, "write_joint_position_newton", counted_position)
    monkeypatch.setattr(target, "write_nonroot_velocity_canonical", counted_velocity)
    monkeypatch.setattr(target.kinematics, "eval_fk_batched_torch", counted_fk)

    evidence = _certify_target_coordinates(target, coordinates, index)

    assert evidence.shape == (1919, len(_TARGET_COORDINATE_QUALITY_NAMES))
    assert calls == {"position": 1, "velocity": 1, "fk": 1}


def test_certificate_matches_separate_clips_and_clip_permutation(target: _Target) -> None:
    """Segmented corpus evidence equals separate calls and follows arbitrary complete-clip order."""
    index, coordinates = _three_frame_corpus((2.0, 4.0, 8.0, 16.0), (0.05, 0.1, 0.15, 0.2))
    corpus_evidence = _certify_target_coordinates(target, coordinates, index)
    separate = []
    for row, clip in enumerate(index.clips):
        start, stop = index.offsets[row : row + 2]
        single_index = MotionClipIndex(
            source_content_sha256=_SHA256,
            skeleton_identity_sha256s=(_SHA256,),
            clips=(clip,),
        )
        separate.append(
            _certify_target_coordinates(
                target,
                MotionGeneralizedCoordinates(coordinates.joint_q[start:stop].contiguous(), None),
                single_index,
            )
        )
    torch.testing.assert_close(corpus_evidence, torch.cat(separate), rtol=0.0, atol=1.0e-6)

    permutation = (2, 0, 3, 1)
    permuted_index = MotionClipIndex(
        source_content_sha256=_SHA256,
        skeleton_identity_sha256s=(_SHA256,),
        clips=tuple(index.clips[row] for row in permutation),
    )
    permuted_joint_q = torch.cat(
        tuple(coordinates.joint_q[index.offsets[row] : index.offsets[row + 1]] for row in permutation)
    ).contiguous()
    permuted_evidence = _certify_target_coordinates(
        target, MotionGeneralizedCoordinates(permuted_joint_q, None), permuted_index
    )
    torch.testing.assert_close(permuted_evidence, corpus_evidence[torch.tensor(permutation)], rtol=0.0, atol=1.0e-6)


@pytest.mark.parametrize(
    ("fault", "column", "well_formed"),
    (
        ("nonfinite_position", "coordinate_finite", False),
        ("root_quaternion", "root_quaternion_norm_error", False),
        ("joint_limit", "joint_position_limits_satisfied", True),
        ("joint_velocity", "canonical_joint_velocity_limits_satisfied", True),
    ),
)
def test_certificate_routes_well_formedness_and_limit_faults(
    target: _Target, fault: str, column: str, well_formed: bool
) -> None:
    """Coordinate evidence routes malformed data and robot-limit policy independently."""
    source = _coordinates()
    joint_q = source.joint_q.clone()
    if fault == "nonfinite_position":
        joint_q[0, 0] = torch.nan
    elif fault == "root_quaternion":
        joint_q[0, 3:7] *= 2.0
    elif fault == "joint_limit":
        joint_q[0, 7] = 1.1
    else:
        joint_q[1, 7] = 0.4
    coordinates = MotionGeneralizedCoordinates(joint_q, None)

    evidence = _certify_target_coordinates(target, coordinates, _clips())
    candidate = SimpleNamespace(target_coordinate_evidence=evidence)
    well_formed_accepted = motion_criterion_target_coordinates(None, candidate, torch.arange(2))
    limit_accepted = motion_criterion_target_coordinate_limits(None, candidate, torch.arange(2))

    assert bool(well_formed_accepted[0]) is well_formed
    assert bool(well_formed_accepted[1])
    if fault in ("joint_limit", "joint_velocity"):
        assert not bool(limit_accepted[0])
    if column.endswith("_satisfied") or column.endswith("_finite"):
        assert evidence[0, _column(column)] == 0.0
    else:
        assert not torch.isfinite(evidence[0, _column(column)]) or evidence[0, _column(column)] > 0.0


def test_certificate_rejects_nonfinite_live_target_fk(target: _Target, monkeypatch: pytest.MonkeyPatch) -> None:
    """Finite coordinates cannot pass when the selected target mechanics produce nonfinite FK."""
    evaluate = target.kinematics.eval_fk_batched_torch

    def nonfinite_fk(joint_q, joint_qd, body_q, body_qd):
        result = evaluate(joint_q, joint_qd, body_q, body_qd)
        body_q[0, 0, 0] = torch.nan
        return result

    monkeypatch.setattr(target.kinematics, "eval_fk_batched_torch", nonfinite_fk)
    evidence = _certify_target_coordinates(target, _coordinates(), _clips())

    assert evidence[:, _column("fk_finite")].tolist() == [0.0, 1.0]
    assert motion_criterion_target_coordinates(
        None, SimpleNamespace(target_coordinate_evidence=evidence), torch.arange(2)
    ).tolist() == [False, True]


def test_certificate_rejects_nonfinite_target_owned_canonical_velocity(
    target: _Target, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finite positions cannot hide a nonfinite target-owned canonical velocity."""
    write_velocity = target.write_nonroot_velocity_canonical

    def nonfinite_velocity(joint_q, clip_offsets, step_seconds, output):
        write_velocity(joint_q, clip_offsets, step_seconds, output)
        output[0, 6] = torch.nan

    monkeypatch.setattr(target, "write_nonroot_velocity_canonical", nonfinite_velocity)
    evidence = _certify_target_coordinates(target, _coordinates(), _clips())

    assert evidence[:, _column("coordinate_finite")].tolist() == [0.0, 1.0]
    assert motion_criterion_target_coordinates(
        None, SimpleNamespace(target_coordinate_evidence=evidence), torch.arange(2)
    ).tolist() == [False, True]


def test_raw_retargeting_has_no_proto_product_mechanisms() -> None:
    """Owned raw-retarget paths retain Proto equations without its product-specific machinery."""
    motion_root = Path(__file__).parents[1] / "motion"
    paths = (
        motion_root / "retarget.py",
        motion_root / "mdp" / "commands" / "motion_task_table_builder.py",
        motion_root / "mdp" / "commands" / "motion_trajectory.py",
        motion_root / "robots" / "g1" / "reference.py",
        motion_root / "robots" / "smpl" / "reference.py",
    )
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names} | {
            node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        }
        assert not any(name == "jax" or name.startswith("jax.") for name in imports)
        assert not any(isinstance(node, ast.Name) and node.id == "source_type" for node in ast.walk(tree))
        assert not any(isinstance(node, ast.Constant) and node.value == 450 for node in ast.walk(tree))
        assert not any(
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Div)
            and isinstance(node.right, ast.Constant)
            and node.right.value in (4, 4.0)
            for node in ast.walk(tree)
        )


@pytest.mark.parametrize(
    ("symbol", "args"),
    (
        (G1FrameBuilder, (None, None, None, False)),
        (g1_frame_builder, (None, None)),
        (SmplFrameBuilder, (None, None, None, False)),
        (smpl_frame_builder, (None, None)),
    ),
)
def test_deprecated_frame_builders_refuse_implicit_contact_policy(symbol, args: tuple[object, ...]) -> None:
    """Deprecated composite builders warn and reject their now-ambiguous hidden policy."""
    with pytest.warns(DeprecationWarning, match="deprecated"), pytest.raises(RuntimeError, match="contact_patches"):
        symbol(*args)


def test_deprecated_frame_builders_have_no_internal_callers() -> None:
    """Only public shim definitions, package exports, and this boundary test retain removed builder names."""
    root = Path(__file__).parents[1]
    symbols = {"G1FrameBuilder", "g1_frame_builder", "SmplFrameBuilder", "smpl_frame_builder"}
    permitted = {
        Path(__file__),
        root / "motion" / "robots" / "g1" / "reference.py",
        root / "motion" / "robots" / "smpl" / "reference.py",
    }
    for path in root.rglob("*.py"):
        if path in permitted:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in symbols
        }
        calls.update(
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in symbols
        )
        assert not calls, f"Deprecated frame-builder caller remains in {path}: {sorted(calls)}"
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
            if alias.name in symbols
        }
        assert not imports, f"Deprecated frame-builder import remains in {path}: {sorted(imports)}"

    expected_exports = {
        root / "motion" / "robots" / "g1" / "__init__.pyi": ("G1FrameBuilder", "g1_frame_builder"),
        root / "motion" / "robots" / "smpl" / "__init__.pyi": ("SmplFrameBuilder", "smpl_frame_builder"),
    }
    for path, names in expected_exports.items():
        source = path.read_text(encoding="utf-8")
        assert all(f'"{name}"' in source for name in names)
