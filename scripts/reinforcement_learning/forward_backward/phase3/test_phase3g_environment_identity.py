# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate direct Phase 3 environment identity without inventing config owners."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from isaaclab_newton.sim import NewtonMjcfFileCfg
from motion_environment_identity import motion_environment_axes

import isaaclab.sim as sim_utils
from isaaclab.managers import ActionTermCfg

from isaaclab_tasks.core.multi_task.motion.data.sources import CmuHumEnvSmplClips
from isaaclab_tasks.core.multi_task.motion.robots.g1.reference import G1LocalBodyPoseFrameBuilder
from isaaclab_tasks.core.multi_task.motion.robots.smpl.articulation import SMPL_MOTION_ARTICULATION_CFG
from isaaclab_tasks.core.multi_task.motion.robots.smpl.reference import SmplGeneralizedCoordinateFrameBuilder
from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionImitationEnvCfg
from isaaclab_tasks.utils import resolve_presets

from isaaclab_assets.robots.smpl.smpl_cfg import SMPL_HUMANOID_CFG
from isaaclab_assets.robots.smpl.smpl_constants import MUJOCO_BODY_NAMES

IDENTITY = Path(__file__).parent / "motion_environment_identity.py"


def _smpl_live_axes() -> tuple[tuple[str, ...], tuple[str, ...]]:
    joints = tuple(
        f"{body}_x_{body}_y_{body}_z:{component}" for body in MUJOCO_BODY_NAMES[1:] for component in range(3)
    )
    return joints, MUJOCO_BODY_NAMES


def _module():
    spec = importlib.util.spec_from_file_location("motion_environment_identity", IDENTITY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _smpl_identity(cfg: MotionImitationEnvCfg) -> dict[str, object]:
    return _module().motion_environment_dependency_identity(
        preset="smpl_cmu",
        cfg=cfg,
        importer_type=CmuHumEnvSmplClips,
        frame_builder_type=SmplGeneralizedCoordinateFrameBuilder,
    )


def _g1_cmu_identities(
    cfg: MotionImitationEnvCfg,
    *,
    frame_builder_identity_sha256: str = "a" * 64,
    reference_artifact_root: Path | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    module = _module()
    common = {
        "preset": "g1_cmu",
        "cfg": cfg,
        "importer_type": CmuHumEnvSmplClips,
        "frame_builder_type": G1LocalBodyPoseFrameBuilder,
        "reference_artifact_root": reference_artifact_root,
    }
    return (
        module.motion_environment_dependency_identity(**common),
        module.motion_composition_dependency_identity(
            **common,
            frame_builder_identity_sha256=frame_builder_identity_sha256,
        ),
    )


def test_smpl_identity_uses_native_asset_axes_without_config_actuator() -> None:
    """Native SMPL control must resolve all live axes with no duplicate actuator owner."""
    cfg = resolve_presets(MotionImitationEnvCfg(), selected=motion_environment_axes("smpl_cmu"))
    assert cfg.scene.robot.actuators == {}

    identity = _module().motion_environment_dependency_identity(
        preset="smpl_cmu",
        cfg=cfg,
        importer_type=CmuHumEnvSmplClips,
        frame_builder_type=SmplGeneralizedCoordinateFrameBuilder,
    )

    robot = identity["resolved_axes"]["robot"]
    construction = identity["resolved_axes"]["construction"]
    assert robot["joint_names"] == list(_smpl_live_axes()[0])
    assert robot["body_names"] == list(_smpl_live_axes()[1])
    assert robot["action_name"] == "control"
    assert robot["action_type"] == "NativeMujocoControlActionCfg"
    assert construction["task_row_mode"] == "source_frames"
    assert len(robot["joint_names"]) == 69
    assert len(identity["bundle_sha256"]) == 64


def test_motion_preset_owns_newton_native_smpl_composition() -> None:
    """Keep the public asset generic and place backend selection at the task boundary."""
    assert isinstance(SMPL_HUMANOID_CFG.spawn, sim_utils.MjcfFileCfg)
    assert SMPL_HUMANOID_CFG.articulation_root_prim_path == "/Geometry/Pelvis"
    assert isinstance(SMPL_MOTION_ARTICULATION_CFG.spawn, NewtonMjcfFileCfg)
    assert SMPL_MOTION_ARTICULATION_CFG.articulation_root_prim_path == "/humanoid"


def test_live_axis_selector_follows_concrete_action_ownership() -> None:
    """Axis ownership must dispatch on the resolved control type, not a preset label."""
    module = _module()
    smpl_cfg = resolve_presets(MotionImitationEnvCfg(), selected=motion_environment_axes("smpl_cmu"))
    g1_cfg = resolve_presets(MotionImitationEnvCfg(), selected=motion_environment_axes("g1_lafan"))

    assert module.motion_action_term_cfg(smpl_cfg)[0] == "control"
    assert module.motion_action_term_cfg(g1_cfg)[0] == "joint_position"
    assert module._motion_live_axes(smpl_cfg) == _smpl_live_axes()
    assert module._motion_live_axes(g1_cfg) == module.motion_g1_live_axes(g1_cfg)

    g1_cfg.actions.joint_position = ActionTermCfg(class_type="unsupported:Action")
    with pytest.raises(TypeError, match="Unsupported motion action ownership"):
        module._motion_live_axes(g1_cfg)


def test_resolved_configuration_closes_solver_ground_and_mdp_drift() -> None:
    """Hydra overrides must change identity without relying on source-byte drift."""
    baseline = _smpl_identity(resolve_presets(MotionImitationEnvCfg(), selected=motion_environment_axes("smpl_cmu")))
    configuration = baseline["resolved_configuration"]
    solver = configuration["sim"]["physics"]["solver_cfg"]
    ground = configuration["scene"]["ground"]["spawn"]
    assert solver["enable_native_ccd"] is False
    assert solver["enable_multiccd"] is False
    assert solver["integrator"] == "implicitfast"
    assert solver["tolerance"] == 1.0e-8
    assert ground["collision_props"]["margin"] == 0.001
    assert ground["collision_props"]["solimp"] == [0.99, 0.99, 0.003, 0.5, 2.0]
    assert ground["collision_props"]["solref"] == [0.015, 1.0]
    assert ground["physics_material"]["static_friction"] == 0.7
    assert {"actions", "commands", "events", "observations", "terminations"} <= set(configuration)

    native_ccd_cfg = resolve_presets(MotionImitationEnvCfg(), selected=motion_environment_axes("smpl_cmu"))
    native_ccd_cfg.sim.physics.solver_cfg.enable_native_ccd = True
    solimp_cfg = resolve_presets(MotionImitationEnvCfg(), selected=motion_environment_axes("smpl_cmu"))
    solimp_cfg.scene.ground.spawn.collision_props.solimp = (0.98, 0.99, 0.003, 0.5, 2.0)

    for changed in (_smpl_identity(native_ccd_cfg), _smpl_identity(solimp_cfg)):
        assert changed["resolved_configuration_sha256"] != baseline["resolved_configuration_sha256"]
        assert changed["bundle_sha256"] != baseline["bundle_sha256"]
        assert changed["python_sources"] == baseline["python_sources"]
        assert changed["runtime_dependencies"] == baseline["runtime_dependencies"]


def test_resolved_configuration_excludes_only_execution_placement() -> None:
    """Scale, device, trial seed, and artifact roots must not split one semantic identity."""
    baseline_cfg = resolve_presets(MotionImitationEnvCfg(), selected=motion_environment_axes("smpl_cmu"))
    baseline = _smpl_identity(baseline_cfg)
    moved_cfg = resolve_presets(MotionImitationEnvCfg(), selected=motion_environment_axes("smpl_cmu"))
    moved_cfg.scene.num_envs = 1024
    moved_cfg.seed = 9917
    moved_cfg.sim.device = "cpu"
    moved_cfg.log_dir = "/host/a"
    moved_cfg.sim.log_dir = "/host/b"
    moved_cfg.commands.motion.task_table.source_artifact_root = "/deployment/source"
    moved_cfg.commands.motion.task_table.reference_artifact_root = "/deployment/reference"

    moved = _smpl_identity(moved_cfg)

    assert moved["resolved_configuration"] == baseline["resolved_configuration"]
    assert moved["resolved_configuration_sha256"] == baseline["resolved_configuration_sha256"]
    assert moved["bundle_sha256"] == baseline["bundle_sha256"]


def test_runtime_identity_closes_concrete_external_build_and_newton_owners() -> None:
    """The evidence must distinguish incompatible Newton/Warp/Isaac Sim installations."""
    identity = _smpl_identity(resolve_presets(MotionImitationEnvCfg(), selected=motion_environment_axes("smpl_cmu")))
    runtime = identity["runtime_dependencies"]

    assert set(runtime) == {
        "h5py",
        "isaac_sim",
        "mujoco",
        "mujoco_warp",
        "newton",
        "newton_owners",
        "numpy",
        "torch",
        "warp",
    }
    assert runtime["isaac_sim"]["version"]
    assert len(runtime["isaac_sim"]["version_sha256"]) == 64
    assert set(runtime["newton_owners"]) == {
        "model_builder_add_mjcf",
        "model_builder_add_usd",
        "schema_resolver_mjc",
        "solver_mujoco",
    }
    assert all(len(owner["source_sha256"]) == 64 for owner in runtime["newton_owners"].values())
    assert len(identity["runtime_dependencies_sha256"]) == 64


@pytest.mark.parametrize(
    ("preset", "profile_packages"),
    (
        ("smpl_cmu", {"h5py", "mujoco", "mujoco_warp"}),
        ("g1_lafan", {"joblib"}),
        ("g1_cmu", {"h5py"}),
    ),
)
def test_runtime_identity_closes_preset_data_and_fall_pool_packages(
    preset: str,
    profile_packages: set[str],
) -> None:
    """Each preset must identify the packages that decode or augment its motion data."""
    runtime = _module()._runtime_dependencies(preset)
    expected = {"isaac_sim", "newton", "newton_owners", "numpy", "torch", "warp"} | profile_packages

    assert set(runtime) == expected
    package_fields = {"module_version", "distribution_version", "module_source_sha256"}
    for name in {"newton", "numpy", "warp"} | profile_packages:
        assert set(runtime[name]) == package_fields
        assert runtime[name]["module_version"]
        assert len(runtime[name]["module_source_sha256"]) == 64


def test_runtime_identity_rejects_an_unknown_preset() -> None:
    """Runtime identity must not silently assign the common package set to a new preset."""
    with pytest.raises(ValueError, match="Unsupported motion dependency preset"):
        _module()._runtime_dependencies("unknown")


def test_g1_identity_records_named_transition_evidence_without_command_history() -> None:
    """The environment receipt exposes stateless evidence and no command transition owner."""
    cfg = resolve_presets(MotionImitationEnvCfg(), selected=motion_environment_axes("g1_cmu"))
    environment, _composition = _g1_cmu_identities(cfg)
    runtime = environment["resolved_axes"]["runtime"]

    assert runtime["transition_evidence_names"] == [
        "penalty_torques",
        "penalty_action_rate",
        "limits_dof_pos",
        "limits_torque",
        "penalty_undesired_contact",
        "penalty_feet_ori",
        "penalty_ankle_roll",
        "penalty_slippage",
    ]
    assert runtime["transition_evidence_concatenated"] is False
    assert "transition_factory" not in runtime
    assert "episode_length_steps" not in runtime
    assert not hasattr(cfg.observations, "history_actor")
    assert not hasattr(cfg.commands.motion.payload, "transition_factory")


def test_composition_identity_excludes_mdp_runtime_but_closes_trajectory_inputs(tmp_path: Path) -> None:
    """Pure retarget identity must close data construction without owning the simulator MDP."""
    baseline_cfg = resolve_presets(MotionImitationEnvCfg(), selected=motion_environment_axes("g1_cmu"))
    baseline_environment, baseline_composition = _g1_cmu_identities(baseline_cfg)

    runtime_cfg = resolve_presets(MotionImitationEnvCfg(), selected=motion_environment_axes("g1_cmu"))
    runtime_cfg.compute_final_obs = not runtime_cfg.compute_final_obs
    runtime_environment, runtime_composition = _g1_cmu_identities(runtime_cfg)
    assert runtime_environment["bundle_sha256"] != baseline_environment["bundle_sha256"]
    assert runtime_composition == baseline_composition

    source_cfg = resolve_presets(MotionImitationEnvCfg(), selected=motion_environment_axes("g1_cmu"))
    source_cfg.commands.motion.task_table.source.evaluation.source_content_sha256 = "b" * 64
    source_environment, source_composition = _g1_cmu_identities(source_cfg)
    assert source_environment["bundle_sha256"] != baseline_environment["bundle_sha256"]
    assert source_composition["bundle_sha256"] != baseline_composition["bundle_sha256"]

    _, changed_builder = _g1_cmu_identities(baseline_cfg, frame_builder_identity_sha256="c" * 64)
    assert changed_builder["bundle_sha256"] != baseline_composition["bundle_sha256"]

    reference_root = tmp_path / "reference"
    reference_model = reference_root / "humanoidverse/data/robots/g1/g1_29dof.xml"
    reference_model.parent.mkdir(parents=True)
    reference_model.write_text("<mujoco model='changed-g1'/>", encoding="utf-8")
    asset_environment, asset_composition = _g1_cmu_identities(
        baseline_cfg,
        reference_artifact_root=reference_root,
    )
    assert asset_environment["bundle_sha256"] != baseline_environment["bundle_sha256"]
    assert asset_composition["bundle_sha256"] != baseline_composition["bundle_sha256"]


def test_g1_live_axes_require_exactly_one_actuator_owner() -> None:
    module = _module()
    missing = resolve_presets(MotionImitationEnvCfg(), selected=motion_environment_axes("g1_lafan"))
    missing.scene.robot.actuators = {}
    with pytest.raises(ValueError, match="exactly one actuator"):
        module.motion_g1_live_axes(missing)

    duplicate = resolve_presets(MotionImitationEnvCfg(), selected=motion_environment_axes("g1_lafan"))
    duplicate.scene.robot.actuators["duplicate"] = next(iter(duplicate.scene.robot.actuators.values()))
    with pytest.raises(ValueError, match="exactly one actuator"):
        module.motion_g1_live_axes(duplicate)
