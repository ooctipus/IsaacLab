# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Content identity for every task-owned motion-environment runtime dependency."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import sys
from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab.managers import ActionTermCfg

_SCHEMA = "forward_backward_phase3_motion_environment_dependency_identity_v9"
_RETAINED_SCHEMAS = frozenset(
    {
        "forward_backward_phase3_motion_environment_dependency_identity_v7",
        _SCHEMA,
    }
)
_COMPOSITION_SCHEMA = "forward_backward_phase3_motion_composition_dependency_identity_v1"
_PROFILE_ENVIRONMENT_AXES = {
    "smpl_cmu": frozenset(
        ("smpl", "cmu", "newton_mjwarp", "timing_sim450_control30_horizon300", "sampling_source_rows")
    ),
    "g1_lafan": frozenset(
        (
            "g1",
            "lafan",
            "physx",
            "timing_sim200_control50_horizon501",
            "sampling_clip_time",
            "evidence_physical_auxiliary",
            "randomization_physics_observation_pose_push",
        )
    ),
    "g1_cmu": frozenset(
        (
            "g1",
            "cmu",
            "physx",
            "timing_sim200_control50_horizon501",
            "sampling_clip_time",
            "evidence_physical_auxiliary",
            "randomization_physics_observation_pose_push",
        )
    ),
}
_PROFILE_RUNNER_AXES = {
    "smpl_cmu": frozenset(
        (
            "helpers_discriminator",
            "tracking_off",
            "model_plain_2x1024",
            "replay_transition_uniform_2m",
            "schedule_50x10_5m",
            "optimization_lr1e4_implied0p1_actor0p01",
            "context_online_10k",
            "exploration_std0p2_range1",
            "seed_0",
            "expert_clock_source_rows",
        )
    ),
    "g1_lafan": frozenset(
        (
            "helpers_discriminator_auxiliary",
            "tracking_reset_frame",
            "tracking_interval_9p6m",
            "model_residual_6x1024",
            "replay_episode_uniform_5120k",
            "schedule_1024x1_211p2m",
            "optimization_lr3e4_implied0_actor0p05",
            "context_expert_half_8192",
            "exploration_std0p05_range5",
            "seed_4728",
            "expert_clock_50hz",
        )
    ),
    "g1_cmu": frozenset(
        (
            "helpers_discriminator_auxiliary",
            "tracking_reset_frame",
            "tracking_interval_9p6m",
            "model_residual_6x1024",
            "replay_episode_uniform_5120k",
            "schedule_1024x1_211p2m",
            "optimization_lr3e4_implied0_actor0p05",
            "context_expert_half_8192",
            "exploration_std0p05_range5",
            "seed_4728",
            "expert_clock_50hz",
        )
    ),
}
_SEMANTIC_CONFIGURATION_AXES = (
    "actions",
    "commands",
    "compute_final_obs",
    "curriculum",
    "decimation",
    "episode_length_s",
    "events",
    "is_finite_horizon",
    "num_rerenders_on_reset",
    "observations",
    "rerender_on_reset",
    "rewards",
    "scene",
    "sim",
    "terminations",
)
_EXECUTION_CONFIGURATION_AXES = {
    "export_io_descriptors",
    "isaac_teleop",
    "log_dir",
    "recorders",
    "seed",
    "teleop_devices",
    "ui_window_class_type",
    "video_recorder",
    "viewer",
    "wait_for_textures",
    "xr",
}
_COMMON_MODULES = (
    "isaaclab.envs.mdp.terminations",
    "isaaclab.sim.schemas.schemas",
    "isaaclab.sim.schemas.schemas_cfg",
    "isaaclab.sim.spawners.materials.physics_materials",
    "isaaclab.sim.spawners.materials.physics_materials_cfg",
    "isaaclab.sim.spawners.shapes.shapes",
    "isaaclab.sim.spawners.shapes.shapes_cfg",
    "isaaclab.sim.utils.prims",
    "isaaclab.utils.math",
    "isaaclab.envs.manager_based_env",
    "isaaclab.envs.manager_based_env_cfg",
    "isaaclab.envs.manager_based_rl_env",
    "isaaclab.envs.manager_based_rl_env_cfg",
    "isaaclab.managers.action_manager",
    "isaaclab.managers.command_manager",
    "isaaclab.managers.curriculum_manager",
    "isaaclab.managers.event_manager",
    "isaaclab.managers.manager_base",
    "isaaclab.managers.observation_manager",
    "isaaclab.managers.reward_manager",
    "isaaclab.managers.termination_manager",
    "isaaclab.assets.articulation.articulation_cfg",
    "isaaclab.assets.articulation.base_articulation",
    "isaaclab.assets.articulation.base_articulation_data",
    "isaaclab.physics.physics_manager",
    "isaaclab.physics.physics_manager_cfg",
    "isaaclab.sim.simulation_context",
    "isaaclab_tasks.core.multi_task.kinematics.kinematic_tree",
    "isaaclab_tasks.core.multi_task.kinematics.newton_kinematics",
    "isaaclab_tasks.core.multi_task.kinematics.newton_kinematics_cfg",
    "isaaclab_tasks.core.multi_task.motion_env_cfg",
    "isaaclab_tasks.core.multi_task.motion.identity",
    "isaaclab_tasks.core.multi_task.motion.data.clip_index",
    "isaaclab_tasks.core.multi_task.motion.data.source",
    "isaaclab_tasks.core.multi_task.motion.data.skeleton",
    "isaaclab_tasks.core.multi_task.motion.mdp.commands.commands_cfg",
    "isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_state_payload",
    "isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table",
    "isaaclab_tasks.core.multi_task.mdp.commands.state_command.state_command",
    "isaaclab_tasks.core.multi_task.mdp.commands.state_command.state_command_cfg",
)
_COMPOSITION_COMMON_MODULES = (
    "isaaclab.utils.math",
    "isaaclab_tasks.core.multi_task.kinematics.kinematic_tree",
    "isaaclab_tasks.core.multi_task.kinematics.newton_kinematics",
    "isaaclab_tasks.core.multi_task.kinematics.newton_kinematics_cfg",
    "isaaclab_tasks.core.multi_task.motion_env_cfg",
    "isaaclab_tasks.core.multi_task.motion.identity",
    "isaaclab_tasks.core.multi_task.motion.data.clip_index",
    "isaaclab_tasks.core.multi_task.motion.data.source",
    "isaaclab_tasks.core.multi_task.motion.data.skeleton",
    "isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table",
)
_G1_MODULES = (
    "isaaclab_tasks.core.multi_task.mdp.curriculums",
    "isaaclab_tasks.core.multi_task.mdp.events",
    "isaaclab_tasks.core.multi_task.mdp.rewards",
    "isaaclab_tasks.core.multi_task.motion.robots.g1.actions",
    "isaaclab_tasks.core.multi_task.motion.robots.g1.actions_cfg",
    "isaaclab_tasks.core.multi_task.motion.robots.g1.articulation",
    "isaaclab_tasks.core.multi_task.motion.robots.g1.frames",
    "isaaclab_tasks.core.multi_task.motion.robots.g1.observations",
    "isaaclab_tasks.core.multi_task.motion.robots.g1.reference",
    "isaaclab_tasks.core.multi_task.motion.robots.g1.reset",
)
_SMPL_MODULES = (
    "isaaclab_assets.robots.smpl.smpl_cfg",
    "isaaclab_assets.robots.smpl.smpl_constants",
    "isaaclab_newton.sim.schemas.schemas_cfg",
    "isaaclab_newton.sim.spawners.mjcf.mjcf",
    "isaaclab_newton.sim.spawners.mjcf.mjcf_cfg",
    "isaaclab_tasks.core.multi_task.mdp.native_mujoco_action",
    "isaaclab_tasks.core.multi_task.mdp.native_mujoco_action_cfg",
    "isaaclab_tasks.core.multi_task.motion.robots.smpl.articulation",
    # bfm-env-20260805 campaign patch: HEAD merged smpl frames into
    # reference.py + reference_warp.py; hash the warp kernel module instead.
    "isaaclab_tasks.core.multi_task.motion.robots.smpl.reference_warp",
    "isaaclab_tasks.core.multi_task.motion.robots.smpl.observations",
    "isaaclab_tasks.core.multi_task.motion.robots.smpl.reference",
    "isaaclab_tasks.core.multi_task.motion.robots.smpl.reset",
)
_PHYSX_MODULES = (
    "isaaclab_physx.assets.articulation.articulation",
    "isaaclab_physx.assets.articulation.articulation_data",
    "isaaclab_physx.assets.articulation.kernels",
    "isaaclab_physx.physics.physx_manager",
    "isaaclab_physx.physics.physx_manager_cfg",
)
_NEWTON_MJWARP_MODULES = (
    "isaaclab_newton.actuators.adapter",
    "isaaclab_newton.actuators.kernels",
    "isaaclab_newton.assets.articulation.articulation",
    "isaaclab_newton.assets.articulation.articulation_data",
    "isaaclab_newton.assets.articulation.kernels",
    "isaaclab_newton.cloner.newton_clone_utils",
    "isaaclab_newton.cloner.replicate",
    "isaaclab_newton.physics.mjwarp_manager",
    "isaaclab_newton.physics.mjwarp_manager_cfg",
    "isaaclab_newton.physics.newton_collision_cfg",
    "isaaclab_newton.physics.newton_manager",
    "isaaclab_newton.physics.newton_manager_cfg",
    "isaaclab_newton.physics.visualization_builder",
)


def motion_environment_axes(profile: str) -> frozenset[str]:
    """Return explicit robot, dataset, and backend axes for an experiment profile."""
    try:
        return _PROFILE_ENVIRONMENT_AXES[profile]
    except KeyError as error:
        raise ValueError(f"Unsupported motion experiment profile: {profile!r}.") from error


def motion_runner_axes(profile: str) -> frozenset[str]:
    """Return explicit learner-policy axes for an experiment profile."""
    try:
        return _PROFILE_RUNNER_AXES[profile]
    except KeyError as error:
        raise ValueError(f"Unsupported motion experiment profile: {profile!r}.") from error


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one required regular file."""
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Motion dependency must be a regular non-symbolic file: {path}.")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _module_source_path(module_name: str) -> Path:
    """Resolve one explicit Python module uniquely without importing the target."""
    parts = module_name.split(".")
    if not parts or any(not part.isidentifier() for part in parts):
        raise ValueError(f"Motion dependency module name is invalid: {module_name!r}.")
    matches: set[Path] = set()
    symbolic: list[Path] = []

    def collect(candidate: Path) -> None:
        if candidate.is_symlink():
            symbolic.append(candidate.absolute())
        elif candidate.is_file():
            matches.add(candidate.resolve())

    relative = Path(*parts).with_suffix(".py")
    for entry in sys.path:
        if isinstance(entry, str):
            collect(Path(entry or ".").expanduser() / relative)

    spec = importlib.util.find_spec(parts[0])
    if spec is not None:
        if len(parts) == 1:
            if spec.origin not in (None, "built-in", "frozen"):
                collect(Path(spec.origin))
        else:
            remainder = Path(*parts[1:]).with_suffix(".py")
            for location in spec.submodule_search_locations or ():
                root = Path(location).expanduser()
                collect(root / remainder)
                collect(root / parts[0] / remainder)

    if symbolic:
        paths = ", ".join(str(path) for path in sorted(set(symbolic)))
        raise ValueError(f"Motion dependency module has symbolic source: {module_name!r}: {paths}.")
    if not matches:
        raise ModuleNotFoundError(f"Motion dependency module source is missing: {module_name!r}.")
    if len(matches) != 1:
        paths = ", ".join(str(path) for path in sorted(matches))
        raise RuntimeError(f"Motion dependency module source is ambiguous: {module_name!r}: {paths}.")
    return matches.pop()


def _native_skeleton_module(source_cfg: object) -> str:
    """Name the coordinates module declaring the native source skeleton.

    bfm-env-20260805 campaign patch: HEAD removed MotionSourceCfg.skeleton_factory;
    native decoders own their skeleton in a sibling ``*_coordinates`` module.
    """
    module_name = f"{source_cfg.open_source.__module__}_coordinates"
    _module_source_path(module_name)
    return module_name


def _python_sources(
    preset: str,
    importer_type: type,
    frame_builder_factory: Callable,
    frame_builder_type: type,
) -> dict[str, str]:
    """Hash the explicit source owners of construction, reset, step, and capture."""
    modules = list(_COMMON_MODULES)
    modules.extend(_G1_MODULES if preset in ("g1_lafan", "g1_cmu") else _SMPL_MODULES)
    modules.extend(_PHYSX_MODULES if preset in ("g1_lafan", "g1_cmu") else _NEWTON_MJWARP_MODULES)
    modules.extend((importer_type.__module__, frame_builder_factory.__module__, frame_builder_type.__module__))
    names = tuple(sorted(set(modules)))
    sources = {name: _sha256(_module_source_path(name)) for name in names}
    sources["motion_environment_identity"] = _sha256(Path(__file__).resolve())
    return dict(sorted(sources.items()))


def _composition_python_sources(
    preset: str,
    importer_type: type,
    frame_builder_type: type,
    table_cfg: object,
) -> dict[str, str]:
    """Hash only code that converts declared source rows into target-robot facts."""
    modules = list(_COMPOSITION_COMMON_MODULES)
    if preset in ("g1_lafan", "g1_cmu"):
        modules.extend(_G1_MODULES)
    else:
        modules.extend(
            (
                "isaaclab_assets.robots.smpl.smpl_constants",
                "isaaclab_tasks.core.multi_task.motion.robots.smpl.articulation",
                # bfm-env-20260805 campaign patch: smpl frames merged into
                # reference.py + reference_warp.py at HEAD.
                "isaaclab_tasks.core.multi_task.motion.robots.smpl.reference_warp",
                "isaaclab_tasks.core.multi_task.motion.robots.smpl.reference",
            )
        )
    source_cfg = table_cfg.source
    # bfm-env-20260805 campaign patch: HEAD moved the per-robot construction
    # factories under table_cfg.target_kinematics (target_factory + kinematics
    # build cfg) and resolves source skeletons from the decoder's coordinates
    # module instead of a MotionSourceCfg.skeleton_factory field.
    modules.extend(
        (
            importer_type.__module__,
            table_cfg.target_kinematics.target_factory.__module__,
            frame_builder_type.__module__,
            type(table_cfg.target_kinematics.kinematics).__module__,
            source_cfg.open_source.__module__,
            _native_skeleton_module(source_cfg),
        )
    )
    names = tuple(sorted(set(modules)))
    sources = {name: _sha256(_module_source_path(name)) for name in names}
    sources["motion_environment_identity"] = _sha256(Path(__file__).resolve())
    return dict(sorted(sources.items()))


def _simulation_robot_assets(robot_cfg: object) -> dict[str, str]:
    """Hash the selected packaged source asset, never a generated cache."""
    spawn = getattr(robot_cfg, "spawn", None)
    asset_path = getattr(spawn, "asset_path", None)
    if isinstance(asset_path, str) and asset_path:
        path = Path(asset_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError("The selected simulation source asset does not exist.")
        return {f"simulation/{path.name}": _sha256(path)}

    usd_path = getattr(spawn, "usd_path", None)
    if not isinstance(usd_path, str) or not usd_path:
        raise ValueError("Motion evidence requires a concrete simulation source or USD path.")
    path = Path(usd_path).expanduser().resolve()
    root = path.parent
    files = tuple(
        sorted(
            candidate
            for candidate in root.rglob("*")
            if candidate.is_file() and candidate.suffix in {".usd", ".usda", ".usdc"}
        )
    )
    if path not in files:
        raise ValueError("The selected simulation USD is absent from its packaged asset directory.")
    return {f"simulation/{candidate.relative_to(root).as_posix()}": _sha256(candidate) for candidate in files}


def _reference_assets(preset: str, reference_artifact_root: str | Path | None) -> dict[str, str]:
    """Hash the exact reference MJCF used to construct trajectory frames."""
    if preset == "smpl_cmu":
        from isaaclab_assets.robots.smpl.smpl_constants import SMPL_HUMENV_MJCF_PATH

        path = Path(SMPL_HUMENV_MJCF_PATH).resolve()
    else:
        if reference_artifact_root is None or not str(reference_artifact_root):
            from isaaclab_tasks.core.multi_task.motion.robots.g1.reference import G1_REFERENCE_MJCF_SHA256

            return {"reference/g1_29dof.xml": G1_REFERENCE_MJCF_SHA256}
        path = Path(reference_artifact_root).expanduser().resolve() / "humanoidverse/data/robots/g1/g1_29dof.xml"
    return {f"reference/{path.name}": _sha256(path)}


def _json_hash(value: object) -> str:
    """Return the canonical JSON SHA-256 for one identity projection."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(name: str, value: object) -> str:
    """Return one validated lowercase SHA-256 digest."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _callable_name(value: object) -> str:
    """Return one stable module-qualified callable name."""
    module = getattr(value, "__module__", None)
    name = getattr(value, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(name, str):
        raise TypeError(f"Resolved motion axis is not a named callable: {value!r}.")
    return f"{module}:{name}"


def _runtime_package_identity(module: object, distribution_name: str) -> dict[str, str | None]:
    """Return version and imported-module source identity for one runtime package."""
    version = getattr(module, "__version__", None)
    if not isinstance(version, str) or not version:
        raise ValueError(f"Runtime package {distribution_name!r} does not expose a version.")
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str):
        raise ValueError(f"Runtime package {distribution_name!r} does not expose a source file.")
    try:
        distribution_version = importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        distribution_version = None
    return {
        "module_version": version,
        "distribution_version": distribution_version,
        "module_source_sha256": _sha256(Path(module_file).resolve()),
    }


def _torch_runtime_identity(torch: object) -> dict[str, str | None]:
    """Return the shared Torch package, CUDA, and source identity."""
    identity: dict[str, str | None] = _runtime_package_identity(torch, "torch")
    identity.update(
        {
            "cuda_version": torch.version.cuda,
            "git_version": torch.version.git_version,
        }
    )
    return identity


def _runtime_owner_identity(owner: object) -> dict[str, str]:
    """Return module-qualified and source-byte identity for one concrete runtime owner."""
    source = inspect.getsourcefile(owner)
    module = getattr(owner, "__module__", None)
    name = getattr(owner, "__qualname__", None)
    if source is None or not isinstance(module, str) or not isinstance(name, str):
        raise ValueError(f"Runtime owner does not expose Python source identity: {owner!r}.")
    return {
        "owner": f"{module}:{name}",
        "source_sha256": _sha256(Path(source).resolve()),
    }


def _isaac_sim_build_identity() -> dict[str, str]:
    """Return the exact Isaac Sim release build from its installed VERSION file."""
    import isaacsim

    package_file = getattr(isaacsim, "__file__", None)
    if not isinstance(package_file, str):
        raise ValueError("Isaac Sim does not expose an installed package path.")
    version_file = next(
        (parent / "VERSION" for parent in Path(package_file).resolve().parents if (parent / "VERSION").is_file()),
        None,
    )
    if version_file is None:
        raise FileNotFoundError("Isaac Sim installation does not contain a VERSION file.")
    version = version_file.read_text(encoding="utf-8").strip()
    if not version:
        raise ValueError("Isaac Sim VERSION file is empty.")
    return {"version": version, "version_sha256": _sha256(version_file)}


def _runtime_dependencies(preset: str) -> dict[str, object]:
    """Return build identity for the external runtime that executes one preset."""
    if preset not in ("smpl_cmu", "g1_lafan", "g1_cmu"):
        raise ValueError(f"Unsupported motion dependency preset: {preset!r}.")

    import newton
    import numpy
    import torch
    import warp

    newton_owners = {
        "model_builder_add_mjcf": _runtime_owner_identity(newton.ModelBuilder.add_mjcf),
    }
    dependencies: dict[str, object] = {
        "isaac_sim": _isaac_sim_build_identity(),
        "newton": _runtime_package_identity(newton, "newton"),
        "newton_owners": newton_owners,
        "numpy": _runtime_package_identity(numpy, "numpy"),
        "torch": _torch_runtime_identity(torch),
        "warp": _runtime_package_identity(warp, "warp-lang"),
    }
    if preset in ("smpl_cmu", "g1_cmu"):
        import h5py

        dependencies["h5py"] = _runtime_package_identity(h5py, "h5py")
    if preset == "g1_lafan":
        import joblib

        dependencies["joblib"] = _runtime_package_identity(joblib, "joblib")
    if preset == "smpl_cmu":
        import mujoco
        import mujoco_warp
        from newton.solvers import SolverMuJoCo
        from newton.usd import SchemaResolverMjc

        dependencies["mujoco"] = _runtime_package_identity(mujoco, "mujoco")
        dependencies["mujoco_warp"] = _runtime_package_identity(mujoco_warp, "mujoco-warp")
        newton_owners.update(
            {
                "model_builder_add_usd": _runtime_owner_identity(newton.ModelBuilder.add_usd),
                "schema_resolver_mjc": _runtime_owner_identity(SchemaResolverMjc),
                "solver_mujoco": _runtime_owner_identity(SolverMuJoCo),
            }
        )
    return dependencies


def motion_composition_runtime_dependencies(preset: str) -> dict[str, object]:
    """Return external builds that decode and construct target motion facts.

    Args:
        preset: One of ``smpl_cmu``, ``g1_lafan``, or ``g1_cmu``.

    Returns:
        Exact package and concrete Newton MJCF-loader identities.
    """
    if preset not in ("smpl_cmu", "g1_lafan", "g1_cmu"):
        raise ValueError(f"Unsupported motion dependency preset: {preset!r}.")

    import newton
    import numpy
    import torch
    import warp

    dependencies: dict[str, object] = {
        "newton": _runtime_package_identity(newton, "newton"),
        "newton_owners": {
            "model_builder_add_mjcf": _runtime_owner_identity(newton.ModelBuilder.add_mjcf),
        },
        "numpy": _runtime_package_identity(numpy, "numpy"),
        "torch": _torch_runtime_identity(torch),
        "warp": _runtime_package_identity(warp, "warp-lang"),
    }
    if preset in ("smpl_cmu", "g1_cmu"):
        import h5py

        dependencies["h5py"] = _runtime_package_identity(h5py, "h5py")
    elif preset == "g1_lafan":
        import joblib

        dependencies["joblib"] = _runtime_package_identity(joblib, "joblib")
    return dependencies


def _canonical_configuration_value(value: object, path: str) -> object:
    """Convert one resolved config value into deterministic JSON data."""
    if isinstance(value, Enum):
        return _canonical_configuration_value(value.value, path)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical_configuration_value(getattr(value, field.name), f"{path}.{field.name}")
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        if any(not isinstance(name, str) for name in value):
            raise TypeError(f"Resolved configuration mapping has a non-string key at {path}.")
        return {name: _canonical_configuration_value(child, f"{path}.{name}") for name, child in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical_configuration_value(child, f"{path}[{index}]") for index, child in enumerate(value)]
    if isinstance(value, (set, frozenset)):
        children = [_canonical_configuration_value(child, f"{path}[]") for child in value]
        return sorted(children, key=lambda child: json.dumps(child, sort_keys=True, separators=(",", ":")))
    if isinstance(value, slice):
        return {
            "start": _canonical_configuration_value(value.start, f"{path}.start"),
            "stop": _canonical_configuration_value(value.stop, f"{path}.stop"),
            "step": _canonical_configuration_value(value.step, f"{path}.step"),
        }
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Resolved configuration contains a non-finite float at {path}.")
        return value
    if callable(value):
        return _callable_name(value)
    raise TypeError(f"Resolved configuration contains unsupported {type(value).__name__} at {path}.")


def _pop_configuration_path(root: dict[str, object], path: tuple[str, ...]) -> object:
    """Remove one required deployment-only leaf from a copied config mapping."""
    current: dict[str, object] = root
    for name in path[:-1]:
        child = current.get(name)
        if not isinstance(child, dict):
            raise ValueError(f"Resolved configuration path is missing: {'.'.join(path)}.")
        current = child
    if path[-1] not in current:
        raise ValueError(f"Resolved configuration path is missing: {'.'.join(path)}.")
    return current.pop(path[-1])


def _resolved_configuration(cfg: object) -> dict[str, object]:
    """Project every MDP/physics axis while excluding only execution placement."""
    to_dict = getattr(cfg, "to_dict", None)
    if not callable(to_dict):
        raise TypeError("Resolved motion configuration must expose to_dict().")
    raw = to_dict()
    if not isinstance(raw, dict):
        raise TypeError("Resolved motion configuration to_dict() must return a mapping.")
    classified = set(_SEMANTIC_CONFIGURATION_AXES) | _EXECUTION_CONFIGURATION_AXES
    if set(raw) != classified:
        missing = tuple(sorted(set(_SEMANTIC_CONFIGURATION_AXES).difference(raw)))
        unknown = tuple(sorted(set(raw).difference(classified)))
        raise ValueError(f"Motion configuration axes require classification; missing={missing}, unknown={unknown}.")
    projection = {name: raw[name] for name in _SEMANTIC_CONFIGURATION_AXES}
    for path in (
        ("scene", "num_envs"),
        ("sim", "device"),
        ("sim", "log_dir"),
        ("commands", "motion", "task_table", "source_artifact_root"),
        # bfm-env-20260805 campaign patch: HEAD renamed reference_artifact_root
        # to target_artifact_root on MotionTaskTableCfg.
        ("commands", "motion", "task_table", "target_artifact_root"),
    ):
        _pop_configuration_path(projection, path)

    robot_spawn = projection["scene"]["robot"]["spawn"]
    robot_source_path = robot_spawn.pop("asset_path", None)
    if robot_source_path is not None:
        if not isinstance(robot_source_path, str) or not robot_source_path:
            raise ValueError("Resolved motion robot must expose a concrete source path.")
        for cache_field in ("force_usd_conversion", "make_instanceable", "usd_dir", "usd_file_name"):
            robot_spawn.pop(cache_field, None)
        robot_spawn["source_asset_file"] = Path(robot_source_path).name
    else:
        robot_usd_path = robot_spawn.pop("usd_path")
        if not isinstance(robot_usd_path, str) or not robot_usd_path:
            raise ValueError("Resolved motion robot must expose a concrete USD path.")
        robot_spawn["usd_asset_file"] = Path(robot_usd_path).name

    solver_cfg = projection["sim"]["physics"].get("solver_cfg")
    if isinstance(solver_cfg, dict) and "save_to_mjcf" in solver_cfg:
        solver_cfg.pop("save_to_mjcf")
    value = _canonical_configuration_value(projection, "environment")
    if not isinstance(value, dict):
        raise TypeError("Canonical resolved motion configuration must be a mapping.")
    return value


def motion_g1_live_axes(cfg: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Derive ordered G1 joint and body axes from concrete articulation ownership."""
    from isaaclab_tasks.core.multi_task.motion.robots.g1.articulation import (
        G1_BEHAVIOR_BODY_NAMES,
        G1_BEHAVIOR_JOINT_NAMES,
    )

    actuators = cfg.scene.robot.actuators
    if len(actuators) != 1:
        raise ValueError("The G1 articulation must declare exactly one actuator group.")
    actuator = next(iter(actuators.values()))
    joint_names = tuple(actuator.joint_names_expr)
    if len(joint_names) != len(G1_BEHAVIOR_JOINT_NAMES) or set(joint_names) != set(G1_BEHAVIOR_JOINT_NAMES):
        raise ValueError("The resolved G1 articulation joint axis differs from the robot behavior axis.")
    body_by_joint = dict(zip(G1_BEHAVIOR_JOINT_NAMES, G1_BEHAVIOR_BODY_NAMES[1:], strict=True))
    body_names = (G1_BEHAVIOR_BODY_NAMES[0], *(body_by_joint[name] for name in joint_names))
    if len(set(body_names)) != len(G1_BEHAVIOR_BODY_NAMES):
        raise ValueError("The resolved G1 joint axis does not map one-to-one onto physical bodies.")
    return joint_names, body_names


def motion_action_term_cfg(cfg: object) -> tuple[str, ActionTermCfg]:
    """Return the one action term declared by a resolved motion environment."""
    from isaaclab.managers import ActionTermCfg

    actions = cfg.actions
    if not is_dataclass(actions):
        raise TypeError("Resolved motion actions must be a configclass.")
    action_terms = tuple(
        (field.name, value)
        for field in fields(actions)
        if isinstance(value := getattr(actions, field.name), ActionTermCfg)
    )
    if len(action_terms) != 1:
        raise ValueError(f"Resolved motion actions must declare exactly one action term; got {len(action_terms)}.")
    return action_terms[0]


def _smpl_live_axes(cfg: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return concrete axes owned by the packaged native SMPL asset."""
    from isaaclab_assets.robots.smpl.smpl_constants import MUJOCO_BODY_NAMES

    if cfg.scene.robot.actuators:
        raise ValueError("The native SMPL asset must be the sole owner of control and passive joint terms.")
    body_names = MUJOCO_BODY_NAMES
    joint_names = tuple(f"{body}_x_{body}_y_{body}_z:{component}" for body in body_names[1:] for component in range(3))
    if len(joint_names) != 69 or len(set(joint_names)) != len(joint_names):
        raise ValueError("The packaged SMPL articulation must declare 69 unique simulator joints.")
    if len(body_names) != 24 or len(set(body_names)) != len(body_names):
        raise ValueError("The packaged SMPL articulation must declare 24 unique simulator bodies.")
    return joint_names, body_names


def _motion_live_axes(cfg: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve live axes from the concrete action/control ownership boundary."""
    from isaaclab_tasks.core.multi_task.mdp import NativeMujocoControlActionCfg
    from isaaclab_tasks.core.multi_task.motion.robots.g1.actions_cfg import G1JointPositionActionCfg

    _action_name, action_cfg = motion_action_term_cfg(cfg)
    if isinstance(action_cfg, NativeMujocoControlActionCfg):
        return _smpl_live_axes(cfg)
    if isinstance(action_cfg, G1JointPositionActionCfg):
        return motion_g1_live_axes(cfg)
    raise TypeError(f"Unsupported motion action ownership: {type(action_cfg).__name__}.")


def _resolved_axes(preset: str, cfg: object, frame_builder_type: type) -> dict[str, Any]:
    """Project the resolved environment configuration into host-independent facts."""
    table_cfg = cfg.commands.motion.task_table
    source_cfg = table_cfg.source
    payload_cfg = cfg.commands.motion.payload
    transition_evidence = getattr(cfg.observations, "transition", None)
    timeout = round(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation))
    if timeout < 1:
        raise ValueError("The resolved motion horizon must contain at least one applied action.")
    evidence_names = None
    if transition_evidence is not None:
        if not is_dataclass(transition_evidence):
            raise TypeError("Transition evidence configuration must be a configclass.")
        evidence_names = [
            field.name
            for field in fields(transition_evidence)
            if type(getattr(transition_evidence, field.name)).__name__ == "ObservationTermCfg"
        ]
    joint_names, body_names = _motion_live_axes(cfg)
    action_name, action_cfg = motion_action_term_cfg(cfg)
    robot_asset_path = getattr(cfg.scene.robot.spawn, "asset_path", None)
    if not isinstance(robot_asset_path, str) or not robot_asset_path:
        robot_asset_path = cfg.scene.robot.spawn.usd_path

    if not joint_names or len(set(joint_names)) != len(joint_names):
        raise ValueError("The resolved articulation must declare one unique ordered joint axis.")
    return {
        "preset": preset,
        "robot": {
            "prim_path": cfg.scene.robot.prim_path,
            "asset_file": Path(robot_asset_path).name,
            "joint_names": list(joint_names),
            "body_names": list(body_names),
            "action_name": action_name,
            "action_type": type(action_cfg).__name__,
        },
        "source": {
            "identifier": source_cfg.identifier,
            "format": source_cfg.format,
            "semantic_level": source_cfg.semantic_level,
            "train_content_sha256": source_cfg.train.source_content_sha256,
            "evaluation_content_sha256": source_cfg.evaluation.source_content_sha256,
        },
        "construction": {
            # bfm-env-20260805 campaign patch: HEAD owns these under target_kinematics.
            "frame_builder_factory": _callable_name(table_cfg.target_kinematics.target_factory),
            "frame_builder_type": _callable_name(frame_builder_type),
            "task_row_mode": table_cfg.task_row_mode,
            "reference_kinematics_factory": _callable_name(type(table_cfg.target_kinematics.kinematics)),
        },
        "sampling": {
            "reset_sources": [[name, probability] for name, probability in payload_cfg.reset_sources],
        },
        "runtime": {
            "physics_dt_seconds": cfg.sim.dt,
            "control_decimation": cfg.decimation,
            "episode_length_seconds": cfg.episode_length_s,
            "applied_actions_before_timeout": timeout,
            "reset_transform_factory": _callable_name(payload_cfg.reset_transform_factory),
            "root_velocity_frame": payload_cfg.root_velocity_frame,
            "transition_evidence_names": evidence_names,
            "transition_evidence_concatenated": (
                None if transition_evidence is None else transition_evidence.concatenate_terms
            ),
            "compute_final_obs": cfg.compute_final_obs,
        },
    }


def _resolved_composition(
    preset: str,
    cfg: object,
    frame_builder_type: type,
    frame_builder_identity_sha256: str,
) -> dict[str, object]:
    """Project only source-to-target trajectory construction into portable facts."""
    table_cfg = cfg.commands.motion.task_table
    source_cfg = table_cfg.source
    source_skeleton = source_cfg.build_skeleton()
    joint_names, body_names = _motion_live_axes(cfg)
    if not joint_names or len(set(joint_names)) != len(joint_names):
        raise ValueError("The target trajectory must declare one unique ordered joint axis.")
    if not body_names or len(set(body_names)) != len(body_names):
        raise ValueError("The target trajectory must declare one unique ordered body axis.")

    def split_value(split: object) -> dict[str, object]:
        return {
            "name": split.name,
            "artifact": split.artifact,
            "artifact_sha256": split.artifact_sha256,
            "source_content_sha256": split.source_content_sha256,
            "clip_count": split.clip_count,
            "frame_count": split.frame_count,
        }

    return {
        "preset": preset,
        "source": {
            "identifier": source_cfg.identifier,
            "format": source_cfg.format,
            "semantic_level": source_cfg.semantic_level,
            "source_fps_hz": source_cfg.source_fps,
            "license": source_cfg.license,
            "open_source": _callable_name(source_cfg.open_source),
            # bfm-env-20260805 campaign patch: skeleton owned by the coordinates module.
            "skeleton_factory": _native_skeleton_module(source_cfg),
            "skeleton_identity_sha256": _require_sha256(
                "source skeleton identity",
                source_skeleton.identity_sha256,
            ),
            "selected_split": table_cfg.motion_split,
            "train": split_value(source_cfg.train),
            "evaluation": split_value(source_cfg.evaluation),
        },
        "target": {
            "joint_names": list(joint_names),
            "body_names": list(body_names),
        },
        "construction": {
            # bfm-env-20260805 campaign patch: HEAD owns these under target_kinematics.
            "frame_builder_factory": _callable_name(table_cfg.target_kinematics.target_factory),
            "frame_builder_type": _callable_name(frame_builder_type),
            "frame_builder_identity_sha256": _require_sha256(
                "frame builder identity",
                frame_builder_identity_sha256,
            ),
            "reference_kinematics_factory": _callable_name(type(table_cfg.target_kinematics.kinematics)),
        },
    }


def _validate_direct_dependency_types(preset: str, importer_type: type, frame_builder_type: type) -> None:
    """Require the importer and builder selected by one resolved preset."""
    # bfm-env-20260805 campaign patch: HEAD renamed the frame builders and unified
    # the two G1 builders into one G1FrameBuilder.
    from isaaclab_tasks.core.multi_task.motion.data.sources import CmuHumEnvSmplClips, LafanG1JoblibClips
    from isaaclab_tasks.core.multi_task.motion.robots.g1.reference import G1FrameBuilder
    from isaaclab_tasks.core.multi_task.motion.robots.smpl.reference import SmplFrameBuilder

    expected_importer = LafanG1JoblibClips if preset == "g1_lafan" else CmuHumEnvSmplClips
    expected_builder = {
        "smpl_cmu": SmplFrameBuilder,
        "g1_lafan": G1FrameBuilder,
        "g1_cmu": G1FrameBuilder,
    }[preset]
    if importer_type is not expected_importer or frame_builder_type is not expected_builder:
        raise ValueError("Motion importer or trajectory-builder type differs from the selected direct axes.")


def motion_composition_dependency_identity(
    *,
    preset: str,
    cfg: object,
    importer_type: type,
    frame_builder_type: type,
    frame_builder_identity_sha256: str,
    reference_artifact_root: str | Path | None = None,
) -> dict[str, object]:
    """Return the exact source-to-target trajectory construction identity.

    This boundary intentionally excludes reset, observation, reward, solver,
    and policy behavior. Those remain owned by
    :func:`motion_environment_dependency_identity`.

    Args:
        preset: Resolved motion preset name.
        cfg: Resolved direct motion environment configuration.
        importer_type: Concrete source importer selected by ``preset``.
        frame_builder_type: Concrete trajectory builder selected by ``preset``.
        frame_builder_identity_sha256: Constructed builder's content identity.
        reference_artifact_root: Optional root containing the reference robot model.

    Returns:
        Closed dependency identity for pure source-to-target composition.
    """
    if preset not in ("smpl_cmu", "g1_lafan", "g1_cmu"):
        raise ValueError(f"Unsupported motion dependency preset: {preset!r}.")
    if hasattr(cfg, "motion"):
        raise ValueError("Motion evidence requires resolved direct axes, not an aggregate motion config.")
    _validate_direct_dependency_types(preset, importer_type, frame_builder_type)

    table_cfg = cfg.commands.motion.task_table
    composition = _resolved_composition(
        preset,
        cfg,
        frame_builder_type,
        frame_builder_identity_sha256,
    )
    runtime_dependencies = motion_composition_runtime_dependencies(preset)
    identity: dict[str, object] = {
        "schema": _COMPOSITION_SCHEMA,
        "preset": preset,
        "composition": composition,
        "composition_sha256": _json_hash(composition),
        "runtime_dependencies": runtime_dependencies,
        "runtime_dependencies_sha256": _json_hash(runtime_dependencies),
        "python_sources": _composition_python_sources(preset, importer_type, frame_builder_type, table_cfg),
        "reference_assets": dict(sorted(_reference_assets(preset, reference_artifact_root).items())),
    }
    return {**identity, "bundle_sha256": _json_hash(identity)}


def motion_environment_dependency_identity(
    *,
    preset: str,
    cfg: object,
    importer_type: type,
    frame_builder_type: type,
    reference_artifact_root: str | Path | None = None,
) -> dict[str, object]:
    """Return one broad host-independent identity for every motion collector."""
    if preset not in ("smpl_cmu", "g1_lafan", "g1_cmu"):
        raise ValueError(f"Unsupported motion dependency preset: {preset!r}.")
    if hasattr(cfg, "motion"):
        raise ValueError("Motion evidence requires resolved direct axes, not an aggregate motion config.")

    _validate_direct_dependency_types(preset, importer_type, frame_builder_type)
    table_cfg = cfg.commands.motion.task_table
    resolved_axes = _resolved_axes(preset, cfg, frame_builder_type)
    resolved_configuration = _resolved_configuration(cfg)
    runtime_dependencies = _runtime_dependencies(preset)
    robot_assets = {
        **_simulation_robot_assets(cfg.scene.robot),
        **_reference_assets(preset, reference_artifact_root),
    }
    identity: dict[str, object] = {
        "schema": _SCHEMA,
        "preset": preset,
        "resolved_axes": resolved_axes,
        "resolved_axes_sha256": _json_hash(resolved_axes),
        "resolved_configuration": resolved_configuration,
        "resolved_configuration_sha256": _json_hash(resolved_configuration),
        "runtime_dependencies": runtime_dependencies,
        "runtime_dependencies_sha256": _json_hash(runtime_dependencies),
        "python_sources": _python_sources(
            preset,
            importer_type,
            # bfm-env-20260805 campaign patch: factory moved under target_kinematics.
            table_cfg.target_kinematics.target_factory,
            frame_builder_type,
        ),
        "robot_assets": dict(sorted(robot_assets.items())),
    }
    return {**identity, "bundle_sha256": _json_hash(identity)}


def motion_environment_semantic_sha256(value: object) -> str:
    """Validate a dependency identity and hash its runtime-independent semantics.

    Args:
        value: Full dependency identity produced by :func:`motion_environment_dependency_identity`.

    Returns:
        SHA-256 digest of configuration, source, and asset semantics with host runtime excluded.
    """
    if not isinstance(value, Mapping):
        raise ValueError("Motion environment dependency identity must be a mapping.")
    expected_fields = {
        "schema",
        "preset",
        "resolved_axes",
        "resolved_axes_sha256",
        "resolved_configuration",
        "resolved_configuration_sha256",
        "runtime_dependencies",
        "runtime_dependencies_sha256",
        "python_sources",
        "robot_assets",
        "bundle_sha256",
    }
    if set(value) != expected_fields or value.get("schema") not in _RETAINED_SCHEMAS:
        raise ValueError("Motion environment dependency identity has an unsupported structure.")
    payload = dict(value)
    bundle_sha256 = payload.pop("bundle_sha256")
    if bundle_sha256 != _json_hash(payload):
        raise ValueError("Motion environment dependency identity is not internally closed.")
    for field in ("resolved_axes", "resolved_configuration", "runtime_dependencies"):
        if value[f"{field}_sha256"] != _json_hash(value[field]):
            raise ValueError(f"Motion environment dependency identity has a stale {field} digest.")
    semantic = {
        name: value[name]
        for name in (
            "schema",
            "preset",
            "resolved_axes",
            "resolved_axes_sha256",
            "resolved_configuration",
            "resolved_configuration_sha256",
            "python_sources",
            "robot_assets",
        )
    }
    return _json_hash(semantic)


def motion_environment_contract_sha256(value: object) -> str:
    """Hash the declared behavior contract of one internally closed identity.

    Unlike :func:`motion_environment_semantic_sha256`, this projection excludes
    producer source bytes. It therefore answers only whether two receipts declare
    the same resolved environment, data, and asset contract. A matching digest is
    not numerical proof after an implementation change; that requires a fresh
    transfer or runtime receipt.

    Args:
        value: Full dependency identity produced by
            :func:`motion_environment_dependency_identity`.

    Returns:
        SHA-256 digest of the declared behavior contract.
    """
    motion_environment_semantic_sha256(value)
    assert isinstance(value, Mapping)
    contract = {
        name: value[name]
        for name in (
            "preset",
            "resolved_axes",
            "resolved_axes_sha256",
            "resolved_configuration",
            "resolved_configuration_sha256",
            "robot_assets",
        )
    }
    return _json_hash(contract)


def motion_environment_compatibility(historical: object, current: object) -> dict[str, object]:
    """Compare current code with an immutable historical environment identity.

    Both identities are first validated against their own stored digests. The
    result is deliberately separate from receipt authenticity: implementation
    drift never authorizes rewriting the historical identity.

    Args:
        historical: Identity stored by the measured receipt.
        current: Identity reconstructed from the current tree.

    Returns:
        Explicit exact-source and declared-contract compatibility status.
    """
    historical_semantic = motion_environment_semantic_sha256(historical)
    current_semantic = motion_environment_semantic_sha256(current)
    historical_contract = motion_environment_contract_sha256(historical)
    current_contract = motion_environment_contract_sha256(current)
    assert isinstance(historical, Mapping)
    assert isinstance(current, Mapping)
    exact_producer = historical["bundle_sha256"] == current["bundle_sha256"]
    contract_matches = historical_contract == current_contract
    if exact_producer:
        status = "exact_producer_match"
    elif contract_matches:
        status = "declared_contract_match_requires_runtime_validation"
    else:
        status = "declared_contract_mismatch"
    return {
        "status": status,
        "historical_bundle_sha256": historical["bundle_sha256"],
        "current_bundle_sha256": current["bundle_sha256"],
        "historical_semantic_sha256": historical_semantic,
        "current_semantic_sha256": current_semantic,
        "historical_contract_sha256": historical_contract,
        "current_contract_sha256": current_contract,
        "producer_sources_match": historical.get("python_sources") == current.get("python_sources"),
        "runtime_dependencies_match": historical.get("runtime_dependencies") == current.get("runtime_dependencies"),
        "declared_contract_matches": contract_matches,
        "runtime_validation_required": not exact_producer,
    }


def motion_composition_semantic_sha256(value: object) -> str:
    """Validate and hash host-independent source-to-target semantics.

    Args:
        value: Identity produced by :func:`motion_composition_dependency_identity`.

    Returns:
        SHA-256 digest excluding only external runtime build identity.
    """
    if not isinstance(value, Mapping):
        raise ValueError("Motion composition dependency identity must be a mapping.")
    expected_fields = {
        "schema",
        "preset",
        "composition",
        "composition_sha256",
        "runtime_dependencies",
        "runtime_dependencies_sha256",
        "python_sources",
        "reference_assets",
        "bundle_sha256",
    }
    if set(value) != expected_fields or value.get("schema") != _COMPOSITION_SCHEMA:
        raise ValueError("Motion composition dependency identity has an unsupported structure.")
    payload = dict(value)
    bundle_sha256 = payload.pop("bundle_sha256")
    if bundle_sha256 != _json_hash(payload):
        raise ValueError("Motion composition dependency identity is not internally closed.")
    for field in ("composition", "runtime_dependencies"):
        if value[f"{field}_sha256"] != _json_hash(value[field]):
            raise ValueError(f"Motion composition dependency identity has a stale {field} digest.")
    semantic = {
        name: value[name]
        for name in (
            "schema",
            "preset",
            "composition",
            "composition_sha256",
            "python_sources",
            "reference_assets",
        )
    }
    return _json_hash(semantic)


def motion_composition_contract_sha256(value: object) -> str:
    """Hash one internally closed source-to-target declaration without code bytes."""
    motion_composition_semantic_sha256(value)
    assert isinstance(value, Mapping)
    return _json_hash(
        {
            "preset": value["preset"],
            "composition": value["composition"],
            "composition_sha256": value["composition_sha256"],
            "reference_assets": value["reference_assets"],
        }
    )


def motion_composition_compatibility(historical: object, current: object) -> dict[str, object]:
    """Compare current composition code without mutating historical provenance."""
    historical_semantic = motion_composition_semantic_sha256(historical)
    current_semantic = motion_composition_semantic_sha256(current)
    historical_contract = motion_composition_contract_sha256(historical)
    current_contract = motion_composition_contract_sha256(current)
    assert isinstance(historical, Mapping)
    assert isinstance(current, Mapping)
    exact_producer = historical["bundle_sha256"] == current["bundle_sha256"]
    contract_matches = historical_contract == current_contract
    if exact_producer:
        status = "exact_producer_match"
    elif contract_matches:
        status = "declared_contract_match_requires_runtime_validation"
    else:
        status = "declared_contract_mismatch"
    return {
        "status": status,
        "historical_bundle_sha256": historical["bundle_sha256"],
        "current_bundle_sha256": current["bundle_sha256"],
        "historical_semantic_sha256": historical_semantic,
        "current_semantic_sha256": current_semantic,
        "historical_contract_sha256": historical_contract,
        "current_contract_sha256": current_contract,
        "producer_sources_match": historical.get("python_sources") == current.get("python_sources"),
        "runtime_dependencies_match": historical.get("runtime_dependencies") == current.get("runtime_dependencies"),
        "declared_contract_matches": contract_matches,
        "runtime_validation_required": not exact_producer,
    }


__all__ = [
    "motion_composition_compatibility",
    "motion_composition_contract_sha256",
    "motion_composition_dependency_identity",
    "motion_composition_runtime_dependencies",
    "motion_composition_semantic_sha256",
    "motion_environment_compatibility",
    "motion_environment_contract_sha256",
    "motion_environment_dependency_identity",
    "motion_environment_semantic_sha256",
    "motion_g1_live_axes",
]
