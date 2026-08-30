# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Architecture gates for the multi-task package layout."""

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "isaaclab_tasks"
_MULTI_TASK_ROOT = _PACKAGE_ROOT / "core" / "multi_task"
_RSL_RL_PUBLIC_STUB = Path(__file__).resolve().parents[2] / "isaaclab_rl" / "isaaclab_rl" / "rsl_rl" / "__init__.pyi"
_TEXT_SUFFIXES = {".md", ".py", ".pyi", ".rst", ".toml"}


def test_removed_position_package_is_not_reintroduced() -> None:
    """Position task ownership should remain under the multi-task composition root."""
    assert not (_PACKAGE_ROOT / "core" / "position").exists()


def test_repository_has_no_rejected_multi_task_legacy_paths() -> None:
    """Multi-task sources and maintained scripts should not reference obsolete package locations."""
    rejected_paths = (
        "commands/multi_task_command.py",
        "impl/multi_task_command_warp",
        "impl.multi_task_command_warp",
        "isaaclab.cloner.cloner_utils",
        "locomotion/position",
        "manager_based/multi_task",
        "manager_based.multi_task",
        "mdp.commands.benchmark",
        "mdp/commands/benchmark",
        "scripts/reinforcement_learning/rsl_rl/train.py",
    )
    violations = []
    source_paths = set(_MULTI_TASK_ROOT.rglob("*"))
    source_paths.update((_REPO_ROOT / "scripts").rglob("*"))
    for path in sorted(source_paths):
        if path.suffix not in _TEXT_SUFFIXES:
            continue
        if path.is_relative_to(_REPO_ROOT / "scripts") and "extras" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for rejected_path in rejected_paths:
            if rejected_path in text:
                violations.append(f"{path.relative_to(_REPO_ROOT)}: {rejected_path}")

    assert not violations, "Rejected legacy paths remain:\n" + "\n".join(violations)


def test_multi_task_core_does_not_depend_on_contrib_factory() -> None:
    """Core multi-task ownership must not depend on the legacy contrib Factory task."""
    violations = []
    for path in sorted(_MULTI_TASK_ROOT.rglob("*")):
        if path.suffix not in {".py", ".pyi"}:
            continue
        if "isaaclab_tasks.contrib.nist" in path.read_text(encoding="utf-8"):
            violations.append(str(path.relative_to(_PACKAGE_ROOT)))

    assert not violations, "Core multi-task imports contrib.nist:\n" + "\n".join(violations)


def test_multi_task_rsl_cfgs_are_not_reexported_by_isaaclab_rl() -> None:
    """Task-owned RSL-RL extensions must not have duplicate public ownership in :mod:`isaaclab_rl`."""
    task_owned_symbols = {
        "RslRlCrlAlgorithmCfg",
        "RslRlHerCfg",
        "RslRlMLPEncoderModelCfg",
        "RslRlOffPolicyRunnerCfg",
        "RslRlResidualMLPCfg",
        "RslRlResidualMLPEncoderModelCfg",
    }
    public_stub = _RSL_RL_PUBLIC_STUB.read_text(encoding="utf-8")
    violations = sorted(symbol for symbol in task_owned_symbols if symbol in public_stub)

    assert not violations, "Task-owned symbols reexported by isaaclab_rl:\n" + "\n".join(violations)


def test_multi_task_does_not_use_deprecated_schema_aliases() -> None:
    """Multi-task configs should name their backend schema ownership explicitly."""
    deprecated = {
        "ArticulationRootPropertiesCfg",
        "CollisionPropertiesCfg",
        "JointDrivePropertiesCfg",
        "RigidBodyPropertiesCfg",
        "RigidBodyMaterialCfg",
    }
    violations = []
    for path in sorted(_MULTI_TASK_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in deprecated:
                violations.append(f"{path.relative_to(_PACKAGE_ROOT)}:{node.lineno}: {node.attr}")
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in deprecated:
                        violations.append(f"{path.relative_to(_PACKAGE_ROOT)}:{node.lineno}: {alias.name}")

    assert not violations, "Deprecated schema aliases remain:\n" + "\n".join(violations)


def test_warp_kernel_modules_do_not_own_torch_boundaries() -> None:
    """Warp kernel modules must remain pure launch modules over Warp arrays."""
    violations = []
    warp_modules = set(_MULTI_TASK_ROOT.rglob("*_warp.py"))
    warp_modules.update(_MULTI_TASK_ROOT.rglob("*_wp.py"))
    for path in sorted(warp_modules):
        if "impl" not in path.relative_to(_MULTI_TASK_ROOT).parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "torch" or alias.name.startswith("torch."):
                        violations.append(f"{path.relative_to(_PACKAGE_ROOT)}:{node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                if node.module == "torch" or node.module.startswith("torch."):
                    violations.append(f"{path.relative_to(_PACKAGE_ROOT)}:{node.lineno}: from {node.module}")
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"from_torch", "to_torch"}
            ):
                violations.append(f"{path.relative_to(_PACKAGE_ROOT)}:{node.lineno}: {node.func.attr} conversion")

    assert not violations, "Torch boundary work found in Warp backends:\n" + "\n".join(violations)


def test_command_composition_root_owns_backend_selection() -> None:
    """Command backend selection should not be hidden behind an implementation factory."""
    command_root = _MULTI_TASK_ROOT / "mdp" / "commands" / "multi_task_command"
    violations = []
    for path in sorted(command_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "build_command_backend":
                violations.append(f"{path.relative_to(_PACKAGE_ROOT)}:{node.lineno}")

    assert not violations, "Hidden command backend factories remain:\n" + "\n".join(violations)


def test_warp_composition_roots_stay_outside_impl() -> None:
    """Torch-to-Warp wrappers should remain at public composition roots."""
    expected = (
        _MULTI_TASK_ROOT / "curriculum" / "sampling" / "sampler.py",
        _MULTI_TASK_ROOT / "mdp" / "commands" / "multi_task_command" / "multi_task_command_warp.py",
    )
    forbidden = (
        _MULTI_TASK_ROOT / "curriculum" / "sampling" / "impl" / "sampler_warp.py",
        _MULTI_TASK_ROOT / "mdp" / "commands" / "multi_task_command" / "impl" / "multi_task_command_warp.py",
    )
    assert not [path.relative_to(_PACKAGE_ROOT) for path in expected if not path.exists()]
    assert not [path.relative_to(_PACKAGE_ROOT) for path in forbidden if path.exists()]


def test_command_public_ownership_stays_outside_impl() -> None:
    """Public command configs and backend-neutral IDs should live at the command composition root."""
    command_root = _MULTI_TASK_ROOT / "mdp" / "commands" / "multi_task_command"
    expected = (command_root / "multi_task_cfg.py", command_root / "kernel_ids.py")
    forbidden = (command_root / "impl" / "multi_task_cfg.py", command_root / "impl" / "kernel_ids.py")

    assert not [path.relative_to(_PACKAGE_ROOT) for path in expected if not path.exists()]
    assert not [path.relative_to(_PACKAGE_ROOT) for path in forbidden if path.exists()]

    for path in (_MULTI_TASK_ROOT / "multi_task_env_cfg.py", _MULTI_TASK_ROOT / "terrain" / "tasks_cfg.py"):
        text = path.read_text(encoding="utf-8")
        assert ".impl.multi_task_cfg" not in text
        assert ".impl.kernels_torch" not in text


def test_incomplete_factory_prototype_scripts_are_absent() -> None:
    """Tracked Factory scripts must not depend on prototype modules that are absent from the repository."""
    script = _MULTI_TASK_ROOT / "factory" / "scripts" / "roundtrip_newton_ik.py"
    assert not script.exists()


def test_controlled_multi_task_compatibility_aliases_are_not_reintroduced() -> None:
    """Known in-repo callers should use the canonical config and geometry symbols directly."""
    terrain_cfg_path = _MULTI_TASK_ROOT / "terrain" / "terrains" / "trimesh" / "mesh_terrains_cfg.py"
    terrain_cfg_tree = ast.parse(terrain_cfg_path.read_text(encoding="utf-8"), filename=str(terrain_cfg_path))
    violations = []
    for node in ast.walk(terrain_cfg_tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Name):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == node.value.id:
                violations.append(f"{terrain_cfg_path.relative_to(_PACKAGE_ROOT)}:{node.lineno}: {target.id}")

    for path in sorted((_MULTI_TASK_ROOT / "factory").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                not isinstance(node, ast.ImportFrom)
                or not node.module
                or not node.module.endswith("assembly_keypoints")
            ):
                continue
            if any(alias.name == "Offset" for alias in node.names):
                violations.append(f"{path.relative_to(_PACKAGE_ROOT)}:{node.lineno}: Offset")

    assert not violations, "Controlled compatibility aliases remain:\n" + "\n".join(violations)


def test_multi_task_relative_imports_resolve_to_existing_modules() -> None:
    """Relative imports should resolve inside the selected multi-task file tree."""
    violations = []
    source_paths = set(_MULTI_TASK_ROOT.rglob("*.py"))
    source_paths.update(_MULTI_TASK_ROOT.rglob("*.pyi"))
    for path in sorted(source_paths):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        package_parts = path.relative_to(_PACKAGE_ROOT).with_suffix("").parts[:-1]
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level == 0:
                continue
            parent_count = node.level - 1
            if parent_count > len(package_parts):
                violations.append(f"{path.relative_to(_PACKAGE_ROOT)}:{node.lineno}: escapes package root")
                continue
            target_parts = package_parts[: len(package_parts) - parent_count]
            if node.module is not None:
                target_parts += tuple(node.module.split("."))
            target = _PACKAGE_ROOT.joinpath(*target_parts)
            candidates = (
                target.with_suffix(".py"),
                target.with_suffix(".pyi"),
                target / "__init__.py",
                target / "__init__.pyi",
            )
            if not any(candidate.exists() for candidate in candidates):
                module = "." * node.level + (node.module or "")
                violations.append(f"{path.relative_to(_PACKAGE_ROOT)}:{node.lineno}: {module}")

    assert not violations, "Relative imports target missing modules:\n" + "\n".join(violations)
