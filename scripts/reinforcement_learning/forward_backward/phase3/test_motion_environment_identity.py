# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate import-free byte identity for explicit motion-environment modules."""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

IDENTITY = Path(__file__).parent / "motion_environment_identity.py"


def _module():
    spec = importlib.util.spec_from_file_location("phase3_motion_environment_identity", IDENTITY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dependency_identity(module, *, runtime_version: str = "one", source_sha256: str = "a" * 64) -> dict:
    resolved_axes = {"preset": "smpl_cmu"}
    resolved_configuration = {"compute_final_obs": True}
    runtime_dependencies = {"newton": {"module_version": runtime_version}}
    identity = {
        "schema": module._SCHEMA,
        "preset": "smpl_cmu",
        "resolved_axes": resolved_axes,
        "resolved_axes_sha256": module._json_hash(resolved_axes),
        "resolved_configuration": resolved_configuration,
        "resolved_configuration_sha256": module._json_hash(resolved_configuration),
        "runtime_dependencies": runtime_dependencies,
        "runtime_dependencies_sha256": module._json_hash(runtime_dependencies),
        "python_sources": {"motion": source_sha256},
        "robot_assets": {"simulation/robot.xml": "b" * 64},
    }
    return {**identity, "bundle_sha256": module._json_hash(identity)}


def test_environment_identity_closes_procedural_ground_contact_and_native_mjcf_runtime() -> None:
    """Ground, contact, and native MJCF owners must participate in byte identity."""
    module = _module()

    assert {
        "isaaclab.sim.schemas.schemas",
        "isaaclab.sim.schemas.schemas_cfg",
        "isaaclab.sim.spawners.shapes.shapes",
        "isaaclab.sim.spawners.materials.physics_materials",
        "isaaclab.sim.spawners.materials.physics_materials_cfg",
        "isaaclab.sim.spawners.shapes.shapes_cfg",
        "isaaclab.sim.utils.prims",
    } <= set(module._COMMON_MODULES)
    assert "isaaclab_tasks.core.multi_task.motion.tracking" not in module._COMMON_MODULES
    assert "isaaclab_newton.sim.schemas.schemas_cfg" in module._SMPL_MODULES

    assert {
        "isaaclab_newton.sim.spawners.mjcf.mjcf",
        "isaaclab_newton.sim.spawners.mjcf.mjcf_cfg",
    } <= set(module._SMPL_MODULES)
    assert {
        "isaaclab.sim.converters.asset_converter_base",
        "isaaclab.sim.converters.mjcf_converter",
        "isaaclab.sim.spawners.from_files.from_files",
    }.isdisjoint(module._SMPL_MODULES)


def test_simulation_robot_assets_hashes_source_without_generated_cache(tmp_path: Path) -> None:
    """Direct-source spawners bind evidence to source bytes, not generated USD caches."""
    module = _module()
    source = tmp_path / "robot.xml"
    source.write_text("<mujoco model='robot'/>")
    generated = tmp_path / "robot" / "robot.usda"
    generated.parent.mkdir()
    generated.write_text("host-specific generated cache")
    robot = SimpleNamespace(spawn=SimpleNamespace(asset_path=str(source), usd_dir=str(tmp_path)))

    assets = module._simulation_robot_assets(robot)

    assert assets == {"simulation/robot.xml": hashlib.sha256(source.read_bytes()).hexdigest()}
    assert all("usda" not in name for name in assets)


def test_environment_semantic_identity_excludes_host_runtime_but_closes_source_bytes() -> None:
    """Host package versions may differ, while source-byte changes must stale portable evidence."""
    module = _module()
    baseline = _dependency_identity(module)
    other_runtime = _dependency_identity(module, runtime_version="two")
    other_source = _dependency_identity(module, source_sha256="c" * 64)

    semantic_sha256 = module.motion_environment_semantic_sha256
    assert baseline["bundle_sha256"] != other_runtime["bundle_sha256"]
    assert semantic_sha256(baseline) == semantic_sha256(other_runtime)
    assert semantic_sha256(baseline) != semantic_sha256(other_source)

    stale = dict(baseline)
    stale["bundle_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="not internally closed"):
        semantic_sha256(stale)


def test_module_source_path_hashes_an_explicit_module_without_importing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A module with import-time Kit failure must remain hashable as inert source bytes."""
    source = tmp_path / "explosive" / "runtime.py"
    source.parent.mkdir()
    source.write_text('raise RuntimeError("must not import")\nVALUE = 3\n')
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])
    module = _module()

    resolved = module._module_source_path("explosive.runtime")

    assert resolved == source.resolve()
    assert hashlib.sha256(resolved.read_bytes()).hexdigest() == module._sha256(resolved)
    assert "explosive" not in sys.modules
    assert "explosive.runtime" not in sys.modules


def test_module_source_path_resolves_an_editable_package_without_importing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A PEP 660 package root outside sys.path must expose inert target bytes."""
    package_root = tmp_path / "editable_package"
    package_root.mkdir()
    source = package_root / "runtime.py"
    source.write_text('raise RuntimeError("must not import")\nVALUE = 4\n')
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if Path(entry or ".") != tmp_path])
    module = _module()

    def find_spec(name: str):
        assert name == "editable_package"
        spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        spec.submodule_search_locations = [str(package_root)]
        return spec

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)

    assert module._module_source_path("editable_package.runtime") == source.resolve()
    assert "editable_package" not in sys.modules
    assert "editable_package.runtime" not in sys.modules


def test_module_source_path_resolves_an_editable_namespace_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A PEP 660 project-root location must resolve its nested package bytes."""
    project_root = tmp_path / "project"
    source = project_root / "editable_package" / "runtime.py"
    source.parent.mkdir(parents=True)
    source.write_text('raise RuntimeError("must not import")\nVALUE = 5\n')
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if Path(entry or ".") != project_root])
    module = _module()

    def find_spec(name: str):
        assert name == "editable_package"
        spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        spec.submodule_search_locations = [str(project_root)]
        return spec

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)

    assert module._module_source_path("editable_package.runtime") == source.resolve()
    assert "editable_package" not in sys.modules
    assert "editable_package.runtime" not in sys.modules


def test_module_source_path_rejects_ambiguous_explicit_modules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two distinct sys.path owners must fail instead of selecting by host order."""
    roots = (tmp_path / "one", tmp_path / "two")
    for index, root in enumerate(roots):
        source = root / "package" / "runtime.py"
        source.parent.mkdir(parents=True)
        source.write_text(f"VALUE = {index}\n")
    monkeypatch.setattr(sys, "path", [*(str(root) for root in roots), *sys.path])
    module = _module()

    with pytest.raises(RuntimeError, match="ambiguous"):
        module._module_source_path("package.runtime")


def test_module_source_path_rejects_missing_and_symbolic_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing and symbolic module files must not weaken content closure."""
    real = tmp_path / "real.py"
    real.write_text("VALUE = 1\n")
    symbolic = tmp_path / "package" / "runtime.py"
    symbolic.parent.mkdir()
    symbolic.symlink_to(real)
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])
    module = _module()

    with pytest.raises(ValueError, match="symbolic"):
        module._module_source_path("package.runtime")
    with pytest.raises(ModuleNotFoundError, match="missing.runtime"):
        module._module_source_path("missing.runtime")
