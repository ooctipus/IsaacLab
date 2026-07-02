# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate independent consumption of native G1 policy-quality artifacts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
CONSUMER = ROOT / "g1_policy_quality_consumer.py"
ARTIFACT = ROOT / "fixtures" / "runtime" / "g1_lafan_policy_quality_v6"
GATE = ROOT / "fixtures" / "g1_lafan_policy_quality_gate_v1.json"
IDENTITY = ROOT / "motion_environment_identity.py"


def _module():
    spec = importlib.util.spec_from_file_location("g1_policy_quality_consumer", CONSUMER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _copy_artifact(tmp_path: Path) -> tuple[Path, object]:
    destination = tmp_path / "artifact"
    shutil.copytree(ARTIFACT, destination)
    module = _module()
    return destination, module


def test_controlled_v6_native_policy_artifact_recomputes_tracking_and_reward_decisions(tmp_path: Path) -> None:
    """Serialized rows must independently recover tracking pass and reward inconclusion."""
    artifact, module = _copy_artifact(tmp_path)
    receipt = module.validate_native_policy_quality_artifact(artifact, GATE)

    assert receipt == {
        "schema": "forward_backward_phase3_g1_policy_quality_consumer_v2",
        "status": "passed",
        "manifest_sha256": module._sha256(artifact / "manifest.json"),
        "tracking_sha256": "e57468132cec5d42f34d028784fbcca116bc674f8167617ea7efbd3aefe380fd",
        "broad_reward_sha256": "d355da0328962cabbdfb6fc800db8ccab3aa56d8ebea13030eff6d9ace842744",
        "physical_gpu_uuid": "GPU-ec08549f-c376-5290-59f3-79125cd3c660",
        "tracking_emd_mean": pytest.approx(1.3172697943286036),
        "tracking_obs_state_emd_mean": pytest.approx(1.1444436957062538),
        "broad_return_mean": pytest.approx(51.10669467049539),
        "decision": "tracking_non_inferiority_passed",
        "broad_reward_status": "inconclusive_protocol_identity",
        "broad_reward_point_gate": "not_met",
    }


def test_consumer_rejects_tracking_bytes_changed_after_publication(tmp_path: Path) -> None:
    """A post-publication tracking edit must fail before its summaries are trusted."""
    artifact, module = _copy_artifact(tmp_path)
    tracking = artifact / "tracking.json"
    tracking.write_bytes(tracking.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="Tracking artifact bytes"):
        module.validate_native_policy_quality_artifact(artifact, GATE)


def test_consumer_rejects_emd_transport_kernel_source_mutation(tmp_path: Path) -> None:
    """Changing the transitive Warp kernel must invalidate otherwise unchanged evidence."""
    artifact, module = _copy_artifact(tmp_path)
    original_source_path = module._module_source_path
    kernel_path = original_source_path(module._EMD_TRANSPORT_MODULE)
    mutated_kernel = tmp_path / "uniform_emd_warp.py"
    mutated_kernel.write_bytes(kernel_path.read_bytes() + b"\n# mutation\n")

    def source_path(module_name: str) -> Path:
        if module_name == module._EMD_TRANSPORT_MODULE:
            return mutated_kernel
        return original_source_path(module_name)

    module._module_source_path = source_path
    with pytest.raises(ValueError, match="emd transport kernel bytes"):
        module.validate_native_policy_quality_artifact(artifact, GATE)


def test_consumer_rejects_a_manifest_decision_not_derived_from_rows(tmp_path: Path) -> None:
    """A producer-declared pass cannot replace independent row-level recomputation."""
    artifact, module = _copy_artifact(tmp_path)
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["decision"]["metrics"]["tracking"]["emd_mean"]["actual"] = 0.0
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="decision differs"):
        module.validate_native_policy_quality_artifact(artifact, GATE)


def test_consumer_rejects_broad_reward_rows_even_if_their_hash_is_redeclared(tmp_path: Path) -> None:
    """Rehashing a changed CSV cannot preserve its published metric summaries."""
    artifact, module = _copy_artifact(tmp_path)
    csv_path = artifact / "broad_reward.csv"
    text = csv_path.read_text()
    csv_path.write_text(text.replace(",return,", ",return,", 1).replace(",0.0\n", ",1.0\n", 1))
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["policy_quality"]["broad_reward"]["sha256"] = module._sha256(csv_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="summaries differ"):
        module.validate_native_policy_quality_artifact(artifact, GATE)


def test_consumer_rejects_broad_reward_called_passed_without_paired_identity() -> None:
    """A manifest cannot promote the diagnostic point result into an authoritative pass."""
    module = _module()
    decision = {
        "diagnostics": {
            "broad_reward": {
                "status": "passed",
                "authoritative": False,
                "identity_closure": {"identity_closed": False},
            }
        }
    }

    with pytest.raises(ValueError, match="cannot be called passed"):
        module._validate_broad_reward_decision_boundary(decision, None)


def test_policy_source_closure_hashes_module_bytes_without_host_path_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Moving equal source bytes may pass, while mutating those bytes must stale evidence."""
    module = _module()
    first = tmp_path / "host_a" / "runtime.py"
    second = tmp_path / "host_b" / "runtime.py"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("VALUE = 1\n")
    second.write_bytes(first.read_bytes())
    sources = {"example.runtime": module._sha256(first)}
    monkeypatch.setattr(module, "_module_source_path", lambda _name: second)

    module._validate_policy_python_sources("environment", sources, IDENTITY)
    second.write_text("VALUE = 2\n")

    with pytest.raises(ValueError, match="environment.*example.runtime.*bytes differ"):
        module._validate_policy_python_sources("environment", sources, IDENTITY)


def test_policy_dependency_checks_both_environment_and_composition_source_maps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both stored dependency layers must pass their complete source maps to byte closure."""
    module = _module()
    manifest = json.loads((ARTIFACT / "manifest.json").read_text())
    code_identity = manifest["code_identity"]
    observed: list[tuple[str, object, Path]] = []

    def validate(owner: str, sources: object, identity_path: Path) -> None:
        observed.append((owner, sources, identity_path))

    monkeypatch.setattr(module, "_validate_policy_python_sources", validate)
    module._policy_dependency_identities(code_identity, IDENTITY, "g1_lafan")

    assert observed == [
        ("environment", code_identity["dependency_identity"]["python_sources"], IDENTITY),
        (
            "composition",
            code_identity["composition_dependency_identity"]["python_sources"],
            IDENTITY,
        ),
    ]


def test_cross_companion_identity_excludes_only_absolute_path_provenance() -> None:
    """Moving an artifact may change paths but never its companion semantic identity."""
    remote = {
        "retarget_fit": {"path": "/remote/retarget.json", "sha256": "a" * 64},
        "reference_controller_simulator": {"path": "/remote/simulator.json", "sha256": "b" * 64},
    }
    local = {
        "retarget_fit": {"path": "/local/retarget.json", "sha256": "a" * 64},
        "reference_controller_simulator": {"path": "/local/simulator.json", "sha256": "b" * 64},
    }
    changed = {
        **local,
        "retarget_fit": {"path": "/local/retarget.json", "sha256": "c" * 64},
    }
    module = _module()

    assert module._companion_identity(remote) == module._companion_identity(local)
    assert module._companion_identity(remote) != module._companion_identity(changed)


_LOCAL_POLICY_CODE_FIELDS = (
    "evaluator_sha256",
    "tracking_evaluator_sha256",
    "emd_transport_kernel_sha256",
    "expert_provider_sha256",
    "expert_buffer_sha256",
    "model_sha256",
    "learner_code_bundle_sha256",
    "python_source_identity_sha256",
    "reward_kinematics_sha256",
    "reward_context_policy_sha256",
    "bfm_reward_runtime_sha256",
    "gpu_ownership_sha256",
    "policy_quality_gate_sha256",
    "policy_quality_protocol_audit_sha256",
)


@pytest.mark.parametrize("field", _LOCAL_POLICY_CODE_FIELDS)
def test_consumer_rejects_each_changed_local_policy_code_owner(field: str) -> None:
    """Every locally available producer source must be independently rehashed."""
    module = _module()
    stored = module._current_policy_code_identity(GATE)
    assert set(stored) == set(_LOCAL_POLICY_CODE_FIELDS)
    stored[field] = "0" * 64

    with pytest.raises(ValueError, match=field.removesuffix("_sha256").replace("_", " ")):
        module._validate_local_policy_code_identity(stored, GATE)


@pytest.mark.parametrize(
    "declared",
    (
        None,
        {},
        {"../escaped.py": "a" * 64},
        {"source.txt": "a" * 64},
        {"source.py": "not-a-digest"},
    ),
)
def test_consumer_rejects_malformed_external_reward_source_provenance(declared: object) -> None:
    """External source hashes remain declared provenance, but cannot be ambiguous."""
    module = _module()

    with pytest.raises((TypeError, ValueError), match="external BFM reward source"):
        module._validate_external_reward_source_identity(declared)


def test_consumer_binds_external_reward_sources_to_one_repository_revision() -> None:
    """The declared files must close one immutable BFM repository revision and tree."""
    module = _module()
    contract = json.loads(module._BFM_REWARD_SOURCE_IDENTITY.read_text())

    assert module._validate_external_reward_source_identity(contract["files"]) == {
        "contract_sha256": module._sha256(module._BFM_REWARD_SOURCE_IDENTITY),
        "repository": contract["repository"],
        "files": contract["files"],
    }


def test_consumer_rejects_valid_looking_external_sources_from_another_revision() -> None:
    """Syntax-valid hashes cannot substitute for the frozen BFM repository identity."""
    module = _module()
    contract = json.loads(module._BFM_REWARD_SOURCE_IDENTITY.read_text())
    declared = dict(contract["files"])
    declared[next(iter(declared))] = "a" * 64

    with pytest.raises(ValueError, match="differ from the frozen repository identity"):
        module._validate_external_reward_source_identity(declared)


def test_external_source_contract_reproduces_the_frozen_bfm_git_revision() -> None:
    """Every declared source digest must be recoverable from the named Git commit."""
    root = Path.cwd().parents[1] / "BFM-Zero"
    if not root.is_dir():
        pytest.skip("The frozen BFM-Zero reference checkout is unavailable.")
    module = _module()
    contract = json.loads(module._BFM_REWARD_SOURCE_IDENTITY.read_text())
    revision = contract["repository"]["revision"]
    tree = subprocess.run(
        ("git", "-C", str(root), "rev-parse", f"{revision}^{{tree}}"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert tree == contract["repository"]["tree"]
    for name, digest in contract["files"].items():
        content = subprocess.run(
            ("git", "-C", str(root), "show", f"{revision}:{name}"),
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(content).hexdigest() == digest


@pytest.mark.parametrize("change", ("missing", "unexpected"))
def test_policy_code_identity_requires_the_exact_closed_field_set(change: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A producer field cannot be silently dropped or added outside the consumer contract."""
    module = _module()
    manifest = json.loads((ARTIFACT / "manifest.json").read_text())
    code_identity = manifest["code_identity"]
    if change == "missing":
        code_identity.pop("expert_buffer_sha256")
    else:
        code_identity["unconsumed_sha256"] = "a" * 64
    monkeypatch.setattr(module, "_validate_local_policy_code_identity", lambda *_args: None)

    with pytest.raises(ValueError, match="code-identity fields differ"):
        module._validate_policy_code_identity(code_identity, GATE, "g1_lafan")


def test_python_package_identity_rejects_a_symbolic_package_root(tmp_path: Path) -> None:
    """Resolving a package-root symlink must not erase its symbolic provenance."""
    module = _module()._source_module(ROOT / "python_source_identity.py")
    package = tmp_path / "package"
    package.mkdir()
    (package / "__init__.py").write_text("VALUE = 1\n")
    linked_package = tmp_path / "linked_package"
    linked_package.symlink_to(package, target_is_directory=True)

    with pytest.raises(ValueError, match="non-symbolic directory"):
        module.python_package_identity(linked_package)


def test_python_package_identity_rejects_a_symbolic_python_member(tmp_path: Path) -> None:
    """Resolving a Python-file symlink must not turn it into a regular member."""
    module = _module()._source_module(ROOT / "python_source_identity.py")
    package = tmp_path / "package"
    package.mkdir()
    (package / "__init__.py").write_text("VALUE = 1\n")
    source = tmp_path / "external.py"
    source.write_text("VALUE = 2\n")
    (package / "external.py").symlink_to(source)

    with pytest.raises(ValueError, match="regular non-symbolic file"):
        module.python_package_identity(package)
