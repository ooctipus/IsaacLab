# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise the Phase 3F callback boundary without launching simulation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parent / "motion_training_receipt.py"


def _module():
    spec = importlib.util.spec_from_file_location("motion_training_callback", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_callback_publishes_each_stage_beside_the_training_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    identity = {"identity": "1" * 64}
    provenance = {"owner": "retained"}
    contract_digest = "2" * 64
    profile: dict[str, object] = {}
    training_root = tmp_path / "training"
    validation_log = tmp_path / "validation-log"
    checkpoint = training_root / "model_4.pt"
    configured_env_cfg = object()
    constructed_env_cfg = object()

    monkeypatch.setattr(module, "_preset", lambda _cfg: "fixture")
    monkeypatch.setattr(module, "_contract_profile", lambda _preset: profile)
    monkeypatch.setattr(module, "_contract_declaration_sha256", lambda: contract_digest)

    def live_identity(_preset: str, identity_cfg: object, *_args: object):
        assert identity_cfg is configured_env_cfg
        return identity, provenance

    monkeypatch.setattr(module, "_live_identity_evidence", live_identity)
    monkeypatch.setattr(
        module,
        "_launch_record",
        lambda *_args: {
            "schema": "launch",
            "preset": "fixture",
            "identity": identity,
            "contract_declaration_sha256": contract_digest,
            "provenance": provenance,
            "lifecycle_extension": None,
        },
    )
    monkeypatch.setattr(
        module,
        "_complete_record",
        lambda *_args: {"schema": "complete", "preset": "fixture", "identity": identity},
    )
    monkeypatch.setattr(
        module,
        "_validation_record",
        lambda *_args: {"schema": "validation", "preset": "fixture", "identity": identity},
    )
    monkeypatch.setattr(
        module,
        "validate_smoke_records",
        lambda *_args: {"schema": "receipt", "status": "passed"},
    )

    callback_values = {
        "env_cfg": constructed_env_cfg,
        "configured_env_cfg": configured_env_cfg,
        "agent_cfg": object(),
        "env": object(),
        "runner": object(),
    }
    module.training_callback(stage="launch", log_dir=training_root, **callback_values)
    module.training_callback(
        stage="complete",
        log_dir=validation_log,
        checkpoint_path=checkpoint,
        **callback_values,
    )
    module.training_callback(
        stage="validate",
        log_dir=validation_log,
        checkpoint_path=checkpoint,
        **callback_values,
    )

    assert json.loads((training_root / "phase3f_launch.json").read_text())["schema"] == "launch"
    assert json.loads((training_root / "phase3f_complete.json").read_text())["schema"] == "complete"
    assert json.loads((training_root / "phase3f_validation.json").read_text())["schema"] == "validation"
    assert json.loads((training_root / "phase3f_receipt.json").read_text())["status"] == "passed"
    assert not validation_log.exists()


def test_training_callback_rejects_contract_drift_after_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Completion must not reinterpret a launch under edited declaration bytes."""
    module = _module()
    identity = {"identity": "1" * 64}
    provenance = {"owner": "retained"}
    configured_env_cfg = object()
    constructed_env_cfg = object()
    digests = iter(("2" * 64, "3" * 64))
    monkeypatch.setattr(module, "_preset", lambda _cfg: "fixture")
    monkeypatch.setattr(module, "_contract_profile", lambda _preset: {})
    monkeypatch.setattr(module, "_contract_declaration_sha256", lambda: next(digests))
    monkeypatch.setattr(
        module,
        "_live_identity_evidence",
        lambda *_args: (identity, provenance),
    )

    def launch_record(*_args: object) -> dict[str, object]:
        return {
            "schema": "launch",
            "preset": "fixture",
            "identity": identity,
            "contract_declaration_sha256": "2" * 64,
            "provenance": provenance,
            "lifecycle_extension": None,
        }

    monkeypatch.setattr(module, "_launch_record", launch_record)
    values = {
        "env_cfg": constructed_env_cfg,
        "configured_env_cfg": configured_env_cfg,
        "agent_cfg": object(),
        "env": object(),
        "runner": object(),
    }
    root = tmp_path / "training"
    module.training_callback(stage="launch", log_dir=root, **values)

    with pytest.raises(ValueError, match="contract declaration"):
        module.training_callback(
            stage="complete",
            log_dir=tmp_path / "ignored",
            checkpoint_path=root / "model_4.pt",
            **values,
        )


def test_evidence_publication_is_exclusive(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "receipt.json"
    module._write_json_exclusive(path, {"status": "first"})

    with pytest.raises(FileExistsError, match="already exists"):
        module._write_json_exclusive(path, {"status": "replacement"})

    assert json.loads(path.read_text()) == {"status": "first"}
    assert not path.with_name(f".{path.name}.tmp").exists()


def test_launch_contract_rejects_identity_and_lifecycle_drift() -> None:
    module = _module()
    profile = {
        "closed_input_identity": {"bundle": "a" * 64},
        "environment_semantic_sha256": "c" * 64,
        "collection": {
            "num_envs": 4,
            "steps_per_iteration": 2,
            "iterations": 4,
            "random_action_transitions": 8,
            "updates_per_group": 3,
            "expected_transitions": 32,
            "expected_update_groups": 2,
            "expected_update_calls": 6,
        },
    }
    env = SimpleNamespace(num_envs=4)
    runner = SimpleNamespace(
        lifecycle_extension=None,
        cfg={"num_steps_per_env": 2},
        random_action_steps=8,
        num_updates_per_iteration=3,
    )
    agent = SimpleNamespace(lifecycle_extension=None, max_iterations=4)

    expected_identity = {"bundle": "a" * 64, "environment_semantic_sha256": "c" * 64}
    drifted_identity = {"bundle": "b" * 64, "environment_semantic_sha256": "c" * 64}
    with pytest.raises(ValueError, match="bundle") as error:
        module._assert_launch_contract(profile, drifted_identity, env, runner, agent)
    assert "a" * 64 in str(error.value)
    assert "b" * 64 in str(error.value)

    agent.lifecycle_extension = {"class_name": "tracking"}
    with pytest.raises(ValueError, match="lifecycle_extension=null"):
        module._assert_launch_contract(profile, expected_identity, env, runner, agent)


def test_validation_record_propagates_actual_post_load_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    checkpoint = tmp_path / "model_4.pt"
    checkpoint.write_bytes(b"checkpoint")
    load_summary = {
        "environment_resume": "restart",
        "environment_state_dict_is_none": True,
        "map_location": "cuda:7",
        "mmap": True,
        "strict": False,
    }
    runner = SimpleNamespace(checkpoint_load_summary=lambda: dict(load_summary))
    learner = {"update_step": 6}
    monkeypatch.setattr(module, "_learner_snapshot", lambda *_args: learner)

    record = module._validation_record("fixture", {"bundle": "a" * 64}, object(), runner, checkpoint, "0" * 64)

    assert record["learner"] == learner
    assert record["contract_declaration_sha256"] == "0" * 64
    assert record["checkpoint"]["map_location"] == "cuda:7"
    assert record["checkpoint"]["mmap"] is True
    assert record["checkpoint"]["strict_load"] is False
    assert record["checkpoint"]["environment_resume"] == "restart"
    assert record["checkpoint"]["environment_state_dict_is_none"] is True


def test_resolved_agent_config_hash_closes_nested_non_schema_hyperparameters() -> None:
    module = _module()

    def expert_provider() -> None:
        pass

    class AgentCfg:
        def __init__(self) -> None:
            self.algorithm = {
                "learning_rate": 3.0e-4,
                "discriminator_gradient_penalty_coefficient": 10.0,
                "value_cfg": {"auxiliary": {"reward_coefficients": (0.0, 0.1, 10.0)}},
            }

        def to_dict(self) -> dict[str, object]:
            return {
                "algorithm": self.algorithm,
                "expert": {"provider": expert_provider},
                "model": {"actor_cfg": {"hidden_dim": 1024, "hidden_layers": 6}},
                "replay": {"capacity_transitions": 12_288},
                "device": "cuda:0",
            }

    agent_cfg = AgentCfg()
    baseline = module._resolved_agent_config_sha256(agent_cfg)
    agent_cfg.algorithm["discriminator_gradient_penalty_coefficient"] = 9.0

    assert module._resolved_agent_config_sha256(agent_cfg) != baseline


def test_learner_package_bundle_covers_every_python_file_and_ignores_checkout_root(tmp_path: Path) -> None:
    module = _module()

    def package(root: Path) -> Path:
        package_root = root / "rsl_rl"
        (package_root / "modules").mkdir(parents=True)
        (package_root / "runners").mkdir()
        (package_root / "__init__.py").write_text("VERSION = 1\n")
        (package_root / "modules" / "forward_backward.py").write_text("def equation(): return 1\n")
        (package_root / "runners" / "off_policy_runner.py").write_text("class OffPolicyRunner: pass\n")
        return package_root

    first = package(tmp_path / "checkout-a")
    second = package(tmp_path / "checkout-b")
    baseline = module._python_package_bundle_sha256(first)

    assert module._python_package_bundle_sha256(second) == baseline

    (first / "modules" / "forward_backward.py").write_text("def equation(): return 2\n")
    changed_equation = module._python_package_bundle_sha256(first)
    assert changed_equation != baseline

    (first / "modules" / "normalization.py").write_text("def normalize(x): return x\n")
    assert module._python_package_bundle_sha256(first) != changed_equation


def test_learner_package_identity_retains_every_hashed_member(tmp_path: Path) -> None:
    """The package digest must remain explainable after source bytes move on."""
    module = _module()
    package_root = tmp_path / "rsl_rl"
    (package_root / "algorithms").mkdir(parents=True)
    (package_root / "__init__.py").write_text("VERSION = 1\n")
    (package_root / "algorithms" / "forward_backward.py").write_text("def equation(): return 1\n")

    identity = module._python_package_identity(package_root)

    assert identity["python_file_count"] == 2
    assert identity["python_files"] == [
        {"path": "__init__.py", "sha256": module._file_sha256(package_root / "__init__.py")},
        {
            "path": "algorithms/forward_backward.py",
            "sha256": module._file_sha256(package_root / "algorithms" / "forward_backward.py"),
        },
    ]
    assert identity["bundle_sha256"] == module._python_package_bundle_sha256(package_root)


def test_task_bridge_bundle_uses_actual_wrapper_config_and_expert_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()

    class Wrapper:
        pass

    class MotionRunnerCfg:
        pass

    def expert_provider() -> None:
        pass

    def other_expert_provider() -> None:
        pass

    agent_cfg = MotionRunnerCfg()
    agent_cfg.expert = {"provider": expert_provider}
    observed: list[object] = []

    def owner_source(owner: object) -> tuple[str, str]:
        observed.append(owner)
        return "fixture.shared", "a" * 64

    monkeypatch.setattr(module, "_owner_source", owner_source)

    identity = module._task_bridge_identity(Wrapper(), agent_cfg)
    digest = identity["bundle_sha256"]

    assert len(digest) == 64
    assert identity["source_owner_count"] == 3
    assert set(identity["source_owners"]) == {
        "environment_wrapper",
        "motion_expert_provider",
        "motion_runner_config",
    }
    assert observed == [Wrapper, MotionRunnerCfg, expert_provider]

    observed.clear()
    agent_cfg.expert["provider"] = other_expert_provider
    assert module._task_bridge_bundle_sha256(Wrapper(), agent_cfg) != digest
    assert observed == [Wrapper, MotionRunnerCfg, other_expert_provider]


def test_learner_runtime_identity_binds_tensordict_and_gymnasium_versions() -> None:
    """Tensor container and environment-boundary packages are learner inputs."""
    module = _module()

    identity = module._learner_runtime_identity()

    assert set(identity["packages"]) == {"gymnasium", "tensordict"}
    for package in identity["packages"].values():
        assert package["module_version"]
        assert package["distribution_version"]
        assert len(package["module_source_sha256"]) == 64
    payload = {name: value for name, value in identity.items() if name != "bundle_sha256"}
    assert identity["bundle_sha256"] == module._json_sha256(payload)


def test_live_identity_uses_only_consolidated_environment_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    broad = SimpleNamespace(
        motion_environment_dependency_identity=lambda **_kwargs: {
            "bundle_sha256": "a" * 64,
            "resolved_axes_sha256": "b" * 64,
        },
        motion_environment_semantic_sha256=lambda _identity: "6" * 64,
    )
    monkeypatch.setattr(
        module, "_sibling_module", lambda name: broad if name == "motion_environment_identity" else None
    )
    monkeypatch.setattr(module, "_native_types", lambda _preset: (type("Importer", (), {}), type("Builder", (), {})))
    monkeypatch.setattr(module, "_owner_source", lambda _owner: ("owner.module", "c" * 64))
    monkeypatch.setattr(module, "_python_package_identity", lambda _root: {"bundle_sha256": "2" * 64})
    monkeypatch.setattr(module, "_task_bridge_identity", lambda _env, _cfg: {"bundle_sha256": "3" * 64})
    monkeypatch.setattr(module, "_learner_runtime_identity", lambda: {"bundle_sha256": "5" * 64})
    monkeypatch.setattr(module, "_resolved_agent_config_sha256", lambda _cfg: "4" * 64)
    monkeypatch.setattr(module, "_file_sha256", lambda _path: "d" * 64)
    monkeypatch.setattr(module, "_motion_table", lambda _env: SimpleNamespace(cache_identity="e" * 64))
    monkeypatch.setattr(module, "_validate_provenance", lambda *_args: None)

    import rsl_rl

    runner_path = Path(next(iter(rsl_rl.__path__))) / "runners" / "off_policy_runner.py"
    monkeypatch.setattr(rsl_rl, "__file__", None)
    monkeypatch.setattr(module.inspect, "getsourcefile", lambda _owner: str(runner_path))

    observation_schema = SimpleNamespace(schema_hash="0" * 64)
    model = SimpleNamespace(observation_schema=observation_schema)
    algorithm = SimpleNamespace(
        model=model,
        replay=object(),
        expert=SimpleNamespace(schema=SimpleNamespace(schema_hash="f" * 64)),
        checkpoint_header=SimpleNamespace(schema_hash="1" * 64),
    )
    runner = SimpleNamespace(alg=algorithm)
    table_cfg = SimpleNamespace(reference_artifact_root="/reference")
    env_cfg = SimpleNamespace(
        scene=SimpleNamespace(robot=object()),
        commands=SimpleNamespace(motion=SimpleNamespace(task_table=table_cfg)),
    )

    identity, provenance = module._live_identity_evidence("fixture", env_cfg, object(), runner, object())

    assert identity["environment_dependency_bundle_sha256"] == "a" * 64
    assert identity["environment_semantic_sha256"] == "6" * 64
    assert identity["resolved_axes_sha256"] == "b" * 64
    assert identity["learner_code_bundle_sha256"] == "2" * 64
    assert identity["learner_runtime_bundle_sha256"] == "5" * 64
    assert identity["task_bridge_code_bundle_sha256"] == "3" * 64
    assert identity["resolved_agent_config_sha256"] == "4" * 64
    assert provenance["environment"]["dependency_identity"]["bundle_sha256"] == "a" * 64
    assert provenance["learner_code"]["bundle_sha256"] == "2" * 64
    assert provenance["learner_runtime"]["bundle_sha256"] == "5" * 64
    assert provenance["task_bridge"]["bundle_sha256"] == "3" * 64
    assert "resolved_environment_bundle_sha256" not in identity


def test_prepare_callback_materializes_one_exclusive_live_identity_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preparation must reuse live production identity without entering launch validation."""
    module = _module()
    profile = {
        "command": ["frozen-command"],
        "collection": {
            "num_envs": 4,
            "steps_per_iteration": 2,
            "iterations": 4,
            "expected_transitions": 32,
            "random_action_transitions": 8,
            "updates_per_group": 3,
            "expected_update_groups": 2,
            "expected_update_calls": 6,
        },
        "environment_semantic_sha256": "0" * 64,
        "closed_input_identity": {"task_table_sha256": "1" * 64},
    }
    identity = {
        "environment_semantic_sha256": "2" * 64,
        "task_table_sha256": "3" * 64,
    }
    provenance = {"owners": "retained"}
    configured_env_cfg = object()
    constructed_env_cfg = object()
    agent_cfg = SimpleNamespace(lifecycle_extension=None, max_iterations=4)
    env = SimpleNamespace(num_envs=4)
    runner = SimpleNamespace(
        lifecycle_extension=None,
        cfg={"num_steps_per_env": 2},
        random_action_steps=8,
        num_updates_per_iteration=3,
    )
    monkeypatch.setattr(module, "_preset", lambda _cfg: "smpl_cmu")
    monkeypatch.setattr(module, "_contract_profile", lambda _preset: profile)
    monkeypatch.setattr(module, "_contract_declaration_sha256", lambda: "4" * 64)
    monkeypatch.setattr(module, "_live_identity_evidence", lambda *_args: (identity, provenance))
    monkeypatch.setattr(module, "_validate_provenance", lambda _identity, value: value)

    runner.random_action_steps = 7
    with pytest.raises(ValueError, match="runner cadence"):
        module.training_callback(
            stage="prepare",
            env_cfg=constructed_env_cfg,
            configured_env_cfg=configured_env_cfg,
            agent_cfg=agent_cfg,
            env=env,
            runner=runner,
            log_dir=tmp_path,
        )
    assert not (tmp_path / "phase3f_identity_freeze.json").exists()
    runner.random_action_steps = 8

    module.training_callback(
        stage="prepare",
        env_cfg=constructed_env_cfg,
        configured_env_cfg=configured_env_cfg,
        agent_cfg=agent_cfg,
        env=env,
        runner=runner,
        log_dir=tmp_path,
    )

    record = json.loads((tmp_path / "phase3f_identity_freeze.json").read_text())
    assert record == {
        "schema": "forward_backward_phase3f_identity_freeze_v1",
        "preset": "smpl_cmu",
        "contract_declaration_sha256": "4" * 64,
        "static_profile_sha256": module._static_profile_sha256(profile),
        "identity": identity,
        "provenance": provenance,
    }
    assert not (tmp_path / "phase3f_launch.json").exists()

    with pytest.raises(FileExistsError, match="already exists"):
        module.training_callback(
            stage="prepare",
            env_cfg=constructed_env_cfg,
            configured_env_cfg=configured_env_cfg,
            agent_cfg=agent_cfg,
            env=env,
            runner=runner,
            log_dir=tmp_path,
        )


def _freeze_fixture(module: object, tmp_path: Path) -> tuple[Path, list[Path], dict[str, object]]:
    profiles = {
        preset: {
            "command": [f"train-{preset}"],
            "collection": {"iterations": index + 1},
            "environment_semantic_sha256": str(index) * 64,
            "closed_input_identity": {
                "task_table_sha256": str(index + 2) * 64,
                "expert_schema_sha256": str(index + 4) * 64,
            },
        }
        for index, preset in enumerate(("smpl_cmu", "g1_lafan"))
    }
    contract = {
        "schema": "forward_backward_phase3_motion_training_smoke_contract_v2",
        "status": "prepared_not_launched",
        "launch_gate": "wait_for_final_phase3e_simulator_and_cloner_identity",
        "claim_scope": "frozen",
        "profiles": profiles,
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    contract_digest = module._file_sha256(contract_path)
    record_paths: list[Path] = []
    for index, preset in enumerate(profiles):
        record = {
            "schema": "forward_backward_phase3f_identity_freeze_v1",
            "preset": preset,
            "contract_declaration_sha256": contract_digest,
            "static_profile_sha256": module._static_profile_sha256(profiles[preset]),
            "identity": {
                "environment_semantic_sha256": str(index + 6) * 64,
                "task_table_sha256": str(index + 7) * 64,
                "expert_schema_sha256": str(index + 8) * 64,
            },
            "provenance": {"preset": preset},
        }
        path = tmp_path / f"{preset}.json"
        path.write_text(json.dumps(record))
        record_paths.append(path)
    return contract_path, record_paths, contract


def test_contract_freeze_replaces_only_derived_identity_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Commands, collection math, learner settings, and gates must remain byte-semantic peers."""
    module = _module()
    contract_path, records, original = _freeze_fixture(module, tmp_path)
    monkeypatch.setattr(module, "_validate_provenance", lambda *_args: None)
    output = tmp_path / "frozen.json"

    module.freeze_contract(contract_path, records, output)

    frozen = json.loads(output.read_text())
    for preset, original_profile in original["profiles"].items():
        identity = json.loads(records[0 if preset == "smpl_cmu" else 1].read_text())["identity"]
        expected_profile = dict(original_profile)
        expected_profile["environment_semantic_sha256"] = identity["environment_semantic_sha256"]
        expected_profile["closed_input_identity"] = {
            name: value for name, value in identity.items() if name != "environment_semantic_sha256"
        }
        assert frozen["profiles"][preset] == expected_profile
    assert {name: value for name, value in frozen.items() if name != "profiles"} == {
        name: value for name, value in original.items() if name != "profiles"
    }


def test_contract_freeze_rejects_static_profile_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A record from different commands or collection math must never rewrite the contract."""
    module = _module()
    contract_path, records, _original = _freeze_fixture(module, tmp_path)
    monkeypatch.setattr(module, "_validate_provenance", lambda *_args: None)
    record = json.loads(records[0].read_text())
    record["static_profile_sha256"] = "f" * 64
