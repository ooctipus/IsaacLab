# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Architecture gates for declarative direct-task scenes."""

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DIRECT_ROOTS = (
    Path(__file__).resolve().parents[2] / "isaaclab_tasks",
    _REPO_ROOT / "source/isaaclab_tasks_experimental/isaaclab_tasks_experimental",
    _REPO_ROOT / "scripts",
)
_DIRECT_TEMPLATES = (
    _REPO_ROOT / "tools/template/templates/tasks/direct_single-agent/env",
    _REPO_ROOT / "tools/template/templates/tasks/direct_multi-agent/env",
)
_SCENE_LIFECYCLE_CALLS = ("clone_plan_from_env_0(", "ReplicateSession(", "cloner.replicate(", ".spawn.func(")


def test_direct_tasks_leave_scene_construction_to_their_cfg() -> None:
    """Repository direct tasks leave construction and cloning to their declared scene."""
    offenders = []
    for root in _DIRECT_ROOTS:
        for path in sorted(root.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            offenders.extend(
                f"{path.relative_to(_REPO_ROOT)}:{node.lineno}"
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "_setup_scene"
            )
            if "benchmarks" not in path.parts and (
                "_env" in path.stem or "DirectRLEnv" in source or "DirectMARLEnv" in source
            ):
                offenders.extend(
                    f"{path.relative_to(_REPO_ROOT)}: {call}" for call in _SCENE_LIFECYCLE_CALLS if call in source
                )
    offenders.extend(
        f"{path.relative_to(_REPO_ROOT)}: {pattern}"
        for path in _DIRECT_TEMPLATES
        for pattern in ("_setup_scene", *_SCENE_LIFECYCLE_CALLS)
        if pattern in path.read_text()
    )

    assert not offenders, "Direct task classes own scene construction or cloning:\n" + "\n".join(offenders)


def test_cartpole_camera_derives_shape_from_scene_owned_camera(monkeypatch) -> None:
    from isaaclab_tasks.core.cartpole.cartpole_direct_camera_env import CartpoleCameraEnv
    from isaaclab_tasks.core.cartpole.cartpole_direct_camera_env_cfg import CartpoleCameraEnvCfg
    from isaaclab_tasks.core.cartpole.cartpole_direct_env import CartpoleEnv
    from isaaclab_tasks.utils.hydra import resolve_config

    cfg = CartpoleCameraEnvCfg.BaseCartpoleCameraEnvCfg(frame_stack=1)
    cfg.scene.camera.default.height = 45
    cfg.scene.camera.default.width = 123
    cfg = resolve_config(cfg, [])

    def init_stub(env, cfg, *_args, **_kwargs):
        env.cfg = cfg
        env._is_closed = True

    monkeypatch.setattr(CartpoleEnv, "__init__", init_stub)

    CartpoleCameraEnv(cfg)

    assert cfg.observation_space == [3, 45, 123]


def test_cartpole_camera_term_declares_shape_without_reading_data() -> None:
    """The Manager camera stack must derive its shape from allocated metadata, not render a frame."""
    from types import SimpleNamespace

    from isaaclab.managers import ObservationTermCfg, SceneEntityCfg

    from isaaclab_tasks.core.cartpole.mdp import CameraImageStack

    class FakeCamera:
        output_shapes = {"albedo": (2, 8, 16, 4)}

        @property
        def data(self):
            raise AssertionError("Shape declaration must not read camera data.")

    env = SimpleNamespace(
        cfg=SimpleNamespace(frame_stack=2),
        num_envs=2,
        device="cpu",
        scene=SimpleNamespace(sensors={"camera": FakeCamera()}),
    )
    cfg = ObservationTermCfg(
        func=CameraImageStack,
        params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "albedo"},
    )

    term = CameraImageStack(cfg, env)

    assert term._output_shape == (6, 8, 16)


def test_cartpole_feature_terms_declare_model_output_shapes(monkeypatch) -> None:
    """Feature observations must not run an encoder to discover their shape."""
    from types import SimpleNamespace

    from isaaclab.envs.mdp import image_features

    from isaaclab_tasks.core.cartpole.cartpole_manager_camera_env_cfg import (
        ResNet18ObservationCfg,
        TheiaTinyObservationCfg,
    )

    class FakeCamera:
        output_shapes = {"rgb": (2, 96, 96, 3)}

        @property
        def data(self):
            raise AssertionError("Shape declaration must not read camera data.")

    env = SimpleNamespace(device="cpu", scene=SimpleNamespace(sensors={"camera": FakeCamera()}))
    resnet = ResNet18ObservationCfg().policy.image
    theia = TheiaTinyObservationCfg().policy.image

    model_config = {"model": object, "inference": lambda *_args, **_kwargs: None}
    monkeypatch.setattr(image_features, "_prepare_resnet_model", lambda *_args: model_config)
    monkeypatch.setattr(image_features, "_prepare_theia_transformer_model", lambda *_args: model_config)

    assert image_features(resnet, env)._output_shape == (1000,)
    assert image_features(theia, env)._output_shape == (36, 192)


def test_factory_and_automate_cfgs_declare_complete_scenes() -> None:
    """Every registered Factory, Forge, and AutoMate variant owns its drawn assets in ``scene``."""
    from isaaclab_tasks.contrib.automate.assembly_env_cfg import AssemblyEnvCfg
    from isaaclab_tasks.contrib.automate.disassembly_env_cfg import DisassemblyEnvCfg
    from isaaclab_tasks.contrib.factory.factory_env_cfg import (
        FactoryTaskGearMeshCfg,
        FactoryTaskNutThreadCfg,
        FactoryTaskPegInsertCfg,
    )
    from isaaclab_tasks.contrib.forge.forge_env_cfg import (
        ForgeTaskGearMeshCfg,
        ForgeTaskNutThreadCfg,
        ForgeTaskPegInsertCfg,
    )

    common = {"ground", "table", "robot", "fixed_asset", "held_asset", "light"}
    cfg_types = (
        AssemblyEnvCfg,
        DisassemblyEnvCfg,
        FactoryTaskPegInsertCfg,
        FactoryTaskNutThreadCfg,
        ForgeTaskPegInsertCfg,
        ForgeTaskNutThreadCfg,
    )
    for cfg_type in cfg_types:
        cfg = cfg_type()
        cfg.validate()
        assert all(getattr(cfg.scene, name) is not None for name in common)
        assert not any(hasattr(cfg, name) for name in ("robot", "tasks", "task_name"))

    for cfg_type in (FactoryTaskGearMeshCfg, ForgeTaskGearMeshCfg):
        cfg = cfg_type()
        cfg.validate()
        assert all(getattr(cfg.scene, name) is not None for name in common | {"small_gear", "large_gear"})


def test_direct_goal_markers_belong_to_the_scene() -> None:
    """Direct goal markers must enter the same clone plan as the task assets."""
    from isaaclab_tasks.core.handover.handover_env_cfg import HandoverEnvCfg
    from isaaclab_tasks.core.reorient.config.allegro_hand.allegro_hand_direct_env_cfg import AllegroHandEnvCfg
    from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_direct_env_cfg import ShadowHandEnvCfg

    for cfg_type in (HandoverEnvCfg, AllegroHandEnvCfg, ShadowHandEnvCfg):
        cfg = cfg_type()
        assert "goal_object_cfg" not in vars(cfg)
        assert "goal_object_cfg" in vars(cfg.scene)
