# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Architecture gates for implementation-owned observation shapes."""

import ast
import math
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parents[3]
_PRODUCTION_ROOTS = (
    _SOURCE_ROOT / "isaaclab_tasks" / "isaaclab_tasks",
    _SOURCE_ROOT / "isaaclab_mimic" / "isaaclab_mimic",
)
_OBSERVATION_CFGS = {"ObservationTermCfg", "ObsTerm"}


def _symbol_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def test_production_task_cfgs_do_not_declare_output_shapes() -> None:
    """Task cfgs configure observations but never own their implementation dimensions."""
    offenders = []
    for root in _PRODUCTION_ROOTS:
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
                if _symbol_name(call.func) not in _OBSERVATION_CFGS:
                    continue
                if any(keyword.arg == "output_shape" for keyword in call.keywords):
                    offenders.append(f"{root.parent.name}/{path.relative_to(root)}:{call.lineno}")

    assert not offenders, "Task configs owning observation shapes:\n" + "\n".join(offenders)


def test_instance_randomized_stack_placeholders_match_declared_shapes() -> None:
    """Pre-reset placeholders must have the same dimensions as post-reset object selections."""
    from types import SimpleNamespace

    from isaaclab_tasks.contrib.stack.stack_instance_randomize_env_cfg import ObservationsCfg

    cfg = ObservationsCfg().policy
    env = SimpleNamespace(num_envs=2)
    for name in ("object", "cube_positions", "cube_orientations"):
        term = getattr(cfg, name)
        assert term.func(env, **term.params).shape == (env.num_envs, *term.func._output_shape)


def test_surface_gripper_grasp_observations_are_scalar(monkeypatch) -> None:
    """Surface-gripper state must not broadcast one scalar grasp result across all environments."""
    import importlib
    from types import SimpleNamespace

    import torch

    from isaaclab.managers import SceneEntityCfg

    place_observations = importlib.import_module("isaaclab_tasks.contrib.place.mdp.observations")
    stack_observations = importlib.import_module("isaaclab_tasks.contrib.stack.mdp.observations")

    class Scene(dict):
        surface_grippers = {"surface_gripper": SimpleNamespace(state=object())}

    scene = Scene(
        robot=SimpleNamespace(),
        ee_frame=SimpleNamespace(data=SimpleNamespace(target_pos_w=SimpleNamespace(torch=torch.zeros(2, 1, 3)))),
        cube=SimpleNamespace(data=SimpleNamespace(root_pos_w=SimpleNamespace(torch=torch.zeros(2, 3)))),
    )
    env = SimpleNamespace(scene=scene)
    entities = {name: SceneEntityCfg(name) for name in ("robot", "ee_frame", "cube")}
    for observations in (stack_observations, place_observations):
        monkeypatch.setattr(observations.wp, "to_torch", lambda _: torch.tensor([1, 0]))
        result = observations.object_grasped(env, entities["robot"], entities["ee_frame"], entities["cube"])
        assert result.shape == (2,)


def test_sensor_shape_resolvers_match_selected_entities() -> None:
    """Implementation resolvers derive dimensions from resolved sensor selections."""
    from types import SimpleNamespace

    from isaaclab_tasks.contrib.velocity.config.digit.rough_env_cfg import DigitRoughEnvCfg
    from isaaclab_tasks.core.lift.config.franka.franka_env_cfg import StateObservationCfg as FrankaObservationsCfg
    from isaaclab_tasks.core.lift.config.kuka_allegro.camera_cfg import StateObservationCfg as KukaObservationsCfg
    from isaaclab_tasks.core.locomotion.ant.ant_manager_env_cfg import AntObservationsCfg
    from isaaclab_tasks.core.locomotion.humanoid.humanoid_manager_env_cfg import HumanoidObservationsCfg
    from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_camera_manager_env_cfg import (
        ShadowHandCameraObservationsCfg,
    )
    from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_manager_env_cfg import (
        ShadowHandAsymmetricObservationsCfg,
    )
    from isaaclab_tasks.core.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

    for cfg_cls in (FrankaObservationsCfg, KukaObservationsCfg):
        term = cfg_cls().proprio.contact
        assert term.func._output_shape(None, **term.params) == (3 * len(term.params["contact_sensor_names"]),)

    for cfg in (AntObservationsCfg().default, HumanoidObservationsCfg().default):
        term = cfg.policy.feet_body_forces
        sensor_cfg = term.params["sensor_cfg"]
        sensor_cfg.body_ids = list(range(len(sensor_cfg.body_names)))
        sensor = SimpleNamespace(num_bodies=8)
        env = SimpleNamespace(scene=SimpleNamespace(sensors={sensor_cfg.name: sensor}))
        assert term.func._output_shape(env, **term.params) == (6 * len(sensor_cfg.body_names),)

    for cfg in (ShadowHandAsymmetricObservationsCfg(), ShadowHandCameraObservationsCfg()):
        term = cfg.critic.fingertip_wrench
        sensor_cfg = term.params["sensor_cfg"]
        sensor_cfg.body_ids = list(range(len(sensor_cfg.body_names)))
        sensor = SimpleNamespace(num_bodies=8)
        env = SimpleNamespace(scene=SimpleNamespace(sensors={sensor_cfg.name: sensor}))
        assert term.func._output_shape(env, **term.params) == (6 * len(sensor_cfg.body_names),)

    for cfg in (LocomotionVelocityRoughEnvCfg(), DigitRoughEnvCfg()):
        term = cfg.observations.policy.height_scan
        pattern = cfg.scene.height_scanner.pattern_cfg
        grid_shape = math.prod(int(round(size / pattern.resolution)) + 1 for size in pattern.size)
        sensor_cfg = term.params["sensor_cfg"]
        sensor = SimpleNamespace(num_rays=grid_shape)
        env = SimpleNamespace(scene=SimpleNamespace(sensors={sensor_cfg.name: sensor}))
        assert term.func._output_shape(env, **term.params) == (grid_shape,)
