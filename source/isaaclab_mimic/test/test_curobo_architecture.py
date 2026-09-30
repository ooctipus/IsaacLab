# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path

from isaaclab import cloner

from isaaclab_mimic.envs.franka_stack_ik_rel_skillgen_env_cfg import FrankaCubeStackIKRelSkillgenEnvCfg

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CUROBO_ROOT = _PACKAGE_ROOT / "isaaclab_mimic" / "motion_planners" / "curobo"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_curobo_cfg_is_a_data_only_clone_plan_declaration() -> None:
    """CuRobo configuration declares construction and geometry without executable policy."""
    cfg_path = _CUROBO_ROOT / "curobo_planner_cfg.py"
    cfg_source = _read(cfg_path)
    cfg_tree = ast.parse(cfg_source, filename=str(cfg_path))
    cfg_classes = [node for node in cfg_tree.body if isinstance(node, ast.ClassDef)]

    assert {node.name for node in cfg_classes} == {"CuroboCollisionCfg", "CuroboPlannerCfg"}
    assert not [
        node
        for cfg_class in cfg_classes
        for node in ast.walk(cfg_class)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
    ]
    assert '"{DIR}.curobo_planner:CuroboPlanner"' in cfg_source
    assert "mesh_prim_paths" in cfg_source


def test_curobo_world_comes_only_from_the_clone_plan() -> None:
    """CuRobo must not rediscover collision geometry from the completed USD stage."""
    planner_path = _CUROBO_ROOT / "curobo_planner.py"
    planner_source = _read(planner_path)
    forbidden = {
        "UsdHelper",
        "get_current_stage",
        "get_obstacles_from_stage",
        "GetPrimAtPath",
        "PrimRange",
        "_discover_object_mappings",
        "_get_object_mappings",
    }

    assert not sorted(symbol for symbol in forbidden if symbol in planner_source)
    assert "ClonePlan" in planner_source
    assert "plan.match_geometry_targets" in planner_source
    assert "plan.match_rigid_body_subtrees" in planner_source


def test_curobo_has_no_hidden_visualizer_lifecycle() -> None:
    """Planner visualization belongs to the explicit simulation composition root."""
    assert not (_CUROBO_ROOT / "plan_visualizer.py").exists()
    sources = [
        _read(_CUROBO_ROOT / "curobo_planner.py"),
        _read(_CUROBO_ROOT / "curobo_planner_cfg.py"),
        _read(_REPO_ROOT / "scripts" / "imitation_learning" / "isaaclab_mimic" / "generate_dataset.py"),
    ]
    forbidden = {"PlanVisualizer", "plan_visualizer", "visualize_plan"}

    assert not sorted(symbol for source in sources for symbol in forbidden if symbol in source)


def test_curobo_consumers_construct_only_from_owned_cfg() -> None:
    """SkillGen owns CuRobo intent before cloning and constructs it through ``class_type``."""
    script_path = _REPO_ROOT / "scripts" / "imitation_learning" / "isaaclab_mimic" / "generate_dataset.py"
    script_source = _read(script_path)
    skillgen_cfg = _read(_PACKAGE_ROOT / "isaaclab_mimic" / "envs" / "franka_stack_ik_rel_skillgen_env_cfg.py")

    assert "motion_planner: CuroboPlannerCfg = CuroboPlannerCfg()" in skillgen_cfg
    assert "planner_cfg = env.cfg.motion_planner" in script_source
    assert "planner_cfg.class_type(planner_cfg, env, env_id)" in script_source
    forbidden = {"CuroboPlanner(", "from_task_name", "franka_config", "getattr(planner", "visualize_spheres"}
    assert not sorted(symbol for symbol in forbidden if symbol in script_source)

    data_generator_path = _PACKAGE_ROOT / "isaaclab_mimic" / "datagen" / "data_generator.py"
    data_generator_source = _read(data_generator_path)
    data_generator_tree = ast.parse(data_generator_source, filename=str(data_generator_path))
    assert "getattr(motion_planner" not in data_generator_source
    assert "hasattr(motion_planner" not in data_generator_source
    assert 'hasattr(self.env, "get_expected_attached_object")' not in data_generator_source
    planning_calls = [
        node
        for node in ast.walk(data_generator_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "update_world_and_plan_motion"
    ]
    assert planning_calls
    assert not [keyword for call in planning_calls for keyword in call.keywords if keyword.arg == "env_id"]


def test_curobo_has_no_direct_constructor_consumers() -> None:
    """Call sites resolve the planner implementation from configuration."""
    roots = [
        _PACKAGE_ROOT / "isaaclab_mimic",
        _PACKAGE_ROOT / "test",
        _REPO_ROOT / "scripts" / "imitation_learning" / "isaaclab_mimic",
    ]
    offenders = []
    for root in roots:
        for path in root.rglob("*.py"):
            tree = ast.parse(_read(path), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "CuroboPlanner":
                    offenders.append(str(path.relative_to(_REPO_ROOT)))

    assert not sorted(set(offenders))


def test_skillgen_clone_plan_contains_the_curobo_collision_world() -> None:
    """The SkillGen scene contributes every CuRobo collision target before cloning."""
    cfg = FrankaCubeStackIKRelSkillgenEnvCfg()
    env_template = "/World/envs/env_{}"
    expected = tuple(
        cloner.expand_env_regex_ns(target.prim_expr, env_template) for target in cfg.motion_planner.mesh_prim_paths
    )
    plan = cloner.make_clone_plan(
        (cfg.scene.table, cfg.scene.cube_1, cfg.scene.cube_2, cfg.scene.cube_3),
        num_clones=2,
        env_spacing=cfg.scene.env_spacing,
        geometry_prim_paths=cfg.scene.geometry_prim_paths,
        env_template=env_template,
    )

    assert tuple(cloner.expand_env_regex_ns(path, env_template) for path in cfg.scene.geometry_prim_paths) == expected
    assert plan.geometry_requests == expected
