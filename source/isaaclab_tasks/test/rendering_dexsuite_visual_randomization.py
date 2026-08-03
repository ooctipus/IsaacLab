# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene, config and golden-image driver for the Dexsuite visual domain randomization."""

from typing import Any

import pytest
from rendering_test_utils import (
    _FLAKY_MARK,
    MAX_DIFFERENT_PIXELS_PERCENTAGE_BY_ENV_NAME,
    _apply_overrides_to_env_cfg,
    _physics_preset_name,
    _redirect_ovrtx_renderer_log_to_stdout,
    maybe_save_stage,
    validate_camera_outputs,
)

# Golden-image name for the visual-randomization lanes.
_VISUAL_DR_TEST_NAME = "dexsuite_kuka_visual_dr"

DEXSUITE_VISUAL_RANDOMIZATION_KITLESS_COMBINATIONS = [
    pytest.param("newton", "newton_renderer", id="newton-newton_warp"),
    pytest.param("newton", "ovrtx_renderer", id="newton-ovrtx_renderer"),
]

# Albedo authored on both robot materials before any write (18% gray). The reset event overwrites it
# per environment; seeing this value in a render means the randomization did not land.
_VISUAL_DR_INITIAL_COLOR = (0.18, 0.18, 0.18)

# Scene-entity names of the two per-environment materials bound to the robot.
_VISUAL_DR_ARM_MATERIAL = "robot_arm_material"
_VISUAL_DR_HAND_MATERIAL = "robot_hand_material"

# Asset-relative visual prims of the KukaAllegro robot, split into the two part groups. Two
# materials rather than one so the golden also pins that the groups are randomized independently;
# the hand parts additionally exercise binding at a deeper nesting level.
_VISUAL_DR_ARM_PARTS = tuple(f"iiwa7_link_{index}/visuals" for index in range(8))
_VISUAL_DR_HAND_PARTS = ("ee_link/palm_link/visuals",) + tuple(
    f"ee_link/{finger}_link_{index}/visuals" for finger in ("index", "middle", "ring", "thumb") for index in range(4)
)

# The camera the golden image is captured from.
_VISUAL_DR_CAMERA = "base_camera"

# Camera presets. 128x128 rather than the 64x64 the other Dexsuite lanes use: the robot's colour is
# what this golden exists to pin, and the extra resolution keeps that readable in a diff.
_VISUAL_DR_CAMERA_PRESETS = "rgb128,single_camera"

# Number of environments. Every environment draws its own colours from the per-environment
# materials, so the golden grid shows four differently coloured robots and per-environment
# divergence is visible in the image itself.
_VISUAL_DR_NUM_ENVS = 4

# Renders performed by ``reset`` after the colour event fires, before the frame is captured. The
# RTX renderers accumulate temporally, so a single re-render still blends the pre-write frame.
_VISUAL_DR_RERENDERS_ON_RESET = 4


def _make_dexsuite_visual_randomization_env_cfg(physics_backend: str, renderer: str) -> Any:
    """Build the Dexsuite KukaAllegro lift camera config with per-environment robot materials.

    Two :class:`~isaaclab.assets.VisualMaterial` entities are declared inside the cloned environment
    namespace and bound to the arm and hand visual prims of the robot, so every environment binds
    its own clone. A ``reset``-mode :func:`~isaaclab.envs.mdp.randomize_visual_color` term re-rolls
    both, per environment; the sampling is seeded by the test's determinism fixture, which is what
    makes the resulting image a stable golden.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.assets import VisualMaterialCfg
    from isaaclab.envs import mdp as env_mdp
    from isaaclab.managers import EventTermCfg, SceneEntityCfg

    from isaaclab_tasks.core.dexsuite.config.kuka_allegro.dexsuite_kuka_allegro_camera_env_cfg import (
        DexsuiteKukaAllegroLiftCameraEnvCfg,
    )

    override_arg = f"presets={_physics_preset_name(physics_backend)},{renderer},{_VISUAL_DR_CAMERA_PRESETS},cube"
    env_cfg = _apply_overrides_to_env_cfg(DexsuiteKukaAllegroLiftCameraEnvCfg(), [override_arg])
    env_cfg.scene.num_envs = _VISUAL_DR_NUM_ENVS

    # The colours are written by a reset-mode event, so the frame the camera holds when reset returns
    # predates them unless the reset re-renders. This defaults to 0, which is why the other Dexsuite
    # lanes never need it: they capture straight after construction and nothing changes afterwards.
    env_cfg.num_rerenders_on_reset = _VISUAL_DR_RERENDERS_ON_RESET

    # Per-environment materials: the ``{ENV_REGEX_NS}`` prim path is what makes the scene clone one
    # material per environment instead of sharing a single bucket prim.
    for material_name in (_VISUAL_DR_ARM_MATERIAL, _VISUAL_DR_HAND_MATERIAL):
        setattr(
            env_cfg.scene,
            material_name,
            VisualMaterialCfg(
                prim_path="{ENV_REGEX_NS}/Materials/" + material_name,
                spawn=sim_utils.PbrMdlCfg(diffuse_color_constant=_VISUAL_DR_INITIAL_COLOR),
            ),
        )

    # Each part binds its own environment's clone through the same token.
    env_cfg.scene.robot.spawn.visual_material_bindings = {
        **{part: "{ENV_REGEX_NS}/Materials/" + _VISUAL_DR_ARM_MATERIAL for part in _VISUAL_DR_ARM_PARTS},
        **{part: "{ENV_REGEX_NS}/Materials/" + _VISUAL_DR_HAND_MATERIAL for part in _VISUAL_DR_HAND_PARTS},
    }

    env_cfg.events.robot_visual_color = EventTermCfg(
        func=env_mdp.randomize_visual_color,
        mode="reset",
        params={
            "materials": [SceneEntityCfg(_VISUAL_DR_ARM_MATERIAL), SceneEntityCfg(_VISUAL_DR_HAND_MATERIAL)],
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
        },
    )

    # Pin the robot pose. This golden exists to pin the randomized colours, and the reset otherwise
    # offsets every joint by +/-0.5 rad (and the wrist by +/-3), which makes the image depend on
    # where the reset lands in the RNG stream - so it shifts with the physics backend and with any
    # change to an event that draws ahead of these terms. Zeroing the ranges leaves the term running
    # and the robot in its default pose, so colour is what the comparison is actually measuring.
    reset_terms = env_cfg.events.conditional_reset.params["terms"]
    for term_name in ("reset_robot_joints", "reset_robot_wrist_joint"):
        if term_name in reset_terms:
            reset_terms[term_name].params["position_range"] = [0.0, 0.0]

    # Disable the observation point-cloud visualisation markers (/Visuals/ObservationPointCloud).
    # The underlying point sampling uses the global numpy/torch RNG, so marker positions shift
    # across processes and show up as random red dots in the rendered camera output.
    point_cloud_term = getattr(env_cfg.observations.perception, "object_point_cloud", None)
    if point_cloud_term is not None:
        point_cloud_term.params["visualize"] = False

    # The success and failure markers are placed exactly at the same location. If both markers are
    # visible, the rendering order will determine which one is visible in the camera output.
    for marker_cfg in env_cfg.commands.object_pose.success_visualizer_cfg.markers.values():
        marker_cfg.visible = False

    return env_cfg


def rendering_test_dexsuite_visual_randomization(
    physics_backend: str,
    renderer: str,
    comparison_scores: list[dict],
) -> None:
    """Test Dexsuite KukaAllegro rendering correctness with per-environment visual randomization.

    The robot's arm and hand bind two per-environment
    :class:`~isaaclab.assets.VisualMaterial` entities that a ``reset``-mode
    :func:`~isaaclab.envs.mdp.randomize_visual_color` term re-rolls. The colours are sampled from
    the seeded RNG, so the reset produces the same four differently coloured robots on every run and
    the camera output can be compared against a golden image like every other rendering lane.

    Args:
        physics_backend: Physics backend label (e.g. ``"physx"``, ``"newton"``).
        renderer: Camera renderer preset name (e.g. ``"isaacsim_rtx_renderer"``).
        comparison_scores: Module-local comparison score storage for the HTML report.
    """
    from isaaclab.envs import ManagerBasedRLEnv

    env_cfg = _make_dexsuite_visual_randomization_env_cfg(physics_backend, renderer)

    if renderer == "ovrtx_renderer":
        _redirect_ovrtx_renderer_log_to_stdout(env_cfg)

    env = None

    try:
        env = ManagerBasedRLEnv(env_cfg)
        # reset fires the per-environment colour term on both materials
        env.reset()
        maybe_save_stage(_VISUAL_DR_TEST_NAME, physics_backend, renderer, "rgb")
        validate_camera_outputs(
            _VISUAL_DR_TEST_NAME,
            physics_backend,
            renderer,
            env.scene.sensors[_VISUAL_DR_CAMERA].data.output,
            max_different_pixels_percentage=MAX_DIFFERENT_PIXELS_PERCENTAGE_BY_ENV_NAME[_VISUAL_DR_TEST_NAME],
            comparison_scores=comparison_scores,
        )
    finally:
        if env is not None:
            env.close()

            # This invokes camera sensor and renderer cleanup explicitly before pytest teardown, otherwise OV
            # native code could probably complain about leaks and trigger segmentation fault.
            env = None
