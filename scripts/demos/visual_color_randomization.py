# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates visual material randomization with per-environment materials.

The same ANYmal-C asset is spawned in five appearance *styles* through
:class:`~isaaclab.sim.spawners.wrappers.MultiAssetSpawnerCfg`, assigned round-robin across
environments by heterogeneous cloning. Every style binds *per-environment*
:class:`~isaaclab.assets.VisualMaterial` entities (``{ENV_REGEX_NS}/Materials/...`` paths, cloned
with the environments), three per style — one each for the body, the legs, and the feet — so part
groups re-roll independently AND every robot of a style draws its own values (``env_ids`` is
honored on partial resets). ``randomize_visual_material`` event terms re-roll each material's
channels on every reset — granularity follows the material declaration, not the term:

* **original**: no bindings — the asset's authored multi-material appearance, untouched.
* **textured**: body, legs, and feet each swap which texture from the declared ``texture_pool``
  is shown and re-roll its tiling scale, per robot. The pool preloads every candidate once, so
  swaps are index writes with no runtime I/O. Pool textures are generic tileable patterns
  projected in object space (``project_uvw``) — a UV-atlas texture authored for another mesh
  would look scrambled by construction.
* **glass**: body, legs, and feet each re-roll their glass tint, frosting roughness, and index
  of refraction (refraction renders on the RTX backends; Newton mirrors only the color).
* **solid**: body, legs, and feet each re-roll a flat color.
* **mixed**: one part group from each family — textured body, solid legs, glass feet.

Run with ``--num_envs 10`` to see two robots of each style side by side and diverge.

.. code-block:: bash

    # Usage with default PhysX physics and default kit visualizer.
    ./isaaclab.sh -p scripts/demos/visual_color_randomization.py

    # Usage with Newton (MJWarp) physics.
    ./isaaclab.sh -p scripts/demos/visual_color_randomization.py --physics newton_mjwarp

"""

"""Parse CLI first so we can decide whether to launch Isaac Sim Kit."""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(
    description="This script demonstrates visual material randomization with per-environment materials.",
    conflict_handler="resolve",
)
parser.add_argument("--num_envs", type=int, default=8, help="Number of environments to spawn.")
parser.add_argument("--physics", default="physx", choices=["physx", "newton_mjwarp"], help="Physics backend.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli = parser.parse_args()

import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, VisualMaterialCfg
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.physics import PhysicsCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.configclass import configclass
from isaaclab.utils.timer import Timer

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort:skip

# generic tileable textures for the textured style: bespoke UV-atlas textures only look right on
# the exact mesh they were authored for, so the pool uses uniform patterns and the bucket enables
# ``project_uvw`` (object-space box projection) so any pool texture maps cleanly on any part
_TEXTURE_POOL_URLS = (
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Walnut_Planks/Walnut_Planks_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Bamboo_Planks/Bamboo_Planks_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Stone/Marble/Marble_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/RustedMetal/RustedMetal_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Brass/Brass_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Masonry/Brick_Pavers/Brick_Pavers_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Masonry/Concrete_Smooth/Concrete_Smooth_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Textiles/Leather_Brown/Leather_Brown_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Natural/Dirt/Dirt_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Roof_Tiles/Roof_Tiles_BaseColor.png",
)

_LEGS = ("LF", "LH", "RF", "RH")


@configclass
class VisualMaterialSceneCfg(InteractiveSceneCfg):
    """One styled ANYmal variant per environment (round-robin); three per-env materials per style."""

    # every material is per-environment ({ENV_REGEX_NS} paths clone them with the envs); each
    # style declares three — body, legs, feet — so part groups re-roll independently and every
    # robot of a style draws its own values
    texture_body = VisualMaterialCfg(
        prim_path="{ENV_REGEX_NS}/Materials/texture_body",
        spawn=sim_utils.PbrMdlCfg(project_uvw=True),
        channels=("uv_scale", "texture"),
        texture_pool=(),  # filled in main() once the pool textures are retrieved
    )
    texture_leg = VisualMaterialCfg(
        prim_path="{ENV_REGEX_NS}/Materials/texture_leg",
        spawn=sim_utils.PbrMdlCfg(project_uvw=True),
        channels=("uv_scale", "texture"),
        texture_pool=(),  # filled in main() once the pool textures are retrieved
    )
    texture_foot = VisualMaterialCfg(
        prim_path="{ENV_REGEX_NS}/Materials/texture_foot",
        spawn=sim_utils.PbrMdlCfg(project_uvw=True),
        channels=("uv_scale", "texture"),
        texture_pool=(),  # filled in main() once the pool textures are retrieved
    )
    glass_body = VisualMaterialCfg(
        prim_path="{ENV_REGEX_NS}/Materials/glass_body",
        spawn=sim_utils.GlassMdlCfg(glass_color=(0.8, 0.9, 1.0), glass_ior=1.5),
        channels=("color", "roughness", "ior"),
    )
    glass_leg = VisualMaterialCfg(
        prim_path="{ENV_REGEX_NS}/Materials/glass_leg",
        spawn=sim_utils.GlassMdlCfg(glass_color=(0.8, 0.9, 1.0), glass_ior=1.5),
        channels=("color", "roughness", "ior"),
    )
    glass_foot = VisualMaterialCfg(
        prim_path="{ENV_REGEX_NS}/Materials/glass_foot",
        spawn=sim_utils.GlassMdlCfg(glass_color=(0.8, 0.9, 1.0), glass_ior=1.5),
        channels=("color", "roughness", "ior"),
    )
    solid_body = VisualMaterialCfg(
        prim_path="{ENV_REGEX_NS}/Materials/solid_body",
        spawn=sim_utils.PbrMdlCfg(diffuse_color_constant=(0.8, 0.3, 0.1)),
    )
    solid_leg = VisualMaterialCfg(
        prim_path="{ENV_REGEX_NS}/Materials/solid_leg",
        spawn=sim_utils.PbrMdlCfg(diffuse_color_constant=(0.2, 0.2, 0.7)),
    )
    solid_foot = VisualMaterialCfg(
        prim_path="{ENV_REGEX_NS}/Materials/solid_foot",
        spawn=sim_utils.PbrMdlCfg(diffuse_color_constant=(0.1, 0.6, 0.2), reflection_roughness_constant=0.9),
    )

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())

    light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0)
    )

    # the same robot in five styled variants, assigned env_id % 5 by heterogeneous cloning
    robot: ArticulationCfg = ANYMAL_C_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                # original: the asset's authored appearance, untouched
                ANYMAL_C_CFG.spawn,
                # textured: body, legs, and feet follow their own per-env texture materials
                ANYMAL_C_CFG.spawn.replace(
                    visual_material_bindings={
                        "base/visuals": "{ENV_REGEX_NS}/Materials/texture_body",
                        **{
                            f"{leg}_{link}/visuals": "{ENV_REGEX_NS}/Materials/texture_leg"
                            for leg in _LEGS
                            for link in ("HIP", "THIGH", "SHANK")
                        },
                        **{f"{leg}_FOOT/visuals": "{ENV_REGEX_NS}/Materials/texture_foot" for leg in _LEGS},
                    }
                ),
                # glass: body, legs, and feet follow their own per-env glass materials
                ANYMAL_C_CFG.spawn.replace(
                    visual_material_bindings={
                        "base/visuals": "{ENV_REGEX_NS}/Materials/glass_body",
                        **{
                            f"{leg}_{link}/visuals": "{ENV_REGEX_NS}/Materials/glass_leg"
                            for leg in _LEGS
                            for link in ("HIP", "THIGH", "SHANK")
                        },
                        **{f"{leg}_FOOT/visuals": "{ENV_REGEX_NS}/Materials/glass_foot" for leg in _LEGS},
                    }
                ),
                # solid: body, legs, and feet follow their own per-env flat-color materials
                ANYMAL_C_CFG.spawn.replace(
                    visual_material_bindings={
                        "base/visuals": "{ENV_REGEX_NS}/Materials/solid_body",
                        **{
                            f"{leg}_{link}/visuals": "{ENV_REGEX_NS}/Materials/solid_leg"
                            for leg in _LEGS
                            for link in ("HIP", "THIGH", "SHANK")
                        },
                        **{f"{leg}_FOOT/visuals": "{ENV_REGEX_NS}/Materials/solid_foot" for leg in _LEGS},
                    }
                ),
                # mixed: one part group from each family — textured body, solid legs, glass feet
                ANYMAL_C_CFG.spawn.replace(
                    visual_material_bindings={
                        "base/visuals": "{ENV_REGEX_NS}/Materials/texture_body",
                        **{
                            f"{leg}_{link}/visuals": "{ENV_REGEX_NS}/Materials/solid_leg"
                            for leg in _LEGS
                            for link in ("HIP", "THIGH", "SHANK")
                        },
                        **{f"{leg}_FOOT/visuals": "{ENV_REGEX_NS}/Materials/glass_foot" for leg in _LEGS},
                    }
                ),
            ],
            random_choice=False,
        ),
    )


@configclass
class ActionsCfg:
    """Hold the default pose with a joint position action."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """One term per style: each fire re-rolls the style's three materials, per environment."""

    randomize_texture_style = EventTerm(
        func=mdp.randomize_visual_material,
        mode="reset",
        params={
            "materials": [
                SceneEntityCfg("texture_body"),
                SceneEntityCfg("texture_leg"),
                SceneEntityCfg("texture_foot"),
            ],
            "channels": {
                "texture": {"choices": []},  # filled in main() alongside the texture pool
                "uv_scale": ((0.5, 0.5), (3.0, 3.0)),
            },
        },
    )
    randomize_glass_style = EventTerm(
        func=mdp.randomize_visual_material,
        mode="reset",
        params={
            "materials": [SceneEntityCfg("glass_body"), SceneEntityCfg("glass_leg"), SceneEntityCfg("glass_foot")],
            # saturated tints and heavy frosting so the change reads even in the RTX real-time
            # viewport mode, which approximates glass (IOR shifts only show in path tracing)
            "channels": {
                "color": {"r": (0.05, 1.0), "g": (0.05, 1.0), "b": (0.05, 1.0)},
                "roughness": (0.0, 0.8),
                "ior": (1.1, 1.8),
            },
        },
    )
    randomize_solid_style = EventTerm(
        func=mdp.randomize_visual_material,
        mode="reset",
        params={
            "materials": [SceneEntityCfg("solid_body"), SceneEntityCfg("solid_leg"), SceneEntityCfg("solid_foot")],
            "channels": {"color": {"r": (0.05, 1.0), "g": (0.05, 1.0), "b": (0.05, 1.0)}},
        },
    )
    # Newton-only: per-SHAPE randomization below the material binding — every leg link re-rolls
    # its own color within whatever material it binds, which no material write can express and
    # only the Newton model represents (dropped on other backends in main())
    randomize_parts_per_env = EventTerm(
        func=mdp.randomize_visual_shape,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=[f"{leg}_{link}" for leg in _LEGS for link in ("SHANK", "FOOT", "HIP", "THIGH")]
            ),
            "channels": {"color": ((0.05, 0.05, 0.05), (1.0, 1.0, 1.0))},
        },
    )


@configclass
class VisualMaterialEnvCfg(ManagerBasedEnvCfg):
    scene: VisualMaterialSceneCfg = VisualMaterialSceneCfg(
        num_envs=4, env_spacing=1.5, random_heterogeneous_cloning=False
    )
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.sim.dt = 0.005


def main():
    """Main function."""
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        # runtime classes import USD; defer until after the launcher decided whether Kit owns it
        from isaaclab.envs import ManagerBasedEnv  # noqa: PLC0415

        env_cfg = VisualMaterialEnvCfg()
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device
        env_cfg.sim.physics = physics_cfg
        # per-shape visual randomization needs per-shape storage, which only the Newton
        # model provides — on other backends the term raises at construction, so drop it
        if not type(physics_cfg).__name__.lower().startswith("newton"):
            env_cfg.events.randomize_parts_per_env = None
        # resolve the texture pool to local files: declaring it preloads every candidate once,
        # so per-reset texture swaps stay free of runtime I/O
        texture_pool = tuple(retrieve_file_path(url) for url in _TEXTURE_POOL_URLS)
        for name in ("texture_body", "texture_leg", "texture_foot"):
            getattr(env_cfg.scene, name).texture_pool = texture_pool
        env_cfg.events.randomize_texture_style.params["channels"]["texture"]["choices"] = list(texture_pool)
        env = ManagerBasedEnv(cfg=env_cfg)

        actions = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        count = 0
        env.reset()
        while env.sim.is_headless_or_exist_active_visualizer():
            # reset every few seconds: every per-env material re-rolls, so each robot restyles
            # independently (the "original" robot never changes — that is its style)
            if count % 50 == 0:
                count = 0
                with Timer(name="reset", time_unit="ms") as reset_timer:
                    env.reset()
                reset_ms = reset_timer.total_run_time * 1000.0
                reset_mean_ms = Timer.timing_info["reset"]["mean"] * 1000.0

                def fmt(name: str, channel: str, env_id: int) -> str:
                    """Read one channel of one environment's material clone and format it."""
                    value = env.scene[name].read_channel(channel, env_id=env_id)
                    if channel == "texture":
                        # "…/Walnut_Planks/Walnut_Planks_BaseColor.png@" -> "Walnut_Planks"
                        return str(value).rsplit("/", 1)[-1].rstrip("@").removesuffix("_BaseColor.png")
                    if hasattr(value, "__len__"):
                        return "(" + ", ".join(f"{float(component):.2f}" for component in value) + ")"
                    return f"{float(value):.2f}"

                # one line per style per environment; values are always body / legs / feet
                lines = [f"[INFO]: Reset took {reset_ms:.1f} ms (mean {reset_mean_ms:.1f} ms)  [body / legs / feet]"]
                for label, names, channel in (
                    ("textured ", ("texture_body", "texture_leg", "texture_foot"), "texture"),
                    ("glass ior", ("glass_body", "glass_leg", "glass_foot"), "ior"),
                    ("solid rgb", ("solid_body", "solid_leg", "solid_foot"), "color"),
                ):
                    for env_id in range(min(env.num_envs, 2)):
                        prefix = f"  {label}" if env_id == 0 else "  " + " " * len(label)
                        values = " / ".join(fmt(name, channel, env_id) for name in names)
                        lines.append(f"{prefix}  env{env_id}  {values}")
                print("\n".join(lines))
            env.step(actions)
            count += 1

        env.close()


if __name__ == "__main__":
    main()
