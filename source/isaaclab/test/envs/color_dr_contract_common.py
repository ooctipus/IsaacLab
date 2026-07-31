# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared scene + assertions for the per-renderer visual-color contract tests.

Not named ``test_*`` so pytest does not collect it. The per-renderer files
(``test_color_randomization_contract_{rtx,newton,ovrtx}.py``) import :func:`assert_color_contract`
and call it with their renderer/physics cfgs; renderers cannot share a process, so each backend lives
in its own file (mirroring ``test_camera_ppisp_gaussian_{newton,ovrtx}``).

The scene declares two shared "bucket" materials (:class:`~isaaclab.assets.VisualMaterialCfg` with
OmniPBR) at ``/World/Materials`` and two cubes per env that bind one bucket each through an absolute
``visual_material_path``. Because the materials live outside the cloned environment namespace, every
environment binds the same two prims on every backend (Kit ``Sdf.CopySpec`` copies the absolute
binding targets verbatim; OVRTX/Newton clones reference the source prims). The contract:

* writing a bucket recolors the bound cube in *all* environments (cross-env propagation),
* the two buckets are independent (per-bucket distinctness),
* re-rolling one bucket leaves the other untouched.

The module also carries the *per-environment* contract (:func:`assert_per_env_color_contract`): a
second scene declares one logical material inside the environment namespace
(``{ENV_REGEX_NS}/Materials/style``) that the scene clones per environment, with each environment's
cube bound to its own clone through the same token. The contract inverts the bucket one:

* one write carrying per-env colors diverges the environments (env ``i`` renders color ``c_i``),
* ``env_ids`` is honored: re-rolling one environment leaves the others untouched,
* a bucket material in the same scene keeps propagating to all environments.

The written colors come from a fixed *saturated* palette and the rendered pixels are read back and
classified (nearest-palette by hue, robust to lighting / tonemapping). The isaac_rtx file launches
Kit (``AppLauncher``) before importing this module; the kit-less renderers (newton_warp / ovrtx)
need no launch.
"""

from __future__ import annotations

import numpy as np
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, VisualMaterialCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

# Saturated palette (linear RGB 0-1), one entry per bucket write. Saturated primaries survive
# lighting + tonemapping so the rendered pixel still classifies to the right entry.
_PALETTE = {
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
    "magenta": (1.0, 0.0, 1.0),
}
# cube_a is sampled at the right of the view, cube_b at the left (see _SceneCfg layout).
_RIGHT, _LEFT = 0.75, 0.25

# The default OmniPBR albedo authored on both buckets before any write (18% gray).
_INITIAL_COLOR = (0.18, 0.18, 0.18)


def _cube(prim_suffix: str, y: float, material_path: str) -> RigidObjectCfg:
    """Spawn a cube bound to one shared bucket material via an absolute material path."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + prim_suffix,
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material_path=material_path,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, y, 0.5)),
    )


@configclass
class _SceneCfg(InteractiveSceneCfg):
    # shared bucket materials: one prim each, outside the environment namespace, bound by the
    # corresponding cube in every environment.
    bucket_a = VisualMaterialCfg(
        prim_path="/World/Materials/bucket_a",
        spawn=sim_utils.PbrMdlCfg(diffuse_color_constant=_INITIAL_COLOR),
    )
    bucket_b = VisualMaterialCfg(
        prim_path="/World/Materials/bucket_b",
        spawn=sim_utils.PbrMdlCfg(diffuse_color_constant=_INITIAL_COLOR),
    )

    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=600.0))
    cube_a = _cube("cube_a", y=-0.8, material_path="/World/Materials/bucket_a")  # right of the view
    cube_b = _cube("cube_b", y=+0.8, material_path="/World/Materials/bucket_b")  # left of the view
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=CameraCfg.OffsetCfg(pos=(-2.5, 0.0, 0.5), rot=(0.0, 0.0, 0.0, 1.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955, clipping_range=(0.1, 100.0)),
        width=256,
        height=256,
    )


@configclass
class _ActionsCfg:
    pass  # no articulation -> no action terms


@configclass
class _ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        cube_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("cube_a")})

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class _EventCfg:
    # one generic term randomizes both buckets; every fire samples one color per bucket
    bucket_colors = EventTerm(
        func=mdp.randomize_visual_color,
        mode="reset",
        params={
            "materials": [SceneEntityCfg("bucket_a"), SceneEntityCfg("bucket_b")],
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
        },
    )


@configclass
class _ContractEnvCfg(ManagerBasedEnvCfg):
    scene = _SceneCfg(num_envs=2, env_spacing=8.0)
    actions = _ActionsCfg()
    observations = _ObservationsCfg()
    events = _EventCfg()

    def __post_init__(self):
        self.decimation = 1
        self.sim.dt = 0.01


@configclass
class _PerEnvSceneCfg(InteractiveSceneCfg):
    # one logical per-env material: spawned in the source environment, cloned per environment by
    # the scene; each environment's cube_a binds its own clone through the same token.
    style = VisualMaterialCfg(
        prim_path="{ENV_REGEX_NS}/Materials/style",
        spawn=sim_utils.PbrMdlCfg(diffuse_color_constant=_INITIAL_COLOR),
    )
    # a bucket material beside it: still shared by every environment (mode coexistence)
    bucket = VisualMaterialCfg(
        prim_path="/World/Materials/bucket",
        spawn=sim_utils.PbrMdlCfg(diffuse_color_constant=_INITIAL_COLOR),
    )

    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=600.0))
    cube_a = _cube("cube_a", y=-0.8, material_path="{ENV_REGEX_NS}/Materials/style")  # right of the view
    cube_b = _cube("cube_b", y=+0.8, material_path="/World/Materials/bucket")  # left of the view
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=CameraCfg.OffsetCfg(pos=(-2.5, 0.0, 0.5), rot=(0.0, 0.0, 0.0, 1.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955, clipping_range=(0.1, 100.0)),
        width=256,
        height=256,
    )


@configclass
class _PerEnvEventCfg:
    # the same generic color term: it follows the per-env granularity the entity declares
    style_colors = EventTerm(
        func=mdp.randomize_visual_color,
        mode="reset",
        params={
            "materials": SceneEntityCfg("style"),
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
        },
    )


@configclass
class _PerEnvContractEnvCfg(ManagerBasedEnvCfg):
    scene = _PerEnvSceneCfg(num_envs=2, env_spacing=8.0)
    actions = _ActionsCfg()
    observations = _ObservationsCfg()
    events = _PerEnvEventCfg()

    def __post_init__(self):
        self.decimation = 1
        self.sim.dt = 0.01


def _classify(pixel: np.ndarray) -> str:
    """Classify a pixel to the nearest palette entry by hue (cosine similarity).

    Direction-based, not Euclidean: brightness-invariant, so it is robust to the large exposure
    differences between renderers (kit-less Newton renders these saturated colors far dimmer than RTX,
    e.g. a dark ``[31, 31, 0]`` is still unmistakably *yellow* by direction).
    """
    vec = pixel.astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return "black"
    vec = vec / norm
    return max(_PALETTE, key=lambda k: float(np.dot(vec, np.asarray(_PALETTE[k]) / np.linalg.norm(_PALETTE[k]))))


def _set_color(env: ManagerBasedEnv, material_name: str, color_name: str) -> None:
    """Write one palette color to a bucket material through its scene entity."""
    material = env.scene[material_name]
    material.write_channels([material], {"color": torch.tensor([_PALETTE[color_name]])})


def _render_rgb(env: ManagerBasedEnv) -> np.ndarray:
    for _ in range(3):
        env.sim.step()
    camera = env.scene["camera"]
    camera.update(env.sim.cfg.dt)
    rgb = camera.data.output["rgb"]
    if not isinstance(rgb, torch.Tensor):
        rgb = rgb.torch
    return rgb.detach().cpu().numpy()


def _sample(rgb: np.ndarray, env_id: int, frac_x: float) -> np.ndarray:
    h, w = rgb.shape[1], rgb.shape[2]
    cx, cy, r = int(w * frac_x), h // 2, 3
    return rgb[env_id, cy - r : cy + r + 1, cx - r : cx + r + 1, :3].reshape(-1, 3).mean(0)


def assert_color_contract(renderer_cfg, physics_cfg=None) -> None:
    """Render the 2-cube scene on the given backend and assert the bucket visual-color contract.

    Args:
        renderer_cfg: The camera renderer cfg for the backend under test (e.g. ``IsaacRtxRendererCfg``,
            ``NewtonWarpRendererCfg``, ``OVRTXRendererCfg``).
        physics_cfg: The physics cfg for the backend (``NewtonCfg`` for the kit-less renderers); ``None``
            keeps the PhysX default (for ``isaac_rtx``).

    Asserts cross-env propagation of bucket writes, per-bucket distinctness, and bucket independence.
    """
    cfg = _ContractEnvCfg()
    cfg.scene.camera.renderer_cfg = renderer_cfg
    if physics_cfg is not None:
        cfg.sim.physics = physics_cfg

    # The caller launches Kit (via AppLauncher) before importing this module for the isaac_rtx renderer;
    # the kit-less renderers (newton_warp / ovrtx) need no launch. See the per-renderer test files.
    env = ManagerBasedEnv(cfg=cfg)
    try:
        # reset fires the randomize_visual_color term: both buckets leave their authored default
        env.reset()
        for name in ("bucket_a", "bucket_b"):
            sampled = env.scene[name].read_channel("color")
            assert sampled is not None and tuple(sampled) != _INITIAL_COLOR, f"event term did not write {name}"

        # fixed palette: bucket_a -> red, bucket_b -> green
        _set_color(env, "bucket_a", "red")
        _set_color(env, "bucket_b", "green")
        rgb = _render_rgb(env)

        env0 = (_classify(_sample(rgb, 0, _RIGHT)), _classify(_sample(rgb, 0, _LEFT)))
        env1 = (_classify(_sample(rgb, 1, _RIGHT)), _classify(_sample(rgb, 1, _LEFT)))

        # cross-env propagation + per-bucket distinctness: one write per bucket recolors every env
        assert env0 == ("red", "green"), f"env0 got {env0}"
        assert env1 == ("red", "green"), f"env1 got {env1}"

        # bucket independence: re-rolling bucket_a must not disturb bucket_b
        _set_color(env, "bucket_a", "magenta")
        rgb = _render_rgb(env)
        for env_id in (0, 1):
            right, left = _classify(_sample(rgb, env_id, _RIGHT)), _classify(_sample(rgb, env_id, _LEFT))
            assert right == "magenta", f"bucket_a re-roll did not land in env {env_id}: got {right}"
            assert left == "green", f"bucket_a re-roll disturbed bucket_b in env {env_id}: got {left}"
    finally:
        env.close()


def _set_env_colors(env: ManagerBasedEnv, material_name: str, color_names, env_ids=None) -> None:
    """Write one palette color per (selected) environment to a per-env material entity."""
    material = env.scene[material_name]
    colors = torch.tensor([[_PALETTE[name] for name in color_names]])
    material.write_channels([material], {"color": colors}, env_ids=env_ids)


def assert_per_env_color_contract(renderer_cfg, physics_cfg=None) -> None:
    """Render the per-env scene on the given backend and assert the per-environment color contract.

    Args:
        renderer_cfg: The camera renderer cfg for the backend under test.
        physics_cfg: The physics cfg for the backend; ``None`` keeps the PhysX default.

    Asserts per-environment divergence of one logical material, ``env_ids`` narrowing, and
    coexistence with a bucket material that still propagates to all environments.
    """
    cfg = _PerEnvContractEnvCfg()
    cfg.scene.camera.renderer_cfg = renderer_cfg
    if physics_cfg is not None:
        cfg.sim.physics = physics_cfg

    env = ManagerBasedEnv(cfg=cfg)
    try:
        # reset fires the randomize_visual_color term per environment
        env.reset()
        for env_id in (0, 1):
            sampled = env.scene["style"].read_channel("color", env_id=env_id)
            assert sampled is not None and tuple(sampled) != _INITIAL_COLOR, f"event did not write env {env_id}"

        # divergence: ONE call writes env 0 -> red and env 1 -> green on the same logical material;
        # the bucket keeps propagating to both environments
        _set_env_colors(env, "style", ("red", "green"))
        _set_color(env, "bucket", "blue")
        rgb = _render_rgb(env)
        env0 = (_classify(_sample(rgb, 0, _RIGHT)), _classify(_sample(rgb, 0, _LEFT)))
        env1 = (_classify(_sample(rgb, 1, _RIGHT)), _classify(_sample(rgb, 1, _LEFT)))
        assert env0 == ("red", "blue"), f"env0 got {env0}"
        assert env1 == ("green", "blue"), f"env1 got {env1}"

        # env_ids narrowing: re-roll only env 1; env 0 and the bucket must be untouched
        _set_env_colors(env, "style", ("magenta",), env_ids=torch.tensor([1]))
        rgb = _render_rgb(env)
        assert _classify(_sample(rgb, 0, _RIGHT)) == "red", "env_ids write leaked into env 0"
        assert _classify(_sample(rgb, 1, _RIGHT)) == "magenta", "env_ids write did not land in env 1"
        assert _classify(_sample(rgb, 0, _LEFT)) == "blue", "env_ids write disturbed the bucket"
    finally:
        env.close()
