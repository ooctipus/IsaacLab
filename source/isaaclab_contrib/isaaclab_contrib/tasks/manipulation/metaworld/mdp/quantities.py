# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Named geometric quantity atoms for Meta-World+ V2 reward composition.

Each atom returns a ``(num_envs,)`` torch tensor in env-local frame. Reward
shapes (in :mod:`reward_shapes`) compose these atoms with the fuzzy
primitives in :mod:`utils` (:func:`tolerance` + :func:`hamacher_product`).

The vocabulary covers MT3 today and most of MT50 with the existing helpers.
Tasks that need a new scalar (e.g. door-angle, dial-angle) add an atom here,
then any reward shape can reference it by passing the atom function in their
cfg.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .utils import hamacher_product, tolerance

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer

    from .commands import MetaworldPairedCommand


# ── helpers ────────────────────────────────────────────────────────────────


def _pad_pos_world(
    env: ManagerBasedRLEnv, frame_transformer_cfg: SceneEntityCfg
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(left_w, right_w, tcp_w)`` each ``(B, 3)``.

    Assumes the FrameTransformer's target frames are ordered
    ``(leftpad, rightpad)``.
    """
    ft: FrameTransformer = env.scene[frame_transformer_cfg.name]
    pos = ft.data.target_pos_w.torch  # (B, 2, 3)
    left = pos[:, 0]
    right = pos[:, 1]
    tcp = 0.5 * (left + right)
    return left, right, tcp


def _keypoint_pos_world(env: ManagerBasedRLEnv, keypoint_frame_cfg: SceneEntityCfg) -> torch.Tensor:
    """World position of the manipulandum keypoint (drawer handle / button top
    / cube root / faucet handle tip / etc.).

    The convention: every task's scene exposes a ``keypoint_frame``
    :class:`FrameTransformer` whose first target body is the manipulandum.
    Cube tasks point at the cube's root prim; real-asset tasks point at a
    welded marker body inside the asset articulation. Atoms and reward
    functions read the keypoint via this single helper — no per-task
    `object_cfg` dispatch required.

    Heterogeneous-scene support: when the keypoint frame transformer is
    cloned only into a subset of envs (i.e. the env uses ``clone_cfg`` to
    spawn the asset per task group), the underlying data buffer has shape
    ``(N_inst, K, 3)`` with ``N_inst < num_envs``. We scatter into a
    ``(num_envs, 3)`` buffer using the cloner layout so callers always see
    ``num_envs`` rows — non-asset envs read zeros, which downstream
    :func:`task_masked_reward` then masks out anyway.
    """
    ft: FrameTransformer = env.scene[keypoint_frame_cfg.name]
    raw = ft.data.target_pos_w.torch[:, 0]  # (N_inst, 3)
    if raw.shape[0] == env.num_envs:
        return raw
    # Heterogeneous: compose a full-env buffer.
    layout = env.scene.layout
    asset_groups = layout.assets.get(keypoint_frame_cfg.name)
    if not asset_groups:
        return raw  # Asset isn't registered with the layout (shouldn't happen).
    full = torch.zeros((env.num_envs, raw.shape[-1]), device=raw.device, dtype=raw.dtype)
    view = layout.get(list(asset_groups), keypoint_frame_cfg.name)
    src = raw if isinstance(view.view_ids, slice) else raw[view.view_ids]
    full[view.env_ids] = src
    return full


def _paired_command(env: ManagerBasedRLEnv, name: str) -> MetaworldPairedCommand:
    return env.command_manager.get_term(name)  # type: ignore[return-value]


# ── distance atoms (each → (B,) tensor) ────────────────────────────────────


def tcp_to_target_dist(
    env: ManagerBasedRLEnv,
    *,
    frame_transformer_cfg: SceneEntityCfg = SceneEntityCfg("tcp_frame"),
    goal_command_name: str = "ee_pose",
) -> torch.Tensor:
    """``‖tcp − goal‖`` in env-local frame [m]."""
    _, _, tcp_w = _pad_pos_world(env, frame_transformer_cfg)
    target_w = env.scene.env_origins + env.command_manager.get_command(goal_command_name)[:, 0:3]
    _viz_keypoint("tcp", tcp_w)
    _viz_keypoint("target", target_w)
    return torch.linalg.norm(tcp_w - target_w, dim=-1)


def obj_to_target_dist(
    env: ManagerBasedRLEnv,
    *,
    keypoint_frame_cfg: SceneEntityCfg = SceneEntityCfg("keypoint_frame"),
    goal_command_name: str = "ee_pose",
) -> torch.Tensor:
    """``‖keypoint − goal‖`` in env-local frame [m]."""
    obj_w = _keypoint_pos_world(env, keypoint_frame_cfg)
    target_w = env.scene.env_origins + env.command_manager.get_command(goal_command_name)[:, 0:3]
    _viz_keypoint("obj", obj_w)
    _viz_keypoint("target", target_w)
    return torch.linalg.norm(obj_w - target_w, dim=-1)


def tcp_to_obj_dist(
    env: ManagerBasedRLEnv,
    *,
    frame_transformer_cfg: SceneEntityCfg = SceneEntityCfg("tcp_frame"),
    keypoint_frame_cfg: SceneEntityCfg = SceneEntityCfg("keypoint_frame"),
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """``‖scale · (tcp − keypoint)‖`` in world frame [m].

    With ``scale=(1, 1, 1)`` this is a vanilla L2 distance. Pass an
    axis-emphasised ``scale`` (e.g. ``(3, 3, 1)`` for drawer-open's caging
    term, or ``(1, 2, 2)`` for peg-insert's in-place) to bias the metric.
    """
    obj_w = _keypoint_pos_world(env, keypoint_frame_cfg)
    _, _, tcp_w = _pad_pos_world(env, frame_transformer_cfg)
    _viz_keypoint("tcp", tcp_w)
    _viz_keypoint("obj", obj_w)
    s = torch.tensor(scale, device=env.device, dtype=tcp_w.dtype)
    return torch.linalg.norm((tcp_w - obj_w) * s, dim=-1)


def axis_obj_to_target_dist(
    env: ManagerBasedRLEnv,
    *,
    keypoint_frame_cfg: SceneEntityCfg = SceneEntityCfg("keypoint_frame"),
    goal_command_name: str = "ee_pose",
    axis: int = 0,
) -> torch.Tensor:
    """``|keypoint[axis] − goal[axis]|`` — single-axis distance.

    MW window/button reward use single-axis in-place: window-x for
    horizontal slider, button-z for vertical press. ``axis`` is one of
    0=x, 1=y, 2=z.
    """
    obj_w = _keypoint_pos_world(env, keypoint_frame_cfg)
    obj_e = obj_w - env.scene.env_origins
    target_e = env.command_manager.get_command(goal_command_name)[:, 0:3]
    return torch.abs(obj_e[:, axis] - target_e[:, axis])


def axis_init_to_target_dist(
    env: ManagerBasedRLEnv,
    *,
    goal_command_name: str = "ee_pose",
    axis: int = 0,
) -> torch.Tensor:
    """``|obj_init[axis] − target[axis]|`` — single-axis margin counterpart
    to :func:`axis_obj_to_target_dist`."""
    cmd = _paired_command(env, goal_command_name)
    target_e = env.command_manager.get_command(goal_command_name)[:, 0:3]
    return torch.abs(cmd.obj_init_pos_e[:, axis] - target_e[:, axis])


def gripper_close_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """``action[-1].clamp(0, 1)`` — the gripper-close command, in ``[0, 1]``.

    Used by drawer-close to gate the reach reward by gripper command
    (``H(reach × grip_close, in_place)``).
    """
    return env.action_manager.action[:, -1].clamp(min=0.0, max=1.0)


def gripper_closed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """``1 − gripper_open`` — the ``tcp_closed`` scalar in MW's button-press
    reward."""
    return 1.0 - gripper_open(env)


def init_tcp_to_keypoint_init_dist(
    env: ManagerBasedRLEnv,
    *,
    goal_command_name: str = "ee_pose",
    keypoint_init_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """``‖scale · (init_tcp − init_keypoint)‖``.

    Drawer-open's MW reward uses a synthesised init-handle position
    ``handle_pos_init = target + (0, max_dist, 0)``. To express that in
    the new keypoint world: pass ``keypoint_init_offset=(0, max_dist, 0)``
    so this returns ``‖scale · (init_tcp − (target + offset))‖``. Defaults
    to ``offset=0`` which gives the literal init-tcp-to-init-keypoint
    distance.
    """
    cmd = _paired_command(env, goal_command_name)
    target_e = env.command_manager.get_command(goal_command_name)[:, 0:3]
    offset = torch.tensor(keypoint_init_offset, device=env.device, dtype=target_e.dtype)
    handle_init_e = target_e + offset
    s = torch.tensor(scale, device=env.device, dtype=target_e.dtype)
    return torch.linalg.norm((cmd.init_tcp_e - handle_init_e) * s, dim=-1)


def obj_init_to_target_dist(
    env: ManagerBasedRLEnv,
    *,
    goal_command_name: str = "ee_pose",
) -> torch.Tensor:
    """``‖obj_init − goal‖`` [m]. Margin for ``in_place`` terms in object-
    centric tasks (push, pick-place)."""
    cmd = _paired_command(env, goal_command_name)
    target_e = env.command_manager.get_command(goal_command_name)[:, 0:3]
    return torch.linalg.norm(cmd.obj_init_pos_e - target_e, dim=-1)


def hand_init_to_target_dist(
    env: ManagerBasedRLEnv,
    *,
    goal_command_name: str = "ee_pose",
    hand_init_pos_e: tuple[float, float, float] = (0.0, 0.6, 0.2),
) -> torch.Tensor:
    """``‖hand_init − goal‖`` [m]. Margin for reach-style tolerance terms,
    using a *fixed* hand_init constant (matches Meta-World's ``hand_init_pos``
    after ``_reset_hand`` drives the mocap there)."""
    target_e = env.command_manager.get_command(goal_command_name)[:, 0:3]
    init_e = torch.tensor(hand_init_pos_e, device=env.device, dtype=target_e.dtype).expand_as(target_e)
    return torch.linalg.norm(init_e - target_e, dim=-1)


def init_tcp_to_target_dist(
    env: ManagerBasedRLEnv,
    *,
    goal_command_name: str = "ee_pose",
) -> torch.Tensor:
    """``‖init_tcp − goal‖`` [m].

    Reads the **actual** post-reset TCP captured by
    :class:`MetaworldPairedCommand`. This is the right margin atom for
    setups that don't drive the arm to ``hand_init_pos`` at reset — the
    margin reflects how far the agent must travel from where it actually
    starts, so the long-tail tolerance gradient is always non-degenerate.
    """
    cmd = _paired_command(env, goal_command_name)
    target_e = env.command_manager.get_command(goal_command_name)[:, 0:3]
    return torch.linalg.norm(cmd.init_tcp_e - target_e, dim=-1)


def obj_z_above_init(
    env: ManagerBasedRLEnv,
    *,
    keypoint_frame_cfg: SceneEntityCfg = SceneEntityCfg("keypoint_frame"),
    goal_command_name: str = "ee_pose",
) -> torch.Tensor:
    """``keypoint.z − init.z`` [m]. Positive when the manipulandum has been
    lifted above its reset height."""
    cmd = _paired_command(env, goal_command_name)
    obj_pos_e = _keypoint_pos_world(env, keypoint_frame_cfg) - env.scene.env_origins
    return obj_pos_e[:, 2] - cmd.obj_init_pos_e[:, 2]


_GRIPPER_OPEN_NORM: float = 0.1
"""Pad-pad distance [m] mapped to ``tcp_opened = 1.0`` (matches MW's clip)."""


# ── Inline reward-keypoint visualization ──────────────────────────────────
#
# Each atom that consumes 3D positions can side-effect-visualize them via
# :func:`_viz_keypoint`. Same pattern as
# ``isaaclab_tasks.manager_based.manipulation.dexsuite.mdp.observations.
# ObjectPointCloud`` — the function does its real work and visualizes the
# points it computed. A module-level toggle (:data:`_VIZ_ENABLED`) gates
# every call so visualization is free at training time and turned on at
# play / video time. Each label gets its own ``VisualizationMarkers`` prim
# instance, lazily created the first time the label is drawn.


_VIZ_ENABLED: bool = False
_VIZ_MARKERS: dict[str, VisualizationMarkers] = {}  # noqa: F821


_KEYPOINT_COLORS: dict[str, tuple[float, float, float]] = {
    "tcp": (1.0, 0.0, 0.0),  # red
    "obj": (0.0, 0.2, 1.0),  # blue
    "target": (0.0, 0.9, 0.2),  # green
    "leftpad": (1.0, 0.5, 0.0),  # orange
    "rightpad": (1.0, 0.0, 0.7),  # magenta
    "obj_init": (0.5, 0.5, 0.5),  # gray
    "init_tcp": (0.7, 0.4, 0.0),  # brown
    "hand_init": (0.6, 0.0, 0.6),  # purple
}


def set_viz_enabled(enabled: bool) -> None:
    """Toggle inline keypoint visualization. Call once at play time, before
    the env's first :meth:`step`. Atom side-effect ``visualize`` calls are a
    no-op when disabled (training default), so there's no perf cost."""
    global _VIZ_ENABLED
    _VIZ_ENABLED = enabled


def _viz_keypoint(label: str, position_w: torch.Tensor) -> None:
    """Draw a single keypoint marker per env at world-frame ``position_w``
    (shape ``(B, 3)``). No-op when :data:`_VIZ_ENABLED` is False."""
    if not _VIZ_ENABLED:
        return
    marker = _VIZ_MARKERS.get(label)
    if marker is None:
        # Lazy-create one PointInstancer per label. The imports are kept
        # local because they pull in ``pxr`` which can't be loaded before
        # AppLauncher boots (cfgs are parsed before kit starts).
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        color = _KEYPOINT_COLORS.get(label, (1.0, 1.0, 0.0))
        marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path=f"/Visuals/RewardKeypoints/{label}",
                markers={
                    "marker": sim_utils.SphereCfg(
                        radius=0.025,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, emissive_color=color),
                    ),
                },
            )
        )
        _VIZ_MARKERS[label] = marker
    marker.visualize(translations=position_w)


def gripper_open(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Meta-World's ``tcp_opened`` — normalised pad-pad distance in ``[0, 1]``.

    MW reads ``tcp_opened = obs[3]``, which is the *physical* pad gap
    normalised to ``[0, 1]`` (1 = fully open, 0 = fully closed). We compute
    it from the FrameTransformer's leftpad/rightpad positions, matching the
    same scalar already exposed in :class:`MetaworldObservation`'s obs[3].

    Earlier we approximated this from ``action[-1]`` (gripper command), but
    that misclassifies the "holding a cube" state: a closed-on-cube gripper
    has ``action[-1] ≈ +1`` (close command) but the actual pad gap is the
    cube width (≈ 3 cm), so ``tcp_opened ≈ 0.3``. Using the action gave a
    binary-like 0/1, breaking pick-place's lift-bonus trigger
    (``tcp_opened > 0`` AND ``obj_z > obj_init_z + 0.01``).
    """
    ft: FrameTransformer = env.scene["tcp_frame"]
    target_e = ft.data.target_pos_source.torch  # (B, 2, 3): (leftpad, rightpad)
    pad_gap = torch.linalg.norm(target_e[:, 0] - target_e[:, 1], dim=-1)
    return (pad_gap / _GRIPPER_OPEN_NORM).clamp(min=0.0, max=1.0)


# ── caging atoms ───────────────────────────────────────────────────────────


@dataclass
class GripperCagingParams:
    """Hyperparameters for the shared caging atom."""

    obj_radius: float = 0.015
    pad_success_thresh: float = 0.05
    object_reach_radius: float = 0.01
    xz_thresh: float = 0.005
    desired_gripper_effort: float = 1.0
    high_density: bool = False
    medium_density: bool = False


def gripper_caging(
    env: ManagerBasedRLEnv,
    *,
    frame_transformer_cfg: SceneEntityCfg = SceneEntityCfg("tcp_frame"),
    keypoint_frame_cfg: SceneEntityCfg = SceneEntityCfg("keypoint_frame"),
    goal_command_name: str = "ee_pose",
    params: GripperCagingParams = GripperCagingParams(),
) -> torch.Tensor:
    """Standard Meta-World caging reward in ``[0, 1]``.

    Torch port of :func:`metaworld.rewards.caging.gripper_caging_reward`.
    Used by push and most contact-heavy tasks (hammer, basketball, sweep, ...).
    """
    if params.high_density and params.medium_density:
        raise ValueError("high_density and medium_density are mutually exclusive")

    cmd = _paired_command(env, goal_command_name)
    env_origins = env.scene.env_origins

    obj_pos_w = _keypoint_pos_world(env, keypoint_frame_cfg)
    obj_pos_e = obj_pos_w - env_origins
    left_w, right_w, tcp_w = _pad_pos_world(env, frame_transformer_cfg)
    left_e = left_w - env_origins
    right_e = right_w - env_origins
    tcp_e = tcp_w - env_origins
    obj_init_pos_e = cmd.obj_init_pos_e
    init_tcp_e = cmd.init_tcp_e
    _viz_keypoint("leftpad", left_w)
    _viz_keypoint("rightpad", right_w)
    _viz_keypoint("obj", obj_pos_w)

    pad_y = torch.stack([left_e[:, 1], right_e[:, 1]], dim=-1)
    pad_to_obj = torch.abs(pad_y - obj_pos_e[:, 1:2])
    pad_to_objinit = torch.abs(pad_y - obj_init_pos_e[:, 1:2])
    caging_lr_margin = torch.abs(pad_to_objinit - params.pad_success_thresh)

    caging_left = tolerance(
        pad_to_obj[:, 0],
        bounds=(params.obj_radius, params.pad_success_thresh),
        margin=caging_lr_margin[:, 0],
        sigmoid="long_tail",
    )
    caging_right = tolerance(
        pad_to_obj[:, 1],
        bounds=(params.obj_radius, params.pad_success_thresh),
        margin=caging_lr_margin[:, 1],
        sigmoid="long_tail",
    )
    caging_y = hamacher_product(caging_left, caging_right)

    xz_idx = torch.tensor([0, 2], device=obj_pos_e.device)
    caging_xz_margin = torch.linalg.norm(obj_init_pos_e[:, xz_idx] - init_tcp_e[:, xz_idx], dim=-1) - params.xz_thresh
    caging_xz = tolerance(
        torch.linalg.norm(tcp_e[:, xz_idx] - obj_pos_e[:, xz_idx], dim=-1),
        bounds=(0.0, params.xz_thresh),
        margin=caging_xz_margin,
        sigmoid="long_tail",
    )

    grip_action = env.action_manager.action[:, -1].clamp(min=0.0, max=params.desired_gripper_effort)
    gripper_closed = grip_action / params.desired_gripper_effort

    caging = hamacher_product(caging_y, caging_xz)
    gripping = torch.where(caging > 0.97, gripper_closed, torch.zeros_like(caging))
    caging_and_gripping = hamacher_product(caging, gripping)

    if params.high_density:
        caging_and_gripping = (caging_and_gripping + caging) / 2
    if params.medium_density:
        tcp_to_obj = torch.linalg.norm(obj_pos_e - tcp_e, dim=-1)
        tcp_to_obj_init = torch.linalg.norm(obj_init_pos_e - init_tcp_e, dim=-1)
        reach_margin = torch.abs(tcp_to_obj_init - params.object_reach_radius)
        reach = tolerance(
            tcp_to_obj,
            bounds=(0.0, params.object_reach_radius),
            margin=reach_margin,
            sigmoid="long_tail",
        )
        caging_and_gripping = (caging_and_gripping + reach) / 2

    return caging_and_gripping


_PP_PAD_SUCCESS_MARGIN = 0.05
_PP_X_Z_SUCCESS_MARGIN = 0.005
_PP_OBJ_RADIUS = 0.015


def pick_place_caging(
    env: ManagerBasedRLEnv,
    *,
    frame_transformer_cfg: SceneEntityCfg = SceneEntityCfg("tcp_frame"),
    keypoint_frame_cfg: SceneEntityCfg = SceneEntityCfg("keypoint_frame"),
    goal_command_name: str = "ee_pose",
) -> torch.Tensor:
    """Pick-place's task-local caging variant.

    Differs from :func:`gripper_caging` by using *per-pad* init positions
    for the Y caging margins (instead of a shared obj_init Y baseline) and
    applying the high-density blend ``(caging+grip)/2`` always.
    """
    cmd = _paired_command(env, goal_command_name)
    env_origins = env.scene.env_origins

    obj_pos_w = _keypoint_pos_world(env, keypoint_frame_cfg)
    obj_pos_e = obj_pos_w - env_origins
    left_w, right_w, tcp_w = _pad_pos_world(env, frame_transformer_cfg)
    left_e = left_w - env_origins
    right_e = right_w - env_origins
    tcp_e = tcp_w - env_origins
    _viz_keypoint("leftpad", left_w)
    _viz_keypoint("rightpad", right_w)
    _viz_keypoint("obj", obj_pos_w)
    _viz_keypoint("tcp", tcp_w)
    init_left_pad_e = cmd.init_left_pad_e
    init_right_pad_e = cmd.init_right_pad_e
    obj_init_pos_e = cmd.obj_init_pos_e
    init_tcp_e = cmd.init_tcp_e

    delta_y_left = left_e[:, 1] - obj_pos_e[:, 1]
    delta_y_right = obj_pos_e[:, 1] - right_e[:, 1]
    right_caging_margin = torch.abs(torch.abs(obj_pos_e[:, 1] - init_right_pad_e[:, 1]) - _PP_PAD_SUCCESS_MARGIN)
    left_caging_margin = torch.abs(torch.abs(obj_pos_e[:, 1] - init_left_pad_e[:, 1]) - _PP_PAD_SUCCESS_MARGIN)
    right_caging = tolerance(
        delta_y_right,
        bounds=(_PP_OBJ_RADIUS, _PP_PAD_SUCCESS_MARGIN),
        margin=right_caging_margin,
        sigmoid="long_tail",
    )
    left_caging = tolerance(
        delta_y_left,
        bounds=(_PP_OBJ_RADIUS, _PP_PAD_SUCCESS_MARGIN),
        margin=left_caging_margin,
        sigmoid="long_tail",
    )
    y_caging = hamacher_product(left_caging, right_caging)

    tcp_xz = tcp_e.clone()
    tcp_xz[:, 1] = 0.0
    obj_xz = obj_pos_e.clone()
    obj_xz[:, 1] = 0.0
    tcp_obj_norm_xz = torch.linalg.norm(tcp_xz - obj_xz, dim=-1)

    init_obj_xz = obj_init_pos_e.clone()
    init_obj_xz[:, 1] = 0.0
    init_tcp_xz = init_tcp_e.clone()
    init_tcp_xz[:, 1] = 0.0
    tcp_obj_xz_margin = torch.linalg.norm(init_obj_xz - init_tcp_xz, dim=-1) - _PP_X_Z_SUCCESS_MARGIN

    x_z_caging = tolerance(
        tcp_obj_norm_xz,
        bounds=(0.0, _PP_X_Z_SUCCESS_MARGIN),
        margin=tcp_obj_xz_margin,
        sigmoid="long_tail",
    )

    # Match MW's ``pick_place_v2_caging`` exactly: hard ``caging > 0.97``
    # gate. If this gate is unreachable in practice for our setup, that's
    # a configuration issue elsewhere (geometry / FT offset / IK precision)
    # — fix it there, not by softening the reward.
    grip_close = env.action_manager.action[:, -1].clamp(min=0.0, max=1.0)
    caging = hamacher_product(y_caging, x_z_caging)
    gripping = torch.where(caging > 0.97, grip_close, torch.zeros_like(caging))
    caging_and_gripping = hamacher_product(caging, gripping)
    return (caging_and_gripping + caging) / 2
