# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Heterogeneous multi-task Meta-World env: 15 real-asset tasks in one scene.

A single :class:`~isaaclab.envs.ManagerBasedRLEnv` runs all 15 real-asset
Meta-World tasks side by side. Each parallel env is assigned a single task
at construction; assets are cloned per-task via :class:`InclusionSet`, so
a "drawer-open" env contains a Sawyer + drawer (no button / window / etc.),
while a "button-press" env contains a Sawyer + button. Reward composition
reuses the archetype primitives in :mod:`mdp.reward_shapes`; per-task
routing is done via the ``@scatterable`` wrappers in
:mod:`mdp.scatter_rewards` — each :class:`RewardTermCfg` passes
``asset_cfg=SceneEntityCfg(<keypoint_frame>, groups=[task_name])`` so the
reward is written only to that task's clone-group envs.

Tasks (15):

* drawer-open / drawer-close                                   (mw_drawer)
* button-press-topdown / coffee-button                          (mw_button)
* window-open / window-close                                    (mw_window)
* faucet-open / faucet-close / dial-turn / lever-pull           (mw_faucet)
* door-open / door-close / door-lock / door-unlock              (mw_door)
* peg-insert-side                                               (mw_peg_block + cube cylinder)

MT3 (reach / push / pick-place) is registered separately because the cube
*is* the manipulandum there — different cube cfg from the placeholder used
by the 15 articulated-asset tasks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.cloner.cloner_strategies import interleaved
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import CloneCfg, InclusionSet
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from ... import metaworld_env_base
from ...mdp import (
    BUTTON_TARGET_RADIUS,
    DOOR_TARGET_RADIUS,
    DRAWER_TARGET_RADIUS,
    PEG_TARGET_RADIUS,
    WINDOW_TARGET_RADIUS,
    CagingTimesInPlaceShapeCfg,
    HamacherShapeCfg,
    MetaworldMultiTaskCommandCfg,
    MetaworldObservation,
    PhaseBonusCfg,
    SuccessOverrideCfg,
    TaskBoxCfg,
    ToleranceShapeCfg,
    TriggerCfg,
    axis_init_to_target_dist,
    axis_obj_to_target_dist,
    gripper_close_action,
    gripper_closed,
    init_tcp_to_keypoint_init_dist,
    metaworld_task_onehot,
    obj_init_to_target_dist,
    obj_to_target_dist,
    pick_place_caging,
    tcp_to_obj_dist,
)
from ...mdp.scatter_rewards import (
    caging_times_in_place_term,
    hamacher_term,
    keypoint_success_termination,
    success_indicator_term,
    tolerance_term,
)
from ...metaworld_assets_cfg import (
    MW_BUTTON_CFG,
    MW_DOOR_CFG,
    MW_DRAWER_CFG,
    MW_FAUCET_CFG,
    MW_PEG_BLOCK_CFG,
    MW_WINDOW_CFG,
    SAWYER_METAWORLD_CFG,
)
from ...metaworld_specs import TASK_SPECS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ── Task list (order is the task_index used by ``MetaworldMultiTaskCommand``) ─

TASK_NAMES: list[str] = [
    "drawer_open",
    "drawer_close",
    "button_press_topdown",
    "coffee_button",
    "window_open",
    "window_close",
    "faucet_open",
    "faucet_close",
    "dial_turn",
    "lever_pull",
    "door_open",
    "door_close",
    "door_lock",
    "door_unlock",
    "peg_insert_side",
]
"""Ordered task list. Index in this list = ``task_id`` written by
:class:`MetaworldMultiTaskCommand`. The reward terms reference the same index
so that each per-task reward fires only on the matching envs."""

TASK_TO_INDEX: dict[str, int] = {n: i for i, n in enumerate(TASK_NAMES)}

# Asset → groups that include it. Drives the InclusionSet membership.
_ASSET_GROUPS: dict[str, list[str]] = {
    "mw_drawer": ["drawer_open", "drawer_close"],
    "mw_button": ["button_press_topdown", "coffee_button"],
    "mw_window": ["window_open", "window_close"],
    "mw_faucet": ["faucet_open", "faucet_close", "dial_turn", "lever_pull"],
    "mw_door": ["door_open", "door_close", "door_lock", "door_unlock"],
    "mw_peg_block": ["peg_insert_side"],
    # Per-asset keypoint frames travel with their asset.
    "drawer_keypoint": ["drawer_open", "drawer_close"],
    "button_keypoint": ["button_press_topdown", "coffee_button"],
    "window_keypoint": ["window_open", "window_close"],
    "faucet_keypoint": ["faucet_open", "faucet_close", "dial_turn", "lever_pull"],
    "door_keypoint": ["door_open", "door_close", "door_lock", "door_unlock"],
    "peg_keypoint": ["peg_insert_side"],
}


def _build_clone_groups() -> dict[str, InclusionSet]:
    """Invert ``_ASSET_GROUPS`` into one ``InclusionSet`` per task."""
    per_group: dict[str, list[str]] = {n: [] for n in TASK_NAMES}
    for asset, groups in _ASSET_GROUPS.items():
        for g in groups:
            per_group[g].append(asset)
    return {g: InclusionSet(assets=assets, weight=1) for g, assets in per_group.items()}


# ── Scene ───────────────────────────────────────────────────────────────────


@configclass
class MetaworldMultiTaskSceneCfg(metaworld_env_base.MetaworldSceneCfg):
    """Heterogeneous scene with all 6 MW manipulanda + per-asset keypoint frames.

    Globals (every env):
        * ``robot``         — Sawyer arm.
        * ``cube``          — invisible 1 mm anchor at (0, 0.6, 0.02). The
          :class:`MetaworldMultiTaskCommand` writes the per-env goal to it
          on reset; reward terms read the manipulandum via the per-asset
          ``*_keypoint`` frames, never via the cube.
        * ``tcp_frame``     — Sawyer leftpad/rightpad ``FrameTransformer``.
        * ``ground``, ``light``, ``goal_marker``.

    Per-task (clone-group-local):
        * ``mw_drawer`` + ``drawer_keypoint``       → drawer_open/close.
        * ``mw_button`` + ``button_keypoint``       → button_press / coffee.
        * ``mw_window`` + ``window_keypoint``       → window_open / close.
        * ``mw_faucet`` + ``faucet_keypoint``       → faucet/dial/lever-pull.
        * ``mw_door``   + ``door_keypoint``         → door_open/close/lock/unlock.
        * ``mw_peg_block`` + ``peg_keypoint``       → peg_insert_side.
    """

    robot = SAWYER_METAWORLD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.001, 0.001, 0.001),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
            collision_props=None,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0), opacity=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.6, 0.02)),
    )

    # ── Per-task articulated assets ──────────────────────────────────────────
    mw_drawer = MW_DRAWER_CFG.replace(prim_path="{ENV_REGEX_NS}/MwDrawer")
    mw_button = MW_BUTTON_CFG.replace(prim_path="{ENV_REGEX_NS}/MwButton")
    mw_window = MW_WINDOW_CFG.replace(prim_path="{ENV_REGEX_NS}/MwWindow")
    mw_faucet = MW_FAUCET_CFG.replace(prim_path="{ENV_REGEX_NS}/MwFaucet")
    mw_door = MW_DOOR_CFG.replace(prim_path="{ENV_REGEX_NS}/MwDoor")
    mw_peg_block = MW_PEG_BLOCK_CFG.replace(prim_path="{ENV_REGEX_NS}/MwPegBlock")

    # ── Per-asset keypoint frames ────────────────────────────────────────────

    drawer_keypoint: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/MwDrawer/drawercase",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/MwDrawer/drawer_handle",
                name="kp",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ),
        ],
    )

    button_keypoint: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/MwButton/button_box",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/MwButton/button_top",
                name="kp",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ),
        ],
    )

    window_keypoint: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/MwWindow/window_frame",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/MwWindow/window_handle",
                name="kp",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ),
        ],
    )

    faucet_keypoint: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/MwFaucet/faucet_base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/MwFaucet/handle_tip",
                name="kp",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ),
        ],
    )

    door_keypoint: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/MwDoor/door_frame",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/MwDoor/door_handle",
                name="kp",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ),
        ],
    )

    peg_keypoint: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/MwPegBlock/block",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/MwPegBlock/hole",
                name="kp",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ),
        ],
    )

    tcp_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/leftpad",
                name="leftpad",
                offset=OffsetCfg(pos=(0.0, 0.0, -0.045)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/rightpad",
                name="rightpad",
                offset=OffsetCfg(pos=(0.0, 0.0, -0.045)),
            ),
        ],
    )

    # The heterogeneous cloner: each task group gets its asset(s) + keypoint.
    # ``interleaved`` aligns env→group with the round-robin task_id from
    # ``MetaworldMultiTaskCommand`` (env i → group i % n_tasks); the default
    # ``random`` strategy would scatter envs so most envs end up with the
    # wrong asset for their assigned task, diluting the gradient signal.
    clone_cfg = CloneCfg(clone_groups=_build_clone_groups(), clone_strategy=interleaved)


# ── Commands: one ``MetaworldMultiTaskCommand`` with 15 task boxes ──────────
#
# Per-task (obj_init, goal) come from the shared :data:`TASK_SPECS` so single-
# task and multi-task envs cannot drift apart.


@configclass
class _MultiTaskCommandsCfg:
    ee_pose = MetaworldMultiTaskCommandCfg(
        resampling_time_range=(1.0e6, 1.0e6),
        debug_vis=False,
        object_name="cube",
        frame_transformer_name="tcp_frame",
        min_xy_separation=0.0,
        tasks=[
            TaskBoxCfg(
                name=name,
                obj_low=TASK_SPECS[name].obj_range()[0],
                obj_high=TASK_SPECS[name].obj_range()[1],
                goal_low=TASK_SPECS[name].goal_range()[0],
                goal_high=TASK_SPECS[name].goal_range()[1],
            )
            for name in TASK_NAMES
        ],
    )


# ── Observations: standard 39-d MW state + task one-hot ────────────────────


@configclass
class _MultiTaskObsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        state = ObservationTermCfg(
            func=MetaworldObservation,
            params={
                "frame_transformer_cfg": SceneEntityCfg("tcp_frame"),
                "object1_cfg": SceneEntityCfg("cube"),
                "object2_cfg": None,
                "goal_command_name": "ee_pose",
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )
        task_onehot = ObservationTermCfg(
            func=metaworld_task_onehot,
            params={"command_name": "ee_pose"},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ── Reward archetype helpers (parameterised by keypoint-frame asset name) ──
#
# Mirror the helpers in env_cfgs.py but accept the ``keypoint_frame`` name as
# a parameter so each task can point at its own per-asset frame transformer.

_TCP_CFG = SceneEntityCfg("tcp_frame")


def _kp(name: str) -> SceneEntityCfg:
    return SceneEntityCfg(name)


def _reach_term(keypoint_frame: str, *, success_radius: float = 0.02, sigmoid: str = "long_tail") -> ToleranceShapeCfg:
    return ToleranceShapeCfg(
        distance=tcp_to_obj_dist,
        distance_kwargs={"keypoint_frame_cfg": _kp(keypoint_frame), "frame_transformer_cfg": _TCP_CFG},
        margin=tcp_to_obj_dist,
        margin_kwargs={"keypoint_frame_cfg": _kp(keypoint_frame), "frame_transformer_cfg": _TCP_CFG},
        success_radius=success_radius,
        sigmoid=sigmoid,
    )


def _in_place_term(keypoint_frame: str, success_radius: float, axis: int | None = None) -> ToleranceShapeCfg:
    if axis is None:
        return ToleranceShapeCfg(
            distance=obj_to_target_dist,
            distance_kwargs={"keypoint_frame_cfg": _kp(keypoint_frame), "goal_command_name": "ee_pose"},
            margin=obj_init_to_target_dist,
            margin_kwargs={"goal_command_name": "ee_pose"},
            success_radius=success_radius,
        )
    return ToleranceShapeCfg(
        distance=axis_obj_to_target_dist,
        distance_kwargs={"keypoint_frame_cfg": _kp(keypoint_frame), "axis": axis},
        margin=axis_init_to_target_dist,
        margin_kwargs={"axis": axis},
        success_radius=success_radius,
    )


def _hamacher_cfg(
    keypoint_frame: str,
    *,
    success_radius: float,
    axis: int | None = None,
    reach_sigmoid: str = "long_tail",
    a_modulator=None,
    success_override_threshold: float | None = None,
    scale: float = 10.0,
) -> HamacherShapeCfg:
    return HamacherShapeCfg(
        term_a=_reach_term(keypoint_frame, sigmoid=reach_sigmoid),
        term_b=_in_place_term(keypoint_frame, success_radius=success_radius, axis=axis),
        a_modulator=a_modulator,
        scale=scale,
        success_override=(
            SuccessOverrideCfg(
                quantity=obj_to_target_dist,
                atom_kwargs={"keypoint_frame_cfg": _kp(keypoint_frame), "goal_command_name": "ee_pose"},
                threshold=success_override_threshold,
                op="<=",
                value=scale,
            )
            if success_override_threshold is not None
            else None
        ),
    )


def _scatter(
    task_name: str,
    scatter_func,
    cfg,
    keypoint_frame: str,
    *,
    weight: float = 1.0,
) -> RewardTermCfg:
    """Build a :class:`RewardTermCfg` that routes ``scatter_func`` to envs in
    clone group ``task_name`` only.

    The shape primitive (``cfg``) computes for all envs (its inner
    quantities use bare :class:`SceneEntityCfg` references), and the
    :func:`scatter_rewards` decorator writes the result back only on
    ``asset_cfg.env_ids`` — i.e. envs where ``keypoint_frame`` is
    spawned, which by construction is the task's clone-group envs.
    """
    return RewardTermCfg(
        func=scatter_func,
        weight=weight,
        params={
            "cfg": cfg,
            "asset_cfg": SceneEntityCfg(keypoint_frame, groups=[task_name]),
        },
    )


def _hamacher_reward(
    task_name: str,
    keypoint_frame: str,
    *,
    success_radius: float,
    axis: int | None = None,
    reach_sigmoid: str = "long_tail",
    a_modulator=None,
    success_override_threshold: float | None = None,
    scale: float = 10.0,
    weight: float = 1.0,
) -> RewardTermCfg:
    """``hamacher_shape(...)`` scattered onto envs in clone group ``task_name``."""
    cfg = _hamacher_cfg(
        keypoint_frame,
        success_radius=success_radius,
        axis=axis,
        reach_sigmoid=reach_sigmoid,
        a_modulator=a_modulator,
        success_override_threshold=success_override_threshold,
        scale=scale,
    )
    return _scatter(task_name, hamacher_term, cfg, keypoint_frame, weight=weight)


def _success_term(task_name: str, keypoint_frame: str, threshold: float) -> RewardTermCfg:
    """``success_indicator_term`` — keypoint-to-goal binary, logged only (weight=0)."""
    return RewardTermCfg(
        func=success_indicator_term,
        weight=0.0,
        params={
            "keypoint_frame_cfg": _kp(keypoint_frame),
            "asset_cfg": SceneEntityCfg(keypoint_frame, groups=[task_name]),
            "goal_command_name": "ee_pose",
            "threshold": threshold,
        },
    )


def _button_press_cfg(keypoint_frame: str) -> HamacherShapeCfg:
    """Button-press reward shape parameterised by keypoint-frame name."""
    return HamacherShapeCfg(
        term_a=ToleranceShapeCfg(
            distance=tcp_to_obj_dist,
            distance_kwargs={"keypoint_frame_cfg": _kp(keypoint_frame), "frame_transformer_cfg": _TCP_CFG},
            margin=tcp_to_obj_dist,
            margin_kwargs={"keypoint_frame_cfg": _kp(keypoint_frame), "frame_transformer_cfg": _TCP_CFG},
            success_radius=0.01,
        ),
        a_modulator=gripper_closed,
        term_b=ToleranceShapeCfg(
            distance=axis_obj_to_target_dist,
            distance_kwargs={"keypoint_frame_cfg": _kp(keypoint_frame), "axis": 2},
            margin=axis_init_to_target_dist,
            margin_kwargs={"axis": 2},
            success_radius=0.005,
        ),
        scale=5.0,
        phase=PhaseBonusCfg(
            triggers=[
                TriggerCfg(
                    atom=tcp_to_obj_dist,
                    op="<=",
                    threshold=0.03,
                    atom_kwargs={"keypoint_frame_cfg": _kp(keypoint_frame), "frame_transformer_cfg": _TCP_CFG},
                )
            ],
            offset=0.0,
            in_place_mult=5.0,
            self_mult=0.0,
        ),
    )


# ── Drawer-Open: two additive ToleranceShape terms (caging xy + opening) ───

_DRAWER_OPEN_CAGING = ToleranceShapeCfg(
    distance=tcp_to_obj_dist,
    distance_kwargs={
        "keypoint_frame_cfg": _kp("drawer_keypoint"),
        "frame_transformer_cfg": _TCP_CFG,
        "scale": (3.0, 3.0, 1.0),
    },
    margin=init_tcp_to_keypoint_init_dist,
    margin_kwargs={
        "goal_command_name": "ee_pose",
        "keypoint_init_offset": (0.0, 0.16, 0.0),
        "scale": (3.0, 3.0, 1.0),
    },
    success_radius=0.01,
    scale=1.0,
)
_DRAWER_OPEN_OPENING = ToleranceShapeCfg(
    distance=obj_to_target_dist,
    distance_kwargs={"keypoint_frame_cfg": _kp("drawer_keypoint"), "goal_command_name": "ee_pose"},
    margin=obj_init_to_target_dist,
    margin_kwargs={"goal_command_name": "ee_pose"},
    success_radius=0.02,
    scale=1.0,
)


# ── Peg-Insert: caging-times-in-place ──────────────────────────────────────

_PEG_INSERT_CFG = CagingTimesInPlaceShapeCfg(
    caging=pick_place_caging,
    caging_kwargs={
        "keypoint_frame_cfg": _kp("peg_keypoint"),
        "frame_transformer_cfg": _TCP_CFG,
        "goal_command_name": "ee_pose",
    },
    distance=obj_to_target_dist,
    distance_kwargs={"keypoint_frame_cfg": _kp("peg_keypoint"), "goal_command_name": "ee_pose"},
    margin=obj_init_to_target_dist,
    margin_kwargs={"goal_command_name": "ee_pose"},
    success_radius=PEG_TARGET_RADIUS,
    success_override=SuccessOverrideCfg(
        quantity=obj_to_target_dist,
        op="<=",
        threshold=PEG_TARGET_RADIUS,
        value=1.0,
        atom_kwargs={"keypoint_frame_cfg": _kp("peg_keypoint"), "goal_command_name": "ee_pose"},
    ),
)


# ── Rewards: 15 task-masked terms + global action-rate ─────────────────────


@configclass
class _MultiTaskRewardsCfg:
    """One ``@scatterable`` reward term per task. Each term computes its
    shape primitive over all envs and writes the result back only on envs
    in clone group ``task_name`` (via ``asset_cfg=SceneEntityCfg(<keypoint>,
    groups=[task_name])``)."""

    # Drawer-Open: two additive tolerance terms (caging xy + opening), weight=5 each.
    drawer_open_caging = _scatter("drawer_open", tolerance_term, _DRAWER_OPEN_CAGING, "drawer_keypoint", weight=5.0)
    drawer_open_opening = _scatter("drawer_open", tolerance_term, _DRAWER_OPEN_OPENING, "drawer_keypoint", weight=5.0)
    drawer_open_success = _success_term("drawer_open", "drawer_keypoint", 0.03)

    # Drawer-Close: hamacher with gripper-close modulator + override.
    drawer_close = _hamacher_reward(
        "drawer_close",
        "drawer_keypoint",
        success_radius=DRAWER_TARGET_RADIUS,
        reach_sigmoid="gaussian",
        a_modulator=gripper_close_action,
        success_override_threshold=DRAWER_TARGET_RADIUS + 0.015,
    )
    drawer_close_success = _success_term("drawer_close", "drawer_keypoint", DRAWER_TARGET_RADIUS + 0.015)

    # Button-Press-Topdown / Coffee-Button: shared hamacher with gripper-closed modulator + phase bonus.
    button_press_topdown = _scatter(
        "button_press_topdown", hamacher_term, _button_press_cfg("button_keypoint"), "button_keypoint", weight=1.0
    )
    button_press_topdown_success = _success_term("button_press_topdown", "button_keypoint", BUTTON_TARGET_RADIUS)
    coffee_button = _scatter(
        "coffee_button", hamacher_term, _button_press_cfg("button_keypoint"), "button_keypoint", weight=1.0
    )
    coffee_button_success = _success_term("coffee_button", "button_keypoint", BUTTON_TARGET_RADIUS)

    # Window-Open / Window-Close: hamacher with axis=0 (x-only) in-place.
    window_open = _hamacher_reward(
        "window_open",
        "window_keypoint",
        success_radius=WINDOW_TARGET_RADIUS,
        axis=0,
        reach_sigmoid="long_tail",
    )
    window_open_success = _success_term("window_open", "window_keypoint", WINDOW_TARGET_RADIUS)
    window_close = _hamacher_reward(
        "window_close",
        "window_keypoint",
        success_radius=WINDOW_TARGET_RADIUS,
        axis=0,
        reach_sigmoid="gaussian",
    )
    window_close_success = _success_term("window_close", "window_keypoint", WINDOW_TARGET_RADIUS)

    # Faucet-Open / Faucet-Close / Dial-Turn / Lever-Pull: shared hamacher (axis=None).
    faucet_open = _hamacher_reward(
        "faucet_open",
        "faucet_keypoint",
        success_radius=0.05,
        axis=None,
        reach_sigmoid="long_tail",
    )
    faucet_open_success = _success_term("faucet_open", "faucet_keypoint", 0.05)
    faucet_close = _hamacher_reward(
        "faucet_close",
        "faucet_keypoint",
        success_radius=0.05,
        axis=None,
        reach_sigmoid="long_tail",
    )
    faucet_close_success = _success_term("faucet_close", "faucet_keypoint", 0.05)
    dial_turn = _hamacher_reward(
        "dial_turn",
        "faucet_keypoint",
        success_radius=0.05,
        axis=None,
        reach_sigmoid="long_tail",
    )
    dial_turn_success = _success_term("dial_turn", "faucet_keypoint", 0.05)
    lever_pull = _hamacher_reward(
        "lever_pull",
        "faucet_keypoint",
        success_radius=0.05,
        axis=None,
        reach_sigmoid="long_tail",
    )
    lever_pull_success = _success_term("lever_pull", "faucet_keypoint", 0.05)

    # Door-Open / Door-Close / Door-Lock / Door-Unlock: shared hamacher.
    door_open = _hamacher_reward(
        "door_open",
        "door_keypoint",
        success_radius=DOOR_TARGET_RADIUS,
        axis=None,
        reach_sigmoid="long_tail",
    )
    door_open_success = _success_term("door_open", "door_keypoint", DOOR_TARGET_RADIUS)
    door_close = _hamacher_reward(
        "door_close",
        "door_keypoint",
        success_radius=DOOR_TARGET_RADIUS,
        axis=None,
        reach_sigmoid="long_tail",
    )
    door_close_success = _success_term("door_close", "door_keypoint", DOOR_TARGET_RADIUS)
    door_lock = _hamacher_reward(
        "door_lock",
        "door_keypoint",
        success_radius=0.05,
        axis=None,
        reach_sigmoid="long_tail",
    )
    door_lock_success = _success_term("door_lock", "door_keypoint", 0.05)
    door_unlock = _hamacher_reward(
        "door_unlock",
        "door_keypoint",
        success_radius=0.05,
        axis=None,
        reach_sigmoid="long_tail",
    )
    door_unlock_success = _success_term("door_unlock", "door_keypoint", 0.05)

    # Peg-Insert-Side: caging-times-in-place with override.
    peg_insert_side = _scatter(
        "peg_insert_side", caging_times_in_place_term, _PEG_INSERT_CFG, "peg_keypoint", weight=10.0
    )
    peg_insert_side_success = _success_term("peg_insert_side", "peg_keypoint", PEG_TARGET_RADIUS)

    # Global action-rate penalty.
    action_rate = RewardTermCfg(func="isaaclab.envs.mdp:action_rate_l2", weight=-1e-4)


# ── Per-task success spec → termination + (logged) success reward share ───
# (task_name, keypoint_frame, threshold). Used to build the success
# ``RewardTermCfg``s above and the matching ``TerminationTermCfg``s below
# without re-typing the per-task thresholds. Order matches ``TASK_NAMES``.

_TASK_SUCCESS_SPECS: list[tuple[str, str, float]] = [
    ("drawer_open", "drawer_keypoint", 0.03),
    ("drawer_close", "drawer_keypoint", DRAWER_TARGET_RADIUS + 0.015),
    ("button_press_topdown", "button_keypoint", BUTTON_TARGET_RADIUS),
    ("coffee_button", "button_keypoint", BUTTON_TARGET_RADIUS),
    ("window_open", "window_keypoint", WINDOW_TARGET_RADIUS),
    ("window_close", "window_keypoint", WINDOW_TARGET_RADIUS),
    ("faucet_open", "faucet_keypoint", 0.05),
    ("faucet_close", "faucet_keypoint", 0.05),
    ("dial_turn", "faucet_keypoint", 0.05),
    ("lever_pull", "faucet_keypoint", 0.05),
    ("door_open", "door_keypoint", DOOR_TARGET_RADIUS),
    ("door_close", "door_keypoint", DOOR_TARGET_RADIUS),
    ("door_lock", "door_keypoint", 0.05),
    ("door_unlock", "door_keypoint", 0.05),
    ("peg_insert_side", "peg_keypoint", PEG_TARGET_RADIUS),
]


def _termination_term(task_name: str, keypoint_frame: str, threshold: float) -> TerminationTermCfg:
    """``keypoint_success_termination`` scattered to the task's clone-group envs."""
    return TerminationTermCfg(
        func=keypoint_success_termination,
        params={
            "keypoint_frame_cfg": _kp(keypoint_frame),
            "asset_cfg": SceneEntityCfg(keypoint_frame, groups=[task_name]),
            "goal_command_name": "ee_pose",
            "threshold": threshold,
        },
    )


def _build_termination_terms(specs: list[tuple[str, str, float]]) -> dict[str, TerminationTermCfg]:
    return {f"{task}_done": _termination_term(task, kp, thr) for task, kp, thr in specs}


@configclass
class _MultiTaskTerminationsCfg:
    """Truncate-on-horizon + per-task done-on-success (matches MW V2)."""

    time_out = TerminationTermCfg(func="isaaclab.envs.mdp:time_out", time_out=True)

    # Splat one per-task termination term as class attributes.
    locals().update(_build_termination_terms(_TASK_SUCCESS_SPECS))


# ── Events: per-asset, task-aware joint resets ─────────────────────────────


def task_indexed_joint_reset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    *,
    asset_cfg: SceneEntityCfg,
    joint_value_by_task: dict[str, float] | dict[int, float],
    command_name: str = "ee_pose",
) -> None:
    """Reset a single joint to a per-task value, vectorised across env_ids.

    ``joint_value_by_task`` maps task index → joint position. Keys are
    accepted as either ``int`` (legacy in-process call) or ``str`` (the form
    that survives Hydra's dict-of-int-keys serialization round-trip), and
    are coerced to ``int`` here so callers don't have to care.
    """
    if env_ids.numel() == 0:
        return
    cmd = env.command_manager.get_term(command_name)
    asset = env.scene[asset_cfg.name]

    task_ids_for_envs = cmd.task_id[env_ids]  # (N,)
    joint_pos = torch.zeros(env_ids.numel(), device=env.device)
    matched = torch.zeros_like(joint_pos, dtype=torch.bool)
    for raw_idx, value in joint_value_by_task.items():
        t_idx = int(raw_idx)
        mask = task_ids_for_envs == t_idx
        joint_pos = torch.where(mask, torch.full_like(joint_pos, value), joint_pos)
        matched |= mask
    target_envs = env_ids[matched]
    if target_envs.numel() == 0:
        return
    target_pos = joint_pos[matched].unsqueeze(-1)  # (M, 1)
    target_vel = torch.zeros_like(target_pos)

    asset.write_joint_position_to_sim_index(position=target_pos, joint_ids=asset_cfg.joint_ids, env_ids=target_envs)
    asset.write_joint_velocity_to_sim_index(velocity=target_vel, joint_ids=asset_cfg.joint_ids, env_ids=target_envs)


def _spec_reset_event(asset_name: str, task_names: list[str]) -> EventTermCfg:
    """Build a ``task_indexed_joint_reset`` event term sourcing the joint
    name + per-task reset values from :data:`TASK_SPECS`.

    All ``task_names`` must share the same ``joint_name`` (i.e. they all
    operate on the same asset's articulated joint) — this is asserted, since
    a mismatch would mean we're trying to reset two different joints with
    one event term.
    """
    joint_names = {TASK_SPECS[t].joint_name for t in task_names}
    assert len(joint_names) == 1, f"{asset_name}: tasks {task_names} disagree on joint_name {joint_names}"
    (joint_name,) = joint_names
    return EventTermCfg(
        func=task_indexed_joint_reset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(asset_name, joint_names=[joint_name]),
            "joint_value_by_task": {str(TASK_TO_INDEX[t]): TASK_SPECS[t].joint_reset_value for t in task_names},
        },
    )


@configclass
class _MultiTaskEventCfg(metaworld_env_base.MetaworldEventCfg):
    """Inherits the global Sawyer joint reset; adds per-asset task-aware
    joint resets that pick the right starting value per env's task_id.

    Per-task joint names + reset values come from :data:`TASK_SPECS` so this
    config doesn't drift from the single-task envs.
    """

    reset_drawer = _spec_reset_event("mw_drawer", ["drawer_open", "drawer_close"])
    reset_button = _spec_reset_event("mw_button", ["button_press_topdown", "coffee_button"])
    reset_window = _spec_reset_event("mw_window", ["window_open", "window_close"])
    reset_faucet = _spec_reset_event("mw_faucet", ["faucet_open", "faucet_close", "dial_turn", "lever_pull"])
    reset_door = _spec_reset_event("mw_door", ["door_open", "door_close", "door_lock", "door_unlock"])


# ── Top-level env cfg ──────────────────────────────────────────────────────


@configclass
class MetaworldMultiTaskSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    """Heterogeneous 15-task Meta-World env. Single ``MetaworldMultiTaskCommand``
    assigns each parallel env a ``task_id``; the cloner spawns the matching
    asset(s) per env via :class:`InclusionSet`; per-task rewards and joint
    resets read ``task_id`` to fire only on their matching envs."""

    scene: MetaworldMultiTaskSceneCfg = MetaworldMultiTaskSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _MultiTaskCommandsCfg()
    observations = _MultiTaskObsCfg()
    rewards = _MultiTaskRewardsCfg()
    events = _MultiTaskEventCfg()
    terminations = _MultiTaskTerminationsCfg()


# ── Smaller multi-task subsets (MT5 / MT10) ────────────────────────────────
#
# Both subsets reuse :class:`MetaworldMultiTaskSawyerEnvCfg`'s scene + reward
# composition; only the **active** task list (passed to
# :class:`MetaworldMultiTaskCommand`) is shorter, so the round-robin
# ``task_id = env_idx % N`` assignment only ever picks the chosen N.
#
# Reward terms are masked by ``TASK_TO_INDEX[name]`` — they're held for all 15
# tasks, but the mask never fires for tasks not in the active subset, so they
# contribute zero. (Marginal compute waste; far simpler than re-deriving a new
# reward graph per subset.)
#
# MT3 (reach / push / pick-place) is not included — those use the cube as the
# manipulandum and need a separate scene cfg.


@configclass
class MetaworldMT5SceneCfg(metaworld_env_base.MetaworldSceneCfg):
    """Scene for MT5 — drawer + button + window only (no faucet/door/peg)."""

    robot = SAWYER_METAWORLD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.001, 0.001, 0.001),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
            collision_props=None,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0), opacity=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.6, 0.02)),
    )

    mw_drawer = MW_DRAWER_CFG.replace(prim_path="{ENV_REGEX_NS}/MwDrawer")
    mw_button = MW_BUTTON_CFG.replace(prim_path="{ENV_REGEX_NS}/MwButton")
    mw_window = MW_WINDOW_CFG.replace(prim_path="{ENV_REGEX_NS}/MwWindow")

    drawer_keypoint: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/MwDrawer/drawercase",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/MwDrawer/drawer_handle",
                name="kp",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ),
        ],
    )
    button_keypoint: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/MwButton/button_box",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/MwButton/button_top",
                name="kp",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ),
        ],
    )
    window_keypoint: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/MwWindow/window_frame",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/MwWindow/window_handle",
                name="kp",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ),
        ],
    )

    tcp_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/leftpad",
                name="leftpad",
                offset=OffsetCfg(pos=(0.0, 0.0, -0.045)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/rightpad",
                name="rightpad",
                offset=OffsetCfg(pos=(0.0, 0.0, -0.045)),
            ),
        ],
    )

    clone_cfg = CloneCfg(
        clone_groups={
            "drawer_open": InclusionSet(assets=["mw_drawer", "drawer_keypoint"], weight=1),
            "drawer_close": InclusionSet(assets=["mw_drawer", "drawer_keypoint"], weight=1),
            "button_press_topdown": InclusionSet(assets=["mw_button", "button_keypoint"], weight=1),
            "coffee_button": InclusionSet(assets=["mw_button", "button_keypoint"], weight=1),
            "window_open": InclusionSet(assets=["mw_window", "window_keypoint"], weight=1),
        },
        clone_strategy=interleaved,
    )


@configclass
class MetaworldMT10SceneCfg(MetaworldMT5SceneCfg):
    """Scene for MT10 — MT5 + faucet (no door/peg)."""

    mw_faucet = MW_FAUCET_CFG.replace(prim_path="{ENV_REGEX_NS}/MwFaucet")

    faucet_keypoint: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/MwFaucet/faucet_base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/MwFaucet/handle_tip",
                name="kp",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ),
        ],
    )

    clone_cfg = CloneCfg(
        clone_groups={
            "drawer_open": InclusionSet(assets=["mw_drawer", "drawer_keypoint"], weight=1),
            "drawer_close": InclusionSet(assets=["mw_drawer", "drawer_keypoint"], weight=1),
            "button_press_topdown": InclusionSet(assets=["mw_button", "button_keypoint"], weight=1),
            "coffee_button": InclusionSet(assets=["mw_button", "button_keypoint"], weight=1),
            "window_open": InclusionSet(assets=["mw_window", "window_keypoint"], weight=1),
            "window_close": InclusionSet(assets=["mw_window", "window_keypoint"], weight=1),
            "faucet_open": InclusionSet(assets=["mw_faucet", "faucet_keypoint"], weight=1),
            "faucet_close": InclusionSet(assets=["mw_faucet", "faucet_keypoint"], weight=1),
            "dial_turn": InclusionSet(assets=["mw_faucet", "faucet_keypoint"], weight=1),
            "lever_pull": InclusionSet(assets=["mw_faucet", "faucet_keypoint"], weight=1),
        },
        clone_strategy=interleaved,
    )


@configclass
class _MT5RewardsCfg:
    """Reward terms for MT5: drawer × 2 + button × 2 + window-open."""

    drawer_open_caging = _scatter("drawer_open", tolerance_term, _DRAWER_OPEN_CAGING, "drawer_keypoint", weight=5.0)
    drawer_open_opening = _scatter("drawer_open", tolerance_term, _DRAWER_OPEN_OPENING, "drawer_keypoint", weight=5.0)
    drawer_open_success = _success_term("drawer_open", "drawer_keypoint", 0.03)

    drawer_close = _hamacher_reward(
        "drawer_close",
        "drawer_keypoint",
        success_radius=DRAWER_TARGET_RADIUS,
        reach_sigmoid="gaussian",
        a_modulator=gripper_close_action,
        success_override_threshold=DRAWER_TARGET_RADIUS + 0.015,
    )
    drawer_close_success = _success_term("drawer_close", "drawer_keypoint", DRAWER_TARGET_RADIUS + 0.015)

    button_press_topdown = _scatter(
        "button_press_topdown", hamacher_term, _button_press_cfg("button_keypoint"), "button_keypoint", weight=1.0
    )
    button_press_topdown_success = _success_term("button_press_topdown", "button_keypoint", BUTTON_TARGET_RADIUS)
    coffee_button = _scatter(
        "coffee_button", hamacher_term, _button_press_cfg("button_keypoint"), "button_keypoint", weight=1.0
    )
    coffee_button_success = _success_term("coffee_button", "button_keypoint", BUTTON_TARGET_RADIUS)

    window_open = _hamacher_reward(
        "window_open",
        "window_keypoint",
        success_radius=WINDOW_TARGET_RADIUS,
        axis=0,
        reach_sigmoid="long_tail",
    )
    window_open_success = _success_term("window_open", "window_keypoint", WINDOW_TARGET_RADIUS)

    action_rate = RewardTermCfg(func="isaaclab.envs.mdp:action_rate_l2", weight=-1e-4)


@configclass
class _MT10RewardsCfg(_MT5RewardsCfg):
    """MT5 + window-close + faucet × 4."""

    window_close = _hamacher_reward(
        "window_close",
        "window_keypoint",
        success_radius=WINDOW_TARGET_RADIUS,
        axis=0,
        reach_sigmoid="gaussian",
    )
    window_close_success = _success_term("window_close", "window_keypoint", WINDOW_TARGET_RADIUS)

    faucet_open = _hamacher_reward(
        "faucet_open", "faucet_keypoint", success_radius=0.05, axis=None, reach_sigmoid="long_tail"
    )
    faucet_open_success = _success_term("faucet_open", "faucet_keypoint", 0.05)
    faucet_close = _hamacher_reward(
        "faucet_close", "faucet_keypoint", success_radius=0.05, axis=None, reach_sigmoid="long_tail"
    )
    faucet_close_success = _success_term("faucet_close", "faucet_keypoint", 0.05)
    dial_turn = _hamacher_reward(
        "dial_turn", "faucet_keypoint", success_radius=0.05, axis=None, reach_sigmoid="long_tail"
    )
    dial_turn_success = _success_term("dial_turn", "faucet_keypoint", 0.05)
    lever_pull = _hamacher_reward(
        "lever_pull", "faucet_keypoint", success_radius=0.05, axis=None, reach_sigmoid="long_tail"
    )
    lever_pull_success = _success_term("lever_pull", "faucet_keypoint", 0.05)


@configclass
class _MT5EventCfg(metaworld_env_base.MetaworldEventCfg):
    """Reset events for MT5 — only drawer / button / window."""

    reset_drawer = _spec_reset_event("mw_drawer", ["drawer_open", "drawer_close"])
    reset_button = _spec_reset_event("mw_button", ["button_press_topdown", "coffee_button"])
    reset_window = _spec_reset_event("mw_window", ["window_open"])


@configclass
class _MT10EventCfg(metaworld_env_base.MetaworldEventCfg):
    """Reset events for MT10 — drawer / button / window / faucet."""

    reset_drawer = _spec_reset_event("mw_drawer", ["drawer_open", "drawer_close"])
    reset_button = _spec_reset_event("mw_button", ["button_press_topdown", "coffee_button"])
    reset_window = _spec_reset_event("mw_window", ["window_open", "window_close"])
    reset_faucet = _spec_reset_event("mw_faucet", ["faucet_open", "faucet_close", "dial_turn", "lever_pull"])


# Subsets of ``_TASK_SUCCESS_SPECS`` for the trimmed curricula. Keep only the
# task spec rows whose clone group exists in the corresponding scene; trying
# to resolve a missing-group ``SceneEntityCfg`` raises ``ValueError``.

_MT5_SUCCESS_SPECS = [s for s in _TASK_SUCCESS_SPECS if s[0] in TASK_NAMES[:5]]
_MT10_SUCCESS_SPECS = [s for s in _TASK_SUCCESS_SPECS if s[0] in TASK_NAMES[:10]]


@configclass
class _MT5TerminationsCfg:
    """Truncate-on-horizon + per-task done-on-success (5 active tasks)."""

    time_out = TerminationTermCfg(func="isaaclab.envs.mdp:time_out", time_out=True)
    locals().update(_build_termination_terms(_MT5_SUCCESS_SPECS))


@configclass
class _MT10TerminationsCfg:
    """Truncate-on-horizon + per-task done-on-success (10 active tasks)."""

    time_out = TerminationTermCfg(func="isaaclab.envs.mdp:time_out", time_out=True)
    locals().update(_build_termination_terms(_MT10_SUCCESS_SPECS))


@configclass
class MetaworldMT5SawyerEnvCfg(MetaworldMultiTaskSawyerEnvCfg):
    """MT5: first 5 of :data:`TASK_NAMES` —
    drawer-open, drawer-close, button-press-topdown, coffee-button, window-open."""

    scene: MetaworldMT5SceneCfg = MetaworldMT5SceneCfg(num_envs=4096, env_spacing=2.5)
    events = _MT5EventCfg()
    rewards = _MT5RewardsCfg()
    terminations = _MT5TerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.ee_pose.tasks = self.commands.ee_pose.tasks[:5]


@configclass
class MetaworldMT10SawyerEnvCfg(MetaworldMultiTaskSawyerEnvCfg):
    """MT10: first 10 of :data:`TASK_NAMES` (drawer ×2, button ×2, window ×2,
    faucet ×4)."""

    scene: MetaworldMT10SceneCfg = MetaworldMT10SceneCfg(num_envs=4096, env_spacing=2.5)
    events = _MT10EventCfg()
    rewards = _MT10RewardsCfg()
    terminations = _MT10TerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.ee_pose.tasks = self.commands.ee_pose.tasks[:10]
