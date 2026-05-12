# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-task env cfgs for the 15 Meta-World tasks with real assets.

Tasks: drawer×2, button×1 (+coffee-button), window×2, faucet×4
(open/close/dial/lever), door×4 (open/close/lock/unlock), peg-insert.
Reach/Push/Pick-Place are in their own files since the cube IS the
manipulandum for those — all the rest are here.

Each env wires:

* The Sawyer arm.
* A hand-authored MW-equivalent asset, verified at 0 mm joint→pose
  error in ``scripts/reinforcement_learning/rsl_rl/verify_mw_assets.py``.
* The corresponding V2 reward, reading the manipulandum via
  ``keypoint_frame_cfg`` (a :class:`FrameTransformer` on the asset's
  welded marker body).
* A reset event that places the asset's joint at the task's starting
  state and the Sawyer joints at their default ±0.05 m offset.

All cfgs share the same hidden anchor cube (required by the existing
:class:`MetaworldPairedCommand` machinery) and the same Sawyer scene
layout; only the asset, marker prim, joint name, and reward differ.
"""

from __future__ import annotations

from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
)
from isaaclab.utils import configclass

from ... import metaworld_env_base
from ...mdp import (
    BUTTON_TARGET_RADIUS,
    DOOR_TARGET_RADIUS,
    DRAWER_TARGET_RADIUS,
    PEG_TARGET_RADIUS,
    PICK_PLACE_SUCCESS_RADIUS,
    PICK_PLACE_TARGET_RADIUS,
    PUSH_TARGET_RADIUS,
    REACH_TARGET_RADIUS,
    WINDOW_TARGET_RADIUS,
    CagingTimesInPlaceShapeCfg,
    GripperCagingParams,
    HamacherShapeCfg,
    LinearComboShapeCfg,
    MetaworldObservation,
    PhaseBonusCfg,
    SuccessOverrideCfg,
    ToleranceShapeCfg,
    TriggerCfg,
    WeightedAtomCfg,
    axis_init_to_target_dist,
    axis_obj_to_target_dist,
    caging_times_in_place_shape,
    gripper_caging,
    gripper_close_action,
    gripper_closed,
    gripper_open,
    hamacher_shape,
    hand_init_to_target_dist,
    init_tcp_to_keypoint_init_dist,
    keypoint_at_target,
    linear_combo_shape,
    obj_init_to_target_dist,
    obj_to_target_dist,
    obj_z_above_init,
    pick_place_caging,
    reach_success,
    tcp_to_obj_dist,
    tcp_to_target_dist,
    tolerance_shape,
)
from ...metaworld_scenes_cfg import (
    SawyerBasketSceneCfg,
    SawyerBoxWithLidSceneCfg,
    SawyerButtonFrontSceneCfg,
    SawyerButtonFrontWithWallSceneCfg,
    SawyerButtonSceneCfg,
    SawyerButtonWithWallSceneCfg,
    SawyerCubeInHoleSceneCfg,
    SawyerCubeSceneCfg,
    SawyerCubeWithBinSceneCfg,
    SawyerCubeWithButtonSceneCfg,
    SawyerCubeWithPegSceneCfg,
    SawyerCubeWithStickSceneCfg,
    SawyerCubeWithWallSceneCfg,
    SawyerDoorSceneCfg,
    SawyerDrawerSceneCfg,
    SawyerFaucetSceneCfg,
    SawyerHandlePressSceneCfg,
    SawyerHandlePressSideSceneCfg,
    SawyerHoleBlockSceneCfg,
    SawyerNailBlockSceneCfg,
    SawyerPegInsertSceneCfg,
    SawyerPegUnplugSceneCfg,
    SawyerPlateSceneCfg,
    SawyerPlateSideSceneCfg,
    SawyerShelfSceneCfg,
    SawyerSoccerSceneCfg,
    SawyerWindowSceneCfg,
)

_TCP_CFG = SceneEntityCfg("tcp_frame")
_HANDLE_CFG = SceneEntityCfg("keypoint_frame")


# ── Generic observations (39-d MW state, cube anchor used for "object1") ────


@configclass
class _ObsCfg:
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

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ── Per-task command + event helpers ──────────────────────────────────────
#
# Pulled out to ``_helpers.py`` so the multi-task cfg can share them without
# re-importing the per-task ``MetaworldDrawerOpenSawyerEnvCfg`` etc.

from ._helpers import (  # noqa: E402
    _paired_from_spec,
    _reset_joint_from_spec,
    _reset_robot,
)

# ── Shared reward-shape factories ─────────────────────────────────────────
#
# Every Window/Door/Faucet/Dial/Lever-Pull task reduces to:
#     reward = α · H(tolerance(tcp→handle), tolerance(handle→goal))
# differing only in:
#   - axis used for the in_place distance (x for windows, full L2 for others
#     where MW used 3-axis already),
#   - success threshold (0.05 / 0.08 / 0.07 / etc.),
#   - sigmoid on the reach term (long_tail vs gaussian).
# The helpers below build the appropriate RewardTermCfg from these params.


def _reach_term(success_radius: float = 0.02, sigmoid: str = "long_tail") -> ToleranceShapeCfg:
    """``tolerance(|tcp − handle|)`` with the given bounds + sigmoid."""
    return ToleranceShapeCfg(
        distance=tcp_to_obj_dist,
        distance_kwargs={"keypoint_frame_cfg": _HANDLE_CFG, "frame_transformer_cfg": _TCP_CFG},
        margin=tcp_to_obj_dist,
        margin_kwargs={"keypoint_frame_cfg": _HANDLE_CFG, "frame_transformer_cfg": _TCP_CFG},
        success_radius=success_radius,
        sigmoid=sigmoid,
    )


def _in_place_term(success_radius: float, axis: int | None = None) -> ToleranceShapeCfg:
    """``tolerance(|handle − goal|)`` with optional single-axis distance.

    Set ``axis`` to 0/1/2 for x/y/z-only (window slider, button press
    z-only). Leave ``None`` for full L2 distance.
    """
    if axis is None:
        return ToleranceShapeCfg(
            distance=obj_to_target_dist,
            distance_kwargs={"keypoint_frame_cfg": _HANDLE_CFG, "goal_command_name": "ee_pose"},
            margin=obj_init_to_target_dist,
            margin_kwargs={"goal_command_name": "ee_pose"},
            success_radius=success_radius,
        )
    return ToleranceShapeCfg(
        distance=axis_obj_to_target_dist,
        distance_kwargs={"keypoint_frame_cfg": _HANDLE_CFG, "axis": axis},
        margin=axis_init_to_target_dist,
        margin_kwargs={"axis": axis},
        success_radius=success_radius,
    )


def _hamacher_reward(
    *,
    success_radius: float,
    axis: int | None = None,
    reach_sigmoid: str = "long_tail",
    a_modulator=None,
    success_override_threshold: float | None = None,
    scale: float = 10.0,
) -> RewardTermCfg:
    """``α · H(tolerance(reach), tolerance(in_place))`` with optional
    gripper modulator on reach and optional success override.

    Captures Window-Open (axis=0), Window-Close (sigmoid='gaussian'),
    Door-Open, Faucet-Open, Drawer-Close (with a_modulator and override),
    Lever-Pull, Dial-Turn, etc.
    """
    return RewardTermCfg(
        func=hamacher_shape,
        weight=1.0,
        params={
            "cfg": HamacherShapeCfg(
                term_a=_reach_term(sigmoid=reach_sigmoid),
                term_b=_in_place_term(success_radius=success_radius, axis=axis),
                a_modulator=a_modulator,
                scale=scale,
                success_override=(
                    SuccessOverrideCfg(
                        quantity=obj_to_target_dist,
                        atom_kwargs={"keypoint_frame_cfg": _HANDLE_CFG, "goal_command_name": "ee_pose"},
                        threshold=success_override_threshold,
                        op="<=",
                        value=scale,
                    )
                    if success_override_threshold is not None
                    else None
                ),
            )
        },
    )


def _success_term(threshold: float) -> RewardTermCfg:
    return RewardTermCfg(
        func=keypoint_at_target,
        weight=0.0,
        params={"keypoint_frame_cfg": _HANDLE_CFG, "goal_command_name": "ee_pose", "threshold": threshold},
    )


_ACTION_RATE = RewardTermCfg(func="isaaclab.envs.mdp:action_rate_l2", weight=-1e-4)


# ── Shared reward archetypes for cube tasks ────────────────────────────────
#
# These three reward shapes come up across many MT50 tasks beyond the original
# 15 articulated families. Each is parameterised so the same archetype can be
# reused with a per-task ``success_radius`` and ``hand_init_pos_e``.


def _push_reward(success_radius: float = PUSH_TARGET_RADIUS) -> RewardTermCfg:
    """Push-style ``linear_combo``: ``2 * caging + in_place + phase_bonus``.

    Used by push, push-back, push-wall, soccer, sweep, sweep-into. Identical
    shape to MW's ``push_v2`` (``self_mult=1`` in the phase bonus reproduces
    MW's ``reward += 1 + reward + 5 * in_place``).
    """
    return RewardTermCfg(
        func=linear_combo_shape,
        weight=1.0,
        params={
            "cfg": LinearComboShapeCfg(
                terms=[
                    WeightedAtomCfg(
                        weight=2.0,
                        atom=gripper_caging,
                        atom_kwargs={
                            "frame_transformer_cfg": _TCP_CFG,
                            "keypoint_frame_cfg": _HANDLE_CFG,
                            "goal_command_name": "ee_pose",
                            "params": GripperCagingParams(
                                obj_radius=0.015,
                                pad_success_thresh=0.05,
                                object_reach_radius=0.01,
                                xz_thresh=0.005,
                                high_density=True,
                            ),
                        },
                    ),
                ],
                in_place_distance=obj_to_target_dist,
                in_place_distance_kwargs={"keypoint_frame_cfg": _HANDLE_CFG, "goal_command_name": "ee_pose"},
                in_place_margin=obj_init_to_target_dist,
                in_place_margin_kwargs={"goal_command_name": "ee_pose"},
                in_place_success_radius=success_radius,
                phase=PhaseBonusCfg(
                    triggers=[
                        TriggerCfg(
                            atom=tcp_to_obj_dist,
                            op="<",
                            threshold=0.02,
                            atom_kwargs={"frame_transformer_cfg": _TCP_CFG, "keypoint_frame_cfg": _HANDLE_CFG},
                        ),
                        TriggerCfg(atom=gripper_open, op=">", threshold=0.0),
                    ],
                    offset=1.0,
                    in_place_mult=5.0,
                    self_mult=1.0,
                ),
                success_override=SuccessOverrideCfg(
                    quantity=obj_to_target_dist,
                    threshold=success_radius,
                    op="<",
                    value=10.0,
                    atom_kwargs={"keypoint_frame_cfg": _HANDLE_CFG, "goal_command_name": "ee_pose"},
                ),
            )
        },
    )


def _pick_place_reward(success_radius: float = PICK_PLACE_TARGET_RADIUS) -> RewardTermCfg:
    """Pick-place ``caging × in_place`` with lift bonus + override.

    Used by pick-place, pick-place-wall, basketball, shelf-place. Identical
    shape to MW's ``pick_place_v2``.
    """
    return RewardTermCfg(
        func=caging_times_in_place_shape,
        weight=1.0,
        params={
            "cfg": CagingTimesInPlaceShapeCfg(
                caging=pick_place_caging,
                caging_kwargs={
                    "frame_transformer_cfg": _TCP_CFG,
                    "keypoint_frame_cfg": _HANDLE_CFG,
                    "goal_command_name": "ee_pose",
                },
                distance=obj_to_target_dist,
                distance_kwargs={"keypoint_frame_cfg": _HANDLE_CFG, "goal_command_name": "ee_pose"},
                margin=obj_init_to_target_dist,
                margin_kwargs={"goal_command_name": "ee_pose"},
                success_radius=success_radius,
                phase=PhaseBonusCfg(
                    triggers=[
                        TriggerCfg(
                            atom=tcp_to_obj_dist,
                            op="<",
                            threshold=0.02,
                            atom_kwargs={"frame_transformer_cfg": _TCP_CFG, "keypoint_frame_cfg": _HANDLE_CFG},
                        ),
                        TriggerCfg(atom=gripper_open, op=">", threshold=0.0),
                        TriggerCfg(
                            atom=obj_z_above_init,
                            op=">",
                            threshold=0.01,
                            atom_kwargs={"keypoint_frame_cfg": _HANDLE_CFG, "goal_command_name": "ee_pose"},
                        ),
                    ],
                    offset=1.0,
                    in_place_mult=5.0,
                ),
                success_override=SuccessOverrideCfg(
                    quantity=obj_to_target_dist,
                    threshold=success_radius,
                    op="<",
                    value=10.0,
                    atom_kwargs={"keypoint_frame_cfg": _HANDLE_CFG, "goal_command_name": "ee_pose"},
                ),
            )
        },
    )


_DEFAULT_HAND_INIT_E = (-0.067, 0.571, 0.132)
"""Realised TCP world position from the Sawyer's default joint config (env-local frame)."""


def _reach_reward(success_radius: float = REACH_TARGET_RADIUS) -> RewardTermCfg:
    """Reach-style ``tolerance(tcp → goal)`` reward.

    Used by reach (MT3) and reach-wall. Margin is ``‖hand_init − goal‖``
    using the realised post-reset TCP position as ``hand_init`` (we don't
    drive the arm to a separate ``hand_init_pos`` like MW does).
    """
    return RewardTermCfg(
        func=tolerance_shape,
        weight=1.0,
        params={
            "cfg": ToleranceShapeCfg(
                distance=tcp_to_target_dist,
                distance_kwargs={"frame_transformer_cfg": _TCP_CFG, "goal_command_name": "ee_pose"},
                margin=hand_init_to_target_dist,
                margin_kwargs={"goal_command_name": "ee_pose", "hand_init_pos_e": _DEFAULT_HAND_INIT_E},
                success_radius=success_radius,
                scale=10.0,
            )
        },
    )


def _reach_success_term(threshold: float = REACH_TARGET_RADIUS) -> RewardTermCfg:
    """1.0 if TCP within ``threshold`` of goal — for reach-style success indicators."""
    return RewardTermCfg(
        func=reach_success,
        weight=0.0,
        params={"frame_transformer_cfg": _TCP_CFG, "goal_command_name": "ee_pose", "threshold": threshold},
    )


# ── Button-Press-Topdown ───────────────────────────────────────────────────


@configclass
class _ButtonCommandsCfg:
    """Button starts extended (z=0.13). Goal: button pressed (z=0.07)."""

    ee_pose = _paired_from_spec("button_press_topdown")


# Button-press reward shape, exported as a module-level RewardTermCfg so it
# can be reused by both _ButtonRewardsCfg (Button-Press-Topdown) and
# _CoffeeButtonRewardsCfg without duplicating the cfg literal.
_BUTTON_PRESS_TERM = RewardTermCfg(
    func=hamacher_shape,
    weight=1.0,
    params={
        "cfg": HamacherShapeCfg(
            # near_button = tolerance(|tcp - obj|, bounds=(0, 0.01))
            term_a=ToleranceShapeCfg(
                distance=tcp_to_obj_dist,
                distance_kwargs={"keypoint_frame_cfg": _HANDLE_CFG, "frame_transformer_cfg": _TCP_CFG},
                margin=tcp_to_obj_dist,
                margin_kwargs={"keypoint_frame_cfg": _HANDLE_CFG, "frame_transformer_cfg": _TCP_CFG},
                success_radius=0.01,
            ),
            a_modulator=gripper_closed,  # tcp_closed = 1 - gripper_open
            # button_pressed = tolerance(|target_z - obj_z|, bounds=(0, 0.005))
            term_b=ToleranceShapeCfg(
                distance=axis_obj_to_target_dist,
                distance_kwargs={"keypoint_frame_cfg": _HANDLE_CFG, "axis": 2},
                margin=axis_init_to_target_dist,
                margin_kwargs={"axis": 2},
                success_radius=0.005,
            ),
            scale=5.0,
            # Bonus: when tcp_to_obj <= 0.03, add 5 * button_pressed.
            phase=PhaseBonusCfg(
                triggers=[
                    TriggerCfg(
                        atom=tcp_to_obj_dist,
                        op="<=",
                        threshold=0.03,
                        atom_kwargs={"keypoint_frame_cfg": _HANDLE_CFG, "frame_transformer_cfg": _TCP_CFG},
                    )
                ],
                offset=0.0,
                in_place_mult=5.0,
                self_mult=0.0,
            ),
        )
    },
)


@configclass
class _ButtonRewardsCfg:
    """Button-press = ``5 · H(tcp_closed, near_button) + 5 · button_pressed``
    when ``tcp_to_obj < 0.03``."""

    button_press = _BUTTON_PRESS_TERM
    success = _success_term(BUTTON_TARGET_RADIUS)
    action_rate = _ACTION_RATE


@configclass
class _ButtonEventCfg:
    reset_robot_joints = _reset_robot()
    reset_button_extended = _reset_joint_from_spec("button_press_topdown")


@configclass
class MetaworldButtonPressTopdownSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerButtonSceneCfg = SawyerButtonSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _ButtonCommandsCfg()
    observations = _ObsCfg()
    rewards = _ButtonRewardsCfg()
    events = _ButtonEventCfg()


# ── Window-Open ─────────────────────────────────────────────────────────────


@configclass
class _WindowOpenCommandsCfg:
    """Window starts closed (joint 0, marker x=-0.04). Goal: open (joint 0.20, marker x=+0.16)."""

    ee_pose = _paired_from_spec("window_open")


@configclass
class _WindowOpenRewardsCfg:
    """``10 · H(reach[long_tail], in_place_x)``."""

    reward = _hamacher_reward(success_radius=WINDOW_TARGET_RADIUS, axis=0, reach_sigmoid="long_tail")
    success = _success_term(WINDOW_TARGET_RADIUS)
    action_rate = _ACTION_RATE


@configclass
class _WindowOpenEventCfg:
    reset_robot_joints = _reset_robot()
    reset_window_closed = _reset_joint_from_spec("window_open")


@configclass
class MetaworldWindowOpenSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerWindowSceneCfg = SawyerWindowSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _WindowOpenCommandsCfg()
    observations = _ObsCfg()
    rewards = _WindowOpenRewardsCfg()
    events = _WindowOpenEventCfg()


# ── Window-Close ────────────────────────────────────────────────────────────


@configclass
class _WindowCloseCommandsCfg:
    """Window starts open (marker x=+0.16). Goal: closed (marker x=-0.04)."""

    ee_pose = _paired_from_spec("window_close")


@configclass
class _WindowCloseRewardsCfg:
    """Same shape as :class:`_WindowOpenRewardsCfg` but reach uses gaussian sigmoid."""

    reward = _hamacher_reward(success_radius=WINDOW_TARGET_RADIUS, axis=0, reach_sigmoid="gaussian")
    success = _success_term(WINDOW_TARGET_RADIUS)
    action_rate = _ACTION_RATE


@configclass
class _WindowCloseEventCfg:
    reset_robot_joints = _reset_robot()
    reset_window_open = _reset_joint_from_spec("window_close")


@configclass
class MetaworldWindowCloseSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerWindowSceneCfg = SawyerWindowSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _WindowCloseCommandsCfg()
    observations = _ObsCfg()
    rewards = _WindowCloseRewardsCfg()
    events = _WindowCloseEventCfg()


# ── Faucet-Open ─────────────────────────────────────────────────────────────


@configclass
class _FaucetOpenCommandsCfg:
    """Faucet handle at angle 0 (tip at -y from base). Goal: angle +π/3 (tip rotated).
    Marker world positions per analytical verification."""

    ee_pose = _paired_from_spec("faucet_open")


@configclass
class _FaucetOpenRewardsCfg:
    """Same shape as :class:`_WindowOpenRewardsCfg` (Hamacher reach + in-place)."""

    reward = _hamacher_reward(success_radius=0.05, axis=None, reach_sigmoid="long_tail")
    success = _success_term(0.05)
    action_rate = _ACTION_RATE


@configclass
class _FaucetOpenEventCfg:
    reset_robot_joints = _reset_robot()
    reset_faucet_zero = _reset_joint_from_spec("faucet_open")


@configclass
class MetaworldFaucetOpenSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerFaucetSceneCfg = SawyerFaucetSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _FaucetOpenCommandsCfg()
    observations = _ObsCfg()
    rewards = _FaucetOpenRewardsCfg()
    events = _FaucetOpenEventCfg()


# ── Faucet-Close ────────────────────────────────────────────────────────────


@configclass
class _FaucetCloseCommandsCfg:
    """Faucet starts at +π/3, goal back to 0."""

    ee_pose = _paired_from_spec("faucet_close")


@configclass
class _FaucetCloseRewardsCfg:
    """Same shape as Faucet-Open."""

    reward = _hamacher_reward(success_radius=0.05, axis=None, reach_sigmoid="long_tail")
    success = _success_term(0.05)
    action_rate = _ACTION_RATE


@configclass
class _FaucetCloseEventCfg:
    reset_robot_joints = _reset_robot()
    reset_faucet_open = _reset_joint_from_spec("faucet_close")


@configclass
class MetaworldFaucetCloseSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerFaucetSceneCfg = SawyerFaucetSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _FaucetCloseCommandsCfg()
    observations = _ObsCfg()
    rewards = _FaucetCloseRewardsCfg()
    events = _FaucetCloseEventCfg()


# ── Door-Open ───────────────────────────────────────────────────────────────


@configclass
class _DoorOpenCommandsCfg:
    """Door closed (handle at (0.32, 0.87, 0.15)). Goal: door open ~60° (handle at (0.237, 0.721, 0.15))."""

    ee_pose = _paired_from_spec("door_open")


@configclass
class _DoorOpenRewardsCfg:
    """Same shape as Window-Open with door-task threshold."""

    reward = _hamacher_reward(success_radius=DOOR_TARGET_RADIUS, axis=None, reach_sigmoid="long_tail")
    success = _success_term(DOOR_TARGET_RADIUS)
    action_rate = _ACTION_RATE


@configclass
class _DoorOpenEventCfg:
    reset_robot_joints = _reset_robot()
    reset_door_closed = _reset_joint_from_spec("door_open")


@configclass
class MetaworldDoorOpenSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerDoorSceneCfg = SawyerDoorSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _DoorOpenCommandsCfg()
    observations = _ObsCfg()
    rewards = _DoorOpenRewardsCfg()
    events = _DoorOpenEventCfg()


# ── Door-Close (same asset, reverse joint init/goal) ───────────────────────


@configclass
class _DoorCloseCommandsCfg:
    """Door open at -π/3, goal: closed (handle at +x, +y)."""

    ee_pose = _paired_from_spec("door_close")


@configclass
class _DoorCloseRewardsCfg:
    """Same shape as Door-Open."""

    reward = _hamacher_reward(success_radius=DOOR_TARGET_RADIUS, axis=None, reach_sigmoid="long_tail")
    success = _success_term(DOOR_TARGET_RADIUS)
    action_rate = _ACTION_RATE


@configclass
class _DoorCloseEventCfg:
    reset_robot_joints = _reset_robot()
    reset_door_open = _reset_joint_from_spec("door_close")


@configclass
class MetaworldDoorCloseSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerDoorSceneCfg = SawyerDoorSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _DoorCloseCommandsCfg()
    observations = _ObsCfg()
    rewards = _DoorCloseRewardsCfg()
    events = _DoorCloseEventCfg()


# ── Peg-Insert-Side (peg block + free peg cylinder) ────────────────────────


@configclass
class _PegInsertCommandsCfg:
    """Peg starts on table (0, 0.65, 0.04). Goal: hole position
    (-0.35, 0.554, 0.13) — verified by mw_peg_block analytical check."""

    ee_pose = _paired_from_spec("peg_insert_side")


@configclass
class _PegInsertRewardsCfg:
    """``10 · H(pick_place_caging, in_place)`` with success override.

    The ``× 10`` is applied via ``RewardTermCfg.weight`` since the archetype
    cfg doesn't carry a separate scale field.
    """

    reward = RewardTermCfg(
        func="isaaclab_contrib.tasks.manipulation.metaworld.mdp.reward_shapes:caging_times_in_place_shape",
        weight=10.0,
        params={
            "cfg": CagingTimesInPlaceShapeCfg(
                caging="isaaclab_contrib.tasks.manipulation.metaworld.mdp.quantities:pick_place_caging",
                caging_kwargs={
                    "keypoint_frame_cfg": _HANDLE_CFG,
                    "frame_transformer_cfg": _TCP_CFG,
                    "goal_command_name": "ee_pose",
                },
                distance=obj_to_target_dist,
                distance_kwargs={"keypoint_frame_cfg": _HANDLE_CFG, "goal_command_name": "ee_pose"},
                margin=obj_init_to_target_dist,
                margin_kwargs={"goal_command_name": "ee_pose"},
                success_radius=PEG_TARGET_RADIUS,
                success_override=SuccessOverrideCfg(
                    quantity=obj_to_target_dist,
                    op="<=",
                    threshold=PEG_TARGET_RADIUS,
                    value=1.0,
                    atom_kwargs={"keypoint_frame_cfg": _HANDLE_CFG, "goal_command_name": "ee_pose"},
                ),
            )
        },
    )
    success = _success_term(PEG_TARGET_RADIUS)
    action_rate = _ACTION_RATE


@configclass
class _PegInsertEventCfg:
    reset_robot_joints = _reset_robot()


@configclass
class MetaworldPegInsertSideSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerPegInsertSceneCfg = SawyerPegInsertSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _PegInsertCommandsCfg()
    observations = _ObsCfg()
    rewards = _PegInsertRewardsCfg()
    events = _PegInsertEventCfg()


# ── Dial-Turn (revolute knob — reuses mw_faucet asset, different goal angle) ─


@configclass
class _DialTurnCommandsCfg:
    """Dial starts at 0, goal: rotated to π/3 (60° turn). Same handle-tip math as faucet."""

    ee_pose = _paired_from_spec("dial_turn")


@configclass
class _DialTurnRewardsCfg:
    """Same shape as Faucet-Open."""

    reward = _hamacher_reward(success_radius=0.05, axis=None, reach_sigmoid="long_tail")
    success = _success_term(0.05)
    action_rate = _ACTION_RATE


@configclass
class _DialTurnEventCfg:
    reset_robot_joints = _reset_robot()
    reset_dial_zero = _reset_joint_from_spec("dial_turn")


@configclass
class MetaworldDialTurnSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerFaucetSceneCfg = SawyerFaucetSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _DialTurnCommandsCfg()
    observations = _ObsCfg()
    rewards = _DialTurnRewardsCfg()
    events = _DialTurnEventCfg()


# ── Coffee-Button (reuses mw_button asset; same task as Button-Press-Topdown
#                   but smaller press distance, MW uses 4 cm vs 6 cm) ────────


@configclass
class _CoffeeButtonCommandsCfg:
    """Button starts extended (z=0.13). Goal: pressed only ~3 cm (MW's coffee button)."""

    ee_pose = _paired_from_spec("coffee_button")


@configclass
class _CoffeeButtonRewardsCfg:
    """Same shape as Button-Press-Topdown — reuses :data:`_BUTTON_PRESS_TERM`."""

    button_press = _BUTTON_PRESS_TERM
    success = _success_term(BUTTON_TARGET_RADIUS)
    action_rate = _ACTION_RATE


@configclass
class _CoffeeButtonEventCfg:
    reset_robot_joints = _reset_robot()
    reset_button_extended = _reset_joint_from_spec("coffee_button")


@configclass
class MetaworldCoffeeButtonSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerButtonSceneCfg = SawyerButtonSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _CoffeeButtonCommandsCfg()
    observations = _ObsCfg()
    rewards = _CoffeeButtonRewardsCfg()
    events = _CoffeeButtonEventCfg()


# ── Lever-Pull (reuses mw_faucet asset; different range — full -π/2 → +π/2) ──


@configclass
class _LeverPullCommandsCfg:
    """Lever starts at -π/3 (-60°), goal: pulled to 0 (handle pointing forward)."""

    ee_pose = _paired_from_spec("lever_pull")


@configclass
class _LeverPullRewardsCfg:
    """Same shape as Faucet-Open."""

    reward = _hamacher_reward(success_radius=0.05, axis=None, reach_sigmoid="long_tail")
    success = _success_term(0.05)
    action_rate = _ACTION_RATE


@configclass
class _LeverPullEventCfg:
    reset_robot_joints = _reset_robot()
    reset_lever_neg = _reset_joint_from_spec("lever_pull")


@configclass
class MetaworldLeverPullSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerFaucetSceneCfg = SawyerFaucetSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _LeverPullCommandsCfg()
    observations = _ObsCfg()
    rewards = _LeverPullRewardsCfg()
    events = _LeverPullEventCfg()


# ── Door-Lock / Door-Unlock (reuses mw_door asset; smaller rotation range) ──


@configclass
class _DoorLockCommandsCfg:
    """Door-lock: rotate handle ~30° downward (lock the latch). Same revolute joint."""

    ee_pose = _paired_from_spec("door_lock")


@configclass
class _DoorLockRewardsCfg:
    """Same shape as Door-Open."""

    reward = _hamacher_reward(success_radius=0.05, axis=None, reach_sigmoid="long_tail")
    success = _success_term(0.05)
    action_rate = _ACTION_RATE


@configclass
class _DoorLockEventCfg:
    reset_robot_joints = _reset_robot()
    reset_door_unlocked = _reset_joint_from_spec("door_lock")


@configclass
class MetaworldDoorLockSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerDoorSceneCfg = SawyerDoorSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _DoorLockCommandsCfg()
    observations = _ObsCfg()
    rewards = _DoorLockRewardsCfg()
    events = _DoorLockEventCfg()


@configclass
class _DoorUnlockCommandsCfg:
    """Door-unlock: from locked (-π/6) back to neutral (0)."""

    ee_pose = _paired_from_spec("door_unlock")


@configclass
class _DoorUnlockRewardsCfg:
    """Same shape as Door-Open."""

    reward = _hamacher_reward(success_radius=0.05, axis=None, reach_sigmoid="long_tail")
    success = _success_term(0.05)
    action_rate = _ACTION_RATE


@configclass
class _DoorUnlockEventCfg:
    reset_robot_joints = _reset_robot()
    reset_door_locked = _reset_joint_from_spec("door_unlock")


@configclass
class MetaworldDoorUnlockSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerDoorSceneCfg = SawyerDoorSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _DoorUnlockCommandsCfg()
    observations = _ObsCfg()
    rewards = _DoorUnlockRewardsCfg()
    events = _DoorUnlockEventCfg()


# ── Drawer-Open / Drawer-Close (use SawyerDrawerSceneCfg from metaworld_scenes_cfg) ─


@configclass
class _DrawerOpenCommandsCfg:
    """Closed-drawer handle world pos: (0, 0.74, 0.09); open: (0, 0.58, 0.09)."""

    ee_pose = _paired_from_spec("drawer_open")


_DRAWER_OPEN_CAGING = ToleranceShapeCfg(
    distance=tcp_to_obj_dist,
    distance_kwargs={"keypoint_frame_cfg": _HANDLE_CFG, "frame_transformer_cfg": _TCP_CFG, "scale": (3.0, 3.0, 1.0)},
    margin=init_tcp_to_keypoint_init_dist,
    margin_kwargs={"goal_command_name": "ee_pose", "keypoint_init_offset": (0.0, 0.16, 0.0), "scale": (3.0, 3.0, 1.0)},
    success_radius=0.01,
)
_DRAWER_OPEN_OPENING = ToleranceShapeCfg(
    distance=obj_to_target_dist,
    distance_kwargs={"keypoint_frame_cfg": _HANDLE_CFG, "goal_command_name": "ee_pose"},
    margin=obj_init_to_target_dist,
    margin_kwargs={"goal_command_name": "ee_pose"},
    success_radius=0.02,
)


@configclass
class _DrawerOpenRewardsCfg:
    """``5 · (caging_xy + opening)`` — sum of two tolerance terms via two
    tolerance_shape RewardTermCfgs (additive across reward terms is what
    the reward manager already does)."""

    caging = RewardTermCfg(
        func=tolerance_shape,
        weight=5.0,
        params={"cfg": ToleranceShapeCfg(**{**_DRAWER_OPEN_CAGING.__dict__, "scale": 1.0})},
    )
    opening = RewardTermCfg(
        func=tolerance_shape,
        weight=5.0,
        params={"cfg": ToleranceShapeCfg(**{**_DRAWER_OPEN_OPENING.__dict__, "scale": 1.0})},
    )
    success = _success_term(0.03)
    action_rate = _ACTION_RATE


@configclass
class _DrawerOpenEventCfg:
    reset_robot_joints = _reset_robot()
    reset_drawer_closed = _reset_joint_from_spec("drawer_open")


@configclass
class MetaworldDrawerOpenSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerDrawerSceneCfg = SawyerDrawerSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _DrawerOpenCommandsCfg()
    observations = _ObsCfg()
    rewards = _DrawerOpenRewardsCfg()
    events = _DrawerOpenEventCfg()


@configclass
class _DrawerCloseCommandsCfg:
    """Drawer starts open (handle at (0, 0.58, 0.09)), goal closed (0, 0.74, 0.09)."""

    ee_pose = _paired_from_spec("drawer_close")


@configclass
class _DrawerCloseRewardsCfg:
    """``10 · H(reach[gaussian] · grip_close, in_place)`` with success override."""

    reward = _hamacher_reward(
        success_radius=DRAWER_TARGET_RADIUS,
        axis=None,
        reach_sigmoid="gaussian",
        a_modulator=gripper_close_action,
        success_override_threshold=DRAWER_TARGET_RADIUS + 0.015,
    )
    success = _success_term(DRAWER_TARGET_RADIUS + 0.015)
    action_rate = _ACTION_RATE


@configclass
class _DrawerCloseEventCfg:
    reset_robot_joints = _reset_robot()
    reset_drawer_open = _reset_joint_from_spec("drawer_close")


@configclass
class MetaworldDrawerCloseSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerDrawerSceneCfg = SawyerDrawerSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _DrawerCloseCommandsCfg()
    observations = _ObsCfg()
    rewards = _DrawerCloseRewardsCfg()
    events = _DrawerCloseEventCfg()


# ─────────────────────────────────────────────────────────────────────────────
# MT50 expansion: cube + obstacle / kinematic destination tasks
# ─────────────────────────────────────────────────────────────────────────────
#
# All cube tasks share: ``observations = _ObsCfg()`` and ``events = _RobotResetEventCfg()``
# (the cube position is set by ``MetaworldPairedCommand`` at reset; no asset
# joint to reset). Per-task reward archetypes come from the helpers above.


@configclass
class _RobotResetEventCfg:
    """Cube tasks: only the Sawyer joint reset is needed at episode start."""

    reset_robot_joints = _reset_robot()


# ── Push-Back ───────────────────────────────────────────────────────────────


@configclass
class _PushBackCommandsCfg:
    ee_pose = _paired_from_spec("push_back")


@configclass
class _PushBackRewardsCfg:
    push_v2 = _push_reward(PUSH_TARGET_RADIUS)
    success = _success_term(PUSH_TARGET_RADIUS)
    action_rate = _ACTION_RATE


@configclass
class MetaworldPushBackSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerCubeSceneCfg = SawyerCubeSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _PushBackCommandsCfg()
    observations = _ObsCfg()
    rewards = _PushBackRewardsCfg()
    events = _RobotResetEventCfg()


# ── Push-Wall ───────────────────────────────────────────────────────────────


@configclass
class _PushWallCommandsCfg:
    ee_pose = _paired_from_spec("push_wall")


@configclass
class _PushWallRewardsCfg:
    push_v2 = _push_reward(PUSH_TARGET_RADIUS)
    success = _success_term(PUSH_TARGET_RADIUS)
    action_rate = _ACTION_RATE


@configclass
class MetaworldPushWallSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerCubeWithWallSceneCfg = SawyerCubeWithWallSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _PushWallCommandsCfg()
    observations = _ObsCfg()
    rewards = _PushWallRewardsCfg()
    events = _RobotResetEventCfg()


# ── Reach-Wall ──────────────────────────────────────────────────────────────


@configclass
class _ReachWallCommandsCfg:
    ee_pose = _paired_from_spec("reach_wall")


@configclass
class _ReachWallRewardsCfg:
    """Reach reward — TCP-to-goal. Cube is just an obs anchor here."""

    reach_v2 = _reach_reward(REACH_TARGET_RADIUS)
    success = _reach_success_term(REACH_TARGET_RADIUS)
    action_rate = _ACTION_RATE


@configclass
class MetaworldReachWallSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerCubeWithWallSceneCfg = SawyerCubeWithWallSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _ReachWallCommandsCfg()
    observations = _ObsCfg()
    rewards = _ReachWallRewardsCfg()
    events = _RobotResetEventCfg()


# ── Pick-Place-Wall ─────────────────────────────────────────────────────────


@configclass
class _PickPlaceWallCommandsCfg:
    ee_pose = _paired_from_spec("pick_place_wall")


@configclass
class _PickPlaceWallRewardsCfg:
    pick_place_v2 = _pick_place_reward(PICK_PLACE_TARGET_RADIUS)
    success = _success_term(PICK_PLACE_SUCCESS_RADIUS)
    action_rate = _ACTION_RATE


@configclass
class MetaworldPickPlaceWallSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerCubeWithWallSceneCfg = SawyerCubeWithWallSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _PickPlaceWallCommandsCfg()
    observations = _ObsCfg()
    rewards = _PickPlaceWallRewardsCfg()
    events = _RobotResetEventCfg()


# ── Basketball ──────────────────────────────────────────────────────────────


@configclass
class _BasketballCommandsCfg:
    ee_pose = _paired_from_spec("basketball")


@configclass
class _BasketballRewardsCfg:
    pick_place_v2 = _pick_place_reward(PICK_PLACE_TARGET_RADIUS)
    success = _success_term(PICK_PLACE_SUCCESS_RADIUS)
    action_rate = _ACTION_RATE


@configclass
class MetaworldBasketballSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerBasketSceneCfg = SawyerBasketSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _BasketballCommandsCfg()
    observations = _ObsCfg()
    rewards = _BasketballRewardsCfg()
    events = _RobotResetEventCfg()


# ── Shelf-Place ─────────────────────────────────────────────────────────────


@configclass
class _ShelfPlaceCommandsCfg:
    ee_pose = _paired_from_spec("shelf_place")


@configclass
class _ShelfPlaceRewardsCfg:
    pick_place_v2 = _pick_place_reward(PICK_PLACE_TARGET_RADIUS)
    success = _success_term(PICK_PLACE_SUCCESS_RADIUS)
    action_rate = _ACTION_RATE


@configclass
class MetaworldShelfPlaceSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerShelfSceneCfg = SawyerShelfSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _ShelfPlaceCommandsCfg()
    observations = _ObsCfg()
    rewards = _ShelfPlaceRewardsCfg()
    events = _RobotResetEventCfg()


# ── Soccer ──────────────────────────────────────────────────────────────────


@configclass
class _SoccerCommandsCfg:
    ee_pose = _paired_from_spec("soccer")


@configclass
class _SoccerRewardsCfg:
    push_v2 = _push_reward(PUSH_TARGET_RADIUS)
    success = _success_term(PUSH_TARGET_RADIUS)
    action_rate = _ACTION_RATE


@configclass
class MetaworldSoccerSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerSoccerSceneCfg = SawyerSoccerSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _SoccerCommandsCfg()
    observations = _ObsCfg()
    rewards = _SoccerRewardsCfg()
    events = _RobotResetEventCfg()


# ── Sweep ───────────────────────────────────────────────────────────────────


@configclass
class _SweepCommandsCfg:
    ee_pose = _paired_from_spec("sweep")


@configclass
class _SweepRewardsCfg:
    push_v2 = _push_reward(PUSH_TARGET_RADIUS)
    success = _success_term(PUSH_TARGET_RADIUS)
    action_rate = _ACTION_RATE


@configclass
class MetaworldSweepSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerCubeSceneCfg = SawyerCubeSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _SweepCommandsCfg()
    observations = _ObsCfg()
    rewards = _SweepRewardsCfg()
    events = _RobotResetEventCfg()


# ── Sweep-Into ──────────────────────────────────────────────────────────────


@configclass
class _SweepIntoCommandsCfg:
    ee_pose = _paired_from_spec("sweep_into")


@configclass
class _SweepIntoRewardsCfg:
    push_v2 = _push_reward(PUSH_TARGET_RADIUS)
    success = _success_term(PUSH_TARGET_RADIUS)
    action_rate = _ACTION_RATE


@configclass
class MetaworldSweepIntoSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerCubeWithBinSceneCfg = SawyerCubeWithBinSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _SweepIntoCommandsCfg()
    observations = _ObsCfg()
    rewards = _SweepIntoRewardsCfg()
    events = _RobotResetEventCfg()


# ── Plate-Slide / Plate-Slide-Back (articulated plate) ──────────────────────


@configclass
class _PlateSlideCommandsCfg:
    ee_pose = _paired_from_spec("plate_slide")


@configclass
class _PlateSlideRewardsCfg:
    """Plate-slide: hamacher of (TCP→plate-marker reach) and (plate-marker→goal in_place)."""

    reward = _hamacher_reward(success_radius=0.05, axis=None, reach_sigmoid="long_tail")
    success = _success_term(0.05)
    action_rate = _ACTION_RATE


@configclass
class _PlateSlideEventCfg:
    reset_robot_joints = _reset_robot()
    reset_plate = _reset_joint_from_spec("plate_slide")


@configclass
class MetaworldPlateSlideSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerPlateSceneCfg = SawyerPlateSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _PlateSlideCommandsCfg()
    observations = _ObsCfg()
    rewards = _PlateSlideRewardsCfg()
    events = _PlateSlideEventCfg()


@configclass
class _PlateSlideBackCommandsCfg:
    ee_pose = _paired_from_spec("plate_slide_back")


@configclass
class _PlateSlideBackEventCfg:
    reset_robot_joints = _reset_robot()
    reset_plate = _reset_joint_from_spec("plate_slide_back")


@configclass
class MetaworldPlateSlideBackSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerPlateSceneCfg = SawyerPlateSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _PlateSlideBackCommandsCfg()
    observations = _ObsCfg()
    rewards = _PlateSlideRewardsCfg()  # same shape, opposite direction
    events = _PlateSlideBackEventCfg()


# ─────────────────────────────────────────────────────────────────────────────
# MT50 expansion (batch 2): handle / peg-unplug / box / hole-block / button-wall
# ─────────────────────────────────────────────────────────────────────────────


# ── Handle-Press / Handle-Pull (top-down handle, prismatic Z) ───────────────


@configclass
class _HandlePressCommandsCfg:
    ee_pose = _paired_from_spec("handle_press")


@configclass
class _HandlePressEventCfg:
    reset_robot_joints = _reset_robot()
    reset_handle = _reset_joint_from_spec("handle_press")


@configclass
class _HandleHamacherRewardsCfg:
    """Hamacher (TCP→handle reach × handle→goal in_place). Used by all 4 handle variants."""

    reward = _hamacher_reward(success_radius=0.04, axis=None, reach_sigmoid="long_tail")
    success = _success_term(0.04)
    action_rate = _ACTION_RATE


@configclass
class MetaworldHandlePressSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerHandlePressSceneCfg = SawyerHandlePressSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _HandlePressCommandsCfg()
    observations = _ObsCfg()
    rewards = _HandleHamacherRewardsCfg()
    events = _HandlePressEventCfg()


@configclass
class _HandlePullCommandsCfg:
    ee_pose = _paired_from_spec("handle_pull")


@configclass
class _HandlePullEventCfg:
    reset_robot_joints = _reset_robot()
    reset_handle = _reset_joint_from_spec("handle_pull")


@configclass
class MetaworldHandlePullSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerHandlePressSceneCfg = SawyerHandlePressSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _HandlePullCommandsCfg()
    observations = _ObsCfg()
    rewards = _HandleHamacherRewardsCfg()
    events = _HandlePullEventCfg()


# ── Handle-Press-Side / Handle-Pull-Side (side handle, prismatic X) ─────────


@configclass
class _HandlePressSideCommandsCfg:
    ee_pose = _paired_from_spec("handle_press_side")


@configclass
class _HandlePressSideEventCfg:
    reset_robot_joints = _reset_robot()
    reset_handle = _reset_joint_from_spec("handle_press_side")


@configclass
class MetaworldHandlePressSideSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerHandlePressSideSceneCfg = SawyerHandlePressSideSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _HandlePressSideCommandsCfg()
    observations = _ObsCfg()
    rewards = _HandleHamacherRewardsCfg()
    events = _HandlePressSideEventCfg()


@configclass
class _HandlePullSideCommandsCfg:
    ee_pose = _paired_from_spec("handle_pull_side")


@configclass
class _HandlePullSideEventCfg:
    reset_robot_joints = _reset_robot()
    reset_handle = _reset_joint_from_spec("handle_pull_side")


@configclass
class MetaworldHandlePullSideSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerHandlePressSideSceneCfg = SawyerHandlePressSideSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _HandlePullSideCommandsCfg()
    observations = _ObsCfg()
    rewards = _HandleHamacherRewardsCfg()
    events = _HandlePullSideEventCfg()


# ── Peg-Unplug-Side (peg pulled out laterally, prismatic X) ─────────────────


@configclass
class _PegUnplugSideCommandsCfg:
    ee_pose = _paired_from_spec("peg_unplug_side")


@configclass
class _PegUnplugSideEventCfg:
    reset_robot_joints = _reset_robot()
    reset_peg = _reset_joint_from_spec("peg_unplug_side")


@configclass
class MetaworldPegUnplugSideSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerPegUnplugSceneCfg = SawyerPegUnplugSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _PegUnplugSideCommandsCfg()
    observations = _ObsCfg()
    rewards = _HandleHamacherRewardsCfg()
    events = _PegUnplugSideEventCfg()


# ── Box-Close (revolute lid, hinge at rear edge) ────────────────────────────


@configclass
class _BoxCloseCommandsCfg:
    ee_pose = _paired_from_spec("box_close")


@configclass
class _BoxCloseRewardsCfg:
    """Hamacher (reach + in_place) with loose threshold (lid arcs through space)."""

    reward = _hamacher_reward(success_radius=0.10, axis=None, reach_sigmoid="long_tail")
    success = _success_term(0.10)
    action_rate = _ACTION_RATE


@configclass
class _BoxCloseEventCfg:
    reset_robot_joints = _reset_robot()
    reset_lid = _reset_joint_from_spec("box_close")


@configclass
class MetaworldBoxCloseSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerBoxWithLidSceneCfg = SawyerBoxWithLidSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _BoxCloseCommandsCfg()
    observations = _ObsCfg()
    rewards = _BoxCloseRewardsCfg()
    events = _BoxCloseEventCfg()


# ── Hand-Insert (TCP reaches inside pocket; no manipulandum) ────────────────


@configclass
class _HandInsertCommandsCfg:
    ee_pose = _paired_from_spec("hand_insert")


@configclass
class _HandInsertRewardsCfg:
    """Reach reward — TCP-to-pocket-marker. Same shape as reach-wall."""

    reach_v2 = _reach_reward(REACH_TARGET_RADIUS)
    success = _reach_success_term(REACH_TARGET_RADIUS)
    action_rate = _ACTION_RATE


@configclass
class MetaworldHandInsertSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerHoleBlockSceneCfg = SawyerHoleBlockSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _HandInsertCommandsCfg()
    observations = _ObsCfg()
    rewards = _HandInsertRewardsCfg()
    events = _RobotResetEventCfg()


# ── Pick-Out-Of-Hole (cube starts inside pocket, agent lifts out) ───────────


@configclass
class _PickOutOfHoleCommandsCfg:
    ee_pose = _paired_from_spec("pick_out_of_hole")


@configclass
class _PickOutOfHoleRewardsCfg:
    pick_place_v2 = _pick_place_reward(PICK_PLACE_TARGET_RADIUS)
    success = _success_term(PICK_PLACE_SUCCESS_RADIUS)
    action_rate = _ACTION_RATE


@configclass
class MetaworldPickOutOfHoleSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerCubeInHoleSceneCfg = SawyerCubeInHoleSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _PickOutOfHoleCommandsCfg()
    observations = _ObsCfg()
    rewards = _PickOutOfHoleRewardsCfg()
    events = _RobotResetEventCfg()


# ── Button-Press-Topdown-Wall (button + wall obstacle) ──────────────────────


@configclass
class _ButtonPressTopdownWallCommandsCfg:
    ee_pose = _paired_from_spec("button_press_topdown_wall")


@configclass
class _ButtonPressTopdownWallEventCfg:
    reset_robot_joints = _reset_robot()
    reset_button = _reset_joint_from_spec("button_press_topdown_wall")


@configclass
class MetaworldButtonPressTopdownWallSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    """Same reward shape as Button-Press-Topdown; only the scene adds a wall."""

    scene: SawyerButtonWithWallSceneCfg = SawyerButtonWithWallSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _ButtonPressTopdownWallCommandsCfg()
    observations = _ObsCfg()
    rewards = _ButtonRewardsCfg()  # reuse the existing button reward
    events = _ButtonPressTopdownWallEventCfg()


# ─────────────────────────────────────────────────────────────────────────────
# MT50 expansion (batch 3): bin / coffee / stick / assembly / hammer
# ─────────────────────────────────────────────────────────────────────────────


# ── Bin-Picking (cube into bin via pick-place) ──────────────────────────────


@configclass
class _BinPickingCommandsCfg:
    ee_pose = _paired_from_spec("bin_picking")


@configclass
class _BinPickingRewardsCfg:
    pick_place_v2 = _pick_place_reward(PICK_PLACE_TARGET_RADIUS)
    success = _success_term(PICK_PLACE_SUCCESS_RADIUS)
    action_rate = _ACTION_RATE


@configclass
class MetaworldBinPickingSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerCubeWithBinSceneCfg = SawyerCubeWithBinSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _BinPickingCommandsCfg()
    observations = _ObsCfg()
    rewards = _BinPickingRewardsCfg()
    events = _RobotResetEventCfg()


# ── Coffee Push / Pull (cube ≈ mug; coffee_button is decoration) ────────────


@configclass
class _CoffeePushCommandsCfg:
    ee_pose = _paired_from_spec("coffee_push")


@configclass
class _CoffeePullCommandsCfg:
    ee_pose = _paired_from_spec("coffee_pull")


@configclass
class _CoffeeRewardsCfg:
    push_v2 = _push_reward(PUSH_TARGET_RADIUS)
    success = _success_term(PUSH_TARGET_RADIUS)
    action_rate = _ACTION_RATE


@configclass
class MetaworldCoffeePushSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerCubeWithButtonSceneCfg = SawyerCubeWithButtonSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _CoffeePushCommandsCfg()
    observations = _ObsCfg()
    rewards = _CoffeeRewardsCfg()
    events = _RobotResetEventCfg()


@configclass
class MetaworldCoffeePullSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerCubeWithButtonSceneCfg = SawyerCubeWithButtonSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _CoffeePullCommandsCfg()
    observations = _ObsCfg()
    rewards = _CoffeeRewardsCfg()
    events = _RobotResetEventCfg()


# ── Stick Push / Pull (cube + free stick tool) ──────────────────────────────


@configclass
class _StickPushCommandsCfg:
    ee_pose = _paired_from_spec("stick_push")


@configclass
class _StickPullCommandsCfg:
    ee_pose = _paired_from_spec("stick_pull")


@configclass
class _StickRewardsCfg:
    push_v2 = _push_reward(PUSH_TARGET_RADIUS)
    success = _success_term(PUSH_TARGET_RADIUS)
    action_rate = _ACTION_RATE


@configclass
class MetaworldStickPushSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerCubeWithStickSceneCfg = SawyerCubeWithStickSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _StickPushCommandsCfg()
    observations = _ObsCfg()
    rewards = _StickRewardsCfg()
    events = _RobotResetEventCfg()


@configclass
class MetaworldStickPullSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerCubeWithStickSceneCfg = SawyerCubeWithStickSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _StickPullCommandsCfg()
    observations = _ObsCfg()
    rewards = _StickRewardsCfg()
    events = _RobotResetEventCfg()


# ── Assembly / Disassemble (cube + assembly_peg) ────────────────────────────


@configclass
class _AssemblyCommandsCfg:
    ee_pose = _paired_from_spec("assembly")


@configclass
class _DisassembleCommandsCfg:
    ee_pose = _paired_from_spec("disassemble")


@configclass
class _AssemblyRewardsCfg:
    pick_place_v2 = _pick_place_reward(PICK_PLACE_TARGET_RADIUS)
    success = _success_term(PICK_PLACE_SUCCESS_RADIUS)
    action_rate = _ACTION_RATE


@configclass
class MetaworldAssemblySawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerCubeWithPegSceneCfg = SawyerCubeWithPegSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _AssemblyCommandsCfg()
    observations = _ObsCfg()
    rewards = _AssemblyRewardsCfg()
    events = _RobotResetEventCfg()


@configclass
class MetaworldDisassembleSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerCubeWithPegSceneCfg = SawyerCubeWithPegSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _DisassembleCommandsCfg()
    observations = _ObsCfg()
    rewards = _AssemblyRewardsCfg()
    events = _RobotResetEventCfg()


# ── Hammer (nail is the articulated manipulandum; hammer-as-tool is implicit) ─


@configclass
class _HammerCommandsCfg:
    ee_pose = _paired_from_spec("hammer")


@configclass
class _HammerEventCfg:
    reset_robot_joints = _reset_robot()
    reset_nail = _reset_joint_from_spec("hammer")


@configclass
class MetaworldHammerSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerNailBlockSceneCfg = SawyerNailBlockSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _HammerCommandsCfg()
    observations = _ObsCfg()
    rewards = _HandleHamacherRewardsCfg()  # same hamacher shape used by handle/peg-unplug
    events = _HammerEventCfg()


# ─────────────────────────────────────────────────────────────────────────────
# MT50 expansion (batch 4): plate-slide-side variants + front-facing button
# ─────────────────────────────────────────────────────────────────────────────


# ── Plate-Slide-Side / -Back-Side (plate slides along world Y) ──────────────


@configclass
class _PlateSlideSideCommandsCfg:
    ee_pose = _paired_from_spec("plate_slide_side")


@configclass
class _PlateSlideSideEventCfg:
    reset_robot_joints = _reset_robot()
    reset_plate = _reset_joint_from_spec("plate_slide_side")


@configclass
class MetaworldPlateSlideSideSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerPlateSideSceneCfg = SawyerPlateSideSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _PlateSlideSideCommandsCfg()
    observations = _ObsCfg()
    rewards = _PlateSlideRewardsCfg()  # same hamacher shape as plate-slide
    events = _PlateSlideSideEventCfg()


@configclass
class _PlateSlideBackSideCommandsCfg:
    ee_pose = _paired_from_spec("plate_slide_back_side")


@configclass
class _PlateSlideBackSideEventCfg:
    reset_robot_joints = _reset_robot()
    reset_plate = _reset_joint_from_spec("plate_slide_back_side")


@configclass
class MetaworldPlateSlideBackSideSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerPlateSideSceneCfg = SawyerPlateSideSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _PlateSlideBackSideCommandsCfg()
    observations = _ObsCfg()
    rewards = _PlateSlideRewardsCfg()
    events = _PlateSlideBackSideEventCfg()


# ── Button-Press / Button-Press-Wall (front-facing button) ──────────────────


@configclass
class _ButtonPressCommandsCfg:
    ee_pose = _paired_from_spec("button_press")


@configclass
class _ButtonPressEventCfg:
    reset_robot_joints = _reset_robot()
    reset_button = _reset_joint_from_spec("button_press")


# Front-facing button uses the same hamacher-press archetype as the topdown
# button: TCP-near-button gated by gripper-closed × button-displaced-along-axis.
# The displacement axis is Y (world) for the front button vs Z for topdown.
_BUTTON_FRONT_PRESS_TERM = RewardTermCfg(
    func=hamacher_shape,
    weight=1.0,
    params={
        "cfg": HamacherShapeCfg(
            term_a=ToleranceShapeCfg(
                distance=tcp_to_obj_dist,
                distance_kwargs={"keypoint_frame_cfg": _HANDLE_CFG, "frame_transformer_cfg": _TCP_CFG},
                margin=tcp_to_obj_dist,
                margin_kwargs={"keypoint_frame_cfg": _HANDLE_CFG, "frame_transformer_cfg": _TCP_CFG},
                success_radius=0.01,
            ),
            a_modulator=gripper_closed,
            term_b=ToleranceShapeCfg(
                distance=axis_obj_to_target_dist,
                distance_kwargs={"keypoint_frame_cfg": _HANDLE_CFG, "axis": 1},  # Y-axis instead of Z
                margin=axis_init_to_target_dist,
                margin_kwargs={"axis": 1},
                success_radius=0.005,
            ),
            scale=5.0,
            phase=PhaseBonusCfg(
                triggers=[
                    TriggerCfg(
                        atom=tcp_to_obj_dist,
                        op="<=",
                        threshold=0.03,
                        atom_kwargs={"keypoint_frame_cfg": _HANDLE_CFG, "frame_transformer_cfg": _TCP_CFG},
                    )
                ],
                offset=0.0,
                in_place_mult=5.0,
                self_mult=0.0,
            ),
        )
    },
)


@configclass
class _ButtonPressRewardsCfg:
    button_press = _BUTTON_FRONT_PRESS_TERM
    success = _success_term(BUTTON_TARGET_RADIUS)
    action_rate = _ACTION_RATE


@configclass
class MetaworldButtonPressSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    scene: SawyerButtonFrontSceneCfg = SawyerButtonFrontSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _ButtonPressCommandsCfg()
    observations = _ObsCfg()
    rewards = _ButtonPressRewardsCfg()
    events = _ButtonPressEventCfg()


@configclass
class _ButtonPressWallCommandsCfg:
    ee_pose = _paired_from_spec("button_press_wall")


@configclass
class _ButtonPressWallEventCfg:
    reset_robot_joints = _reset_robot()
    reset_button = _reset_joint_from_spec("button_press_wall")


@configclass
class MetaworldButtonPressWallSawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    """Front-facing button + wall obstacle. Same reward as button-press."""

    scene: SawyerButtonFrontWithWallSceneCfg = SawyerButtonFrontWithWallSceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _ButtonPressWallCommandsCfg()
    observations = _ObsCfg()
    rewards = _ButtonPressRewardsCfg()
    events = _ButtonPressWallEventCfg()
