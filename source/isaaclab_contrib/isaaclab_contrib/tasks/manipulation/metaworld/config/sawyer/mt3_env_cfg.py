# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MT3 multi-task env: reach / push / pick-place sharing one cube scene.

The three MT3 tasks all use the cube as the manipulandum (cube IS the
object being reached / pushed / picked), so they share one
:class:`SawyerCubeSceneCfg`. Per-task reward routing uses the
``@scatterable`` atoms in :mod:`mdp.scatter_rewards` — a minimal
``clone_cfg`` partitions envs into three task groups (``reach`` /
``push`` / ``pick_place``), each containing the same shared assets
(cube, keypoint_frame, tcp_frame). Each :class:`RewardTermCfg` then
passes ``asset_cfg=SceneEntityCfg("keypoint_frame", groups=[task])``
so its result is written only on that task's envs.

Each parallel env is round-robin-assigned a task index by
:class:`MetaworldMultiTaskCommand` (env i → task i % 3); the
clone-group env partition aligns with this round-robin.
"""

from __future__ import annotations

from isaaclab.cloner.cloner_strategies import interleaved

# Local import — break the import cycle via the leaf re-export.
from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import CloneCfg, InclusionSet
from isaaclab.utils import configclass

from ... import metaworld_env_base
from ...mdp import (
    PICK_PLACE_TARGET_RADIUS,
    PUSH_TARGET_RADIUS,
    REACH_TARGET_RADIUS,
    CagingTimesInPlaceShapeCfg,
    GripperCagingParams,
    LinearComboShapeCfg,
    MetaworldMultiTaskCommandCfg,
    MetaworldObservation,
    PhaseBonusCfg,
    SuccessOverrideCfg,
    TaskBoxCfg,
    ToleranceShapeCfg,
    TriggerCfg,
    WeightedAtomCfg,
    gripper_caging,
    gripper_open,
    hand_init_to_target_dist,
    metaworld_task_onehot,
    obj_init_to_target_dist,
    obj_to_target_dist,
    pick_place_caging,
    tcp_to_obj_dist,
    tcp_to_target_dist,
)
from ...mdp.scatter_rewards import (
    caging_times_in_place_term,
    keypoint_success_termination,
    linear_combo_term,
    reach_success_term,
    reach_success_termination,
    success_indicator_term,
    tolerance_term,
)
from ...metaworld_scenes_cfg import SawyerCubeSceneCfg

# ── MT3 task index mapping ─────────────────────────────────────────────────

MT3_TASK_NAMES: list[str] = ["reach", "push", "pick_place"]
MT3_TASK_TO_INDEX: dict[str, int] = {n: i for i, n in enumerate(MT3_TASK_NAMES)}


_TCP_FRAME_CFG = SceneEntityCfg("tcp_frame")
_OBJECT_CFG = SceneEntityCfg("keypoint_frame")


# ── Commands: one MetaworldMultiTaskCommand with 3 task boxes ─────────────

_REACH_OBJ_LOW = (-0.1, 0.6, 0.02)
_REACH_OBJ_HIGH = (0.1, 0.7, 0.02)
_REACH_GOAL_LOW = (-0.1, 0.8, 0.05)
_REACH_GOAL_HIGH = (0.1, 0.9, 0.30)

_PUSH_OBJ_LOW = (-0.1, 0.6, 0.02)
_PUSH_OBJ_HIGH = (0.1, 0.7, 0.02)
_PUSH_GOAL_LOW = (-0.1, 0.8, 0.01)
_PUSH_GOAL_HIGH = (0.1, 0.9, 0.02)

_PP_OBJ_LOW = (-0.1, 0.6, 0.02)
_PP_OBJ_HIGH = (0.1, 0.7, 0.02)
_PP_GOAL_LOW = (-0.1, 0.8, 0.05)
_PP_GOAL_HIGH = (0.1, 0.9, 0.30)


@configclass
class _MT3CommandsCfg:
    ee_pose = MetaworldMultiTaskCommandCfg(
        resampling_time_range=(1.0e6, 1.0e6),
        debug_vis=False,
        object_name="cube",
        frame_transformer_name="tcp_frame",
        min_xy_separation=0.0,
        tasks=[
            TaskBoxCfg(
                name="reach",
                obj_low=_REACH_OBJ_LOW,
                obj_high=_REACH_OBJ_HIGH,
                goal_low=_REACH_GOAL_LOW,
                goal_high=_REACH_GOAL_HIGH,
            ),
            TaskBoxCfg(
                name="push",
                obj_low=_PUSH_OBJ_LOW,
                obj_high=_PUSH_OBJ_HIGH,
                goal_low=_PUSH_GOAL_LOW,
                goal_high=_PUSH_GOAL_HIGH,
            ),
            TaskBoxCfg(
                name="pick_place",
                obj_low=_PP_OBJ_LOW,
                obj_high=_PP_OBJ_HIGH,
                goal_low=_PP_GOAL_LOW,
                goal_high=_PP_GOAL_HIGH,
            ),
        ],
    )


# ── Observations: 39-d state + task one-hot ───────────────────────────────


@configclass
class _MT3ObsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        state = ObservationTermCfg(
            func=MetaworldObservation,
            params={
                "frame_transformer_cfg": _TCP_FRAME_CFG,
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


# ── Reward shapes (per-task; pulled from the single-task env_cfg files) ───


_REACH_SHAPE = ToleranceShapeCfg(
    distance=tcp_to_target_dist,
    distance_kwargs={"frame_transformer_cfg": _TCP_FRAME_CFG, "goal_command_name": "ee_pose"},
    margin=hand_init_to_target_dist,
    margin_kwargs={
        "goal_command_name": "ee_pose",
        # Pinned to the realised TCP at the default Sawyer joint config.
        "hand_init_pos_e": (-0.067, 0.571, 0.132),
    },
    success_radius=REACH_TARGET_RADIUS,
    scale=10.0,
)


_PUSH_SHAPE = LinearComboShapeCfg(
    terms=[
        WeightedAtomCfg(
            weight=2.0,
            atom=gripper_caging,
            atom_kwargs={
                "frame_transformer_cfg": _TCP_FRAME_CFG,
                "keypoint_frame_cfg": _OBJECT_CFG,
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
    in_place_distance_kwargs={"keypoint_frame_cfg": _OBJECT_CFG, "goal_command_name": "ee_pose"},
    in_place_margin=obj_init_to_target_dist,
    in_place_margin_kwargs={"goal_command_name": "ee_pose"},
    in_place_success_radius=PUSH_TARGET_RADIUS,
    phase=PhaseBonusCfg(
        triggers=[
            TriggerCfg(
                atom=tcp_to_obj_dist,
                op="<",
                threshold=0.02,
                atom_kwargs={"frame_transformer_cfg": _TCP_FRAME_CFG, "keypoint_frame_cfg": _OBJECT_CFG},
            ),
            TriggerCfg(atom=gripper_open, op=">", threshold=0.0),
        ],
        offset=1.0,
        in_place_mult=5.0,
        self_mult=1.0,
    ),
    success_override=SuccessOverrideCfg(
        quantity=obj_to_target_dist,
        threshold=PUSH_TARGET_RADIUS,
        op="<",
        value=10.0,
        atom_kwargs={"keypoint_frame_cfg": _OBJECT_CFG, "goal_command_name": "ee_pose"},
    ),
)


_PP_SHAPE = CagingTimesInPlaceShapeCfg(
    caging=pick_place_caging,
    caging_kwargs={
        "frame_transformer_cfg": _TCP_FRAME_CFG,
        "keypoint_frame_cfg": _OBJECT_CFG,
        "goal_command_name": "ee_pose",
    },
    distance=obj_to_target_dist,
    distance_kwargs={"keypoint_frame_cfg": _OBJECT_CFG, "goal_command_name": "ee_pose"},
    margin=obj_init_to_target_dist,
    margin_kwargs={"goal_command_name": "ee_pose"},
    success_radius=PICK_PLACE_TARGET_RADIUS,
    success_override=SuccessOverrideCfg(
        quantity=obj_to_target_dist,
        threshold=PICK_PLACE_TARGET_RADIUS,
        op="<",
        value=10.0,
        atom_kwargs={"keypoint_frame_cfg": _OBJECT_CFG, "goal_command_name": "ee_pose"},
    ),
)


def _scatter(task_name: str, scatter_func, cfg, *, weight: float = 1.0) -> RewardTermCfg:
    """``scatter_func(cfg=cfg, asset_cfg=...)`` routed to envs in clone group ``task_name``."""
    return RewardTermCfg(
        func=scatter_func,
        weight=weight,
        params={
            "cfg": cfg,
            "asset_cfg": SceneEntityCfg("keypoint_frame", groups=[task_name]),
        },
    )


def _scatter_obj_success(task_name: str, threshold: float) -> RewardTermCfg:
    """Keypoint-to-goal success indicator routed to ``task_name``'s envs (logged-only, weight=0)."""
    return RewardTermCfg(
        func=success_indicator_term,
        weight=0.0,
        params={
            "keypoint_frame_cfg": _OBJECT_CFG,
            "asset_cfg": SceneEntityCfg("keypoint_frame", groups=[task_name]),
            "goal_command_name": "ee_pose",
            "threshold": threshold,
        },
    )


@configclass
class _MT3RewardsCfg:
    """One scattered reward per task — written only on envs in clone group ``task_name``."""

    reach = _scatter("reach", tolerance_term, _REACH_SHAPE, weight=1.0)
    reach_success = RewardTermCfg(
        func=reach_success_term,
        weight=0.0,
        params={
            "frame_transformer_cfg": _TCP_FRAME_CFG,
            "asset_cfg": SceneEntityCfg("keypoint_frame", groups=["reach"]),
            "goal_command_name": "ee_pose",
            "threshold": REACH_TARGET_RADIUS,
        },
    )

    push = _scatter("push", linear_combo_term, _PUSH_SHAPE, weight=1.0)
    push_success = _scatter_obj_success("push", PUSH_TARGET_RADIUS)

    pick_place = _scatter("pick_place", caging_times_in_place_term, _PP_SHAPE, weight=1.0)
    pick_place_success = _scatter_obj_success("pick_place", PICK_PLACE_TARGET_RADIUS)

    action_rate = RewardTermCfg(func="isaaclab.envs.mdp:action_rate_l2", weight=-1e-4)


# ── Terminations: time-out + per-task success (matches MW V2 done-on-success) ─


@configclass
class _MT3TerminationsCfg:
    """Truncate on horizon + terminate on first per-task success step.

    Inherits ``time_out`` from :class:`MetaworldTerminationsCfg`; adds one
    success-termination per task, scattered to its clone-group envs so
    only that task's envs can fire it.
    """

    time_out = TerminationTermCfg(func="isaaclab.envs.mdp:time_out", time_out=True)

    reach_success = TerminationTermCfg(
        func=reach_success_termination,
        params={
            "frame_transformer_cfg": _TCP_FRAME_CFG,
            "asset_cfg": SceneEntityCfg("keypoint_frame", groups=["reach"]),
            "goal_command_name": "ee_pose",
            "threshold": REACH_TARGET_RADIUS,
        },
    )
    push_success = TerminationTermCfg(
        func=keypoint_success_termination,
        params={
            "keypoint_frame_cfg": _OBJECT_CFG,
            "asset_cfg": SceneEntityCfg("keypoint_frame", groups=["push"]),
            "goal_command_name": "ee_pose",
            "threshold": PUSH_TARGET_RADIUS,
        },
    )
    pick_place_success = TerminationTermCfg(
        func=keypoint_success_termination,
        params={
            "keypoint_frame_cfg": _OBJECT_CFG,
            "asset_cfg": SceneEntityCfg("keypoint_frame", groups=["pick_place"]),
            "goal_command_name": "ee_pose",
            "threshold": PICK_PLACE_TARGET_RADIUS,
        },
    )


# ── Top-level env cfg ──────────────────────────────────────────────────────


_MT3_SHARED_ASSETS: list[str] = ["cube", "keypoint_frame", "tcp_frame"]


def _build_mt3_clone_groups() -> dict[str, InclusionSet]:
    """One clone group per task, all containing the shared cube/keypoint/tcp assets.

    The groups exist purely to partition envs by task — they have no
    asset-cloning effect because every group includes the same asset list.
    The :func:`@scatterable` reward atoms read each group's env_ids to
    route per-task rewards.
    """
    return {name: InclusionSet(assets=_MT3_SHARED_ASSETS, weight=1) for name in MT3_TASK_NAMES}


@configclass
class _MT3SceneCfg(SawyerCubeSceneCfg):
    """:class:`SawyerCubeSceneCfg` + a clone-group partition by task name.

    Uses :func:`~isaaclab.cloner.interleaved` so that env ``i`` is
    assigned to clone group ``i % n_tasks`` — matching the round-robin
    ``task_id`` written by :class:`MetaworldMultiTaskCommand`. Without
    this alignment, scatter rewards (which read group env_ids) and
    success indicators (which read task_id) would route to different
    envs and the gradient signal would be diluted.
    """

    clone_cfg = CloneCfg(clone_groups=_build_mt3_clone_groups(), clone_strategy=interleaved)


@configclass
class MetaworldMT3SawyerEnvCfg(metaworld_env_base.MetaworldEnvCfg):
    """MT3 multi-task — reach / push / pick-place on a single shared cube scene.

    Each parallel env is assigned one of the three task indices via the
    :class:`MetaworldMultiTaskCommand` round-robin (env i → task i % 3).
    Per-task rewards route via the ``@scatterable`` atoms in
    :mod:`mdp.scatter_rewards` against a clone-group partition that
    matches the round-robin task assignment. Observation is 39-d state
    + 3-d task one-hot."""

    scene: _MT3SceneCfg = _MT3SceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _MT3CommandsCfg()
    observations = _MT3ObsCfg()
    rewards = _MT3RewardsCfg()
    terminations = _MT3TerminationsCfg()
