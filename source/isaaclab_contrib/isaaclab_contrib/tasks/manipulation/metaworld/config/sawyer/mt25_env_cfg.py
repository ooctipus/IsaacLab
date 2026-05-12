# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MT25 multi-task env: MT15 + 10 articulated tasks.

Adds (over MT15):
* handle_press / handle_pull          (mw_handle_press)
* handle_press_side / handle_pull_side (mw_handle_press_side)
* peg_unplug_side                     (mw_peg_unplug)
* box_close                           (mw_box_with_lid)
* plate_slide / plate_slide_back      (mw_plate)
* button_press_topdown_wall           (mw_button — reused, just adds task index)
* hammer                              (mw_nail_block)

Each task gets a hamacher (TCP→keypoint reach × keypoint→goal in_place)
masked reward, a per-asset joint reset event, and a clone-group entry.
"""

from __future__ import annotations

from isaaclab.cloner.cloner_strategies import interleaved
from isaaclab.managers import EventTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.scene import CloneCfg, InclusionSet
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from ...mdp import (
    HamacherShapeCfg,
    ToleranceShapeCfg,
    axis_init_to_target_dist,
    axis_obj_to_target_dist,
    init_tcp_to_keypoint_init_dist,
    obj_init_to_target_dist,
    obj_to_target_dist,
    tcp_to_obj_dist,
)
from ...mdp.scatter_rewards import (
    hamacher_term,
    keypoint_success_termination,
    success_indicator_term,
)
from ...metaworld_assets_cfg import (
    MW_BOX_WITH_LID_CFG,
    MW_HANDLE_PRESS_CFG,
    MW_HANDLE_PRESS_SIDE_CFG,
    MW_NAIL_BLOCK_CFG,
    MW_PEG_UNPLUG_CFG,
    MW_PLATE_CFG,
    MW_WALL_CFG,
)
from ...metaworld_specs import TASK_SPECS
from .multi_task_env_cfg import (
    TASK_NAMES as MT15_TASK_NAMES,
)
from .multi_task_env_cfg import (
    MetaworldMultiTaskSawyerEnvCfg,
    MetaworldMultiTaskSceneCfg,
    _MultiTaskCommandsCfg,
    _MultiTaskEventCfg,
    _MultiTaskObsCfg,
    _MultiTaskRewardsCfg,
    _MultiTaskTerminationsCfg,
)

# ── Task list (MT15 + 10 articulated) ──────────────────────────────────────

MT25_NEW_TASKS: list[str] = [
    "handle_press",
    "handle_pull",
    "handle_press_side",
    "handle_pull_side",
    "peg_unplug_side",
    "box_close",
    "plate_slide",
    "plate_slide_back",
    "button_press_topdown_wall",
    "hammer",
]

MT25_TASK_NAMES: list[str] = list(MT15_TASK_NAMES) + MT25_NEW_TASKS
MT25_TASK_TO_INDEX: dict[str, int] = {n: i for i, n in enumerate(MT25_TASK_NAMES)}


# ── Asset groups for MT25 (extends MT15) ───────────────────────────────────

_MT25_ASSET_GROUPS: dict[str, list[str]] = {
    # MT15 assets — reused.
    "mw_drawer": ["drawer_open", "drawer_close"],
    "mw_button": ["button_press_topdown", "coffee_button", "button_press_topdown_wall"],
    "mw_window": ["window_open", "window_close"],
    "mw_faucet": ["faucet_open", "faucet_close", "dial_turn", "lever_pull"],
    "mw_door": ["door_open", "door_close", "door_lock", "door_unlock"],
    "mw_peg_block": ["peg_insert_side"],
    "drawer_keypoint": ["drawer_open", "drawer_close"],
    "button_keypoint": ["button_press_topdown", "coffee_button", "button_press_topdown_wall"],
    "window_keypoint": ["window_open", "window_close"],
    "faucet_keypoint": ["faucet_open", "faucet_close", "dial_turn", "lever_pull"],
    "door_keypoint": ["door_open", "door_close", "door_lock", "door_unlock"],
    "peg_keypoint": ["peg_insert_side"],
    # MT25 new assets.
    "mw_handle_press": ["handle_press", "handle_pull"],
    "mw_handle_press_side": ["handle_press_side", "handle_pull_side"],
    "mw_peg_unplug": ["peg_unplug_side"],
    "mw_box_with_lid": ["box_close"],
    "mw_plate": ["plate_slide", "plate_slide_back"],
    "mw_wall": ["button_press_topdown_wall"],
    "mw_nail_block": ["hammer"],
    # Per-asset keypoint frames.
    "handle_press_keypoint": ["handle_press", "handle_pull"],
    "handle_press_side_keypoint": ["handle_press_side", "handle_pull_side"],
    "peg_unplug_keypoint": ["peg_unplug_side"],
    "box_close_keypoint": ["box_close"],
    "plate_keypoint": ["plate_slide", "plate_slide_back"],
    "nail_keypoint": ["hammer"],
}


def _build_mt25_clone_groups() -> dict[str, InclusionSet]:
    per_group: dict[str, list[str]] = {n: [] for n in MT25_TASK_NAMES}
    for asset, groups in _MT25_ASSET_GROUPS.items():
        for g in groups:
            per_group[g].append(asset)
    return {g: InclusionSet(assets=assets, weight=1) for g, assets in per_group.items()}


# ── Scene: MT15 scene + 6 new assets + 6 new keypoint frames ───────────────


def _kp_frame(asset: str, source_body: str, marker_body: str) -> FrameTransformerCfg:
    return FrameTransformerCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{asset}/{source_body}",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{asset}/{marker_body}",
                name="kp",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ),
        ],
    )


@configclass
class MetaworldMT25SceneCfg(MetaworldMultiTaskSceneCfg):
    """MT15 scene + 6 new articulated assets + their keypoint frames."""

    # New articulated assets.
    mw_handle_press = MW_HANDLE_PRESS_CFG.replace(prim_path="{ENV_REGEX_NS}/MwHandlePress")
    mw_handle_press_side = MW_HANDLE_PRESS_SIDE_CFG.replace(prim_path="{ENV_REGEX_NS}/MwHandlePressSide")
    mw_peg_unplug = MW_PEG_UNPLUG_CFG.replace(prim_path="{ENV_REGEX_NS}/MwPegUnplug")
    mw_box_with_lid = MW_BOX_WITH_LID_CFG.replace(prim_path="{ENV_REGEX_NS}/MwBoxWithLid")
    mw_plate = MW_PLATE_CFG.replace(prim_path="{ENV_REGEX_NS}/MwPlate")
    mw_wall = MW_WALL_CFG.replace(prim_path="{ENV_REGEX_NS}/MwWall")
    mw_nail_block = MW_NAIL_BLOCK_CFG.replace(prim_path="{ENV_REGEX_NS}/MwNailBlock")

    # New keypoint frames — see SawyerHandlePressSceneCfg / SawyerPlateSceneCfg
    # in metaworld_scenes_cfg.py for the canonical body names.
    handle_press_keypoint: FrameTransformerCfg = _kp_frame("MwHandlePress", "handle_base", "handle_top")
    handle_press_side_keypoint: FrameTransformerCfg = _kp_frame("MwHandlePressSide", "handle_base", "handle_tip")
    peg_unplug_keypoint: FrameTransformerCfg = _kp_frame("MwPegUnplug", "wall_base", "peg_tip")
    box_close_keypoint: FrameTransformerCfg = _kp_frame("MwBoxWithLid", "box_base", "lid_marker")
    plate_keypoint: FrameTransformerCfg = _kp_frame("MwPlate", "groove_base", "plate_marker")
    nail_keypoint: FrameTransformerCfg = _kp_frame("MwNailBlock", "block_base", "nail_head")

    # Replace clone_cfg with the MT25 task list.
    clone_cfg = CloneCfg(clone_groups=_build_mt25_clone_groups(), clone_strategy=interleaved)


# ── Commands ───────────────────────────────────────────────────────────────


def _mt25_task_boxes():
    """One :class:`TaskBoxCfg` per task in MT25_TASK_NAMES, sourced from TASK_SPECS."""
    from ...mdp import TaskBoxCfg

    return [
        TaskBoxCfg(
            name=name,
            obj_low=TASK_SPECS[name].obj_range()[0],
            obj_high=TASK_SPECS[name].obj_range()[1],
            goal_low=TASK_SPECS[name].goal_range()[0],
            goal_high=TASK_SPECS[name].goal_range()[1],
        )
        for name in MT25_TASK_NAMES
    ]


@configclass
class _MT25CommandsCfg(_MultiTaskCommandsCfg):
    def __post_init__(self) -> None:
        # Override the 15-task list with the 25-task one.
        self.ee_pose.tasks = _mt25_task_boxes()


# ── Reward composition ────────────────────────────────────────────────────


_TCP_CFG = SceneEntityCfg("tcp_frame")


def _kp_cfg(name: str) -> SceneEntityCfg:
    return SceneEntityCfg(name)


def _hamacher_cfg(keypoint_frame: str, *, success_radius: float, axis: int | None = None) -> HamacherShapeCfg:
    """Hamacher (reach × in_place) parameterised by keypoint-frame name."""
    if axis is None:
        in_place = ToleranceShapeCfg(
            distance=obj_to_target_dist,
            distance_kwargs={"keypoint_frame_cfg": _kp_cfg(keypoint_frame), "goal_command_name": "ee_pose"},
            margin=obj_init_to_target_dist,
            margin_kwargs={"goal_command_name": "ee_pose"},
            success_radius=success_radius,
        )
    else:
        in_place = ToleranceShapeCfg(
            distance=axis_obj_to_target_dist,
            distance_kwargs={"keypoint_frame_cfg": _kp_cfg(keypoint_frame), "axis": axis},
            margin=axis_init_to_target_dist,
            margin_kwargs={"axis": axis},
            success_radius=success_radius,
        )
    reach = ToleranceShapeCfg(
        distance=tcp_to_obj_dist,
        distance_kwargs={"keypoint_frame_cfg": _kp_cfg(keypoint_frame), "frame_transformer_cfg": _TCP_CFG},
        margin=init_tcp_to_keypoint_init_dist,
        margin_kwargs={"goal_command_name": "ee_pose"},
        success_radius=0.02,
        sigmoid="long_tail",
    )
    return HamacherShapeCfg(term_a=reach, term_b=in_place, scale=10.0)


def _masked_hamacher(task_name: str, keypoint_frame: str, *, success_radius: float = 0.05) -> RewardTermCfg:
    """``hamacher_term`` scattered onto envs in clone group ``task_name``."""
    cfg = _hamacher_cfg(keypoint_frame, success_radius=success_radius)
    return RewardTermCfg(
        func=hamacher_term,
        weight=1.0,
        params={
            "cfg": cfg,
            "asset_cfg": SceneEntityCfg(keypoint_frame, groups=[task_name]),
        },
    )


def _masked_success(task_name: str, keypoint_frame: str, threshold: float = 0.05) -> RewardTermCfg:
    """``success_indicator_term`` — keypoint-to-goal binary, logged only (weight=0)."""
    return RewardTermCfg(
        func=success_indicator_term,
        weight=0.0,
        params={
            "keypoint_frame_cfg": _kp_cfg(keypoint_frame),
            "asset_cfg": SceneEntityCfg(keypoint_frame, groups=[task_name]),
            "goal_command_name": "ee_pose",
            "threshold": threshold,
        },
    )


@configclass
class _MT25RewardsCfg(_MultiTaskRewardsCfg):
    """MT15 rewards + 10 new masked hamacher rewards (one per added task)."""

    # Handle (top-down) — both press and pull use the same keypoint frame.
    handle_press = _masked_hamacher("handle_press", "handle_press_keypoint", success_radius=0.04)
    handle_press_success = _masked_success("handle_press", "handle_press_keypoint", threshold=0.04)
    handle_pull = _masked_hamacher("handle_pull", "handle_press_keypoint", success_radius=0.04)
    handle_pull_success = _masked_success("handle_pull", "handle_press_keypoint", threshold=0.04)

    # Handle (side) — same shape on the side-mounted keypoint frame.
    handle_press_side = _masked_hamacher("handle_press_side", "handle_press_side_keypoint", success_radius=0.04)
    handle_press_side_success = _masked_success("handle_press_side", "handle_press_side_keypoint", threshold=0.04)
    handle_pull_side = _masked_hamacher("handle_pull_side", "handle_press_side_keypoint", success_radius=0.04)
    handle_pull_side_success = _masked_success("handle_pull_side", "handle_press_side_keypoint", threshold=0.04)

    # Peg-unplug-side.
    peg_unplug_side = _masked_hamacher("peg_unplug_side", "peg_unplug_keypoint", success_radius=0.04)
    peg_unplug_side_success = _masked_success("peg_unplug_side", "peg_unplug_keypoint", threshold=0.04)

    # Box-close — looser threshold (lid arcs through space).
    box_close = _masked_hamacher("box_close", "box_close_keypoint", success_radius=0.10)
    box_close_success = _masked_success("box_close", "box_close_keypoint", threshold=0.10)

    # Plate (slide / slide-back) — same shape, opposite direction.
    plate_slide = _masked_hamacher("plate_slide", "plate_keypoint", success_radius=0.05)
    plate_slide_success = _masked_success("plate_slide", "plate_keypoint", threshold=0.05)
    plate_slide_back = _masked_hamacher("plate_slide_back", "plate_keypoint", success_radius=0.05)
    plate_slide_back_success = _masked_success("plate_slide_back", "plate_keypoint", threshold=0.05)

    # Button-press-topdown-wall — same as button-press-topdown but a different
    # task index (so the wall obstacle is respected via the spawned mw_wall asset).
    button_press_topdown_wall = _masked_hamacher("button_press_topdown_wall", "button_keypoint", success_radius=0.05)
    button_press_topdown_wall_success = _masked_success("button_press_topdown_wall", "button_keypoint", threshold=0.05)

    # Hammer — drive the nail joint into the block (axis 2 = z-prismatic).
    hammer = _masked_hamacher("hammer", "nail_keypoint", success_radius=0.04)
    hammer_success = _masked_success("hammer", "nail_keypoint", threshold=0.04)


# ── Joint reset events for the new assets ─────────────────────────────────


def _spec_reset_event_with_index(asset_name: str, task_names: list[str]) -> EventTermCfg:
    """Same as :func:`_spec_reset_event` but uses MT25_TASK_TO_INDEX so the
    task-id mask matches the MT25 ordering."""
    from .multi_task_env_cfg import task_indexed_joint_reset

    joint_names = {TASK_SPECS[t].joint_name for t in task_names}
    assert len(joint_names) == 1, f"{asset_name}: tasks {task_names} disagree on joint_name {joint_names}"
    (joint_name,) = joint_names
    return EventTermCfg(
        func=task_indexed_joint_reset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(asset_name, joint_names=[joint_name]),
            "joint_value_by_task": {str(MT25_TASK_TO_INDEX[t]): TASK_SPECS[t].joint_reset_value for t in task_names},
        },
    )


@configclass
class _MT25EventCfg(_MultiTaskEventCfg):
    """MT15 events + 6 new per-asset joint resets."""

    reset_handle_press = _spec_reset_event_with_index("mw_handle_press", ["handle_press", "handle_pull"])
    reset_handle_press_side = _spec_reset_event_with_index(
        "mw_handle_press_side", ["handle_press_side", "handle_pull_side"]
    )
    reset_peg_unplug = _spec_reset_event_with_index("mw_peg_unplug", ["peg_unplug_side"])
    reset_box_close = _spec_reset_event_with_index("mw_box_with_lid", ["box_close"])
    reset_plate = _spec_reset_event_with_index("mw_plate", ["plate_slide", "plate_slide_back"])
    reset_nail_block = _spec_reset_event_with_index("mw_nail_block", ["hammer"])


# ── Per-task success spec (extends MT15) → done-on-success termination ─────


_MT25_NEW_SUCCESS_SPECS: list[tuple[str, str, float]] = [
    ("handle_press", "handle_press_keypoint", 0.04),
    ("handle_pull", "handle_press_keypoint", 0.04),
    ("handle_press_side", "handle_press_side_keypoint", 0.04),
    ("handle_pull_side", "handle_press_side_keypoint", 0.04),
    ("peg_unplug_side", "peg_unplug_keypoint", 0.04),
    ("box_close", "box_close_keypoint", 0.10),
    ("plate_slide", "plate_keypoint", 0.05),
    ("plate_slide_back", "plate_keypoint", 0.05),
    ("button_press_topdown_wall", "button_keypoint", 0.05),
    ("hammer", "nail_keypoint", 0.04),
]


def _mt25_termination_term(task_name: str, keypoint_frame: str, threshold: float) -> TerminationTermCfg:
    return TerminationTermCfg(
        func=keypoint_success_termination,
        params={
            "keypoint_frame_cfg": _kp_cfg(keypoint_frame),
            "asset_cfg": SceneEntityCfg(keypoint_frame, groups=[task_name]),
            "goal_command_name": "ee_pose",
            "threshold": threshold,
        },
    )


@configclass
class _MT25TerminationsCfg(_MultiTaskTerminationsCfg):
    """MT15 terminations + 10 new per-task done-on-success."""

    locals().update({f"{t}_done": _mt25_termination_term(t, kp, thr) for t, kp, thr in _MT25_NEW_SUCCESS_SPECS})


# ── Top-level env cfg ─────────────────────────────────────────────────────


@configclass
class MetaworldMT25SawyerEnvCfg(MetaworldMultiTaskSawyerEnvCfg):
    """MT25 = MT15 + 10 articulated tasks (handle × 4 + peg-unplug + box-close
    + plate × 2 + button-wall + hammer)."""

    scene: MetaworldMT25SceneCfg = MetaworldMT25SceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _MT25CommandsCfg()
    observations = _MultiTaskObsCfg()
    rewards = _MT25RewardsCfg()
    events = _MT25EventCfg()
    terminations = _MT25TerminationsCfg()
