# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MT50 multi-task env: MT25 + 25 cube-as-manipulandum tasks.

Adds (over MT25):
* MT3 cube tasks                   — reach, push, pick_place
* Cube + obstacle                  — push_back, push_wall, reach_wall, pick_place_wall
* Cube + kinematic destination     — basketball, shelf_place, soccer, sweep, sweep_into,
                                     coffee_push, coffee_pull, stick_push, stick_pull,
                                     bin_picking, hand_insert, pick_out_of_hole,
                                     assembly, disassemble
* Plate-side                       — plate_slide_side, plate_slide_back_side
* Button-front                     — button_press, button_press_wall

Architectural notes:

* The cube is now **visible 4 cm** (not 1 mm hidden anchor as in MT15 / MT25)
  so cube tasks have a graspable manipulandum. For articulated tasks the
  cube is placed at the per-task ``obj_init`` (set by
  :class:`MetaworldMultiTaskCommand`); on those tasks it's visual clutter
  near the keypoint but doesn't interfere physically (the gripper's
  caging/reach reward reads the per-asset keypoint frame, not the cube).
* For cube tasks (reach / push / pick / etc.) the reward reads the cube
  via a single shared ``cube_keypoint`` frame transformer.
* Per-task rewards use the same MW V2 archetypes as the single-task envs:
  push/sweep/coffee/stick/soccer use ``LinearComboShapeCfg``
  (2 × caging + phase bonus); pick-place/basketball/shelf/bin/assembly
  use ``CagingTimesInPlaceShapeCfg`` (caging × in_place + lift bonus);
  reach/reach-wall/hand-insert use ``ToleranceShapeCfg``; plate-side and
  button-front use ``HamacherShapeCfg``. Each task is decomposed into a
  *monotonic* term + a *success bonus* term (``success_indicator_term``)
  — IsaacLab's reward manager sums them. MW's success-override
  semantics (``reward = 10`` replace) becomes ``reward += 10`` on top
  of the monotonic shape; the policy still maximises success, just with
  a stronger gradient at the converged state.
* All terms are group-scoped via ``asset_cfg = SceneEntityCfg(name,
  groups=[task_name])`` and the ``@scatterable`` wrappers in
  ``mdp/scatter_rewards.py`` write only matching env_ids — no
  ``task_masked_reward`` indirection.
* New kinematic assets: basket, bin, shelf, soccer_goal, hole_block,
  assembly_peg, plate_side, button_front.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.cloner.cloner_strategies import interleaved
from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.scene import CloneCfg, InclusionSet
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from ...mdp import (
    CagingTimesInPlaceShapeCfg,
    GripperCagingParams,
    HamacherShapeCfg,
    LinearComboShapeCfg,
    PhaseBonusCfg,
    ToleranceShapeCfg,
    TriggerCfg,
    WeightedAtomCfg,
    axis_init_to_target_dist,
    axis_obj_to_target_dist,
    gripper_caging,
    gripper_closed,
    gripper_open,
    hand_init_to_target_dist,
    init_tcp_to_keypoint_init_dist,
    obj_init_to_target_dist,
    obj_to_target_dist,
    obj_z_above_init,
    pick_place_caging,
    tcp_to_obj_dist,
    tcp_to_target_dist,
)
from ...mdp.scatter_rewards import (
    caging_times_in_place_term,
    hamacher_term,
    keypoint_success_termination,
    linear_combo_term,
    success_indicator_term,
    tolerance_term,
)
from ...metaworld_assets_cfg import (
    MW_ASSEMBLY_PEG_CFG,
    MW_BASKET_CFG,
    MW_BIN_CFG,
    MW_BUTTON_FRONT_CFG,
    MW_HOLE_BLOCK_CFG,
    MW_PLATE_SIDE_CFG,
    MW_SHELF_CFG,
    MW_SOCCER_GOAL_CFG,
)
from ...metaworld_specs import TASK_SPECS
from .mt25_env_cfg import (
    MT25_TASK_NAMES,
    MetaworldMT25SawyerEnvCfg,
    MetaworldMT25SceneCfg,
    _MT25CommandsCfg,
    _MT25EventCfg,
    _MT25RewardsCfg,
    _MT25TerminationsCfg,
)

# ── Task list (MT25 + 25 cube tasks) ───────────────────────────────────────

MT50_NEW_TASKS: list[str] = [
    # MT3 cube
    "reach",
    "push",
    "pick_place",
    # Cube + obstacle
    "push_back",
    "push_wall",
    "reach_wall",
    "pick_place_wall",
    # Cube + destination
    "basketball",
    "shelf_place",
    "soccer",
    "sweep",
    "sweep_into",
    "coffee_push",
    "coffee_pull",
    "stick_push",
    "stick_pull",
    "bin_picking",
    "hand_insert",
    "pick_out_of_hole",
    "assembly",
    "disassemble",
    # Plate-side / button-front (new USDs)
    "plate_slide_side",
    "plate_slide_back_side",
    "button_press",
    "button_press_wall",
]

MT50_TASK_NAMES: list[str] = list(MT25_TASK_NAMES) + MT50_NEW_TASKS
MT50_TASK_TO_INDEX: dict[str, int] = {n: i for i, n in enumerate(MT50_TASK_NAMES)}

# Tasks where the **cube** is the keypoint (not an articulated asset).
_CUBE_KEYPOINT_TASKS = [
    "reach",
    "push",
    "pick_place",
    "push_back",
    "push_wall",
    "reach_wall",
    "pick_place_wall",
    "basketball",
    "shelf_place",
    "soccer",
    "sweep",
    "sweep_into",
    "coffee_push",
    "coffee_pull",
    "stick_push",
    "stick_pull",
    "bin_picking",
    "hand_insert",
    "pick_out_of_hole",
    "assembly",
    "disassemble",
]


# ── Asset groups (extend MT25's) ──────────────────────────────────────────

# Kinematic assets that travel with their cube tasks.
_MT50_NEW_ASSET_GROUPS: dict[str, list[str]] = {
    "mw_basket": ["basketball"],
    "mw_shelf": ["shelf_place"],
    "mw_soccer_goal": ["soccer"],
    "mw_bin": ["bin_picking"],
    "mw_hole_block": ["hand_insert", "pick_out_of_hole"],
    "mw_assembly_peg": ["assembly", "disassemble"],
    "mw_plate_side": ["plate_slide_side", "plate_slide_back_side"],
    "mw_button_front": ["button_press", "button_press_wall"],
    # Per-asset keypoints for the new articulated tasks.
    "plate_side_keypoint": ["plate_slide_side", "plate_slide_back_side"],
    "button_front_keypoint": ["button_press", "button_press_wall"],
    # cube_keypoint travels with every cube task.
    "cube_keypoint": _CUBE_KEYPOINT_TASKS,
}


def _build_mt50_clone_groups() -> dict[str, InclusionSet]:
    """Extend MT25 clone groups with the MT50 task assignments."""
    # Start from MT25's clone groups, then add MT50 entries.
    from .mt25_env_cfg import _MT25_ASSET_GROUPS

    combined: dict[str, list[str]] = {}
    for asset, groups in _MT25_ASSET_GROUPS.items():
        combined[asset] = list(groups)
    for asset, groups in _MT50_NEW_ASSET_GROUPS.items():
        combined.setdefault(asset, []).extend(groups)

    per_group: dict[str, list[str]] = {n: [] for n in MT50_TASK_NAMES}
    for asset, groups in combined.items():
        for g in groups:
            if g in per_group:
                per_group[g].append(asset)
    return {g: InclusionSet(assets=assets, weight=1) for g, assets in per_group.items()}


# ── Scene: MT25 + visible cube + 8 kinematic destinations ─────────────────


# Override cube to be visible 4 cm (so cube tasks have a graspable manipulandum).
_VISIBLE_CUBE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Cube",
    spawn=sim_utils.CuboidCfg(
        size=(0.04, 0.04, 0.04),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.75),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.6)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.6, 0.02)),
)


@configclass
class MetaworldMT50SceneCfg(MetaworldMT25SceneCfg):
    """MT25 scene + visible cube + 8 new kinematic / front-button assets."""

    cube = _VISIBLE_CUBE_CFG

    # Cube keypoint — for cube tasks.
    cube_keypoint: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Cube",
                name="kp",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ),
        ],
    )

    # Kinematic destinations (no joints).
    mw_basket = MW_BASKET_CFG.replace(prim_path="{ENV_REGEX_NS}/MwBasket")
    mw_bin = MW_BIN_CFG.replace(prim_path="{ENV_REGEX_NS}/MwBin")
    mw_shelf = MW_SHELF_CFG.replace(prim_path="{ENV_REGEX_NS}/MwShelf")
    mw_soccer_goal = MW_SOCCER_GOAL_CFG.replace(prim_path="{ENV_REGEX_NS}/MwSoccerGoal")
    mw_hole_block = MW_HOLE_BLOCK_CFG.replace(prim_path="{ENV_REGEX_NS}/MwHoleBlock")
    mw_assembly_peg = MW_ASSEMBLY_PEG_CFG.replace(prim_path="{ENV_REGEX_NS}/MwAssemblyPeg")

    # Articulated (the two new USDs: plate-side and front-button).
    mw_plate_side = MW_PLATE_SIDE_CFG.replace(prim_path="{ENV_REGEX_NS}/MwPlateSide")
    mw_button_front = MW_BUTTON_FRONT_CFG.replace(prim_path="{ENV_REGEX_NS}/MwButtonFront")

    plate_side_keypoint: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/MwPlateSide/groove_base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/MwPlateSide/plate_marker",
                name="kp",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ),
        ],
    )
    button_front_keypoint: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/MwButtonFront/button_box",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/MwButtonFront/button_top",
                name="kp",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ),
        ],
    )

    # Replace the clone_cfg with MT50's expanded task set.
    clone_cfg = CloneCfg(clone_groups=_build_mt50_clone_groups(), clone_strategy=interleaved)


# ── Commands ───────────────────────────────────────────────────────────────


def _mt50_task_boxes():
    from ...mdp import TaskBoxCfg

    return [
        TaskBoxCfg(
            name=name,
            obj_low=TASK_SPECS[name].obj_range()[0] if name in TASK_SPECS else (-0.1, 0.6, 0.02),
            obj_high=TASK_SPECS[name].obj_range()[1] if name in TASK_SPECS else (0.1, 0.7, 0.02),
            goal_low=TASK_SPECS[name].goal_range()[0] if name in TASK_SPECS else (-0.1, 0.8, 0.05),
            goal_high=TASK_SPECS[name].goal_range()[1] if name in TASK_SPECS else (0.1, 0.9, 0.30),
        )
        for name in MT50_TASK_NAMES
    ]


@configclass
class _MT50CommandsCfg(_MT25CommandsCfg):
    def __post_init__(self) -> None:
        # Skip MT25's __post_init__ — go straight to the 50-task list.
        self.ee_pose.tasks = _mt50_task_boxes()


# ── Reward composition (no stubs) ──────────────────────────────────────────
#
# Per-task rewards decompose into multiple ``RewardTermCfg``s (typically
# 2 — a monotonic shaped term + a success bonus). All atoms read scene
# entities scoped via ``SceneEntityCfg(name, groups=[task_name])`` and
# the ``@scatterable`` wrappers in ``mdp/scatter_rewards.py`` write only
# matching env_ids into the reward buffer; non-matching envs get zero
# from this task's terms automatically. No ``task_masked_reward`` wrapper
# is needed.
#
# Module-level shape literals are shared across tasks that use the same
# archetype (push / pick-place / reach / plate-slide / button-front) —
# the SHAPE is the composition; per-task RewardTermCfg only varies the
# ``asset_cfg.groups``.

_TCP_CFG = SceneEntityCfg("tcp_frame")
_CUBE_KP = SceneEntityCfg("cube_keypoint")
_PLATE_SIDE_KP = SceneEntityCfg("plate_side_keypoint")
_BUTTON_FRONT_KP = SceneEntityCfg("button_front_keypoint")
_DEFAULT_HAND_INIT_E = (-0.067, 0.571, 0.132)


# ── Push V2 (linear-combo: 2*caging + phase_bonus, no override) ───────────

_PUSH_V2_SHAPE = LinearComboShapeCfg(
    terms=[
        WeightedAtomCfg(
            weight=2.0,
            atom=gripper_caging,
            atom_kwargs={
                "frame_transformer_cfg": _TCP_CFG,
                "keypoint_frame_cfg": _CUBE_KP,
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
    in_place_distance_kwargs={"keypoint_frame_cfg": _CUBE_KP, "goal_command_name": "ee_pose"},
    in_place_margin=obj_init_to_target_dist,
    in_place_margin_kwargs={"goal_command_name": "ee_pose"},
    in_place_success_radius=0.05,
    phase=PhaseBonusCfg(
        triggers=[
            TriggerCfg(
                atom=tcp_to_obj_dist,
                op="<",
                threshold=0.02,
                atom_kwargs={"frame_transformer_cfg": _TCP_CFG, "keypoint_frame_cfg": _CUBE_KP},
            ),
            TriggerCfg(atom=gripper_open, op=">", threshold=0.0),
        ],
        offset=1.0,
        in_place_mult=5.0,
        self_mult=1.0,
    ),
    success_override=None,  # Decomposed: success bonus is its own term.
)


# ── Pick-Place V2 (caging × in_place + lift bonus, no override) ───────────

_PICK_PLACE_V2_SHAPE = CagingTimesInPlaceShapeCfg(
    caging=pick_place_caging,
    caging_kwargs={
        "frame_transformer_cfg": _TCP_CFG,
        "keypoint_frame_cfg": _CUBE_KP,
        "goal_command_name": "ee_pose",
    },
    distance=obj_to_target_dist,
    distance_kwargs={"keypoint_frame_cfg": _CUBE_KP, "goal_command_name": "ee_pose"},
    margin=obj_init_to_target_dist,
    margin_kwargs={"goal_command_name": "ee_pose"},
    success_radius=0.07,
    phase=PhaseBonusCfg(
        triggers=[
            TriggerCfg(
                atom=tcp_to_obj_dist,
                op="<",
                threshold=0.02,
                atom_kwargs={"frame_transformer_cfg": _TCP_CFG, "keypoint_frame_cfg": _CUBE_KP},
            ),
            TriggerCfg(atom=gripper_open, op=">", threshold=0.0),
            TriggerCfg(
                atom=obj_z_above_init,
                op=">",
                threshold=0.01,
                atom_kwargs={"keypoint_frame_cfg": _CUBE_KP, "goal_command_name": "ee_pose"},
            ),
        ],
        offset=1.0,
        in_place_mult=5.0,
    ),
    success_override=None,
)


# ── Reach V2 (10 * tolerance(tcp → goal)) ──────────────────────────────────

_REACH_V2_SHAPE = ToleranceShapeCfg(
    distance=tcp_to_target_dist,
    distance_kwargs={"frame_transformer_cfg": _TCP_CFG, "goal_command_name": "ee_pose"},
    margin=hand_init_to_target_dist,
    margin_kwargs={"goal_command_name": "ee_pose", "hand_init_pos_e": _DEFAULT_HAND_INIT_E},
    success_radius=0.05,
    scale=10.0,
)


# ── Plate-Slide V2 (hamacher(reach × in_place)) for plate-side variants ───

_PLATE_SLIDE_V2_SHAPE = HamacherShapeCfg(
    term_a=ToleranceShapeCfg(
        distance=tcp_to_obj_dist,
        distance_kwargs={"keypoint_frame_cfg": _PLATE_SIDE_KP, "frame_transformer_cfg": _TCP_CFG},
        margin=init_tcp_to_keypoint_init_dist,
        margin_kwargs={"goal_command_name": "ee_pose"},
        success_radius=0.02,
        sigmoid="long_tail",
    ),
    term_b=ToleranceShapeCfg(
        distance=obj_to_target_dist,
        distance_kwargs={"keypoint_frame_cfg": _PLATE_SIDE_KP, "goal_command_name": "ee_pose"},
        margin=obj_init_to_target_dist,
        margin_kwargs={"goal_command_name": "ee_pose"},
        success_radius=0.05,
    ),
    scale=10.0,
)


# ── Button-Front V2 (hamacher(near_button × axis-1 displacement)) ─────────
#
# MW's front-button: cap protrudes -Y at joint=0, pressed +Y at joint=+0.06.
# Reward gates the press (axis=1 displacement) on the gripper being closed
# and near the cap. Phase bonus fires ``+5 * in_place`` when within 3 cm.

_BUTTON_FRONT_V2_SHAPE = HamacherShapeCfg(
    term_a=ToleranceShapeCfg(
        distance=tcp_to_obj_dist,
        distance_kwargs={"keypoint_frame_cfg": _BUTTON_FRONT_KP, "frame_transformer_cfg": _TCP_CFG},
        margin=tcp_to_obj_dist,
        margin_kwargs={"keypoint_frame_cfg": _BUTTON_FRONT_KP, "frame_transformer_cfg": _TCP_CFG},
        success_radius=0.01,
    ),
    a_modulator=gripper_closed,
    term_b=ToleranceShapeCfg(
        distance=axis_obj_to_target_dist,
        distance_kwargs={"keypoint_frame_cfg": _BUTTON_FRONT_KP, "axis": 1},
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
                atom_kwargs={"keypoint_frame_cfg": _BUTTON_FRONT_KP, "frame_transformer_cfg": _TCP_CFG},
            ),
        ],
        offset=0.0,
        in_place_mult=5.0,
        self_mult=0.0,
    ),
)


# ── Helpers that compose a (monotonic, success) RewardTermCfg pair ────────
#
# Returning a dict of ``RewardTermCfg`` keyed by name lets us splat into
# ``_MT50RewardsCfg`` as class attributes via dict-unpacking-at-class-time.
# Each helper is just a literal composition — no task-specific logic.


def _push_terms(task: str) -> dict[str, RewardTermCfg]:
    asset = SceneEntityCfg("cube_keypoint", groups=[task])
    return {
        f"{task}_main": RewardTermCfg(
            func=linear_combo_term,
            weight=1.0,
            params={"cfg": _PUSH_V2_SHAPE, "asset_cfg": asset},
        ),
        f"{task}_success": RewardTermCfg(
            func=success_indicator_term,
            weight=10.0,
            params={
                "keypoint_frame_cfg": _CUBE_KP,
                "asset_cfg": asset,
                "goal_command_name": "ee_pose",
                "threshold": 0.05,
            },
        ),
    }


def _pick_place_terms(task: str) -> dict[str, RewardTermCfg]:
    asset = SceneEntityCfg("cube_keypoint", groups=[task])
    return {
        f"{task}_main": RewardTermCfg(
            func=caging_times_in_place_term,
            weight=1.0,
            params={"cfg": _PICK_PLACE_V2_SHAPE, "asset_cfg": asset},
        ),
        f"{task}_success": RewardTermCfg(
            func=success_indicator_term,
            weight=10.0,
            params={
                "keypoint_frame_cfg": _CUBE_KP,
                "asset_cfg": asset,
                "goal_command_name": "ee_pose",
                "threshold": 0.07,
            },
        ),
    }


def _reach_terms(task: str) -> dict[str, RewardTermCfg]:
    asset = SceneEntityCfg("cube_keypoint", groups=[task])
    return {
        f"{task}_main": RewardTermCfg(
            func=tolerance_term,
            weight=1.0,
            params={"cfg": _REACH_V2_SHAPE, "asset_cfg": asset},
        ),
        f"{task}_success": RewardTermCfg(
            func=success_indicator_term,
            weight=10.0,
            params={
                "keypoint_frame_cfg": _CUBE_KP,
                "asset_cfg": asset,
                "goal_command_name": "ee_pose",
                "threshold": 0.05,
            },
        ),
    }


def _plate_slide_terms(task: str) -> dict[str, RewardTermCfg]:
    asset = SceneEntityCfg("plate_side_keypoint", groups=[task])
    return {
        f"{task}_main": RewardTermCfg(
            func=hamacher_term,
            weight=1.0,
            params={"cfg": _PLATE_SLIDE_V2_SHAPE, "asset_cfg": asset},
        ),
        f"{task}_success": RewardTermCfg(
            func=success_indicator_term,
            weight=10.0,
            params={
                "keypoint_frame_cfg": _PLATE_SIDE_KP,
                "asset_cfg": asset,
                "goal_command_name": "ee_pose",
                "threshold": 0.05,
            },
        ),
    }


def _button_front_terms(task: str) -> dict[str, RewardTermCfg]:
    asset = SceneEntityCfg("button_front_keypoint", groups=[task])
    return {
        f"{task}_main": RewardTermCfg(
            func=hamacher_term,
            weight=1.0,
            params={"cfg": _BUTTON_FRONT_V2_SHAPE, "asset_cfg": asset},
        ),
        f"{task}_success": RewardTermCfg(
            func=success_indicator_term,
            weight=10.0,
            params={
                "keypoint_frame_cfg": _BUTTON_FRONT_KP,
                "asset_cfg": asset,
                "goal_command_name": "ee_pose",
                "threshold": 0.05,
            },
        ),
    }


# ── Per-task reward decomposition ─────────────────────────────────────────
#
# Each task contributes ≥1 ``RewardTermCfg`` (most contribute 2 — a
# monotonic shaped term and a success bonus). Reward manager sums them.
# Group-scoping via ``asset_cfg=SceneEntityCfg(..., groups=[task])`` makes
# each term fire only on its own envs.


def _all_mt50_cube_terms() -> dict[str, RewardTermCfg]:
    """Build every (task, term) RewardTermCfg pair for MT50's cube tasks
    and the plate-side / button-front articulated tasks."""
    out: dict[str, RewardTermCfg] = {}

    # MT3 cube + obstacle/destination → push V2.
    for t in (
        "push",
        "push_back",
        "push_wall",
        "soccer",
        "sweep",
        "sweep_into",
        "coffee_push",
        "coffee_pull",
        "stick_push",
        "stick_pull",
    ):
        out.update(_push_terms(t))

    # Pick-and-place family (cube goes from a start pose to a destination).
    for t in (
        "pick_place",
        "pick_place_wall",
        "basketball",
        "shelf_place",
        "bin_picking",
        "pick_out_of_hole",
        "assembly",
        "disassemble",
    ):
        out.update(_pick_place_terms(t))

    # Reach-style (TCP-target tasks).
    for t in ("reach", "reach_wall", "hand_insert"):
        out.update(_reach_terms(t))

    # Plate-side (Y-axis sliding, articulated).
    for t in ("plate_slide_side", "plate_slide_back_side"):
        out.update(_plate_slide_terms(t))

    # Front-facing button.
    for t in ("button_press", "button_press_wall"):
        out.update(_button_front_terms(t))

    return out


@configclass
class _MT50RewardsCfg(_MT25RewardsCfg):
    """MT25 rewards + per-task MW V2 rewards for the 25 cube / plate-side /
    button-front tasks. Decomposed into 2 ``RewardTermCfg``s per task
    (monotonic shape + success bonus), all group-scoped via
    ``SceneEntityCfg(groups=[task_name])``."""

    # Splat the 50 RewardTermCfgs (25 tasks × 2 terms each) as class attrs.
    locals().update(_all_mt50_cube_terms())


# ── Events: MT25 events + new joint resets for plate-side / button-front ──


def _mt50_spec_reset_event(asset_name: str, task_names: list[str]):
    from isaaclab.managers import EventTermCfg

    from .multi_task_env_cfg import task_indexed_joint_reset

    joint_names = {TASK_SPECS[t].joint_name for t in task_names}
    assert len(joint_names) == 1, f"{asset_name}: tasks {task_names} disagree on joint_name {joint_names}"
    (joint_name,) = joint_names
    return EventTermCfg(
        func=task_indexed_joint_reset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(asset_name, joint_names=[joint_name]),
            "joint_value_by_task": {str(MT50_TASK_TO_INDEX[t]): TASK_SPECS[t].joint_reset_value for t in task_names},
        },
    )


@configclass
class _MT50EventCfg(_MT25EventCfg):
    reset_plate_side = _mt50_spec_reset_event("mw_plate_side", ["plate_slide_side", "plate_slide_back_side"])
    reset_button_front = _mt50_spec_reset_event("mw_button_front", ["button_press", "button_press_wall"])


# ── Per-task done-on-success terminations (extends MT25) ────────────────────
#
# Each cube/articulated task gets a ``TerminationTermCfg`` that fires on its
# own clone-group envs only (via ``asset_cfg.env_ids``). The success
# threshold mirrors the per-task ``_success`` reward term in
# :class:`_MT50RewardsCfg`.

_MT50_NEW_SUCCESS_SPECS: list[tuple[str, str, str, float]] = [
    # (task_name, keypoint_asset_name, keypoint_frame_attr_var, threshold)
    *[
        (t, "cube_keypoint", "_CUBE_KP", 0.05)
        for t in (
            "push",
            "push_back",
            "push_wall",
            "soccer",
            "sweep",
            "sweep_into",
            "coffee_push",
            "coffee_pull",
            "stick_push",
            "stick_pull",
            "reach",
            "reach_wall",
            "hand_insert",
        )
    ],
    *[
        (t, "cube_keypoint", "_CUBE_KP", 0.07)
        for t in (
            "pick_place",
            "pick_place_wall",
            "basketball",
            "shelf_place",
            "bin_picking",
            "pick_out_of_hole",
            "assembly",
            "disassemble",
        )
    ],
    *[
        (t, "plate_side_keypoint", "_PLATE_SIDE_KP", 0.05)
        for t in (
            "plate_slide_side",
            "plate_slide_back_side",
        )
    ],
    *[
        (t, "button_front_keypoint", "_BUTTON_FRONT_KP", 0.05)
        for t in (
            "button_press",
            "button_press_wall",
        )
    ],
]


def _mt50_termination_term(task_name: str, keypoint_asset: str, threshold: float) -> TerminationTermCfg:
    return TerminationTermCfg(
        func=keypoint_success_termination,
        params={
            "keypoint_frame_cfg": SceneEntityCfg(keypoint_asset),
            "asset_cfg": SceneEntityCfg(keypoint_asset, groups=[task_name]),
            "goal_command_name": "ee_pose",
            "threshold": threshold,
        },
    )


@configclass
class _MT50TerminationsCfg(_MT25TerminationsCfg):
    """MT25 terminations + 25 new per-task done-on-success."""

    locals().update(
        {f"{t}_done": _mt50_termination_term(t, kp_asset, thr) for t, kp_asset, _, thr in _MT50_NEW_SUCCESS_SPECS}
    )


# ── Top-level env cfg ─────────────────────────────────────────────────────


@configclass
class MetaworldMT50SawyerEnvCfg(MetaworldMT25SawyerEnvCfg):
    """MT50 = MT25 + 25 cube-tail / kinematic-destination tasks.

    Cube is now visible 4 cm (vs 1 mm hidden anchor in MT15 / MT25). For
    articulated tasks, the cube ends up positioned at the per-task
    ``obj_init`` — cosmetic clutter near the asset, but the reward reads
    the asset's keypoint frame so it doesn't influence the policy.
    """

    scene: MetaworldMT50SceneCfg = MetaworldMT50SceneCfg(num_envs=4096, env_spacing=2.5)
    commands = _MT50CommandsCfg()
    rewards = _MT50RewardsCfg()
    events = _MT50EventCfg()
    terminations = _MT50TerminationsCfg()
