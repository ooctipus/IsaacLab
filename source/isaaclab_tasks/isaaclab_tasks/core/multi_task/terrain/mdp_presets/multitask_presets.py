# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-task task-set presets for :class:`MultiTaskCfg.tasks`.

Named task configurations selectable via ``presets=<name>`` on the CLI or by
explicit override in code. Structure mirrors :class:`CommandsPresetCfg` — each
named class attribute is one preset alternative; ``default`` is the fallback
when no CLI override is given.

Starting point (``simple_pos_vel``): one instant subtask (body position) + one
tracking subtask (linear velocity). Enough to exercise the multi-task composer
end-to-end without the full 8-task surface. Once that's training stably,
users can switch to richer presets (``pose_vel``, ``locomotion``).
"""

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from ..tasks_cfg import (
    ANG_VEL_TRACKING,
    BODY_POS_INSTANT,
    BODY_QUAT_INSTANT,
    FEET_CONTACT_COUNT_INSTANT,
    LIN_VEL_TRACKING,
    MECH_POWER,
    PAIR_CONTACT_DIFF_TRACKING,
    STANDING_Z,
    UNDESIRED_CONTACT,
)

# ---------------------------------------------------------------------------
# Shared subtask builders — reused across presets so edits land in one place.
# ---------------------------------------------------------------------------

TWO_METER_POSITON_INSTANT = BODY_POS_INSTANT.replace(
    activation_kernel_param=0.3,
    sampler=BODY_POS_INSTANT.sampler.replace(
        minimum=[-2.0, -2.0, STANDING_Z[0]],
        maximum=[2.0, 2.0, STANDING_Z[1]],
    ),
)


# ---------------------------------------------------------------------------
# Preset — simple starter (1 instant + 1 tracking).
# ---------------------------------------------------------------------------

_SIMPLE_POS_VEL = {
    # Instant: reach a body-position target near standing height.
    "position": [BODY_POS_INSTANT, UNDESIRED_CONTACT, MECH_POWER],
    # Tracking: hold a linear-velocity target over transit.
    "lin_vel": [LIN_VEL_TRACKING, UNDESIRED_CONTACT, MECH_POWER],
}


# ---------------------------------------------------------------------------
# Single-task presets — useful for isolating one supervision signal.
# ---------------------------------------------------------------------------

_VELOCITY_ONLY = {
    # Pure tracking: lin + ang velocity. No instant subtask, so episodes
    # only terminate at time_out / drop / base_contact — the composer's
    # G is just the transit-mean activation over both tracking slots,
    # discounted by the safety multiplicative factors.
    "velocity": [ANG_VEL_TRACKING, LIN_VEL_TRACKING, UNDESIRED_CONTACT, MECH_POWER],
}

_POSITION_ONLY = {
    # Pure instant: reach a body-position target.
    "position": [BODY_POS_INSTANT, UNDESIRED_CONTACT, MECH_POWER],
}


# ---------------------------------------------------------------------------
# Preset — pose + velocity (2 instant + 2 tracking).
# ---------------------------------------------------------------------------

_POSE_VEL = {
    "pose": [BODY_POS_INSTANT, BODY_QUAT_INSTANT, UNDESIRED_CONTACT, MECH_POWER],
    "velocity": [LIN_VEL_TRACKING, ANG_VEL_TRACKING, UNDESIRED_CONTACT, MECH_POWER],
}


# ---------------------------------------------------------------------------
# Preset — full locomotion suite (the previously-hardcoded 8 tasks).
#
# Every task includes the same shared safety set via ``*make_safety_subtasks()``
# — explicit at the call site, no hidden post-processing of the preset dict.
# Subtask dedup at :func:`spec.build_spec` collapses identical safety subtasks
# across tasks into a single shared spec row, so per-step compute cost is
# unchanged regardless of how many tasks list them.
# ---------------------------------------------------------------------------

_LOCOMOTION = {
    # Velocity tracking — two tracking subtasks (lin + ang).
    "velocity": [LIN_VEL_TRACKING, ANG_VEL_TRACKING, UNDESIRED_CONTACT, MECH_POWER],
    # Reach a body-position target near standing height.
    "position": [BODY_POS_INSTANT, UNDESIRED_CONTACT, MECH_POWER],
    # Reach an elevated body-position target (z lifted into the air).
    "reach_point_in_air": [
        BODY_POS_INSTANT.replace(
            activation_kernel_param=0.25,
            sampler=BODY_POS_INSTANT.sampler.replace(
                minimum=[-2.0, -2.0, 1.0],
                maximum=[2.0, 2.0, 1.5],
            ),
        ),
        UNDESIRED_CONTACT,
        MECH_POWER,
    ],
    # Reach body-position AND body-orientation — two instant subtasks.
    "pose": [BODY_POS_INSTANT, BODY_QUAT_INSTANT, UNDESIRED_CONTACT, MECH_POWER],
    # Tip the base to near-vertical pitch — one instant subtask.
    # Pitch ≈ -π/2 tips Anymal-C back onto its rear feet under the
    # x-forward, z-up convention (axis-angle about +y is negative pitch).
    "two_feet_stand": [
        BODY_QUAT_INSTANT.replace(
            sampler=BODY_QUAT_INSTANT.sampler.replace(
                minimum=[0.0, -1.7, -3.14],
                maximum=[0.0, -1.3, 3.14],
            ),
        ),
        UNDESIRED_CONTACT,
        MECH_POWER,
    ],
    # Walking-tripod stance — exactly 3 feet in contact, 1 in the air.
    # Which foot is lifted is free; the policy chooses.
    "tripod_walk": [
        TWO_METER_POSITON_INSTANT,
        FEET_CONTACT_COUNT_INSTANT,
        UNDESIRED_CONTACT,
        MECH_POWER,
    ],
    # Gaits expressed as "one pair fully planted, the other fully
    # airborne". BODY_CONTACT_COUNT_DIFF + GREATER activation fires on
    # either sign, so one subtask covers both alternating phases without
    # hard-coding which half is which.
    # Run (bound): front pair vs hind pair.
    "run": [
        TWO_METER_POSITON_INSTANT,
        PAIR_CONTACT_DIFF_TRACKING.replace(
            asset_cfg=SceneEntityCfg(
                "contact_forces",
                body_names=["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"],
            ),
        ),
        UNDESIRED_CONTACT,
        MECH_POWER,
    ],
    # Trot: diagonal pairs (LF+RH vs RF+LH).
    "trot": [
        TWO_METER_POSITON_INSTANT,
        PAIR_CONTACT_DIFF_TRACKING.replace(
            asset_cfg=SceneEntityCfg(
                "contact_forces",
                body_names=["LF_FOOT", "RH_FOOT", "RF_FOOT", "LH_FOOT"],
            ),
        ),
        UNDESIRED_CONTACT,
        MECH_POWER,
    ],
}


# ---------------------------------------------------------------------------
# Preset cfg class.
# ---------------------------------------------------------------------------


@configclass
class MultiTaskTasksPresetCfg(PresetCfg):
    """Named task-set presets for :class:`MultiTaskCfg.tasks`.

    Each class-level attribute is a preset alternative; ``default`` is the
    fallback chosen when no CLI preset name matches. The preset resolver
    (:func:`isaaclab_tasks.utils.hydra.resolve_presets`) walks the env cfg,
    finds this object, and replaces it with the selected dict.

    Presets:

    - :attr:`simple_pos_vel` — one instant (body pos) + one tracking (lin vel).
      Minimal starter for verifying the pipeline end-to-end.
    - :attr:`pose_vel` — two instant (body pos + body quat) + two tracking
      (lin vel + ang vel). Exercises the composer's AND-gate on instant side.
    - :attr:`locomotion` — the full 8-task production suite (velocity,
      position, reach_point_in_air, pose, two_feet_stand, tripod_walk,
      run, trot).
    - :attr:`default` — aliased to :attr:`simple_pos_vel` so a fresh cfg
      trains on the simplest task before you opt into richer ones.

    Select via CLI::

        ./isaaclab.sh -p train.py ... presets=locomotion

    or in code::

        cfg = MultiTaskEnvCfg()
        cfg.commands.goal_point.tasks = MultiTaskTasksPresetCfg.locomotion
    """

    simple_pos_vel = _SIMPLE_POS_VEL
    pose_vel = _POSE_VEL
    locomotion = _LOCOMOTION
    velocity = _VELOCITY_ONLY
    position = _POSITION_ONLY

    # ``default`` selects the starter preset — flip to ``_LOCOMOTION`` once
    # the richer suite is the training target.
    default = _SIMPLE_POS_VEL
