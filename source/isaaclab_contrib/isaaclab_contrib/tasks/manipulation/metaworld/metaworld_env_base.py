# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base ManagerBasedRLEnvCfg shared across every Meta-World+ task.

Inspired by ``factory_v1/factory_env_base.py`` — base env only. Per-task
``commands`` / ``rewards`` are wired in the leaf cfgs (the 3 MT3 envs in
``config/sawyer/{reach,push,pick_place}_env_cfg.py`` and the 15 real-asset
envs in ``config/sawyer/env_cfgs.py``).

Pieces shared by every task:

* :class:`MetaworldSceneCfg` — robot + tcp_frame + lighting + ground +
  goal_marker. The manipulandum (cube for MT3, hidden anchor for
  real-asset, peg cylinder for peg-insert) is supplied by per-family
  scene subclasses in :mod:`metaworld_scenes_cfg`.
* :class:`MetaworldActionsCfg` — 4-d action (DiffIK xyz delta + gripper).
* :class:`MetaworldObservationsCfg` — 39-d Meta-World+ V2 state.
* :class:`MetaworldEventCfg` — Sawyer joint reset on episode start.
* :class:`MetaworldTerminationsCfg` — truncate-only (paper App. A.4).
* :class:`MetaworldEnvCfg` — top-level, ``commands`` / ``rewards`` set per task.
"""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from . import mdp

# ── Scene ───────────────────────────────────────────────────────────────────


@configclass
class MetaworldSceneCfg(InteractiveSceneCfg):
    """Globals shared by every MW scene: robot anchor, lighting, goal marker.

    The ``robot`` and ``tcp_frame`` are :data:`MISSING` here — set by the
    per-family scene subclasses in :mod:`metaworld_scenes_cfg`. The
    manipulandum (``cube``, ``cabinet``, etc.) is also set there because the
    cube cfg differs by task family (visible 4 cm cube for MT3, 1 mm hidden
    anchor for real-asset, 6 cm cylinder for peg-insert).
    """

    robot: ArticulationCfg = MISSING
    tcp_frame: FrameTransformerCfg = MISSING

    goal_marker: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/GoalMarker",
        spawn=sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.0)),
            collision_props=None,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.8, 0.2)),
    )

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


# ── Actions ─────────────────────────────────────────────────────────────────


@configclass
class MetaworldActionsCfg:
    """4-d action: 3-d world-frame XYZ delta (with workspace clamp) + 1-d gripper scalar.

    The arm action mirrors Meta-World's ``set_xyz_action``:
        ``new_target = clip(current_ee_world + 0.01 * action, mocap_low, mocap_high)``
    """

    arm_xyz_delta = mdp.MetaworldArmActionCfg(
        asset_name="robot",
        joint_names=["right_j[0-6]"],
        body_name="hand",
        scale=0.01,  # Meta-World action_scale = 0.01 m/step
        controller=DifferentialIKControllerCfg(
            command_type="position",
            use_relative_mode=True,
            # ``pinv`` instead of ``dls``: pinv exposes a direct gain
            # ``k_val`` (default 1.0) that scales the joint-space delta. With
            # ``dls`` (lambda=0.01 default) the IK achieved only ~5% of the
            # commanded Δ per step. ``k_val=25`` matches MW's mocap-weld
            # tracking rate (~4.5 mm/step) so PPO can hit the mm-level
            # pad/cube alignment that ``caging > 0.97`` requires.
            ik_method="pinv",
            # Bumped 25 → 100 after the push-dynamics probe showed TCP plateau
            # at z≈0.06 with k_val=25. Cube is at z=0.02, so push couldn't
            # engage; higher gain lets the IK fully track the commanded
            # ``set_xyz_action`` 1 cm/step delta.
            ik_params={"k_val": 100.0},
        ),
        # Per-task ``hand_low``/``hand_high`` (override the class-level mocap
        # defaults). Identical for all three MT3 tasks.
        workspace_low=(-0.5, 0.40, 0.05),
        workspace_high=(0.5, 1.0, 0.5),
    )
    gripper = mdp.MetaworldGripperActionCfg(asset_name="robot")


# ── Observations ────────────────────────────────────────────────────────────


@configclass
class MetaworldObservationsCfg:
    """Single 39-d policy observation; layout matches Meta-World+ App. A.4."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        state = ObservationTermCfg(
            func=mdp.MetaworldObservation,
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


# ── Events ──────────────────────────────────────────────────────────────────


@configclass
class MetaworldEventCfg:
    """Reset events. Cube and goal are placed by :class:`mdp.MetaworldPairedCommand`,
    not here."""

    reset_robot_joints = EventTermCfg(
        func="isaaclab.envs.mdp:reset_joints_by_offset",
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
        },
    )


# ── Terminations ────────────────────────────────────────────────────────────


@configclass
class MetaworldTerminationsCfg:
    """Truncate-only — no early termination (paper App. A.4)."""

    time_out = TerminationTermCfg(func="isaaclab.envs.mdp:time_out", time_out=True)


# ── Top-level env ───────────────────────────────────────────────────────────


@configclass
class MetaworldEnvCfg(ManagerBasedRLEnvCfg):
    """Base env shared across every task. Per-task leaf cfgs supply
    ``scene`` (with concrete robot + manipulandum), ``commands``, and
    ``rewards``."""

    scene: MetaworldSceneCfg = MetaworldSceneCfg(num_envs=4096, env_spacing=2.5)
    actions: MetaworldActionsCfg = MetaworldActionsCfg()
    observations: MetaworldObservationsCfg = MetaworldObservationsCfg()
    events: MetaworldEventCfg = MetaworldEventCfg()
    terminations: MetaworldTerminationsCfg = MetaworldTerminationsCfg()

    # Per-task, set in the leaf cfg.
    commands: object = MISSING
    rewards: object = MISSING

    def __post_init__(self) -> None:
        super().__post_init__()
        # Match Meta-World's control rate: MuJoCo runs at dt=0.002 with
        # frame_skip=5 → 10 ms per env step (100 Hz). We use dt=1/200 with
        # decimation=2 to keep the physics substep at 5 ms (PhysX likes a
        # finer substep than the 10 ms env-step rate, and stiff implicit
        # drives are more stable that way).
        # Episode horizon: paper's max_path_length=500 × 0.01 s = 5 s.
        self.decimation = 2
        self.sim = SimulationCfg(dt=1.0 / 200.0)
        self.episode_length_s = 5.0
        # Zoom-out perspective for ``--video``: orbiting view from the front-
        # right that frames the Sawyer base, the cube starting region
        # (y≈0.6) and the goal region (y≈0.85).
        self.viewer.eye = (2.4, -1.4, 1.6)
        self.viewer.lookat = (0.0, 0.65, 0.15)

        # IsaacLab's RewardManager multiplies every term by ``step_dt`` (here
        # 0.01 s) so weights are interpreted as reward-per-second. Meta-World
        # rewards are reward-per-step. Compensate by scaling each non-zero
        # reward weight by ``1 / step_dt`` so the per-step reward we hand to
        # the policy matches MW's value. Done as a post-construction pass so
        # it covers every per-task ``RewardTermCfg`` regardless of where it
        # was declared.
        from isaaclab.managers import RewardTermCfg as _RT

        step_dt = self.sim.dt * self.decimation
        scale = 1.0 / step_dt  # = 100 for dt=1/200, decimation=2
        for term_name in dir(self.rewards):
            if term_name.startswith("_"):
                continue
            term = getattr(self.rewards, term_name, None)
            if isinstance(term, _RT) and term.weight != 0.0:
                term.weight = term.weight * scale
