# Meta-World port — implementation roadmap

This document is the canonical state of the Sawyer Meta-World port. No
stubs, no `-Real-v0` suffix: every registered env uses real assets. If
a task isn't here, it hasn't been ported yet — the recipe is at the
bottom of this file.

## Current state — 18 task envs + 3 multi-task envs

```
Isaac-Metaworld-Reach-Sawyer-v0              # MT3 cube (cube IS the manipulandum)
Isaac-Metaworld-Push-Sawyer-v0               # MT3 cube
Isaac-Metaworld-Pick-Place-Sawyer-v0         # MT3 cube
Isaac-Metaworld-Drawer-Open-Sawyer-v0        # mw_drawer.usda
Isaac-Metaworld-Drawer-Close-Sawyer-v0       # mw_drawer.usda
Isaac-Metaworld-Button-Press-Topdown-Sawyer-v0   # mw_button.usda
Isaac-Metaworld-Coffee-Button-Sawyer-v0      # mw_button.usda (variant — different press depth)
Isaac-Metaworld-Window-Open-Sawyer-v0        # mw_window.usda
Isaac-Metaworld-Window-Close-Sawyer-v0       # mw_window.usda
Isaac-Metaworld-Faucet-Open-Sawyer-v0        # mw_faucet.usda
Isaac-Metaworld-Faucet-Close-Sawyer-v0       # mw_faucet.usda
Isaac-Metaworld-Dial-Turn-Sawyer-v0          # mw_faucet.usda (variant)
Isaac-Metaworld-Lever-Pull-Sawyer-v0         # mw_faucet.usda (variant)
Isaac-Metaworld-Door-Open-Sawyer-v0          # mw_door.usda
Isaac-Metaworld-Door-Close-Sawyer-v0         # mw_door.usda
Isaac-Metaworld-Door-Lock-Sawyer-v0          # mw_door.usda (variant)
Isaac-Metaworld-Door-Unlock-Sawyer-v0        # mw_door.usda (variant)
Isaac-Metaworld-Peg-Insert-Side-Sawyer-v0    # mw_peg_block.usda + free peg cylinder

Isaac-Metaworld-MT3-Sawyer-MultiTask-v0
Isaac-Metaworld-MT5-Sawyer-MultiTask-v0
Isaac-Metaworld-MT10-Sawyer-MultiTask-v0
```

The MT3 envs use the cube directly (cube *is* the puck/block in MW too —
no asset substitution involved). All 15 non-MT3 envs use one of 5
hand-authored MW-equivalent USDs, verified at 0 mm joint→pose error in
`scripts/reinforcement_learning/rsl_rl/verify_mw_assets.py`.

The 3 multi-task envs run MT3 + the MT10 reward shapes against a shared
cube scene with task-onehot conditioning. They're research envs for
shared-policy RL experiments, not intended as byte-equivalent ports of
each MT10 task.

## Directory layout

```
source/isaaclab_contrib/isaaclab_contrib/tasks/manipulation/metaworld/
├── metaworld_env_cfg.py                 # MT3 base (Scene/Action/Obs/Event/Term + 3 task subclasses)
├── sawyer.py                            # SAWYER_METAWORLD_CFG
│
├── assets/                              # Hand-authored MW-equivalent USDs
│   ├── _usd_helpers.py                  # shared USD authoring helpers
│   ├── build_all_mw_assets.py           # builder script (one-shot)
│   ├── sawyer/   sawyer_with_gripper.usda
│   ├── drawer/   __init__.py + usd/mw_drawer.usda
│   ├── button/   __init__.py + usd/mw_button.usda
│   ├── window/   __init__.py + usd/mw_window.usda
│   ├── faucet/   __init__.py + usd/mw_faucet.usda
│   ├── door/     __init__.py + usd/mw_door.usda
│   ├── peg/      __init__.py + usd/mw_peg_block.usda
│   └── reward_parity/                   # MW reward parity-test fixtures
│
├── mdp/                                 # MDP terms (lazy_export)
│   ├── __init__.pyi                     # __all__ + re-exports
│   ├── actions.py / arm_action_impl.py  # Meta-World 4-d action
│   ├── commands.py                      # MetaworldPairedCommand
│   ├── multitask_command.py             # MetaworldMultiTaskCommand
│   ├── multitask_obs_rewards.py         # task_onehot, task_masked_reward
│   ├── observations.py                  # MetaworldObservation (39-d)
│   ├── quantities.py                    # atoms — read keypoint_frame_cfg
│   ├── reward_shapes.py                 # tolerance_shape, linear_combo_shape, caging_times_in_place_shape
│   ├── rewards.py                       # 7 V2 reward functions + keypoint_at_target
│   └── utils.py                         # tolerance() + hamacher_product()
│
└── config/sawyer/                       # gym registrations + per-task env cfgs
    ├── __init__.py                      # gym.register loops (clean: 18 + 3)
    ├── _scene.py                        # SawyerSceneCfg (cube + tcp_frame + keypoint_frame@cube)
    ├── _drawer_scene.py                 # SawyerDrawerSceneCfg (drawer asset + keypoint_frame@handle)
    ├── reach_env_cfg.py / push_env_cfg.py / pick_place_env_cfg.py    # MT3 leaves
    ├── env_cfgs.py                      # 15 real-asset env cfgs
    ├── multi_task_env_cfg.py            # MT3/MT5/MT10 heterogeneous
    └── agents/rsl_rl_ppo_cfg.py
```

## Architectural pattern

Every env's reward reads the manipulandum through a single
`FrameTransformer` named `keypoint_frame`:

* **MT3** (`_scene.py`): `keypoint_frame.target_prim = .../Cube`.
* **Real-asset** (`_drawer_scene.py` + `env_cfgs.py` `_build_scene_class`):
  `keypoint_frame.target_prim = .../Cabinet/<marker_body>`.

All `mdp/quantities.py` atoms (`obj_to_target_dist`, `tcp_to_obj_dist`,
`obj_z_above_init`, `gripper_caging`, `pick_place_caging`) and all 7
`mdp/rewards.py` V2 reward functions take `keypoint_frame_cfg` — so
the same reward code runs on every task and the only thing that varies
is **which body the scene declares as the keypoint**.

To audit "what is the reward measuring on this env" → grep `keypoint_frame`
in the scene cfg and look at the marker prim path. One source of truth.

## Verification

Two harnesses cover the asset/keypoint correctness — the parts you said
are easiest to mess up:

* `scripts/reinforcement_learning/rsl_rl/verify_mw_assets.py` —
  asset-only joint→pose sweep. Loads each asset USD, writes joints to
  known values, compares the `keypoint_frame`'s reported world position
  against the analytical expectation.
* `scripts/reinforcement_learning/rsl_rl/verify_real_envs.py` —
  end-to-end env-level. Boots the env, reads `keypoint_frame` after a
  reset (matches `init` command), writes the joint to its goal-state
  value, reads `keypoint_frame` again (matches `goal` command). Pass
  threshold is sub-mm.

Current status: all 5 hand-authored assets PASS the asset harness at
0 mm error. Drawer-Open, Drawer-Close, Button-Press-Topdown, Window-Open,
Window-Close, Faucet-Open, Faucet-Close, Door-Open, Door-Close,
Door-Lock, Door-Unlock, Dial-Turn, Lever-Pull, Coffee-Button,
Peg-Insert-Side all PASS the env harness at 0.0–0.3 mm error.

## Recipe to add a new task

```
1. Author a USD using assets/_usd_helpers.py:
   - kinematic base body welded to world
   - moving body with the actuated joint (prismatic or revolute)
   - zero-extent marker body fixed-jointed to the moving body
   Add a build function to assets/build_all_mw_assets.py.

2. Wrap as ArticulationCfg in assets/<task>/__init__.py:
   MW_<TASK>_CFG = ArticulationCfg(
       spawn=UsdFileCfg(usd_path=...),
       init_state=ArticulationCfg.InitialStateCfg(joint_pos={...}),
       actuators={"<task>": ImplicitActuatorCfg(joint_names_expr=[...], ...)},
   )

3. Declare keypoint_frame in the scene cfg.
   For an asset that fits the existing scene factory, reuse:
       _MyScene = _build_scene_class(MW_<TASK>_CFG, "<base_body>", "<marker_body>")
   For something more custom, write a SceneCfg subclass with
   `keypoint_frame = FrameTransformerCfg(prim_path=..., target_frames=[FrameCfg(prim_path=..., name="kp")])`.

4. Compose the reward in env_cfgs.py using V2 atoms with
   keypoint_frame_cfg=SceneEntityCfg("keypoint_frame"):
       @configclass
       class _MyRewardsCfg:
           my_v2 = RewardTermCfg(
               func=mdp.<task>_v2,                # one of the 7 V2 functions
               weight=1.0,
               params={"keypoint_frame_cfg": _HANDLE_CFG, ...},
           )
           success = RewardTermCfg(func=mdp.keypoint_at_target, weight=0.0,
               params={"keypoint_frame_cfg": _HANDLE_CFG, "threshold": 0.05, ...})

5. Add the env class + register in config/sawyer/__init__.py:
       Isaac-Metaworld-<Task>-Sawyer-v0  →  Metaworld<Task>SawyerEnvCfg

6. Verify with verify_real_envs.py:
       ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/verify_real_envs.py \
           --task Isaac-Metaworld-<Task>-Sawyer-v0 \
           --joint <joint_name> --joint_value <goal_state>
   Expect <1 mm error.
```

No new reward functions per task. The 7 V2 functions in `mdp/rewards.py`
cover MW's task families:

* `tolerance_shape` archetype — used by Reach (single tolerance).
* `linear_combo_shape` archetype — used by Push (caging + phase bonus + override).
* `caging_times_in_place_shape` archetype — used by Pick-Place (caging × in-place).
* `drawer_open_v2` / `drawer_close_v2` — drawer slide reward.
* `button_press_topdown_v2` — vertical button press.
* `window_open_v2` / `window_close_v2` — slider with x-only in-place.
* `door_open_v2` — delegates to window_open math (handle reach + in-place).
* `faucet_open_v2` — delegates to window_open math.
* `peg_insert_side_v2` — caging × scale-emphasized in-place.

## Tasks not yet ported

The 32 MW tasks not in the registered list need:

* Composite assets (hammer + nail, basketball + hoop, mug + saucer, bin
  walls, peg + connector, plate + tabletop track, sweep target, soccer
  goal box, shelf, hand-insert receptacle, peg-unplug socket).
* Variant geometry (button-press horizontal, button-press-wall,
  pick-place-wall, push-back, push-wall, reach-wall, stick-pull,
  stick-push, plate-slide×4, sweep, sweep-into, pick-out-of-hole).
* Multi-step composites (assembly, disassemble).

For each: follow the 6-step recipe above. The asset authoring helpers
in `assets/_usd_helpers.py` cover the common cases (boxes, cylinders,
prismatic + revolute joints, fixed welds).
