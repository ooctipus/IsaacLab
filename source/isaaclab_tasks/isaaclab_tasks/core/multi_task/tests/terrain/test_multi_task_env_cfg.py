# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static verification that the new :class:`MultiTaskEnvCfg` constructs cleanly.

Covers only the "the cfg loads" layer — no sim required. Catches regressions in:

- Import graph (env cfg module + transitive imports).
- Configclass instantiation + ``__post_init__`` arithmetic.
- Gym registration landing with the expected entry point.
- Every :class:`DoneTerm` declared with ``time_out=False`` (finite-horizon
  framing — a regression on this flag silently breaks value bootstrap semantics).

Live-sim validation (actual Articulation reads, rollouts, rewards) is the job of
``utils/tools/smoke_multi_task_command.py``, which requires Isaac Sim.
"""

from __future__ import annotations

import gymnasium as gym

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env_cfg_with_locomotion_preset():
    """Build :class:`MultiTaskEnvCfg` and pin the full 8-task ``locomotion`` preset.

    The default cfg uses the :class:`MultiTaskTasksPresetCfg.simple_pos_vel`
    starter (one instant + one tracking). Tests that assert the full 8-task
    structure have to opt in explicitly — this helper does that. Tests that
    check the *default* behavior should use :class:`MultiTaskEnvCfg` directly
    (see ``test_default_preset_is_simple_pos_vel`` below).
    """
    from isaaclab_tasks.core.multi_task.multi_task_env_cfg import MultiTaskEnvCfg
    from isaaclab_tasks.core.multi_task.terrain.mdp_presets.multitask_presets import (
        MultiTaskTasksPresetCfg,
    )

    cfg = MultiTaskEnvCfg()
    # ``configclass`` moves preset alternatives to instance fields, so access
    # via an instance (``MultiTaskTasksPresetCfg().locomotion``), not the class.
    cfg.commands.goal_point.tasks = MultiTaskTasksPresetCfg().locomotion
    return cfg


def test_gym_task_registers():
    """``Isaac-Position-MultiTask-v0`` is registered with the expected entry point."""
    import isaaclab_tasks  # noqa: F401 — triggers registration

    assert "Isaac-Position-MultiTask-v0" in gym.envs.registry
    spec = gym.envs.registry["Isaac-Position-MultiTask-v0"]
    assert spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"
    assert spec.kwargs["env_cfg_entry_point"] == "isaaclab_tasks.core.multi_task.multi_task_env_cfg:MultiTaskEnvCfg"


def test_env_cfg_imports_and_constructs():
    """Importing and instantiating :class:`MultiTaskEnvCfg` does not raise."""
    from isaaclab_tasks.core.multi_task.multi_task_env_cfg import MultiTaskEnvCfg

    cfg = MultiTaskEnvCfg()
    # Sanity: key fields landed on the instance after ``__post_init__``.
    assert cfg.decimation > 0
    assert cfg.episode_length_s > 0
    assert cfg.sim.dt > 0
    assert cfg.sim.render_interval == cfg.decimation


def test_timeout_flag_is_per_task_type():
    """Regression gate for the per-task-type timeout semantic.

    Reach / mixed tasks use ``time_out_reach_truncate`` with ``time_out=True``
    (rsl_rl bootstraps ``γ·V(s_T)`` — infinite-horizon goal-reaching). Pure-
    tracking tasks use ``time_out_track_terminate`` with ``time_out=False``
    (finite-horizon interval reward; bootstrap would double-count). Failure
    terminations (``drop``, ``base_contact``) and ``success`` stay ``False``.
    """
    from isaaclab.managers import TerminationTermCfg as DoneTerm

    from isaaclab_tasks.core.multi_task.multi_task_env_cfg import MultiTaskEnvCfg

    cfg = MultiTaskEnvCfg()
    terminations = cfg.terminations
    done_terms = {name: term for name, term in vars(terminations).items() if isinstance(term, DoneTerm)}
    assert done_terms, "Env cfg has no DoneTerms — sanity failure."
    truncating = {name for name, term in done_terms.items() if term.time_out}
    terminating = {name for name, term in done_terms.items() if not term.time_out}
    # Reach/mixed timeout is a truncation — the only DoneTerm allowed to flag True.
    assert truncating == {"time_out_reach_truncate"}, (
        f"Unexpected set of truncating DoneTerms: {truncating}. Only "
        "``time_out_reach_truncate`` may bootstrap; track timeout + success + failures "
        "must all be real terminations (time_out=False)."
    )
    # Track timeout + success + every failure DoneTerm must be a real termination.
    for name in ("time_out_track_terminate", "success", "drop", "base_contact"):
        assert name in terminating, f"Expected DoneTerm ``{name}`` with time_out=False; got {terminating}."


def test_commands_cfg_is_multitask():
    """The ``goal_point`` command resolves to :class:`MultiTaskCfg` by default.

    ``cfg.commands.goal_point`` is a :class:`PresetCfg` after construction
    (so hydra can swap it for :class:`MinimalVelocityCommandCfg` via CLI);
    ``resolve_presets`` picks the default alternative — ``MultiTaskCfg``.
    """
    from isaaclab_tasks.core.multi_task.mdp.commands.multi_task_command.impl.multi_task_cfg import MultiTaskCfg
    from isaaclab_tasks.core.multi_task.multi_task_env_cfg import MultiTaskEnvCfg
    from isaaclab_tasks.utils import resolve_presets

    cfg = MultiTaskEnvCfg()
    resolve_presets(cfg)
    assert isinstance(cfg.commands.goal_point, MultiTaskCfg)


def test_default_preset_is_simple_pos_vel():
    """The default :class:`MultiTaskEnvCfg` uses the ``simple_pos_vel`` preset.

    One instant (body position) + one tracking (linear velocity) — the minimal
    starter configuration for training the multi-task pipeline end-to-end.
    Richer presets are opt-in via CLI ``presets=locomotion`` or direct
    assignment to :attr:`MultiTaskCfg.tasks`.

    Note: ``cfg.commands.goal_point.tasks`` remains a :class:`PresetCfg` after
    ``MultiTaskEnvCfg()`` — it's resolved either by hydra's ``register_task``
    (CLI path) or by :meth:`MultiTaskCommand.__init__` (runtime path). Here we
    call :func:`resolve_presets` explicitly to mirror the hydra resolution.
    """
    from isaaclab_tasks.core.multi_task.multi_task_env_cfg import MultiTaskEnvCfg
    from isaaclab_tasks.utils import resolve_presets

    cfg = MultiTaskEnvCfg()
    resolve_presets(cfg)  # no ``selected`` argument → picks the ``default`` alternative
    task_names = set(cfg.commands.goal_point.tasks.keys())
    assert task_names == {"position", "lin_vel"}, (
        f"default preset should be simple_pos_vel (position + lin_vel); got {task_names}"
    )


def test_locomotion_preset_has_all_eight_tasks():
    """Opting into the ``locomotion`` preset exposes the full 8-task suite."""
    cfg = _env_cfg_with_locomotion_preset()
    task_names = set(cfg.commands.goal_point.tasks.keys())
    expected = {
        "velocity",
        "position",
        "reach_point_in_air",
        "pose",
        "two_feet_stand",
        "tripod_walk",
        "run",
        "trot",
    }
    assert task_names == expected, f"task key mismatch: got {task_names}, expected {expected}"


def test_tripod_walk_structure():
    """``tripod_walk`` is base-pos + a single permutation-invariant feet-count subtask.

    The foot constraint is ``BODY_CONTACT_COUNT = 3`` over the 4 feet — which foot
    is airborne is left to the policy. This matches the Anymal walking gait where
    at any instant three feet are planted and one is transitioning.

    The full subtask list also includes the shared safety subtasks
    (``DEFAULT_SAFETY_SUBTASKS``) appended at preset-build time. We validate the
    task-specific (instant) prefix here and let
    :func:`test_default_safety_subtasks_appended_to_every_task` cover the safety
    suffix.
    """
    from isaaclab_tasks.core.multi_task.mdp.commands.multi_task_command.impl.kernels_torch import STATE_KERNEL_ID
    from isaaclab_tasks.core.multi_task.mdp.commands.multi_task_command.impl.multi_task_cfg import MultiTaskCfg

    cfg = _env_cfg_with_locomotion_preset()
    subtasks = cfg.commands.goal_point.tasks["tripod_walk"]
    instant_subtasks = [st for st in subtasks if isinstance(st, MultiTaskCfg.InstantaneousTaskCfg)]
    assert len(instant_subtasks) == 2

    # Second instant subtask is the feet-count aggregator targeting 3-on-ground.
    feet_sub = instant_subtasks[1]
    assert feet_sub.state_kernel == int(STATE_KERNEL_ID.BODY_CONTACT_COUNT)
    assert feet_sub.sampler.minimum == [3.0] and feet_sub.sampler.maximum == [3.0]
    # Must reference all four feet so the count ranges over the full set.
    feet_names = set(feet_sub.asset_cfg.body_names) if isinstance(feet_sub.asset_cfg.body_names, list) else set()
    assert feet_names == {"LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"}


def test_gait_tasks_use_count_diff_tracking_with_mirrored_splits():
    """``run`` and ``trot`` each wire gait as a single tracking ``BODY_CONTACT_COUNT_DIFF`` subtask.

    Tracking (not instant) so the composer time-averages activation across
    transit — the gait must hold while the body moves, not just fire once.
    The kernel splits ``body_names`` in half; ``GREATER`` at threshold 1.5
    fires on either diagonal / pair sign.

    - ``run``:  first half = front pair ``(LF, RF)``; second half = hind pair ``(LH, RH)``
    - ``trot``: first half = one diagonal ``(LF, RH)``; second half = other diagonal ``(RF, LH)``
    """
    from isaaclab_tasks.core.multi_task.mdp.commands.multi_task_command.impl.kernels_torch import (
        ACTIVATION_KERNEL_ID,
        STATE_KERNEL_ID,
    )
    from isaaclab_tasks.core.multi_task.mdp.commands.multi_task_command.impl.multi_task_cfg import MultiTaskCfg

    cfg = _env_cfg_with_locomotion_preset()

    expected_splits = {
        "run": (["LF_FOOT", "RF_FOOT"], ["LH_FOOT", "RH_FOOT"]),
        "trot": (["LF_FOOT", "RH_FOOT"], ["RF_FOOT", "LH_FOOT"]),
    }
    for task_name, (first_half, second_half) in expected_splits.items():
        subtasks = cfg.commands.goal_point.tasks[task_name]
        diff_subs = [st for st in subtasks if st.state_kernel == int(STATE_KERNEL_ID.BODY_CONTACT_COUNT_DIFF)]
        assert len(diff_subs) == 1, f"{task_name}: expected exactly one count-diff subtask"
        sub = diff_subs[0]
        assert isinstance(sub, MultiTaskCfg.TrackingTaskCfg), (
            f"{task_name}: gait subtask must be TrackingTaskCfg — gait is maintained "
            "during transit, not an instant latch"
        )
        assert sub.asset_cfg.body_names == first_half + second_half, (
            f"{task_name}: split ordering must place {first_half} first, {second_half} second"
        )
        assert sub.activation_kernel == int(ACTIVATION_KERNEL_ID.GREATER)
        assert abs(sub.activation_kernel_param - 1.5) < 1e-9


def test_task_type_composition_is_correct():
    """Each task's task-specific subtask prefix uses the expected instant/tracking types.

    Every preset also appends :data:`DEFAULT_SAFETY_SUBTASKS` (covered by
    :func:`test_default_safety_subtasks_appended_to_every_task`); we filter
    those out here so the assertions stay focused on the task definition.
    """
    from isaaclab_tasks.core.multi_task.mdp.commands.multi_task_command.impl.multi_task_cfg import MultiTaskCfg

    cfg = _env_cfg_with_locomotion_preset()
    tasks = cfg.commands.goal_point.tasks

    def _non_safety(subs):
        # Soft-safety subtasks are TrackingTaskCfg with expose_in_obs=False;
        # filter them out so the assertions stay focused on the task definition.
        return [
            st
            for st in subs
            if not (isinstance(st, MultiTaskCfg.TrackingTaskCfg) and not getattr(st, "expose_in_obs", True))
        ]

    # velocity = two tracking subtasks (lin_vel, ang_vel).
    vel = _non_safety(tasks["velocity"])
    assert len(vel) == 2
    assert all(isinstance(st, MultiTaskCfg.TrackingTaskCfg) for st in vel)

    # position / reach_point_in_air = one instant body-pos subtask each.
    for name in ("position", "reach_point_in_air"):
        subs = _non_safety(tasks[name])
        assert len(subs) == 1
        assert isinstance(subs[0], MultiTaskCfg.InstantaneousTaskCfg)

    # pose = body-pos + body-quat, both instant.
    pose = _non_safety(tasks["pose"])
    assert len(pose) == 2
    assert all(isinstance(st, MultiTaskCfg.InstantaneousTaskCfg) for st in pose)

    # two_feet_stand = one body-quat instant subtask.
    stand = _non_safety(tasks["two_feet_stand"])
    assert len(stand) == 1
    assert isinstance(stand[0], MultiTaskCfg.InstantaneousTaskCfg)


def test_default_safety_subtasks_appended_to_every_task():
    """Every task in the production preset carries the shared safety set.

    Soft-safety subtasks are :class:`MultiTaskCfg.TrackingTaskCfg` instances
    with ``expose_in_obs=False`` — internal-only quality dimensions that
    discount ``G`` but never reach the policy obs. We verify presence + count
    + kernel-id + target here; their actual scales come from the templates
    in :mod:`isaaclab_tasks.core.multi_task.terrain.tasks_cfg`.
    """
    from isaaclab_tasks.core.multi_task.mdp.commands.multi_task_command.impl.kernels_torch import STATE_KERNEL_ID
    from isaaclab_tasks.core.multi_task.mdp.commands.multi_task_command.impl.multi_task_cfg import MultiTaskCfg

    cfg = _env_cfg_with_locomotion_preset()
    for task_name, subtasks in cfg.commands.goal_point.tasks.items():
        safety = [
            st
            for st in subtasks
            if isinstance(st, MultiTaskCfg.TrackingTaskCfg) and not getattr(st, "expose_in_obs", True)
        ]
        assert len(safety) == 2, (
            f"{task_name}: expected exactly 2 safety subtasks (undesired-contact + mech-power), got {len(safety)}"
        )
        kernels = {st.state_kernel for st in safety}
        assert kernels == {
            int(STATE_KERNEL_ID.BODY_CONTACT_COUNT),
            int(STATE_KERNEL_ID.JOINT_MECH_POWER),
        }, f"{task_name}: safety kernels {kernels} mismatch"
        # Targets must be 0 (no-violation point) for both safety dims.
        for st in safety:
            assert st.sampler.minimum == [0.0] and st.sampler.maximum == [0.0]


def test_reach_point_in_air_z_target_is_elevated():
    """Regression guard: the air-reach sampler must have z > standing height.

    If someone copy-pastes the ground-level position sampler into this task, this
    test fails loudly.
    """
    cfg = _env_cfg_with_locomotion_preset()
    air = cfg.commands.goal_point.tasks["reach_point_in_air"][0]
    # sampler min/max for z is the 3rd component.
    assert air.sampler.minimum[2] >= 1.0
    assert air.sampler.maximum[2] >= air.sampler.minimum[2]


def test_two_feet_stand_targets_large_pitch():
    """The two-feet-stand quat sampler must target near-vertical pitch."""
    cfg = _env_cfg_with_locomotion_preset()
    stand = cfg.commands.goal_point.tasks["two_feet_stand"][0]
    # Sampler's minimum/maximum encode (roll, pitch, yaw) Euler mins/maxs.
    pitch_min, pitch_max = stand.sampler.minimum[1], stand.sampler.maximum[1]
    assert min(abs(pitch_min), abs(pitch_max)) > 1.0, (
        f"two_feet_stand pitch target should be ≈ ±π/2, got [{pitch_min}, {pitch_max}]"
    )


def test_pose_quat_sampler_has_out_dim_override():
    """The quat sampler in the pose task must declare ``out_dim=4``.

    Without that override, ``target_dim_max`` derived from the 3-Euler param count
    would be 3, which can't hold a 4-vec quaternion — the reference dispatch would
    raise at ``_compute_state_delta_error_reference`` when dim_x_cur=4.
    """
    cfg = _env_cfg_with_locomotion_preset()
    quat_subtask = cfg.commands.goal_point.tasks["pose"][1]
    assert quat_subtask.sampler.out_dim == 4


def test_pose_quat_sampler_param_length_matches_four_pairs():
    """``get_kernel_input`` must emit 8 floats (4 interleaved min/range pairs) for pose quat.

    Guards the tight coupling between ``out_dim`` and the padded param tensor length —
    regression here would quietly produce a ``target_dim_max < 4`` and crash at runtime.
    """
    cfg = _env_cfg_with_locomotion_preset()
    quat_subtask = cfg.commands.goal_point.tasks["pose"][1]
    params = quat_subtask.sampler.get_kernel_input(device="cpu")
    assert params.numel() == 8  # 4 (min, range) pairs


def test_canonical_widths_are_split_by_reach_and_track():
    """Regression gate: canonical obs splits into reach + track tensors.

    Reach (instant) subtasks contribute to ``reach_canonical_width``:
      base POS (3) + base QUAT (4) + 4-feet-composite CONTACT_COUNT (1) = 8

    Track (tracking) subtasks contribute to ``track_canonical_width``:
      base LIN_VEL (3) + base ANG_VEL (3)
      + run-entity CONTACT_COUNT_DIFF (1) + trot-entity CONTACT_COUNT_DIFF (1)
      = 8

    POS and POS_Z would get disjoint channels if both used (no z-slot aliasing),
    though the current cfg doesn't exercise that case on the same entity.
    """
    # Minimal scene stub sufficient for SceneEntityCfg.resolve — mirrors the mock env.
    import re as _re

    from isaaclab_tasks.core.multi_task.mdp.commands.multi_task_command.spec import build_spec

    class _StubArticulation:
        def __init__(self, body_names: list[str]):
            self.body_names = body_names
            self.joint_names: list[str] = []
            self.num_bodies = len(body_names)
            self.num_joints = 0
            self.fixed_tendon_names: list[str] = []
            self.num_fixed_tendons = 0

        def _find(self, names, patterns, preserve_order=False):
            if isinstance(patterns, str):
                patterns = [patterns]
            ids, matched = [], []
            for pat in patterns:
                rx = _re.compile(pat)
                for i, n in enumerate(names):
                    if rx.fullmatch(n) and i not in ids:
                        ids.append(i)
                        matched.append(n)
            return ids, matched

        def find_bodies(self, patterns, preserve_order=False):
            return self._find(self.body_names, patterns, preserve_order)

        def find_joints(self, patterns, preserve_order=False):
            return [], []

        def find_fixed_tendons(self, patterns, preserve_order=False):
            return [], []

    class _StubScene:
        """Routes ``scene["robot"]`` (articulation) and ``scene["contact_forces"]``
        (contact sensor) to stubs sharing the same body namespace — enough for
        :meth:`SceneEntityCfg.resolve` to run against either entity."""

        def __init__(self, bodies: list[str]):
            self._art = _StubArticulation(bodies)
            self._sensor = _StubArticulation(bodies)

        def __getitem__(self, key):
            return {"robot": self._art, "contact_forces": self._sensor}[key]

        def __contains__(self, key):
            return key in ("robot", "contact_forces")

        def keys(self):
            return ("robot", "contact_forces")

    scene = _StubScene(["base", "LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"])

    env_cfg = _env_cfg_with_locomotion_preset()
    spec = build_spec(env_cfg.commands.goal_point, scene, device="cpu")
    assert spec.reach_canonical_width == 8, (
        f"expected reach_canonical_width = 3 (base POS) + 4 (base QUAT) + 1 (feet count) = 8, "
        f"got {spec.reach_canonical_width}"
    )
    assert spec.track_canonical_width == 8, (
        f"expected track_canonical_width = 3 (base LIN_VEL) + 3 (base ANG_VEL) "
        f"+ 1 (run count_diff) + 1 (trot count_diff) = 8, "
        f"got {spec.track_canonical_width}"
    )


def test_reward_term_points_at_task_reward():
    """The task reward term resolves to ``mdp.command_task_reward``."""
    from isaaclab_tasks.core.multi_task.multi_task_env_cfg import MultiTaskEnvCfg
    from isaaclab_tasks.core.multi_task.terrain import mdp

    cfg = MultiTaskEnvCfg()
    assert cfg.rewards.task.func is mdp.command_task_reward
    assert cfg.rewards.task.params == {"command_name": "goal_point"}


def test_success_done_term_points_at_task_done():
    """The success termination resolves to ``mdp.command_task_done``."""
    from isaaclab_tasks.core.multi_task.multi_task_env_cfg import MultiTaskEnvCfg
    from isaaclab_tasks.core.multi_task.terrain import mdp

    cfg = MultiTaskEnvCfg()
    assert cfg.terminations.success.func is mdp.command_task_done
    assert cfg.terminations.success.params == {"command_name": "goal_point"}
