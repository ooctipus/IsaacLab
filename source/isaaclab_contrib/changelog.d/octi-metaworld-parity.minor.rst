Added
^^^^^

* Added Meta-World V2 reward parity tooling: per-task fixture dump
  (``utils/parity/mw_dump.py``), system-level comparator
  (``scripts/reinforcement_learning/rsl_rl/parity_compare.py``),
  formula-level reward audit
  (``scripts/reinforcement_learning/rsl_rl/parity_reward_audit.py``),
  punch-list aggregator (``utils/parity/aggregate.py``), MW per-task
  sampling-range bake-out (``utils/parity/bake_ranges.py``).
* Added ``MetaworldMT3SawyerEnvCfg`` (cube manipulandum) plus
  ``MetaworldMT5SawyerEnvCfg`` and ``MetaworldMT10SawyerEnvCfg`` (subset
  curricula sharing the heterogeneous-asset scene).
* Added ``MetaworldMT25SawyerEnvCfg`` (MT15 + 10 articulated tasks) and
  ``MetaworldMT50SawyerEnvCfg`` (MT25 + 25 cube-as-manipulandum tasks
  including obstacle / kinematic-destination variants).
* Added ``@scatterable`` group-scoped reward atoms in
  ``metaworld/mdp/scatter_rewards.py`` (``linear_combo_term``,
  ``caging_times_in_place_term``, ``hamacher_term``, ``tolerance_term``,
  ``success_indicator_term``, ``reach_success_term``). All five
  multi-task envs (MT3 / MT5 / MT10 / MT25 / MT50) now route per-task
  rewards through these atoms; MT3 gained a minimal ``clone_cfg`` to
  partition envs by task name (no asset-cloning difference, just env
  partition).
* Added Meta-World V2 done-on-success termination to every multi-task
  env. New bool-typed scatter atoms ``keypoint_success_termination`` and
  ``reach_success_termination`` (in ``metaworld/mdp/scatter_rewards.py``)
  drive per-task ``TerminationTermCfg``s on MT3 / MT5 / MT10 / MT15 /
  MT25 / MT50; each task's termination reads its own clone-group
  ``env_ids`` so only that task's envs can fire it.

Deprecated
^^^^^^^^^^

* Deprecated :func:`~isaaclab_contrib.tasks.manipulation.metaworld.mdp.task_masked_reward`
  in favor of the ``@scatterable`` atoms in
  :mod:`~isaaclab_contrib.tasks.manipulation.metaworld.mdp.scatter_rewards`
  with ``asset_cfg=SceneEntityCfg(<keypoint>, groups=[task_name])``.
  Migrate by replacing each ``RewardTermCfg(func=task_masked_reward,
  params={"inner_func": F, "inner_kwargs": K, "task_index": i})`` with
  ``RewardTermCfg(func=<scatter atom matching F>, params={...K, "asset_cfg":
  SceneEntityCfg(<keypoint>, groups=[task_name])})``.
* Added MW per-task ``obj_init_pos`` / ``target_pos`` sampling ranges to
  :class:`~isaaclab_contrib.tasks.manipulation.metaworld.metaworld_specs.MetaworldTaskSpec`,
  populated at module-import time from ``mw_ranges_baked.py``.

Changed
^^^^^^^

* **Breaking:** :class:`~isaaclab_contrib.tasks.manipulation.metaworld.metaworld_env_base.MetaworldEnvCfg`
  now multiplies every non-zero reward weight by ``1 / step_dt`` in
  ``__post_init__`` so per-step reward magnitudes match Meta-World's V2
  convention. IsaacLab's ``RewardManager`` interprets weights as
  per-second; without the compensation our policy saw a 100 × weaker
  reward signal than MW.
* Bumped DiffIK ``k_val`` from 25 → 100 in
  :class:`~isaaclab_contrib.tasks.manipulation.metaworld.metaworld_env_base.MetaworldActionsCfg`.
  At ``k_val = 25`` Sawyer descent plateaued at TCP z ≈ 0.06 — the cube
  (z = 0.02) couldn't be engaged. ``k_val = 100`` lets the IK fully
  track the commanded 1 cm/step delta and a deterministic-action push
  probe moved the cube 315 mm vs 0 mm prior.
* Faucet-close ``joint_reset_value`` set to 0 (was ``π/3``); MW resets
  the knob at 0 and encodes the close direction in the goal.

Fixed
^^^^^

* Fixed ``task_indexed_joint_reset`` failing under Hydra serialisation by
  switching ``joint_value_by_task`` from ``dict[int, float]`` to
  ``dict[str, float]`` and coercing keys back to ``int`` on read.
* Fixed Meta-World multi-task envs (MT3 / MT5 / MT10 / MT15 / MT25 / MT50)
  silently misaligning env → clone-group assignment with the round-robin
  ``task_id`` written by :class:`MetaworldMultiTaskCommand`. The
  :class:`~isaaclab.scene.CloneCfg` default ``clone_strategy=random``
  shuffled env IDs across groups, so most envs had the wrong asset for
  their assigned task and the per-task scatter rewards were routed to
  the wrong envs. Set ``clone_strategy=interleaved`` (from
  :mod:`isaaclab.cloner.cloner_strategies`) on every ``CloneCfg`` so
  env ``i`` lands in clone group ``i % n_tasks`` — matching the
  ``MetaworldMultiTaskCommand`` ``task_id`` assignment exactly.
