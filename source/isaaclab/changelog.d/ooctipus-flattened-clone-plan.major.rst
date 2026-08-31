Changed
^^^^^^^

* **Breaking:** Made ``ReplicateSession`` the single cloning lifecycle. Create an active
  ``SimulationContext``, enter one session rooted in the complete cfg, and construct cfg-owned
  objects with ``cfg.class_type(cfg)`` inside it. The session derives its stage and shared assets;
  remove the former ``stage``, ``global_paths``, and ``clone_strategy`` arguments.
* **Breaking:** Flattened ``ClonePlan`` to one row per independently copied asset variant. Shared
  assets are plan rows and nested cfgs map to their nearest parent's rows. Read the active plan with
  ``SimulationContext.get_clone_plan()`` instead of ``ReplicateSession.plan`` or
  ``ClonePlan.global_paths``.

Removed
^^^^^^^

* **Breaking:** Removed the public replication queue and standalone plan/dispatch functions:
  ``REPLICATION_QUEUE``, ``queue_replication()``, ``replicate()``, ``usd_replicate()``,
  ``clone_plan_from_env_0()``, ``make_clone_plan()``, ``make_valid_clone_combinations()``,
  ``num_spawn_variants()``, ``random()``, and ``sequential()``. Use ``ReplicateSession``; low-level
  clone-context callers pass one complete mapping directly to ``context.replicate(...)``.
  ``disabled_fabric_change_notifies()`` is now internal.
* **Breaking:** Removed per-asset and per-sensor ``cloning_contexts``;
  ``CloneCfg.clone_strategy``, ``CloneCfg.device``, and ``CloneCfg.replicate_physics``;
  ``InteractiveSceneCfg.clone_in_fabric``; and spawner ``spawn_path``, ``spawn_paths``, and
  ``random_choice``. Pass device and physics policy to ``ReplicateSession``, declare variant
  combinations with ``CloneCfg.clone_combinations`` and ``InclusionSet.weight``, and pass exact
  path sequences directly to multi-asset spawner functions.
