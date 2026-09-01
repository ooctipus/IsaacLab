Changed
^^^^^^^

* **Breaking:** Consolidated cfg-derived cloning into one lifecycle. Construct a declarative or
  heterogeneous scene with ``scene_cfg.class_type(scene_cfg)``; ``InteractiveScene`` owns its
  lifecycle. Homogeneous direct environment bases instead own a ``from_env_0`` lifecycle around a
  plain scene and ``_setup_scene()``. Keep every authored asset cfg on the direct env cfg, construct
  its prototype with ``cfg.class_type(cfg)``, and register runtime entities on the scene. Standalone
  homogeneous workflows may use ``from_env_0`` explicitly; ``ReplicateSession`` remains the
  lower-level lifecycle for general cloning tools. The lifecycle obtains stage and device from the
  active ``SimulationContext``, dispatches each required stage/native context once on exit, and
  queues model-role backends for the first hard reset after any intervening stage edits.
* **Breaking:** Made ``ClonePlan`` exhaustive: shared assets now use plan rows and nested cfgs map
  to their nearest parent's rows. Read the active plan with
  ``SimulationContext.get_clone_plan()`` instead of ``ReplicateSession.plan`` or
  ``ClonePlan.global_paths``. Clone contexts now receive the published plan once and read their
  selected rows from ``ClonePlan.context_rows``. ``ClonePlan.replicate_physics`` carries the
  declared policy into deferred model construction.
* **Breaking:** Moved ``replicate_physics`` and ``filter_collisions`` from
  ``InteractiveSceneCfg`` to its nested ``CloneCfg``. Configure either policy through
  ``scene.clone_cfg``. ``ReplicateSession`` now applies collision filtering after clone dispatch.
* **Breaking:** Added declarative whole-scene ``cloning_contexts`` to renderer and visualizer
  configs. ``IsaacRtxRendererCfg`` and ``KitVisualizerCfg`` request USD cloning explicitly;
  installing Kit alone no longer causes a USD copy.
* **Breaking:** Stopped ``CameraCfg`` from probing the stage or redirecting a body path to a
  ``/camera`` child. Declare the exact camera child ``prim_path`` and set ``spawn=None`` for a
  preauthored camera.
* **Breaking:** Changed ``RayCasterCfg.spawn`` from ``SensorFrameCfg()`` to ``None``. Set
  ``SensorFrameCfg`` explicitly only when the sensor cfg owns a new Xform.

Removed
^^^^^^^

* **Breaking:** Removed the public replication queue and standalone plan/dispatch functions:
  ``REPLICATION_QUEUE``, ``queue_replication()``, ``replicate()``, ``usd_replicate()``,
  ``clone_plan_from_env_0()``, ``make_clone_plan()``, ``make_valid_clone_combinations()``,
  ``num_spawn_variants()``, ``random()``, and ``sequential()``. Use a declarative
  ``InteractiveScene`` or ``from_env_0`` for a homogeneous prototype; low-level clone-context
  callers pass the active plan to ``context.replicate(plan)``.
  ``disabled_fabric_change_notifies()`` is now internal.
* **Breaking:** Removed ``ReplicateSession.plan`` and its ``device``, ``stage``, ``global_paths``,
  ``clone_strategy``, ``valid_set``, ``replicate_physics``, and ``env_template`` arguments. Read the
  plan from ``SimulationContext`` and configure cloning through the sole nested ``CloneCfg``.
* **Breaking:** Removed ``CloneCfg.clone_strategy`` and ``CloneCfg.device``;
  ``InteractiveSceneCfg.clone_in_fabric``; the public ``grid_transforms()`` and
  ``filter_collisions()`` helpers; and spawner ``spawn_path``, ``spawn_paths``, and
  ``random_choice``. Declare variant combinations with ``CloneCfg.clone_combinations`` and
  ``InclusionSet.weight``, route cfg rows with ``cloning_contexts``, and pass exact path sequences
  directly to multi-asset spawner functions.
