Changed
^^^^^^^

* **Breaking:** Consolidated cfg-derived cloning into one lifecycle. Construct a declarative or
  heterogeneous scene with ``scene_cfg.class_type(scene_cfg)``; ``InteractiveScene`` owns its
  lifecycle. A homogeneous direct ``_setup_scene()`` instead calls ``clone_plan_from_env_0()`` before
  cfg-owned constructors and ``replicate()`` afterward; its plain scene is only the runtime registry.
  Keep every authored asset cfg on the direct env cfg, pass those cfgs explicitly, construct each
  prototype with ``cfg.class_type(cfg)``, and register runtime entities on the scene. Standalone homogeneous
  workflows may use the same two-phase sequence explicitly; ``ReplicateSession`` remains the
  lower-level lifecycle for general cloning tools. Planning obtains stage and device from the active
  ``SimulationContext``; replication dispatches each required stage/native context once and queues
  model-role backends for the first hard reset after any intervening stage edits.
* Moved Warp-mask scene reset support into ``InteractiveScene`` so every frontend uses the
  cfg-selected scene class instead of a parallel Warp-only scene implementation.
* **Breaking:** Changed ``clone_plan_from_env_0()`` to accept the explicit ``CloneCfg`` and prim-author
  cfgs, environment count, and spacing and to publish its exhaustive plan before prototype construction. The former source,
  destination, device, positions, and global-path arguments were removed. Call ``replicate(plan)``
  after construction; it now obtains the stage and backend registry from the active
  ``SimulationContext`` and the physics policy from the published plan instead of accepting
  ``stage`` and ``replicate_physics`` arguments.
* **Breaking:** Added ``replicate_physics``, ``filter_collisions``, and ``collision_paths`` to
  ``ClonePlan`` as compiled lifecycle data so ``replicate(plan)`` needs no parallel policy inputs.
  Nested cfgs map to their nearest parent's rows. Read the active plan with
  ``SimulationContext.get_clone_plan()`` instead of ``ReplicateSession.plan``. Manual plan
  construction now also requires ``replicate_physics``.
* **Breaking:** Moved ``replicate_physics`` and ``filter_collisions`` from
  ``InteractiveSceneCfg`` to its nested ``CloneCfg``. Configure either policy through
  ``scene.clone_cfg``. ``replicate()`` now applies collision filtering after clone dispatch.
* Changed clone planning to materialize the complete USD scene only when the cfg-resolved renderer
  or visualizer requires a USD stage. Installing Kit alone no longer causes a USD copy.
* **Breaking:** Stopped ``CameraCfg`` from probing the stage or redirecting a body path to a
  ``/camera`` child. Declare the exact camera child ``prim_path`` and set ``spawn=None`` for a
  preauthored camera.
* **Breaking:** Changed ``RayCasterCfg.spawn`` from ``SensorFrameCfg()`` to ``None``. Set
  ``SensorFrameCfg`` explicitly only when the sensor cfg owns a new Xform.

Removed
^^^^^^^

* **Breaking:** Removed the public replication queue and legacy plan helpers:
  ``REPLICATION_QUEUE``, ``queue_replication()``, ``add()``, ``make_clone_plan()``,
  ``make_valid_clone_combinations()``, ``num_spawn_variants()``, ``random()``, and ``sequential()``.
  Use a declarative ``InteractiveScene`` or ``clone_plan_from_env_0()`` followed by ``replicate()``
  for a homogeneous prototype; low-level clone-context callers pass the active plan to
  ``context.replicate(plan)``. Raw ``usd_replicate()`` remains available for low-level tooling and
  the ``InteractiveScene`` environment-root bootstrap. ``disabled_fabric_change_notifies()`` is now
  internal.
* **Breaking:** Removed ``ReplicateSession.plan`` and its ``device``, ``global_paths``,
  ``clone_strategy``, ``valid_set``, ``replicate_physics``, and ``env_template`` arguments. Read the
  plan from ``SimulationContext`` and configure cloning through the sole nested ``CloneCfg``.
* **Breaking:** Removed ``CloneCfg.clone_strategy`` and ``CloneCfg.device``;
  ``InteractiveSceneCfg.clone_in_fabric``; the public ``grid_transforms()`` and
  ``filter_collisions()`` helpers; and spawner ``spawn_path``, ``spawn_paths``, and
  ``random_choice``. Declare variant combinations with ``CloneCfg.clone_combinations`` and
  ``InclusionSet.weight``, route cfg rows with ``cloning_contexts``, and pass exact path sequences
  directly to multi-asset spawner functions. Use ``grid_positions()`` when only a standalone grid
  layout is needed; cloning consumers read positions from ``ClonePlan.positions`` or
  ``InteractiveScene.env_origins``.
* **Breaking:** Removed ``AssetBaseCfg._post_spawn()``. Asset implementations that author
  additional schemas after spawning now do so in their runtime constructor after
  ``super().__init__()``, once for every exact path in ``self._source_prim_paths``.
