Changed
^^^^^^^

* **Breaking:** :func:`~isaaclab.app.launch_simulation` alone starts the simulation runtime, using the
  launcher each resolved config names in ``launcher_type``. ``isaaclab.app.AppLauncher`` is now a no-op kept
  so scripts still import; migrate to :func:`~isaaclab.app.add_launcher_args` plus
  ``with launch_simulation(cfg, args_cli):``.
* Changed Kit exit handling: an unhandled exception exits with 1 and ``SIGINT`` raises
  :class:`KeyboardInterrupt`; ``SIGTERM``, ``SIGABRT``, and ``SIGSEGV`` keep their default actions.
* **Breaking:** Removed ``SettingsManager.initialize_carb_settings``, the module-level
  ``initialize_carb_settings``, and ``SettingsManager.is_omniverse_mode``. The Kit launcher now passes
  ``carb.settings`` to :meth:`~isaaclab.app.SettingsManager.set_backend`.
* **Breaking:** Removed ``isaaclab.sim.utils.is_current_stage_in_memory`` and the hidden ``--cpu`` launcher
  argument. Use ``--device cpu`` instead of ``--cpu``.
* Changed the environments to seed Replicator through a hook the Kit launcher registers with
  :func:`~isaaclab.utils.seed.register_seed_hook`, so core modules no longer import ``omni.replicator``.
* Changed the ``run_usd_camera`` and ``run_ray_caster_camera`` tutorials to save images as PNG files with
  :func:`~isaaclab.utils.save_images_to_file` instead of Replicator writers.
* Removed the ``check_*`` scripts under the core and PhysX test folders, which duplicated pytest coverage.

Added
^^^^^

* Added :class:`~isaaclab.app.SimulationLauncher` for backend runtimes and
  :func:`isaaclab.test.utils.launch_test_simulation` to start Kit in test modules.
* Added :func:`~isaaclab.utils.seed.register_seed_hook` so a runtime can seed its own random number generators.
* Added ``class_type`` to :class:`~isaaclab.envs.ManagerBasedEnvCfg` and :class:`~isaaclab.envs.ManagerBasedRLEnvCfg`,
  so ``env_cfg.class_type(env_cfg)`` constructs the environment without importing its class.

Fixed
^^^^^

* Fixed ``{DIR}`` in an inherited ``class_type`` resolving against the module of a config subclass that is
  missing ``@configclass``, which made ``cfg.class_type(cfg)`` fail with ``ModuleNotFoundError``.
