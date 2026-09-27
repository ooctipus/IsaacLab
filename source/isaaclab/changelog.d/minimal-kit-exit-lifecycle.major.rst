Changed
^^^^^^^

* **Breaking:** :func:`~isaaclab.app.launch_simulation` alone starts the simulation runtime, using the
  launcher each resolved config names in ``launcher_type``. ``isaaclab.app.AppLauncher`` is now a no-op kept
  so scripts still import; migrate to :func:`~isaaclab.app.add_launcher_args` plus
  ``with launch_simulation(cfg, args_cli):``.
* Changed Kit exit handling: an unhandled exception exits with 1 and ``SIGINT`` raises
  :class:`KeyboardInterrupt`; ``SIGTERM``, ``SIGABRT``, and ``SIGSEGV`` keep their default actions.

Added
^^^^^

* Added :class:`~isaaclab.app.SimulationLauncher` for backend runtimes and
  :func:`isaaclab.test.utils.launch_test_simulation` to start Kit in test modules.
