Added
^^^^^

* Added :class:`~isaaclab_newton.physics.NewtonDebugCaptureCfg` and
  :class:`~isaaclab_newton.physics.NewtonDebugReplayCfg` for strict,
  schema-driven incident capture with focused failed-world artifacts, workflow
  context, transition replay, applied-control evidence, operation providers,
  and the unified
  ``scripts/tools/physics_debug.py`` archive tool.
* Added :class:`~isaaclab_newton.physics.MJWarpDebugOperationProvider` to
  capture auto-discovered transient solver and collision contexts, pre-solve
  data, and first-non-finite iteration evidence without modifying the installed
  MuJoCo Warp package.
* Added opt-in :attr:`~isaaclab_newton.physics.NewtonCfg.solver_reset` to clear
  solver-owned state after task-authored environment writes, independently of
  physics incident capture. MJWarp clears selected worlds, while CPU MuJoCo
  requires a single-world simulation.

Changed
^^^^^^^

* **Breaking:** Replaced ``NewtonCfg.nan_replay``, ``NanReplayCfg``, and
  ``ReplayBufferCfg`` with
  :attr:`~isaaclab_newton.physics.NewtonCfg.debug_capture`,
  :class:`~isaaclab_newton.physics.NewtonDebugCaptureCfg`, and
  :class:`~isaaclab_newton.physics.NewtonDebugReplayCfg`. Migrate
  ``buffer_size`` to ``history_length``, ``export_path`` to ``output_dir``,
  ``export_envs_only`` to ``failed_worlds_only``, ``max_exports`` to
  ``max_incidents``, ``per_substep`` to ``capture_per_substep``, and
  ``record_mjwarp_context`` to ``record_operations``; configure the shared
  memory limit with
  :attr:`~isaaclab_newton.physics.NewtonDebugCaptureCfg.max_gpu_bytes`.
