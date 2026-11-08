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
