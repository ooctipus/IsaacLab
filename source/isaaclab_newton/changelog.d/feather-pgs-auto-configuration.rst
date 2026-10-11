Added
^^^^^

* Added automatic collision and native sparse-factor choices to ``FeatherPGSSolverCfg``
  and connected its kernel overrides to Newton. Collision defaults to ``auto``;
  set ``convex_collision="ordinary"`` to retain ordinary queries, ``"rejection"``
  to force screening, or ``"convex_cells"`` to retain the existing cell method.
  These fields override legacy collision-selection environment variables.

Changed
^^^^^^^

* Changed ``pgs_kernel`` to default to ``auto``. Explicit unsupported kernels now
  raise instead of silently selecting another implementation; use ``auto`` for
  CPU portability and for matrix-free sweeps. Added
  ``NewtonFeatherPGSManager.get_path_selection()`` for inspecting the selections.
