Added
^^^^^

* Added automatic collision choices to ``FeatherPGSSolverCfg``. Collision
  defaults to ``auto``; set ``convex_collision="ordinary"`` to retain ordinary
  queries, ``"rejection"`` to force screening, or ``"convex_cells"`` to retain
  the existing cell method. These fields override legacy collision-selection
  environment variables. Added :meth:`~isaaclab_newton.physics.FeatherPGSSolver.path_selection`
  for inspecting the selections.
