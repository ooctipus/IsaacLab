Removed
^^^^^^^

* Removed :attr:`~isaaclab_newton.physics.FeatherPGSSolverCfg.grouped_dynamics` and
  :attr:`~isaaclab_newton.physics.FeatherPGSSolverCfg.lazy_kinematics`. FeatherPGS chooses these paths from the
  scene data and no longer takes the options; delete them from your configurations.
