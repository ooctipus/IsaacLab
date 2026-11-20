Added
^^^^^

* Added :class:`~isaaclab_newton.physics.FeatherPGSSolverCfg` and
  :class:`~isaaclab_newton.physics.FeatherPGSSolver` for Newton's
  reduced-coordinate FeatherPGS solver.

Changed
^^^^^^^

* Changed the default FeatherPGS position solve from twelve to eight iterations
  so task families share the measured large-batch baseline. Set
  ``pgs_iterations=12`` explicitly to retain the previous behavior.
* Changed the default FeatherPGS position-correction factor from ``0.2`` to
  ``0.05`` for stable shared task presets. Set ``pgs_beta=0.2`` explicitly to
  retain the previous correction strength.

Fixed
^^^^^

* Fixed the default FeatherPGS joint-limit activation gap to avoid allocating
  inactive lower and upper limit rows on every simulation step.
