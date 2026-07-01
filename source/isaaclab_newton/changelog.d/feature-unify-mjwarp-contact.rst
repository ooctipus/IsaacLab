Added
^^^^^

* Added :class:`~isaaclab_newton.sim.schemas.MujocoCollisionPropertiesCfg`
  to author raw MuJoCo geometry contact parameters through ``MjcCollisionAPI``.
* Added :attr:`~isaaclab_newton.physics.MJWarpSolverCfg.enable_multiccd` to expose
  native MuJoCo Warp multi-contact generation.
* Added :attr:`~isaaclab_newton.physics.MJWarpSolverCfg.enable_native_ccd` to
  select the margin-compatible primitive box collision path when native CCD is
  disabled.

Fixed
^^^^^

* Fixed MJWarp model construction to retain authored MuJoCo contact parameters
  through USD import and environment replication.
* Fixed Newton environment cloning duplicating MuJoCo actuator and equality
  rows from the source template in replicated models.
* Fixed MuJoCo contact arrays to retain their declared USD storage type when
  an applied schema token has no registered attribute definition.
