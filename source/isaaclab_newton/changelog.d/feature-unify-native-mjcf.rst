Added
^^^^^

* Added :class:`~isaaclab_newton.sim.NewtonMjcfFileCfg` for loading local MJCF
  assets directly into Newton model builders without Kit-based USD conversion.

Fixed
^^^^^

* Fixed native MJCF equality constraints so MJWarp preserves one solver-owned
  row per world without duplicate loop-joint conversion.
