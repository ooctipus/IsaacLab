Added
^^^^^

* Added :class:`~isaaclab.sim.PlaneCfg` and :func:`~isaaclab.sim.spawn_plane`
  for local solver-common collision planes without an external USD asset.
* Added :func:`~isaaclab.utils.math.quat_from_rotation_vector` for stable
  rotation-vector conversion.

Fixed
^^^^^

* Fixed USD schema writers to preserve declared or explicitly provided array
  types when setting existing or absent attributes.
