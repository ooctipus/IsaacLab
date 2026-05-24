Added
^^^^^

* Added :func:`~isaaclab.cloner.cloner_utils.source_prim`,
  :func:`~isaaclab.cloner.cloner_utils.descend_source_prims`,
  :func:`~isaaclab.cloner.cloner_utils.ascend_source_prims`, and
  :func:`~isaaclab.cloner.cloner_utils.expand_clone_plan_paths` to resolve sensor
  and asset topology source-side from the active :class:`~isaaclab.cloner.ClonePlan`.

Changed
^^^^^^^

* **Breaking:** :class:`~isaaclab.sensors.SensorBase` and asset post-spawn validation
  now require an active :class:`~isaaclab.cloner.ClonePlan`. Sensors and assets
  constructed outside :class:`~isaaclab.scene.InteractiveScene` must publish a
  covering plan via :meth:`~isaaclab.sim.SimulationContext.set_clone_plan` before
  initialization; destination-prim wildcard discovery is no longer a fallback.
* Per-env world poses (raycaster static path, camera helpers, VBD managers) now
  compose :attr:`~isaaclab.cloner.ClonePlan.env_pose` with source-relative poses
  instead of reading destination prims.

Fixed
^^^^^

* Fixed :func:`~isaaclab.cloner.cloner_utils.iter_clone_plan_matches` to yield
  every plan entry that covers a queried path (removed nearest-template filter).
