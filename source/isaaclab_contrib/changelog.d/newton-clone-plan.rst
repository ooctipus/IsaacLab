Changed
^^^^^^^

* Changed contrib visuotactile sensor and deformable-object hooks to resolve
  source prims through the :class:`~isaaclab.cloner.ClonePlan` instead of
  walking destination prims or substituting ``env_0`` strings.
* VBD-manager Fabric particle sync now uses
  :func:`~isaaclab.cloner.cloner_utils.expand_clone_plan_paths` for vis-mesh
  resolution.
