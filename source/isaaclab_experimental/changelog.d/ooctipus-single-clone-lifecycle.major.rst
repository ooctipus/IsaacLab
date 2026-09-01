Changed
^^^^^^^

* **Breaking:** Made experimental manager-based environments construct declarative scene cfgs through
  the scene-owned clone lifecycle. Homogeneous direct environments now keep asset cfgs on the direct
  env cfg. Their base publishes the cfg-derived plan before ``_setup_scene()`` constructs the
  prototype, then dispatches that same plan. Custom environment roots must supply all clone-owned
  cfgs before constructing assets.
