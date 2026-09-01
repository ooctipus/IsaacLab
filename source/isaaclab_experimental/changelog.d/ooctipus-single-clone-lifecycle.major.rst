Changed
^^^^^^^

* **Breaking:** Made experimental manager-based environments construct declarative scene cfgs through
  the scene-owned clone lifecycle. Homogeneous direct environments now keep asset cfgs on the direct
  env cfg and construct their prototype in ``_setup_scene()`` inside the base-owned ``from_env_0``
  lifecycle. Custom environment roots must supply all clone-owned cfgs before constructing assets.
