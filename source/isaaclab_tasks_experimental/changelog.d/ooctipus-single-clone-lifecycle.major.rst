Changed
^^^^^^^

* **Breaking:** Moved homogeneous experimental direct-task asset declarations onto their direct env
  cfgs. The direct environment base owns the ``from_env_0`` clone lifecycle; construct asset
  prototypes in ``_setup_scene()`` and register runtime entities on the plain scene. Use a scene
  subclass when the workflow is declarative or heterogeneous.
