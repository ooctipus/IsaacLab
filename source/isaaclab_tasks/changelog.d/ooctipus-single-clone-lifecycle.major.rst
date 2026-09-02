Changed
^^^^^^^

* **Breaking:** Moved homogeneous direct-task asset declarations onto their direct env cfgs. The
  task's ``_setup_scene()`` publishes the cfg-derived plan before asset construction and dispatches it
  afterward. Construct asset prototypes with ``cfg.class_type(cfg)`` and register runtime entities on
  the plain scene. Put entities on a declarative scene cfg when the workflow is heterogeneous.
