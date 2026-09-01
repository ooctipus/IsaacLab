Changed
^^^^^^^

* **Breaking:** Moved homogeneous direct-task asset declarations onto their direct env cfgs. The
  direct environment base publishes the cfg-derived plan before ``_setup_scene()`` and dispatches it
  afterward; construct asset prototypes with ``cfg.class_type(cfg)`` in ``_setup_scene()`` and
  register runtime entities on the plain scene. Use a scene subclass when the workflow is declarative
  or heterogeneous.
