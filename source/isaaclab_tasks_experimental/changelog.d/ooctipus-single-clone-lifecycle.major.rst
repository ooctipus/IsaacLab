Changed
^^^^^^^

* **Breaking:** Moved homogeneous experimental direct-task asset declarations onto their direct env
  cfgs. The task's ``_setup_scene()`` publishes the cfg-derived plan before asset construction and
  dispatches it afterward. Register runtime entities on the plain scene, or put them on a declarative
  scene cfg when the workflow is heterogeneous.
