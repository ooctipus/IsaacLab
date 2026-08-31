Changed
^^^^^^^

* **Breaking:** Moved direct-task asset declarations into their scene cfgs so each environment has
  one cfg-owned clone lifecycle. Custom direct tasks should declare assets on ``scene`` and use
  ``_setup_scene()`` only to bind the constructed scene entities.
