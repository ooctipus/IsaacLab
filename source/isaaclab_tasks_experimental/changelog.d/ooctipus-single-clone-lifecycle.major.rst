Changed
^^^^^^^

* **Breaking:** Moved experimental direct-task assets into their scene cfgs for the single clone
  lifecycle. Custom tasks should declare assets on ``scene`` and use ``_setup_scene()`` only to
  bind the constructed entities.
