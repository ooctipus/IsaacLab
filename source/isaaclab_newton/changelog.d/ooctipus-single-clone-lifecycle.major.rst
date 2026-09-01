Changed
^^^^^^^

* **Breaking:** Made ``NewtonManager`` register one simulation-owned
  ``NewtonReplicateContext`` that consumes the active clone plan. Removed ``PHYSICS_CONTEXT`` and
  ``newton_physics_replicate()``; put assets on a declarative scene cfg or homogeneous direct env cfg
  instead of calling a Newton replication entry point directly. Low-level callers now construct the
  context with the simulation and pass the published plan to ``replicate(plan)`` once;
  ``queue_mapping()`` was removed. The first hard reset now builds the model after intervening stage
  edits. With physics replication disabled it imports exact per-environment paths derived from the
  plan instead of rebuilding scene ownership by walking the completed USD stage.
