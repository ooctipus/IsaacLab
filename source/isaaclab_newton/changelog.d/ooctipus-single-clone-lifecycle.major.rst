Changed
^^^^^^^

* **Breaking:** Made ``NewtonManager`` register one simulation-owned
  ``NewtonReplicateContext`` that consumes the active clone plan. Removed ``PHYSICS_CONTEXT`` and
  ``newton_physics_replicate()``; construct assets inside ``ReplicateSession`` instead of calling
  a Newton replication entry point directly. Low-level callers now construct the context with the
  simulation and pass the complete mapping to ``replicate(...)`` once; ``queue_mapping()`` was
  removed.
