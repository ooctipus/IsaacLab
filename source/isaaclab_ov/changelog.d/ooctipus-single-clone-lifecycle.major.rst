Changed
^^^^^^^

* **Breaking:** Made ``OvPhysxManager`` register one simulation-owned ``OvReplicateContext`` that
  consumes the active clone plan. Renamed ``OvPhysxReplicateContext`` and removed
  ``PHYSICS_CONTEXT`` and ``ovphysx_replicate()``; put assets on a declarative scene cfg or
  homogeneous direct env cfg instead of calling an OvPhysX replication entry point directly.
  Low-level callers now construct the context with the simulation and pass the published plan to
  ``replicate(plan)`` once; ``queue()`` and ``queue_mapping()`` were removed.
