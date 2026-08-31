Changed
^^^^^^^

* **Breaking:** Made ``PhysxManager`` register one simulation-owned PhysX clone context and one
  shared USD clone context. Removed ``PHYSICS_CONTEXT`` and ``physx_replicate()``; construct assets
  inside ``ReplicateSession`` instead of calling a PhysX replication entry point directly.
  Low-level callers now pass the complete mapping to ``PhysxReplicateContext.replicate(...)`` once;
  ``queue()`` and ``queue_mapping()`` were removed.
