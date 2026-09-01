Changed
^^^^^^^

* **Breaking:** Made ``PhysxManager`` register one simulation-owned native PhysX clone context and
  one shared USD scene context. Native PhysX replication requires plan-authored destination topology,
  so the USD context copies that topology once before the native context runs; renderers and
  visualizers reuse the same registered USD context. Removed ``PHYSICS_CONTEXT`` and
  ``physx_replicate()``; put assets on a declarative scene cfg or homogeneous direct env cfg instead
  of calling a PhysX replication entry point directly. Low-level callers now pass the published plan
  to ``PhysxReplicateContext.replicate(plan)`` once; ``queue()`` and ``queue_mapping()`` were removed.

Fixed
^^^^^

* Fixed rigid-object and rigid-object-collection resets to honor Warp environment masks when
  clearing external wrenches.
