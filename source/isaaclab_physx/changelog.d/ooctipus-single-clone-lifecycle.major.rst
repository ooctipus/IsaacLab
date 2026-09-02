Changed
^^^^^^^

* Registered one shared USD scene context for the single cfg-owned clone lifecycle. It authors
  destination topology once before native PhysX registration, and renderers and visualizers reuse
  the same context.

Fixed
^^^^^

* Fixed rigid-object and rigid-object-collection resets to honor Warp environment masks when
  clearing external wrenches.
