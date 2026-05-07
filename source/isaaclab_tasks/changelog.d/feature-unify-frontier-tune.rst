Changed
^^^^^^^

* Retuned the terrain ``frontier*`` curriculum presets to widen the frontier band (``dilation_steps`` 1 → 2) and shrink the soft floor (``eps`` 1e-3 → 1e-6). Diagnostic logs showed the previous configuration putting ~65% of sampling mass on already-mastered states via ``eps``; the wider frontier dilation routes that mass back to the actual learning edge while the small ``eps`` retains a long-timescale refresh on stale states.
