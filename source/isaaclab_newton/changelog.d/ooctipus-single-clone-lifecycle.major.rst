Changed
^^^^^^^

* Deferred Newton model construction to the first hard reset so it includes intervening stage edits.
  With physics replication disabled, Newton imports exact per-environment paths from the published
  plan instead of rebuilding scene ownership by walking the completed USD stage.

Fixed
^^^^^

* Fixed rigid-object and rigid-object-collection resets to honor Warp environment masks when
  clearing external wrenches.
