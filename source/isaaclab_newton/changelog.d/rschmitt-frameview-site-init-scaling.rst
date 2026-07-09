Fixed
^^^^^

* Fixed quadratic (``O(num_envs^2)``) startup scaling in
  :class:`~isaaclab_newton.sim.views.NewtonSiteFrameView` when a frame resolves to a
  per-environment body path (e.g. a body-mounted camera). Body-anchored frames are now
  resolved structurally from per-source body offsets recorded during replication, so
  pattern matching stays confined to the source builders and the fully cloned body-label
  list is never scanned; frames without replication metadata fall back to an exact label
  lookup. This reduces simulation-start time for camera-heavy scenes at high environment
  counts (8192 environments dropped from ~29 min to seconds).
