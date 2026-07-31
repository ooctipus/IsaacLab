Added
^^^^^

* Added runtime texture swapping to the Newton viewer: declared texture pools are registered with
  the viewer's GPU texture array once at initialization, and ``texture``-channel writes from
  :class:`~isaaclab.assets.VisualMaterial` update the affected shape instances per reset with no
  decode or re-upload. Requires a Newton build with pooled viewer texture swapping; older builds
  warn once and skip texture updates.
