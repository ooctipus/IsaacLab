Changed
^^^^^^^

* Keyboard selections consume Newton's prepared ``MuJoCoModelMapping`` instead
  of reconstructing private solver indices. Scalar-control and root-frame
  restrictions remain task-owned. Contact reductions use public MJWarp operations.
* Replaced task-local domain aliases with ``Model.AttributeFrequency``. Paths
  still resolve to integer indices before numeric selection operations.
