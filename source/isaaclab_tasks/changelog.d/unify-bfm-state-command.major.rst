Changed
^^^^^^^

* Made preset resolution in :mod:`isaaclab_tasks.utils.hydra` traverse tuples and copy
  selected preset values, so nested :class:`PresetCfg` instances inside tuple fields
  (for example task-table ``families``) resolve without mutating the shared templates.
* Added an optional ``generator`` argument to
  :func:`isaaclab_tasks.core.multi_task.utils.grid_downsample.grid_bucket_downsample` and
  to the morphological flat-patch sampler so table construction consumes one explicit
  random stream.
