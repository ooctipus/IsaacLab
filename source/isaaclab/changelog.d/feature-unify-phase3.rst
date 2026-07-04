Added
^^^^^

* Added a ``processed`` option to :func:`~isaaclab.envs.mdp.last_action` for
  observing a named action term after its configured transformation.
* Added support for :func:`~isaaclab.utils.configclass` classes to preserve an
  explicit or inherited custom ``to_dict`` serializer.
* Added arbitrary leading-batch support to
  :func:`~isaaclab.utils.math.quat_mul`,
  :func:`~isaaclab.utils.math.quat_apply`, and
  :func:`~isaaclab.utils.math.quat_apply_inverse`, plus batched tensor
  interpolation fractions to :func:`~isaaclab.utils.math.quat_slerp`.

Fixed
^^^^^

* Fixed multi-body inputs to
