Fixed
^^^^^

* Fixed :func:`~isaaclab.utils.dict.class_to_dict` and
  :func:`~isaaclab.utils.dict.update_class_from_dict` choking on dictionaries
  with non-string keys. ``class_to_dict``'s ``"__"``-prefix filter now
  guards on ``isinstance(key, str)``, and ``update_class_from_dict``'s
  namespace path concatenation casts keys via ``str()``. Without these
  guards a ``RewardTermCfg.params`` entry that maps ``int`` task indices
  to per-task values (used by manager-based multi-task envs) raised
  ``AttributeError`` / ``TypeError`` during Hydra serialisation.
