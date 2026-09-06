Added
^^^^^

* Added a ``--visualizer`` selector to the shared task-table inspector
  ``core/multi_task/scripts/inspect_task_table.py`` choosing between the ``viser`` web viewer
  (default) and the ``newton_gl`` OpenGL window. Static tables are now re-submitted at a low
  rate so windowed viewers keep processing input until closed.
* Made the inspector show the complete table: static tables draw every referenced reset
  state exactly once at its table-owned placement, and timed tables loop every sequence.
  The former sixteen-sequence display limit and the ``sequence_limit`` argument of
  ``StateCommandCfg.TaskTableCfg.build_inspection_view`` were removed; table density is
  controlled by the task configuration, for example
  ``env.commands.goal_point.task_table.pool_spacing`` for Position.
