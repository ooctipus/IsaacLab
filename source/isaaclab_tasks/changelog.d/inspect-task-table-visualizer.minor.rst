Added
^^^^^

* Added a ``--visualizer`` selector to the shared task-table inspector
  ``core/multi_task/scripts/inspect_task_table.py`` choosing between the ``viser`` web viewer
  (default) and the ``newton_gl`` OpenGL window. Static tables are now re-submitted at a low
  rate so windowed viewers keep processing input until closed.
